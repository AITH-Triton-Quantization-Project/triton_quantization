import triton
import triton.language as tl


@triton.jit
def quantize_pack_int4_kernel(
    x_ptr,                 # fp16 [M, N]
    packed_ptr,            # int8  [M, N//2]
    scale_ptr,             # fp16  [M, N//GROUP_SIZE] (или int32 для субнормалей)
    M, N,
    stride_xm, stride_xn,
    stride_pm, stride_pn,
    stride_sm, stride_sn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Квантует пары fp16-значений и упаковывает их в INT4.

    Четное значение помещается в младшие четыре бита байта, нечетное —
    в старшие. Масштаб выбирается по группе столбцов элемента.

    Args:
        x_ptr: Указатель на входной тензор fp16 формы ``[M, N]``.
        packed_ptr: Указатель на выходной тензор int8 формы ``[M, N // 2]``.
        scale_ptr: Указатель на тензор масштабов.
        M: Количество строк входного тензора.
        N: Количество столбцов входного тензора.
        stride_xm: Шаг входного тензора по строкам.
        stride_xn: Шаг входного тензора по столбцам.
        stride_pm: Шаг упакованного тензора по строкам.
        stride_pn: Шаг упакованного тензора по столбцам.
        stride_sm: Шаг тензора масштабов по строкам.
        stride_sn: Шаг тензора масштабов по столбцам.
        GROUP_SIZE: Размер группы столбцов с общим масштабом.
        BLOCK_M: Количество строк, обрабатываемых одной программой.
        BLOCK_N: Количество пар столбцов в одном блоке.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    offs_pairs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n_even = offs_pairs * 2
    offs_n_odd  = offs_pairs * 2 + 1

    mask_m  = offs_m < M
    mask_ne = offs_n_even < N
    mask_no = offs_n_odd  < N

    xe = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_n_even[None, :] * stride_xn,
        mask=mask_m[:, None] & mask_ne[None, :], other=0.0,
    ).to(tl.float32)
    xo = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_n_odd[None, :] * stride_xn,
        mask=mask_m[:, None] & mask_no[None, :], other=0.0,
    ).to(tl.float32)

    # загрузка масштабов, scale имеет форму [M, N // GROUP_SIZE]
    group_idx = offs_pairs // (GROUP_SIZE // 2)   # индекс группы вдоль N
    mask_s = mask_m[:, None] & (group_idx[None, :] < (N // GROUP_SIZE + 1))
    scale = tl.load(
        scale_ptr + offs_m[:, None] * stride_sm + group_idx[None, :] * stride_sn,
        mask=mask_s, other=1.0,
    )

    qe = tl.floor(xe / scale + 0.5)
    qo = tl.floor(xo / scale + 0.5)
    qe = tl.clamp(qe, -8.0, 7.0).to(tl.int32)
    qo = tl.clamp(qo, -8.0, 7.0).to(tl.int32)

    packed = (qe & 0x0F) | ((qo & 0x0F) << 4)

    packed_i8 = packed.to(tl.int8)
    tl.store(
        packed_ptr + offs_m[:, None] * stride_pm + offs_pairs[None, :] * stride_pn,
        packed_i8,
        mask=mask_m[:, None] & (offs_pairs[None, :] < (N // 2)),
    )


@triton.jit
def compute_scale_kernel(
    x_ptr, scale_ptr,
    M, N,
    stride_xm, stride_xn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr,
):
    """Вычисляет масштабы для групп столбцов входного тензора.

    Для каждой строки и группы столбцов определяется максимум абсолютного
    значения и нормируется на верхнюю границу знакового четырехбитного
    диапазона.

    Args:
        x_ptr: Указатель на входной тензор fp16 формы ``[M, N]``.
        scale_ptr: Указатель на выходной тензор fp16 масштабов.
        M: Количество строк входного тензора.
        N: Количество столбцов входного тензора.
        stride_xm: Шаг входного тензора по строкам.
        stride_xn: Шаг входного тензора по столбцам.
        GROUP_SIZE: Количество столбцов в каждой группе.
        BLOCK_M: Количество строк, обрабатываемых одной программой.
        BLOCK_G: Количество групп столбцов в одном блоке.
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    offs_n = offs_g[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]

    mask = (offs_m[:, None, None] < M) & (offs_n[None, :, :] < N)
    x = tl.load(
        x_ptr + offs_m[:, None, None] * stride_xm + offs_n[None, :, :] * stride_xn,
        mask=mask, other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=2)
    scale = tl.maximum(amax / 7.0, 1e-5)

    mask_s = (offs_m[:, None] < M) & (offs_g[None, :] < tl.cdiv(N, GROUP_SIZE))
    # Строка scales имеет длину N // GROUP_SIZE. Лишний +1 сдвигал каждую
    # следующую строку и писал мимо буфера на больших матрицах.
    tl.store(
        scale_ptr + offs_m[:, None] * (N // GROUP_SIZE) + offs_g[None, :],
        scale.to(tl.float16),
        mask=mask_s,
    )
