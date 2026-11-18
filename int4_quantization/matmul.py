import torch
import triton
import triton.language as tl

from int4_quantization.kernels import QuantizedWeight

# общие константы тайлинга
_BLOCK_M, _BLOCK_N, _BLOCK_K, _GROUP_M = 64, 64, 64, 8


@triton.jit
def _w4_gemm_kernel(
    x_ptr, wq_ptr, s_ptr, z_ptr, out_ptr,
    M, N, K,
    sxm, sxk, swqk, swqn, ssg, ssn, szg, szn, som, son,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Выполнение GEMM с квантованными четырехбитными весами.

    Активируемый тензор загружается блоками, упакованные веса распаковываются
    из младшего и старшего ниблов, к ним применяются масштаб и нулевая точка,
    после чего произведение накапливается в float32.

    Args:
        x_ptr: Указатель на входной тензор X формы ``[M, K]``.
        wq_ptr: Указатель на упакованные веса int8 формы ``[K // 2, N]``.
        s_ptr: Указатель на масштабы весов по группам оси K.
        z_ptr: Указатель на нулевые точки весов по группам оси K.
        out_ptr: Указатель на выходной тензор формы ``[M, N]``.
        M: Количество строк X.
        N: Количество столбцов весов и выходной матрицы.
        K: Количество столбцов X и строк исходной весовой матрицы.
        sxm: Шаг X по строкам.
        sxk: Шаг X по столбцам.
        swqk: Шаг упакованных весов по строкам.
        swqn: Шаг упакованных весов по столбцам.
        ssg: Шаг масштабов по группам.
        ssn: Шаг масштабов по столбцам.
        szg: Шаг нулевых точек по группам.
        szn: Шаг нулевых точек по столбцам.
        som: Шаг выходного тензора по строкам.
        son: Шаг выходного тензора по столбцам.
        GROUP_SIZE: Количество элементов оси K в одной группе.
        BLOCK_M: Размер блока по строкам.
        BLOCK_N: Размер блока по столбцам.
        BLOCK_K: Размер блока по оси накопления.
        GROUP_M: Количество блоков по M в одной группе расписания.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    BLOCK_KH: tl.constexpr = BLOCK_K // 2
    offs_kh = tl.arange(0, BLOCK_KH)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):

        a = tl.load(
            x_ptr + offs_m[:, None] * sxm + (k0 + offs_k)[None, :] * sxk,
            mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
            other=0.0,
        )
        a = a.to(tl.bfloat16)   # страховка, если X пришёл fp16

        wq = tl.load(
            wq_ptr + (k0 // 2 + offs_kh)[:, None] * swqk + offs_n[None, :] * swqn,
            mask=((k0 // 2 + offs_kh)[:, None] < (K // 2)) & (offs_n[None, :] < N),
            other=0,
        )
        w_lo = (wq & 0x0F).to(tl.int32)
        w_hi = ((wq >> 4) & 0x0F).to(tl.int32)

        g = k0 // GROUP_SIZE
        s = tl.load(s_ptr + g * ssg + offs_n * ssn,
                    mask=offs_n < N, other=1.0).to(tl.bfloat16)
        z = tl.load(z_ptr + g * szg + offs_n * szn,
                    mask=offs_n < N, other=0.0).to(tl.bfloat16)

        w_lo_bf = (w_lo.to(tl.bfloat16) - z[None, :]) * s[None, :]
        w_hi_bf = (w_hi.to(tl.bfloat16) - z[None, :]) * s[None, :]

        w_bf = tl.reshape(tl.join(w_lo_bf, w_hi_bf), (BLOCK_K, BLOCK_N))
        acc += tl.dot(a, w_bf, out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _w16_gemm_kernel():
    """Объявляет зарезервированный Triton-ядра для GEMM с весами W16.

    В текущей реализации вычисление не выполняется.
    """
    pass


def _grid(M, N):
    """Вычисляет одномерный размер сетки для GEMM.

    Args:
        M: Количество строк левой матрицы.
        N: Количество столбцов правой матрицы.

    Returns:
        Кортеж из одного элемента с общим количеством блоков.
    """
    return (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)

def _launch_w16_gemm(X, W, out):
    """Запускает зарезервированную реализацию GEMM с весами W16.

    Args:
        X: Входной тензор формы ``[M, K]``.
        W: Весовая матрица формы ``[K, N]``.
        out: Выходной тензор формы ``[M, N]``.
    """
    M, K = X.shape; _, N = W.shape
    _w16_gemm_kernel[_grid(M, N)](
        X, W, out, M, N, K,
        X.stride(0), X.stride(1), W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, GROUP_M=_GROUP_M,
    )

def _launch_w4_gemm(X, W: QuantizedWeight, out):
    """Запускает ядро Triton для умножения на квантованные веса W4.

    Args:
        X: Входной тензор формы ``[M, K]``.
        W: Объект с упакованными весами и параметрами квантизации.
        out: Выходной тензор формы ``[M, N]``.

    Raises:
        AssertionError: Если размер упакованной оси не соответствует
            размеру оси K входной матрицы.
    """
    M, K = X.shape
    Kh, N = W.q.shape
    assert Kh * 2 == K

    _w4_gemm_kernel[_grid(M, N)](
        X, W.q, W.scales, W.zeros, out, M, N, K,
        X.stride(0), X.stride(1),
        W.q.stride(0),      W.q.stride(1),
        W.scales.stride(0), W.scales.stride(1),
        W.zeros.stride(0),  W.zeros.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=W.group_size,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, GROUP_M=_GROUP_M,
    )


def matmul(X: torch.Tensor, W, kind: str, out_dtype=torch.bfloat16):
    """Выполняет матричное умножение X на веса W16 или W4.

    Args:
        X: Двумерная входная матрица формы ``[M, K]``.
        W: Полная матрица весов для ``kind="w16"`` или объект
            :class:`QuantizedWeight` для ``kind="w4"``.
        kind: Тип весов: ``"w16"`` или ``"w4"``.
        out_dtype: dtype выходного тензора.

    Returns:
        Тензор формы ``[M, N]`` на том же устройстве, что и X.

    Raises:
        AssertionError: Если X не двумерен, тип весов неизвестен,
            размерности K не совпадают или для W4 передан неверный тип W.
    """
    assert X.dim() == 2 and kind in ("w16", "w4")
    M, K = X.shape
    K_w, N = W.shape
    assert K == K_w, f"K mismatch: X={K}, W={K_w}"

    out = torch.empty((M, N), dtype=out_dtype, device=X.device)

    if kind == "w16":
        W_bf = W if W.dtype == torch.bfloat16 else W.to(torch.bfloat16)
        _launch_w16_gemm(X, W_bf, out)
    else:
        assert isinstance(W, QuantizedWeight)
        _launch_w4_gemm(X, W, out)
    return out
