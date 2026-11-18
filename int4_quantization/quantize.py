from dataclasses import dataclass

import torch
import triton

from .kernels import compute_scale_kernel, quantize_pack_int4_kernel


def quantize_pack(x: torch.Tensor, group_size=128) -> (torch.Tensor, torch.Tensor):
    """Квантование FP16-матрицы и упаковывание значений в INT4.

    Args:
        x: Двумерный входной тензор dtype ``torch.float16``.
        group_size: Размер группы столбцов с общим масштабом.

    Returns:
        Кортеж ``(packed, scales)``. Упакованный тензор имеет dtype
        ``torch.int8`` и форму ``[M, N // 2]``; масштабы имеют dtype
        ``torch.float16`` и форму ``[M, N // group_size]``.

    Raises:
        AssertionError: Если входной тензор не двумерен или имеет
            неподходящий dtype и размеры.
    """
    assert x.dtype == torch.float16 and x.dim() == 2
    M, N = x.shape
    assert N % 2 == 0 and N % group_size == 0

    G = N // group_size
    scales = torch.empty((M, G), dtype=torch.float16, device=x.device)
    packed = torch.empty((M, N // 2), dtype=torch.int8, device=x.device)

    BLOCK_M, BLOCK_G = 16, 4
    grid_s = (triton.cdiv(M, BLOCK_M), triton.cdiv(G, BLOCK_G))
    compute_scale_kernel[grid_s](
        x, scales, M, N, x.stride(0), x.stride(1),
        GROUP_SIZE=group_size, BLOCK_M=BLOCK_M, BLOCK_G=BLOCK_G,
    )

    BLOCK_M, BLOCK_N = 32, 64
    grid_q = (triton.cdiv(M, BLOCK_M), triton.cdiv(N // 2, BLOCK_N))
    quantize_pack_int4_kernel[grid_q](
        x, packed, scales, M, N,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return packed, scales


@dataclass
class QuantizedWeight:
    """Контейнер для INT4-квантованных весов в групповом формате.

    Хранит упакованные 4-битные коды (по два на байт), а также параметры
    деквантования (scales/zeros), необходимые для восстановления весов
    и для fused dequant + GEMM.

    Раскладка предполагает, что квантование выполнено **вдоль оси K**
    (оси сокращения в GEMM). Каждый байт в ``q`` содержит два соседних
    значения вдоль K: младший ниббл — чётный индекс, старший — нечётный.

    Attributes:
        q (torch.Tensor): Упакованные INT4-коды, форма ``[K // 2, N]``,
            dtype ``torch.int8`` (или ``torch.uint8``). Каждый элемент —
            два INT4.
        scales (torch.Tensor): Масштабы квантования по группам, форма
            ``[K // GROUP_SIZE, N]``, dtype ``torch.bfloat16``.
            Восстановление: ``x_hat = (q - z) * s``.
        zeros (torch.Tensor): Zero-point (точка нуля) по группам, форма
            ``[K // GROUP_SIZE, N]``, dtype ``torch.bfloat16``.
            Для симметричного квантования — нули.
        group_size (int): Количество элементов вдоль K, разделяющих один
            scale/zero. По умолчанию 128.
        orig_dtype (torch.dtype): Исходный dtype весов до квантования.
            По умолчанию ``torch.bfloat16``.

    Note:
        ``zeros`` в текущей реализации всегда нулевые (симметричное
        квантование). Поле оставлено для совместимости с асимметричными
        схемами (GPTQ/AWQ) и для универсального контракта GEMM-кернела.
    """
    # int8  [K//2, N]
    q: torch.Tensor
    # bf16  [K//GROUP_SIZE, N]      
    scales: torch.Tensor
    # bf16  [K//GROUP_SIZE, N]
    zeros: torch.Tensor
    group_size: int = 128
    orig_dtype: torch.dtype = torch.bfloat16

    @property
    def shape(self):  # (K, N)
        """Возвращает логическую форму исходной весовой матрицы.

        Returns:
            Кортеж ``(K, N)``.
        """
        return (self.q.shape[0] * 2, self.q.shape[1])

    @property
    def dtype(self):
        """Возвращает dtype исходной весовой матрицы.

        Returns:
            Значение поля ``orig_dtype``.
        """
        return self.orig_dtype


def quantize_weight(W: torch.Tensor, group_size: int = 128) -> QuantizedWeight:
    """Транспонирует и квантует двумерную матрицу весов.

    Args:
        W: Двумерная матрица весов формы ``[K, N]``.
        group_size: Размер группы по оси K для общего масштаба.

    Returns:
        Объект :class:`QuantizedWeight` с логической формой ``[K, N]``.

    Raises:
        AssertionError: Если W не двумерен или его размеры не кратны
            ``group_size`` и двум.
    """
    assert W.dim() == 2
    K, N = W.shape
    assert K % group_size == 0 and K % 2 == 0

    Wt_fp16 = W.detach().T.contiguous().to(torch.float16)   # [N, K]
    packed_t, scales_t = quantize_pack(Wt_fp16, group_size=group_size)

    q = packed_t.T.contiguous()
    scales = scales_t.T.contiguous().to(torch.bfloat16)
    zeros = torch.zeros_like(scales, dtype=torch.bfloat16)

    return QuantizedWeight(
        q=q, scales=scales, zeros=zeros,
        group_size=group_size, orig_dtype=torch.bfloat16,
    )