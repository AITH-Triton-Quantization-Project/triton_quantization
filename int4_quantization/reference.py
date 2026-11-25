"""PyTorch-эталон симметричного INT4 вдоль оси K."""

from __future__ import annotations

import torch

GROUP_SIZE = 128
INT4_MIN = -8
INT4_MAX = 7
SCALE_EPS = 1e-5


def quantize_weight_reference(
    weight: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Квантует матрицу ``[K, N]`` и упаковывает коды в INT4.

    Args:
        weight: Двумерная матрица формы ``[K, N]``.
        group_size: Число элементов вдоль ``K`` с общим масштабом.

    Returns:
        Кортеж ``(q, scales, zeros)``. ``q`` имеет dtype ``torch.int8`` и
        форму ``[K // 2, N]``. ``scales`` и ``zeros`` имеют dtype
        ``torch.bfloat16`` и форму ``[K // group_size, N]``.

    Raises:
        ValueError: Если матрица не двумерна или ``K`` не делится на
            ``group_size`` и на 2.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [K, N], got shape {tuple(weight.shape)}")
    rows, _cols = weight.shape
    if group_size <= 0 or rows % group_size != 0 or rows % 2 != 0:
        raise ValueError(
            f"K={rows} must be divisible by group_size={group_size} and by 2"
        )

    values = weight.detach().to(dtype=torch.float16).to(dtype=torch.float32)
    groups = rows // group_size
    grouped = values.reshape(groups, group_size, -1)
    amax = grouped.abs().amax(dim=1)
    scale = torch.clamp_min(amax / INT4_MAX, SCALE_EPS)
    scale_fp16 = scale.to(dtype=torch.float16)
    codes = torch.floor(grouped / scale_fp16.to(dtype=torch.float32)[:, None, :] + 0.5)
    codes = codes.clamp(INT4_MIN, INT4_MAX).to(dtype=torch.int16).reshape(rows, -1)

    low = codes[0::2] & 0x0F
    high = codes[1::2] & 0x0F
    packed = (low | (high << 4)).to(dtype=torch.int8).contiguous()
    scales = scale_fp16.to(dtype=torch.bfloat16).contiguous()
    zeros = torch.zeros_like(scales)
    return packed, scales, zeros


def dequantize_weight_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Восстанавливает знаковые INT4-веса в ``float32`` формы ``[K, N]``.

    Args:
        packed: Упакованные коды ``int8`` формы ``[K // 2, N]``.
        scales: Масштабы ``[K // group_size, N]``.
        zeros: Нулевые точки той же формы, что и ``scales``.
        group_size: Размер группы вдоль ``K``.

    Returns:
        Тензор ``float32`` формы ``[K, N]``.
    """
    bits = packed.to(dtype=torch.int16) & 0xFF
    low = bits & 0x0F
    high = (bits >> 4) & 0x0F
    # 8..15 в ниббле — это -8..-1
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    rows = packed.shape[0] * 2
    codes = torch.empty(rows, packed.shape[1], dtype=torch.int16, device=packed.device)
    codes[0::2] = low
    codes[1::2] = high

    scale = scales.to(dtype=torch.float32).repeat_interleave(group_size, dim=0)
    zero = zeros.to(dtype=torch.float32).repeat_interleave(group_size, dim=0)
    return (codes.to(dtype=torch.float32) - zero) * scale


def matmul_reference(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = GROUP_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Считает ``inputs @ W`` после распаковки INT4.

    Активации и веса приводятся к bf16, накопление идёт во float32.

    Args:
        inputs: Активации формы ``[..., K]``.
        packed: Упакованные веса ``[K // 2, N]``.
        scales: Масштабы ``[K // group_size, N]``.
        zeros: Нулевые точки ``[K // group_size, N]``.
        group_size: Размер группы вдоль ``K``.
        out_dtype: dtype результата.

    Returns:
        Тензор формы ``[..., N]``.
    """
    leading = inputs.shape[:-1]
    features = inputs.shape[-1]
    flat = inputs.reshape(-1, features).to(dtype=torch.bfloat16)
    weight = dequantize_weight_reference(packed, scales, zeros, group_size)
    weight = weight.to(dtype=torch.bfloat16)
    product = torch.matmul(flat.float(), weight.float()).to(dtype=out_dtype)
    return product.reshape(*leading, weight.shape[1])
