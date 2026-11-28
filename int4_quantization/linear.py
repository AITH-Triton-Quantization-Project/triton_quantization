"""Линейный слой с весами INT4.

Вес ``nn.Linear`` лежит как ``[out, in]``. Квантуется ``weight.T``, то есть
вдоль входной оси. ``backend="triton"`` зовёт кернели, ``"reference"`` — эталон.
"""

from __future__ import annotations

import torch
from torch import nn

from int4_quantization.reference import (
    GROUP_SIZE,
    matmul_reference,
    quantize_weight_reference,
)


class QuantizedLinear(nn.Module):
    """INT4-веса без bias. Коды в 4 раза меньше той же матрицы в fp16."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int = GROUP_SIZE,
        backend: str = "reference",
    ) -> None:
        super().__init__()
        if backend not in ("reference", "triton"):
            raise ValueError(f"unsupported backend {backend!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.backend = backend
        self.register_buffer("q", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = GROUP_SIZE,
        backend: str = "triton",
    ) -> QuantizedLinear:
        """Строит слой из ``nn.Linear``, квантуя ``weight.T``.

        Args:
            linear: Исходный слой без bias. Вес имеет форму ``[out, in]``.
            group_size: Размер группы вдоль входной оси.
            backend: ``"triton"`` для кернелей или ``"reference"`` для PyTorch.

        Returns:
            Квантованный слой с логической формой весов ``[in, out]``.

        Raises:
            TypeError: Если передан не ``nn.Linear``.
            ValueError: Если у слоя есть bias, форма веса неожиданная или
                backend неизвестен.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
        if linear.bias is not None:
            raise ValueError("QuantizedLinear supports bias-free nn.Linear only")
        if backend not in ("reference", "triton"):
            raise ValueError(f"unsupported backend {backend!r}")

        out_features, in_features = linear.weight.shape
        if (in_features, out_features) != (linear.in_features, linear.out_features):
            raise ValueError(
                f"weight shape {tuple(linear.weight.shape)} does not match "
                f"in_features={linear.in_features}, out_features={linear.out_features}"
            )

        weight_kn = linear.weight.detach().T.contiguous()
        if backend == "triton":
            from int4_quantization.quantize import quantize_weight

            quantized = quantize_weight(weight_kn, group_size=group_size)
            packed, scales, zeros = quantized.q, quantized.scales, quantized.zeros
        else:
            packed, scales, zeros = quantize_weight_reference(weight_kn, group_size)

        return cls(
            in_features,
            out_features,
            packed,
            scales,
            zeros,
            group_size,
            backend,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Считает ``inputs @ W4`` для входа формы ``[..., in_features]``.

        Returns:
            Тензор формы ``[..., out_features]``. Вход ``float16`` или
            ``bfloat16`` сохраняет свой dtype, остальные входы дают ``bfloat16``.
        """
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dim {self.in_features}, got {inputs.shape[-1]}"
            )
        out_dtype = (
            inputs.dtype
            if inputs.dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
        )
        leading = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features)
        if self.backend == "reference":
            product = matmul_reference(
                flat,
                self.q,
                self.scales,
                self.zeros,
                self.group_size,
                out_dtype=out_dtype,
            )
        else:
            from int4_quantization.matmul import matmul
            from int4_quantization.quantize import QuantizedWeight

            quantized = QuantizedWeight(
                q=self.q,
                scales=self.scales,
                zeros=self.zeros,
                group_size=self.group_size,
                orig_dtype=torch.bfloat16,
            )
            product = matmul(flat, quantized, kind="w4")
            if product.dtype != out_dtype:
                product = product.to(dtype=out_dtype)
        return product.reshape(*leading, self.out_features)
