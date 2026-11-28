import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="сверка Triton GEMM требует CUDA",
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().mean().clamp(min=1e-3)
    return float((difference.mean() / scale).item())


@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [
        (8, 128, 32),
        (7, 256, 3),
        (128, 2048, 512),
    ],
)
def test_triton_gemm_matches_signed_reference(rows, in_features, out_features):
    from int4_quantization.matmul import matmul
    from int4_quantization.quantize import quantize_weight
    from int4_quantization.reference import dequantize_weight_reference

    torch.manual_seed(0)
    weight = torch.randn(
        in_features, out_features, device="cuda", dtype=torch.bfloat16
    )
    activations = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)
    quantized = quantize_weight(weight, group_size=128)
    restored = dequantize_weight_reference(
        quantized.q, quantized.scales, quantized.zeros, quantized.group_size
    )
    expected = torch.matmul(activations.float(), restored).to(dtype=torch.bfloat16)

    actual = matmul(activations, quantized, kind="w4")
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert _relative_error(actual, expected) < 0.02


def test_quantized_linear_triton_matches_reference():
    from int4_quantization.linear import QuantizedLinear

    torch.manual_seed(1)
    linear = nn.Linear(256, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    triton_layer = QuantizedLinear.from_linear(linear, backend="triton")
    reference_layer = QuantizedLinear.from_linear(linear, backend="reference")
    activations = torch.randn(4, 9, 256, device="cuda", dtype=torch.bfloat16)

    actual = triton_layer(activations)
    expected = reference_layer(activations)
    torch.cuda.synchronize()

    assert actual.shape == (4, 9, 64)
    assert _relative_error(actual, expected) < 0.02
