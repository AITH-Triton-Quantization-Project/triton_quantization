import pytest
import torch
from torch import nn

from int4_quantization.linear import QuantizedLinear
from int4_quantization.reference import (
    dequantize_weight_reference,
    quantize_weight_reference,
)

# Llama-3.2-1B-Instruct: q_proj и down_proj, формы веса nn.Linear [out, in].
LLAMA_LINEAR_SHAPES = {
    "q_proj": (2048, 2048),
    "down_proj": (2048, 8192),
}


def test_reference_packs_signed_int4_and_restores_values():
    weight = torch.tensor(
        [
            [0.0, 1.0],
            [7.0, -7.0],
        ],
        dtype=torch.float32,
    )

    packed, scales, zeros = quantize_weight_reference(weight, group_size=2)

    assert packed.shape == (1, 2)
    assert packed.dtype == torch.int8
    assert scales.shape == (1, 2)
    assert scales.dtype == torch.bfloat16
    assert torch.count_nonzero(zeros) == 0
    # scale = 1. Коды столбцов: (0, 7) и (1, -7).
    # -7 в младших четырёх битах — это 0b1001.
    assert int(packed[0, 0].item() & 0xFF) == 0x70
    assert int(packed[0, 1].item() & 0xFF) == 0x91

    restored = dequantize_weight_reference(packed, scales, zeros, group_size=2)
    expected = torch.tensor([[0.0, 1.0], [7.0, -7.0]], dtype=torch.float32)
    assert torch.equal(restored, expected)


def test_quantized_linear_rejects_bias_and_unknown_backend():
    biased = nn.Linear(128, 16, bias=True)
    with pytest.raises(ValueError):
        QuantizedLinear.from_linear(biased, backend="reference")

    with pytest.raises(ValueError):
        QuantizedLinear.from_linear(nn.Linear(128, 16, bias=False), backend="cuda")


def test_reference_preserves_float16_activation_dtype():
    linear = nn.Linear(128, 64, bias=False)
    layer = QuantizedLinear.from_linear(linear, group_size=128, backend="reference")
    output = layer(torch.randn(2, 128, dtype=torch.float16))

    assert output.dtype == torch.float16
    assert output.shape == (2, 64)


def test_llama_q_proj_and_down_proj_match_signed_reference():
    torch.manual_seed(0)
    for name, (out_features, in_features) in LLAMA_LINEAR_SHAPES.items():
        linear = nn.Linear(in_features, out_features, bias=False)
        layer = QuantizedLinear.from_linear(linear, group_size=128, backend="reference")

        assert layer.q.shape == (in_features // 2, out_features), name
        assert layer.scales.shape == (in_features // 128, out_features), name
        assert layer.q.dtype == torch.int8
        assert layer.q.nbytes * 4 == in_features * out_features * 2

        restored = dequantize_weight_reference(
            layer.q, layer.scales, layer.zeros, layer.group_size
        )
        original = (
            linear.weight.detach().T.to(dtype=torch.float16).to(dtype=torch.float32)
        )
        scale = layer.scales.to(dtype=torch.float32).repeat_interleave(128, dim=0)
        error = (restored - original).abs()
        assert torch.all(error <= 0.5 * scale + 1e-3), name

        tokens = 4
        activations = torch.randn(2, tokens, in_features)
        output = layer(activations)
        flat = activations.to(dtype=torch.bfloat16).reshape(-1, in_features)
        expected = torch.matmul(flat.float(), restored.to(dtype=torch.bfloat16).float())
        expected = expected.to(dtype=torch.bfloat16).reshape(2, tokens, out_features)

        assert output.shape == (2, tokens, out_features), name
        assert output.dtype == torch.bfloat16
        assert torch.equal(output, expected), name
        assert torch.equal(layer(activations[0]), expected[0]), name
