import sys
import types

import pytest
import torch


try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_module = types.ModuleType("triton")
    triton_module.cdiv = lambda numerator, denominator: (
        numerator + denominator - 1
    ) // denominator
    triton_module.jit = lambda fn: fn

    triton_language = types.ModuleType("triton.language")
    triton_language.constexpr = object
    triton_module.language = triton_language

    sys.modules["triton"] = triton_module
    sys.modules["triton.language"] = triton_language

from int4_quantization import quantize as quantize_module
from int4_quantization.quantize import (
    QuantizedWeight,
    quantize_pack,
    quantize_weight,
)


class KernelSpy:
    def __init__(self, fill_output):
        self.fill_output = fill_output
        self.grid = None
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid
        return self.launch

    def launch(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.fill_output(args)


def test_quantize_pack_launches_kernels_and_returns_expected_shapes(monkeypatch):
    def fill_scales(args):
        scales = args[1]
        scales.fill_(0.5)

    def fill_packed(args):
        packed = args[1]
        packed.fill_(0x21)

    scale_kernel = KernelSpy(fill_scales)
    pack_kernel = KernelSpy(fill_packed)
    monkeypatch.setattr(quantize_module, "compute_scale_kernel", scale_kernel)
    monkeypatch.setattr(quantize_module, "quantize_pack_int4_kernel", pack_kernel)

    x = torch.zeros((32, 128), dtype=torch.float16)
    packed, scales = quantize_pack(x, group_size=32)

    assert packed.shape == (32, 64)
    assert packed.dtype == torch.int8
    assert packed.device == x.device
    assert torch.all(packed == 0x21)

    assert scales.shape == (32, 4)
    assert scales.dtype == torch.float16
    assert scales.device == x.device
    assert torch.all(scales == 0.5)

    assert scale_kernel.grid == (2, 1)
    assert scale_kernel.args[0] is x
    assert scale_kernel.args[1].shape == (32, 4)
    assert scale_kernel.kwargs == {
        "GROUP_SIZE": 32,
        "BLOCK_M": 16,
        "BLOCK_G": 4,
    }

    assert pack_kernel.grid == (1, 1)
    assert pack_kernel.args[0] is x
    assert pack_kernel.args[1].shape == (32, 64)
    assert pack_kernel.kwargs == {
        "GROUP_SIZE": 32,
        "BLOCK_M": 32,
        "BLOCK_N": 64,
    }


@pytest.mark.parametrize(
    ("x", "group_size"),
    [
        (torch.zeros((2, 128), dtype=torch.float32), 128),
        (torch.zeros(128, dtype=torch.float16), 128),
        (torch.zeros((2, 127), dtype=torch.float16), 128),
        (torch.zeros((2, 130), dtype=torch.float16), 64),
    ],
)
def test_quantize_pack_rejects_invalid_inputs(x, group_size):
    with pytest.raises(AssertionError):
        quantize_pack(x, group_size=group_size)


def test_quantize_weight_transposes_and_builds_quantized_weight(monkeypatch):
    seen = {}

    def fake_quantize_pack(x, group_size):
        seen["x"] = x
        seen["group_size"] = group_size
        packed_t = torch.full((x.shape[0], x.shape[1] // 2), 0x21, dtype=torch.int8)
        scales_t = torch.ones(
            (x.shape[0], x.shape[1] // group_size),
            dtype=torch.float16,
        )
        return packed_t, scales_t

    monkeypatch.setattr(quantize_module, "quantize_pack", fake_quantize_pack)

    W = torch.arange(512, dtype=torch.float32).reshape(128, 4)
    weight = quantize_weight(W, group_size=32)

    assert seen["group_size"] == 32
    assert seen["x"].shape == (4, 128)
    assert seen["x"].dtype == torch.float16
    assert not seen["x"].requires_grad
    assert seen["x"].is_contiguous()

    assert weight.q.shape == (64, 4)
    assert weight.q.dtype == torch.int8
    assert torch.all(weight.q == 0x21)

    assert weight.scales.shape == (4, 4)
    assert weight.scales.dtype == torch.bfloat16
    assert torch.all(weight.scales == 1)

    assert weight.zeros.shape == (4, 4)
    assert weight.zeros.dtype == torch.bfloat16
    assert torch.count_nonzero(weight.zeros) == 0

    assert weight.group_size == 32
    assert weight.orig_dtype == torch.bfloat16
    assert weight.shape == (128, 4)
    assert weight.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("shape", "group_size"),
    [
        ((128,), 128),
        ((126, 4), 128),
        ((129, 4), 32),
        ((128, 4), 30),
    ],
)
def test_quantize_weight_rejects_invalid_shapes(shape, group_size):
    W = torch.zeros(shape, dtype=torch.float32)

    with pytest.raises(AssertionError):
        quantize_weight(W, group_size=group_size)


def test_quantized_weight_properties():
    weight = QuantizedWeight(
        q=torch.zeros((8, 3), dtype=torch.int8),
        scales=torch.zeros((2, 3), dtype=torch.bfloat16),
        zeros=torch.zeros((2, 3), dtype=torch.bfloat16),
        group_size=64,
        orig_dtype=torch.float32,
    )

    assert weight.shape == (16, 3)
    assert weight.dtype == torch.float32
