import math

import pytest
import torch
from torch import nn

from int4_quantization.linear import QuantizedLinear
from int4_quantization.llama_eval import (
    benchmark_forward,
    check_triton_against_reference,
    perplexity,
    replace_linear_layers,
    weight_footprint,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 128)
        self.q_proj = nn.Linear(128, 64, bias=False)
        self.down = nn.Linear(256, 32, bias=False)
        self.norm = nn.LayerNorm(64)

    def forward(self, inputs):
        return self.q_proj(inputs)


def test_replace_linear_layers_packs_weights_and_keeps_embedding():
    block = Block()
    fp16_bytes = sum(
        module.weight.numel() * 2
        for module in block.modules()
        if isinstance(module, nn.Linear)
    )
    names = replace_linear_layers(block, group_size=128, backend="reference")

    assert names == ["q_proj", "down"]
    assert isinstance(block.embed, nn.Embedding)
    assert isinstance(block.norm, nn.LayerNorm)
    assert isinstance(block.q_proj, QuantizedLinear)
    assert isinstance(block.down, QuantizedLinear)

    after = weight_footprint(block)
    assert after["linear_layers"] == 0
    assert after["quantized_layers"] == 2
    assert after["code_bytes"] * 4 == fp16_bytes

    output = block(torch.randn(2, 5, 128))
    assert output.shape == (2, 5, 64)
    assert output.dtype == torch.bfloat16


class ConstantLoss(nn.Module):
    def forward(self, input_ids, labels=None):
        return type("Out", (), {"loss": torch.tensor(1.0)})()


def test_perplexity_weights_windows_by_target_length():
    result = perplexity(
        ConstantLoss(),
        torch.zeros(1, 100, dtype=torch.long),
        max_length=40,
        stride=20,
    )

    assert result["tokens"] == 100
    assert result["windows"] == 4
    assert result["perplexity"] == pytest.approx(math.e)


def test_perplexity_rejects_empty_ids():
    with pytest.raises(ValueError):
        perplexity(ConstantLoss(), torch.zeros(1, 0, dtype=torch.long))


def test_benchmark_forward_reports_median_milliseconds():
    rows = benchmark_forward(
        ConstantLoss(),
        seq_lens=(8,),
        vocab_size=10,
        device=torch.device("cpu"),
        warmup=0,
        iters=2,
    )

    assert rows[0]["seq_len"] == 8
    assert rows[0]["ms"] > 0
    assert rows[0]["tokens_per_s"] > 0


def test_triton_check_skips_without_cuda():
    result = check_triton_against_reference(device=torch.device("cpu"))

    assert result["status"] == "skipped"
