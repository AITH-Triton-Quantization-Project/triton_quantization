import pytest

from int4_quantization.llama_shapes import (
    GROUP_SIZE,
    LinearLayerSpec,
    dense_weight_bytes,
    gemm_flops,
    group_by_geometry,
    int4_payload_bytes,
    linear_layers_ms,
    llama32_1b_linear_layers,
    packed_int4_bytes,
)


def test_layer_catalog_matches_llama32_1b_config():
    layers = llama32_1b_linear_layers()
    by_name = {layer.name: layer for layer in layers}

    assert list(by_name) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
    assert by_name["q_proj"] == LinearLayerSpec("q_proj", 2048, 2048, 16)
    assert by_name["o_proj"].geometry == (2048, 2048)
    assert by_name["k_proj"] == LinearLayerSpec("k_proj", 2048, 512, 16)
    assert by_name["v_proj"].geometry == by_name["k_proj"].geometry
    assert by_name["gate_proj"] == LinearLayerSpec("gate_proj", 2048, 8192, 16)
    assert by_name["up_proj"].geometry == by_name["gate_proj"].geometry
    assert by_name["down_proj"] == LinearLayerSpec("down_proj", 8192, 2048, 16)
    assert by_name["lm_head"] == LinearLayerSpec("lm_head", 2048, 128256, 1)
    assert sum(layer.count for layer in layers) == 16 * 7 + 1


def test_reduction_axis_is_quantizable():
    for layer in llama32_1b_linear_layers():
        assert layer.in_features % GROUP_SIZE == 0
        assert layer.in_features % 2 == 0


def test_identical_geometries_are_grouped():
    groups = group_by_geometry(llama32_1b_linear_layers())

    assert set(groups) == {
        (2048, 2048),
        (2048, 512),
        (2048, 8192),
        (8192, 2048),
        (2048, 128256),
    }
    assert [layer.name for layer in groups[(2048, 2048)]] == ["q_proj", "o_proj"]
    assert sum(layer.count for layer in groups[(2048, 512)]) == 32


def test_int4_packing_is_four_times_smaller_than_bf16():
    dense = dense_weight_bytes(2048, 8192)
    packed = packed_int4_bytes(2048, 8192)
    payload = int4_payload_bytes(2048, 8192, group_size=128)

    assert packed * 4 == dense
    assert payload > packed
    assert dense / payload == pytest.approx(3.764, rel=1e-3)
    assert gemm_flops(128, 2048, 8192) == 2 * 128 * 2048 * 8192


def test_linear_latency_weights_each_layer_by_how_often_it_occurs():
    layers = llama32_1b_linear_layers()
    per_call = {layer.name: 1.0 for layer in layers}

    assert linear_layers_ms(per_call, layers) == 16 * 7 + 1

    per_call["lm_head"] = 10.0
    assert linear_layers_ms(per_call, layers) == 16 * 7 + 10
