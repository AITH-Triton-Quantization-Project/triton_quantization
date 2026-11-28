from int4_quantization.benchmark import GemmMeasurement, format_report


def _row(**overrides) -> GemmMeasurement:
    payload = dict(
        names=("q_proj", "o_proj"),
        count=32,
        m=128,
        k=2048,
        n=2048,
        group_size=128,
        flops=2 * 128 * 2048 * 2048,
        dense_bytes=2048 * 2048 * 2,
        int4_bytes=1,
        w16_ms=1.0,
        w4_ms=0.5,
        w16_kernel_ms=0.4,
        w4_kernel_ms=0.2,
        w16_top_op="gemm",
        w4_top_op="w4",
        w16_profile="",
        w4_profile="",
    )
    payload.update(overrides)
    return GemmMeasurement(**payload)


def test_report_scales_latency_by_how_often_the_layer_occurs():
    report = format_report(
        [_row()],
        device_name="NVIDIA GeForce RTX 4060 Ti",
        w16_backend="torch",
    )

    assert "q_proj+o_proj" in report
    assert "W16      32.00 ms" in report
    assert "W4      16.00 ms" in report
    assert "speedup  2.00" in report


def test_speedup_is_w16_time_divided_by_w4_time():
    assert _row(w16_ms=1.0, w4_ms=0.5).speedup == 2.0
