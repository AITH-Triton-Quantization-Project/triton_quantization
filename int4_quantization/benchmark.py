"""Замеры GEMM bf16 и int4 для линейных слоёв Llama-3.2-1B.

Сравнивается ``X[M, K] @ W[K, N]``:

* W16 — плотные веса bf16 через ``torch.matmul`` (cuBLAS). Triton-кернель
  W16 в ``matmul.py`` пока не выполняет вычисление, поэтому базой служит
  библиотечный GEMM. Его можно заменить флагом ``--w16-backend triton``,
  когда кернель будет готов.
* W4 — ``matmul(..., kind="w4")``: упакованные int4 и fused-деквантование.

Квантизация выполняется один раз и в тайминг не входит. Медиана CUDA events
— это latency вызова. ``torch.profiler`` снимается отдельным проходом и
даёт self CUDA time самого умножения.

Запуск из корня репозитория::

    python -m int4_quantization.benchmark
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile

from .llama_shapes import (
    GROUP_SIZE,
    MODEL_ID,
    TOKEN_COUNTS,
    LinearLayerSpec,
    dense_weight_bytes,
    gemm_flops,
    group_by_geometry,
    int4_payload_bytes,
    llama32_1b_linear_layers,
)
from .matmul import matmul
from .quantize import QuantizedWeight, quantize_weight

_PROFILE_SORT_KEYS = (
    "self_device_time_total",
    "self_cuda_time_total",
    "cuda_time_total",
)


@dataclass(frozen=True)
class ProfileResult:
    """Сводка одного прохода ``torch.profiler``."""

    table: str
    self_cuda_ms: float
    top_op: str
    top_op_ms: float
    iters: int


@dataclass(frozen=True)
class GemmMeasurement:
    """Один замер пары W16/W4 для геометрии ``(M, K, N)``."""

    names: tuple[str, ...]
    count: int
    m: int
    k: int
    n: int
    group_size: int
    flops: int
    dense_bytes: int
    int4_bytes: int
    w16_ms: float
    w4_ms: float
    w16_kernel_ms: float | None
    w4_kernel_ms: float | None
    w16_top_op: str
    w4_top_op: str
    w16_profile: str
    w4_profile: str

    @property
    def speedup(self) -> float:
        """Во сколько раз вызов W4 быстрее вызова W16. Больше 1 — W4 быстрее."""
        return self.w16_ms / self.w4_ms

    def tflops(self, latency_ms: float) -> float:
        """Достигнутая производительность по времени CUDA events."""
        return self.flops / latency_ms / 1e9

    def weight_gbps(self, latency_ms: float, nbytes: int) -> float:
        """Пропускная способность чтения только весов, ГБ/с."""
        return nbytes / latency_ms / 1e6


def time_cuda_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    """Медиана GPU-времени ``fn`` в миллисекундах.

    Прогрев не входит в выборку. ``fn`` должен ставить работу в текущий
    CUDA-поток: и Triton, и ``torch.matmul`` так и делают. После каждой
    пары событий поток синхронизируется, поэтому в время входит и запуск,
    и само ядро, и выделение выхода.
    """
    if warmup < 1 or iters < 1:
        raise ValueError("warmup и iters должны быть положительными")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _profile_table(prof: profile) -> str:
    averages = prof.key_averages()
    for sort_key in _PROFILE_SORT_KEYS:
        try:
            return averages.table(sort_by=sort_key, row_limit=20)
        except (KeyError, ValueError):
            continue
    return averages.table(row_limit=20)


def profile_call(fn: Callable[[], object], *, warmup: int, iters: int) -> ProfileResult:
    """Снимает ``torch.profiler`` и возвращает CUDA-время на один вызов.

    Прогрев идёт до включения профайлера, чтобы компиляция Triton не
    попадала в таблицу. ``self_cuda_ms`` — сумма self-времени событий
    с ``device_type == CUDA`` (сами ядра, без родительских aten-операций),
    делённая на число вызовов.
    """
    if warmup < 1 or iters < 1:
        raise ValueError("warmup и iters должны быть положительными")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, acc_events=True) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    total_us = 0.0
    top_op = ""
    top_us = 0.0
    for event in prof.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        elapsed_us = float(event.self_device_time_total or 0)
        total_us += elapsed_us
        if elapsed_us > top_us:
            top_us = elapsed_us
            top_op = event.key

    return ProfileResult(
        table=_profile_table(prof),
        self_cuda_ms=total_us / 1000.0 / iters,
        top_op=top_op,
        top_op_ms=top_us / 1000.0 / iters,
        iters=iters,
    )


def _prepare_inputs(
    m: int,
    k: int,
    n: int,
    group_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, QuantizedWeight]:
    torch.manual_seed(seed)
    activations = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    quantized = quantize_weight(weight, group_size=group_size)
    torch.cuda.synchronize()
    return activations, weight, quantized


def _w16_runner(
    activations: torch.Tensor,
    weight: torch.Tensor,
    backend: str,
) -> Callable[[], torch.Tensor]:
    if backend == "torch":

        def run_torch() -> torch.Tensor:
            return torch.matmul(activations, weight)

        return run_torch
    if backend == "triton":

        def run_triton() -> torch.Tensor:
            return matmul(activations, weight, kind="w16")

        return run_triton
    raise ValueError(f"неизвестный backend W16: {backend}")


def measure_gemm(
    m: int,
    k: int,
    n: int,
    *,
    names: Sequence[str] = (),
    count: int = 1,
    group_size: int = GROUP_SIZE,
    warmup: int = 10,
    iters: int = 30,
    profile_iters: int = 8,
    w16_backend: str = "torch",
    device: str | torch.device = "cuda",
    seed: int = 0,
    profile: bool = True,
) -> GemmMeasurement:
    """Замеряет один GEMM ``[M, K] @ [K, N]`` в bf16 и в int4.

    Возвращает медиану CUDA events и, если ``profile=True``, таблицы
    профайлера. Тензоры после замера освобождаются.
    """
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError("замер Triton-кернеля требует CUDA-устройство")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна для этой сборки PyTorch")

    activations, weight, quantized = _prepare_inputs(
        m, k, n, group_size, cuda_device, seed
    )
    run_w16 = _w16_runner(activations, weight, w16_backend)
    run_w4 = partial(matmul, activations, quantized, kind="w4")

    with torch.inference_mode():
        w16_ms = time_cuda_ms(run_w16, warmup=warmup, iters=iters)
        w4_ms = time_cuda_ms(run_w4, warmup=warmup, iters=iters)
        if profile:
            w16_profile = profile_call(run_w16, warmup=1, iters=profile_iters)
            w4_profile = profile_call(run_w4, warmup=1, iters=profile_iters)
        else:
            w16_profile = None
            w4_profile = None

    del activations, weight, quantized, run_w16, run_w4
    torch.cuda.empty_cache()

    return GemmMeasurement(
        names=tuple(names),
        count=count,
        m=m,
        k=k,
        n=n,
        group_size=group_size,
        flops=gemm_flops(m, k, n),
        dense_bytes=dense_weight_bytes(k, n),
        int4_bytes=int4_payload_bytes(k, n, group_size),
        w16_ms=w16_ms,
        w4_ms=w4_ms,
        w16_kernel_ms=None if w16_profile is None else w16_profile.self_cuda_ms,
        w4_kernel_ms=None if w4_profile is None else w4_profile.self_cuda_ms,
        w16_top_op="" if w16_profile is None else w16_profile.top_op,
        w4_top_op="" if w4_profile is None else w4_profile.top_op,
        w16_profile="" if w16_profile is None else w16_profile.table,
        w4_profile="" if w4_profile is None else w4_profile.table,
    )


def run_suite(
    layers: Sequence[LinearLayerSpec] | None = None,
    token_counts: Sequence[int] = TOKEN_COUNTS,
    **kwargs,
) -> list[GemmMeasurement]:
    """Прогоняет уникальные геометрии слоёв для каждого числа токенов.

    Слои с одинаковыми ``(K, N)`` замеряются один раз. В результате у
    такой строки ``names`` содержит все исходные имена, а ``count`` —
    сумму вхождений в модель.
    """
    if layers is None:
        layers = llama32_1b_linear_layers()

    measurements: list[GemmMeasurement] = []
    groups = group_by_geometry(layers)
    for tokens in token_counts:
        for (k, n), grouped in groups.items():
            names = tuple(layer.name for layer in grouped)
            count = sum(layer.count for layer in grouped)
            label = "+".join(names)
            print(
                f"замер {label}: M={tokens} K={k} N={n}",
                flush=True,
            )
            measurements.append(
                measure_gemm(
                    tokens,
                    k,
                    n,
                    names=names,
                    count=count,
                    **kwargs,
                )
            )
    return measurements


def _ms(value: float | None) -> str:
    if value is None:
        return "        —"
    return f"{value:9.3f}"


def format_report(
    measurements: Sequence[GemmMeasurement],
    *,
    device_name: str,
    w16_backend: str,
) -> str:
    """Текстовый отчёт: таблица GEMM и сумма линейных слоёв одного forward."""
    header = (
        f"Модель: {MODEL_ID}\n"
        f"GPU: {device_name}\n"
        "Операция: X[M, K] bf16 @ W[K, N]. "
        "W в этой раскладке — это weight.T слоя nn.Linear.\n"
        f"W16: {w16_backend}. W4: Triton int4, group_size="
        f"{measurements[0].group_size if measurements else GROUP_SIZE}.\n"
        "Квантизация в замер не входит. "
        "ms — медиана CUDA events одного вызова, вместе с запуском ядра. "
        "kernel ms — чистое GPU-время ядер из torch.profiler.\n"
        "speedup = w16_ms / w4_ms; значение больше 1 означает, что W4 быстрее.\n"
    )
    columns = (
        f"{'layers':<22} {'cnt':>4} {'M':>6} {'K':>6} {'N':>8} "
        f"{'W16 ms':>8} {'W4 ms':>8} {'speedup':>8} "
        f"{'W16 kern':>9} {'W4 kern':>9} "
        f"{'W16 TF/s':>9} {'W4 TF/s':>9}"
    )
    lines = [header, columns, "-" * len(columns)]
    for row in measurements:
        lines.append(
            f"{'+'.join(row.names):<22} {row.count:4d} {row.m:6d} "
            f"{row.k:6d} {row.n:8d} "
            f"{row.w16_ms:8.3f} {row.w4_ms:8.3f} {row.speedup:8.2f} "
            f"{_ms(row.w16_kernel_ms)} {_ms(row.w4_kernel_ms)} "
            f"{row.tflops(row.w16_ms):9.2f} {row.tflops(row.w4_ms):9.2f}"
        )

    lines.append("")
    lines.append(
        "Сумма линейных слоёв одного prefill (attention, нормы и RoPE не входят):"
    )
    by_tokens: dict[int, list[GemmMeasurement]] = {}
    for row in measurements:
        by_tokens.setdefault(row.m, []).append(row)
    for tokens, rows in by_tokens.items():
        w16 = sum(row.w16_ms * row.count for row in rows)
        w4 = sum(row.w4_ms * row.count for row in rows)
        lines.append(
            f"  M={tokens:<6d}  W16 {w16:10.2f} ms    W4 {w4:10.2f} ms    "
            f"speedup {w16 / w4:5.2f}"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, measurements: Sequence[GemmMeasurement]) -> None:
    """Пишет таблицу замера без текстов профайлера."""
    fieldnames = [
        "layers",
        "count",
        "m",
        "k",
        "n",
        "group_size",
        "flops",
        "dense_bytes",
        "int4_bytes",
        "w16_ms",
        "w4_ms",
        "speedup",
        "w16_kernel_ms",
        "w4_kernel_ms",
        "w16_tflops",
        "w4_tflops",
        "w16_weight_gbps",
        "w4_weight_gbps",
        "w16_top_op",
        "w4_top_op",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in measurements:
            payload = asdict(row)
            payload["layers"] = "+".join(row.names)
            payload["speedup"] = row.speedup
            payload["w16_tflops"] = row.tflops(row.w16_ms)
            payload["w4_tflops"] = row.tflops(row.w4_ms)
            payload["w16_weight_gbps"] = row.weight_gbps(row.w16_ms, row.dense_bytes)
            payload["w4_weight_gbps"] = row.weight_gbps(row.w4_ms, row.int4_bytes)
            for drop in ("names", "w16_profile", "w4_profile"):
                payload.pop(drop, None)
            writer.writerow({key: payload[key] for key in fieldnames})


def write_profiles(directory: Path, measurements: Sequence[GemmMeasurement]) -> None:
    """Сохраняет текстовые таблицы ``torch.profiler`` по одному файлу на прогон."""
    directory.mkdir(parents=True, exist_ok=True)
    for row in measurements:
        stem = f"{'+'.join(row.names)}_M{row.m}"
        for kind, table in (("w16", row.w16_profile), ("w4", row.w4_profile)):
            if not table:
                continue
            path = directory / f"{stem}_{kind}.txt"
            path.write_text(table, encoding="utf-8")


def _select_layers(names: Sequence[str] | None) -> tuple[LinearLayerSpec, ...]:
    layers = llama32_1b_linear_layers()
    if not names:
        return layers
    wanted = set(names)
    selected = tuple(layer for layer in layers if layer.name in wanted)
    missing = wanted - {layer.name for layer in selected}
    if missing:
        known = ", ".join(layer.name for layer in layers)
        unknown = ", ".join(sorted(missing))
        raise SystemExit(f"неизвестные слои: {unknown}. Доступны: {known}")
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Сравнение X16@W16 и X16@W4 на формах линейных слоёв "
            "Llama-3.2-1B-Instruct."
        )
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(TOKEN_COUNTS),
        help="число строк активаций X, по умолчанию 128 512 2048",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="какие слои мерить; по умолчанию все линейные слои модели",
    )
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--profile-iters", type=int, default=8)
    parser.add_argument(
        "--w16-backend",
        choices=("torch", "triton"),
        default="torch",
        help="база W16: torch.matmul или Triton-кернель, когда он будет готов",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_results"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="снять только CUDA events, без torch.profiler",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Triton 3.9 иначе предупреждает на каждом первом запуске ядра.
    # 1 сохраняет текущее поведение: FP-fusion включён.
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    if not torch.cuda.is_available():
        print(
            "CUDA недоступна. Замер кернелей на этой машине не выполнить.",
            file=sys.stderr,
        )
        return 1

    layers = _select_layers(args.layers)
    measurements = run_suite(
        layers,
        token_counts=args.tokens,
        group_size=args.group_size,
        warmup=args.warmup,
        iters=args.iters,
        profile_iters=args.profile_iters,
        w16_backend=args.w16_backend,
        seed=args.seed,
        profile=not args.no_profile,
    )
    report = format_report(
        measurements,
        device_name=torch.cuda.get_device_name(0),
        w16_backend=args.w16_backend,
    )
    print(report, flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.txt").write_text(report, encoding="utf-8")
    write_csv(args.out_dir / "gemm.csv", measurements)
    write_profiles(args.out_dir / "profiles", measurements)
    print(f"результаты: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
