"""Формы линейных слоёв Llama-3.2-1B и арифметика замера GEMM.

Числа взяты из конфига ``unsloth/Llama-3.2-1B-Instruct``:
hidden 2048, intermediate 8192, 32 головы по 64, 8 KV-голов, 16 слоёв,
словарь 128256. ``lm_head`` делит веса с эмбеддингами, но матричное
умножение на логиты выполняется отдельно, поэтому слой учтён один раз.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
TOKEN_COUNTS = (128, 512, 2048)
GROUP_SIZE = 128

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 8192
NUM_HIDDEN_LAYERS = 16
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256


@dataclass(frozen=True)
class LinearLayerSpec:
    """Линейный слой, который в модели встречается ``count`` раз.

    ``in_features`` — ось K (сокращение в GEMM), ``out_features`` — ось N.
    Вес ``nn.Linear`` хранится как ``[out, in]``; кернель ждёт уже
    транспонированную матрицу ``[in, out] = [K, N]``.
    """

    name: str
    in_features: int
    out_features: int
    count: int

    @property
    def geometry(self) -> tuple[int, int]:
        """Пара ``(K, N)``, по которой совпадает стоимость GEMM."""
        return (self.in_features, self.out_features)


def llama32_1b_linear_layers() -> tuple[LinearLayerSpec, ...]:
    """Возвращает линейные слои Llama-3.2-1B в порядке внутри блока.

    ``q_proj`` и ``o_proj`` совпадают по форме, как и пары ``k/v`` и
    ``gate/up``. ``count`` равен числу блоков, кроме ``lm_head``.
    """
    hidden = HIDDEN_SIZE
    intermediate = INTERMEDIATE_SIZE
    layers = NUM_HIDDEN_LAYERS
    q_out = NUM_ATTENTION_HEADS * HEAD_DIM
    kv_out = NUM_KEY_VALUE_HEADS * HEAD_DIM
    return (
        LinearLayerSpec("q_proj", hidden, q_out, layers),
        LinearLayerSpec("k_proj", hidden, kv_out, layers),
        LinearLayerSpec("v_proj", hidden, kv_out, layers),
        LinearLayerSpec("o_proj", hidden, hidden, layers),
        LinearLayerSpec("gate_proj", hidden, intermediate, layers),
        LinearLayerSpec("up_proj", hidden, intermediate, layers),
        LinearLayerSpec("down_proj", intermediate, hidden, layers),
        LinearLayerSpec("lm_head", hidden, VOCAB_SIZE, 1),
    )


def group_by_geometry(
    layers: Sequence[LinearLayerSpec],
) -> dict[tuple[int, int], list[LinearLayerSpec]]:
    """Группирует слои с одинаковыми ``(K, N)``.

    Одинаковую геометрию достаточно прогнать один раз: время GEMM от
    значений весов не зависит.
    """
    groups: dict[tuple[int, int], list[LinearLayerSpec]] = {}
    for layer in layers:
        groups.setdefault(layer.geometry, []).append(layer)
    return groups


def gemm_flops(m: int, k: int, n: int) -> int:
    """Число FLOPs для ``X[M, K] @ W[K, N]`` (умножение и сложение)."""
    return 2 * m * k * n


def dense_weight_bytes(k: int, n: int, element_size: int = 2) -> int:
    """Байты плотной матрицы весов. По умолчанию bf16/fp16 — 2 байта."""
    return k * n * element_size


def packed_int4_bytes(k: int, n: int) -> int:
    """Байты упакованных int4-кодов: два значения на один байт вдоль K."""
    return (k // 2) * n


def int4_payload_bytes(k: int, n: int, group_size: int = GROUP_SIZE) -> int:
    """Байты, которые читает W4-кернель: коды, масштабы и нулевые точки.

    Масштабы и нули лежат в bf16 и имеют форму ``[K / group_size, N]``.
    """
    groups = k // group_size
    param_bytes = groups * n * 2
    return packed_int4_bytes(k, n) + param_bytes + param_bytes


def linear_layers_ms(
    per_call_ms: Mapping[str, float],
    layers: Sequence[LinearLayerSpec] | None = None,
) -> float:
    """Суммарное время линейных слоёв одного forward.

    ``per_call_ms`` — время одного GEMM слоя. Оно умножается на ``count``,
    то есть на число блоков, в которых слой встречается.
    """
    if layers is None:
        layers = llama32_1b_linear_layers()
    return sum(per_call_ms[layer.name] * layer.count for layer in layers)
