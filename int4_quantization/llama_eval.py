"""Замена Linear в Llama и замер перплексии со скоростью."""

from __future__ import annotations

import math
import statistics
import time

import torch
from torch import nn

from int4_quantization.linear import QuantizedLinear
from int4_quantization.reference import (
    GROUP_SIZE,
    matmul_reference,
    quantize_weight_reference,
)

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
LLAMA_1B_LINEAR_LAYERS = 16 * 7 + 1


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """Выбирает устройство и dtype весов.

    На CUDA с compute capability 8 и выше (L4, A100, RTX 30xx) берётся
    ``bfloat16``. На T4 и младше — ``float16``: аппаратного bf16 там нет.
    На Apple MPS тоже ``float16``.

    Returns:
        Пара ``(device, dtype)``.
    """
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.device("cuda"), dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.bfloat16


def replace_linear_layers(
    model: nn.Module,
    group_size: int = GROUP_SIZE,
    backend: str = "reference",
) -> list[str]:
    """Заменяет каждый ``nn.Linear`` на ``QuantizedLinear``.

    Embedding и нормализация остаются как есть. Вес берётся транспонированным,
    поэтому квантование идёт вдоль входной оси слоя.

    Args:
        model: Модель, внутри которой ищутся линейные слои.
        group_size: Размер группы INT4.
        backend: ``"reference"`` или ``"triton"``.

    Returns:
        Имена заменённых модулей в порядке обхода.
    """
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                quantized = QuantizedLinear.from_linear(
                    child, group_size=group_size, backend=backend
                )
                setattr(parent, name, quantized)
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model, "")
    return replaced


def weight_footprint(model: nn.Module) -> dict[str, int]:
    """Считает память линейных весов до и после упаковки.

    Returns:
        Словарь с числом обычных и квантованных слоёв и байтами весов.
        ``code_bytes`` — только упакованные INT4. ``quantized_bytes`` включает
        ещё масштабы и нулевые точки.
    """
    linear_bytes = 0
    quantized_bytes = 0
    code_bytes = 0
    n_linear = 0
    n_quantized = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            n_quantized += 1
            code_bytes += module.q.nbytes
            quantized_bytes += (
                module.q.nbytes + module.scales.nbytes + module.zeros.nbytes
            )
        elif isinstance(module, nn.Linear) and module.weight is not None:
            n_linear += 1
            linear_bytes += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                linear_bytes += module.bias.numel() * module.bias.element_size()
    return {
        "linear_layers": n_linear,
        "quantized_layers": n_quantized,
        "linear_bytes": linear_bytes,
        "quantized_bytes": quantized_bytes,
        "code_bytes": code_bytes,
    }


def load_causal_lm(model_id: str, dtype: torch.dtype, device: torch.device):
    """Загружает CausalLM и переносит её на устройство.

    Args:
        model_id: Имя модели на Hugging Face.
        dtype: dtype весов.
        device: Куда класть модель.

    Returns:
        Пара ``(tokenizer, model)``. Модель в режиме ``eval``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_reply(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 32,
) -> str:
    """Генерирует продолжение одного пользовательского сообщения.

    Args:
        model: Языковая модель с методом ``generate``.
        tokenizer: Токенизатор с шаблоном чата.
        prompt: Текст пользователя.
        device: Устройство входных токенов.
        max_new_tokens: Сколько новых токенов дописать.

    Returns:
        Ответ модели без специальных токенов.
    """
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    input_ids = input_ids.to(device)
    with torch.inference_mode():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_tokens = output[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_wikitext2(tokenizer, max_tokens: int | None = None) -> torch.Tensor:
    """Токенизирует тест WikiText-2 raw одной лентой.

    Args:
        tokenizer: Токенизатор модели.
        max_tokens: Обрезать ленту до этого числа токенов. ``None`` — весь тест.

    Returns:
        ``input_ids`` формы ``[1, T]`` на CPU.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    return input_ids


def perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_length: int = 2048,
    stride: int = 512,
    log_every: int = 0,
) -> dict[str, float | int]:
    """Перплексия скользящим окном.

    Перекрытие окон маскируется через ``labels == -100``. Итог взвешен по числу токенов.

    Args:
        model: Модель, чей ``forward`` принимает ``input_ids`` и ``labels``
            и возвращает объект с полем ``loss``.
        input_ids: Токены формы ``[1, T]``.
        max_length: Длина контекста.
        stride: Шаг окна.
        log_every: Печатать промежуточную перплексию каждые столько окон.
            ``0`` отключает печать.

    Returns:
        Словарь с ключами ``perplexity``, ``tokens`` и ``windows``.

    Raises:
        ValueError: Если токенов нет.
    """
    if input_ids.numel() == 0:
        raise ValueError("input_ids is empty")
    device = _module_device(model)
    model.eval()
    sequence_length = input_ids.size(1)
    nll_sum = 0.0
    token_count = 0
    windows = 0
    previous_end = 0
    for begin in range(0, sequence_length, stride):
        end = min(begin + max_length, sequence_length)
        target_length = end - previous_end
        chunk = input_ids[:, begin:end].to(device)
        labels = chunk.clone()
        labels[:, :-target_length] = -100
        with torch.inference_mode():
            loss = model(chunk, labels=labels).loss
        nll_sum += float(loss.item()) * target_length
        token_count += target_length
        windows += 1
        previous_end = end
        if log_every and windows % log_every == 0:
            running = math.exp(nll_sum / token_count)
            print(
                f"windows={windows} tokens={token_count} perplexity={running:.3f}",
                flush=True,
            )
        if end == sequence_length:
            break
    return {
        "perplexity": math.exp(nll_sum / token_count),
        "tokens": token_count,
        "windows": windows,
    }


def benchmark_forward(
    model: nn.Module,
    seq_lens: tuple[int, ...] = (128, 512, 2048),
    vocab_size: int = 128256,
    device: torch.device | None = None,
    warmup: int = 2,
    iters: int = 5,
) -> list[dict[str, float | int]]:
    """Время полного forward, batch 1.

    CUDA events, на MPS и CPU — ``perf_counter`` после синхронизации. Берётся медиана.

    Args:
        model: Модель, принимающая ``input_ids``.
        seq_lens: Длины последовательности.
        vocab_size: Верхняя граница случайных токенов.
        device: Куда класть токены. По умолчанию устройство модели.
        warmup: Число прогревочных прогонов.
        iters: Число измеряемых прогонов.

    Returns:
        Список словарей ``seq_len``, ``ms``, ``tokens_per_s``.

    Raises:
        ValueError: Если измеряемых прогонов меньше одного.
    """
    if iters < 1:
        raise ValueError("iters must be positive")
    if device is None:
        device = _module_device(model)
    model.eval()
    rows: list[dict[str, float | int]] = []
    for seq_len in seq_lens:
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        for _ in range(warmup):
            with torch.inference_mode():
                model(input_ids)
            _synchronize(device)
        samples = [_time_forward_ms(model, input_ids, device) for _ in range(iters)]
        elapsed_ms = statistics.median(samples)
        rows.append(
            {
                "seq_len": seq_len,
                "ms": elapsed_ms,
                "tokens_per_s": seq_len / (elapsed_ms / 1000.0),
            }
        )
    return rows


def check_triton_against_reference(
    device: torch.device | None = None,
    rows: int = 128,
    in_features: int = 256,
    out_features: int = 256,
) -> dict[str, object]:
    """Сверяет Triton-GEMM с эталоном на одном небольшом умножении.

    Args:
        device: Устройство. Без CUDA сверка пропускается.
        rows: Число строк активаций.
        in_features: Входная ось, делится на 128.
        out_features: Выходная ось.

    Returns:
        Словарь со статусом ``skipped``, ``ok`` или ``mismatch``. При ошибке
        запуска кернеля статус ``error``, текст лежит в ``message``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {"status": "skipped", "message": "только CUDA"}
    weight = torch.randn(in_features, out_features, device=device, dtype=torch.float16)
    packed, scales, zeros = quantize_weight_reference(weight, group_size=GROUP_SIZE)
    activations = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)
    reference = matmul_reference(activations, packed, scales, zeros, GROUP_SIZE)
    try:
        from int4_quantization.matmul import matmul
        from int4_quantization.quantize import QuantizedWeight

        quantized = QuantizedWeight(
            q=packed,
            scales=scales,
            zeros=zeros,
            group_size=GROUP_SIZE,
            orig_dtype=torch.bfloat16,
        )
        started = time.perf_counter()
        produced = matmul(activations, quantized, kind="w4")
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except Exception as error:
        return {"status": "error", "message": f"{type(error).__name__}: {error}"}
    difference = (produced.float() - reference.float()).abs()
    scale = reference.float().abs().mean().clamp(min=1e-3)
    relative = float((difference.mean() / scale).item())
    status = "ok" if relative < 0.05 else "mismatch"
    return {
        "status": status,
        "relative_mean_abs": relative,
        "max_abs": float(difference.max().item()),
        "triton_ms": elapsed_ms,
        "message": "совпало" if status == "ok" else "не совпало с эталоном",
    }


def _module_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _time_forward_ms(
    model: nn.Module, input_ids: torch.Tensor, device: torch.device
) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            model(input_ids)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        model(input_ids)
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0
