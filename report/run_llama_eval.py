"""Замер Llama-3.2-1B на GPU-ноде: обычная модель против INT4-эталона."""

from __future__ import annotations

import gc
import json
import os
import signal
from pathlib import Path

import torch

from int4_quantization.llama_eval import (
    LLAMA_1B_LINEAR_LAYERS,
    MODEL_ID,
    benchmark_forward,
    check_triton_against_reference,
    generate_reply,
    load_causal_lm,
    load_wikitext2,
    perplexity,
    replace_linear_layers,
    select_device_and_dtype,
    weight_footprint,
)

PROMPT = "What is the capital of France? Answer in one short sentence."


def main() -> None:
    quick = os.environ.get("EVAL_QUICK", "0") == "1"
    max_length = 2048
    stride = 512
    max_tokens = 2048 if quick else None
    seq_lens = (128,) if quick else (128, 512, 2048)
    warmup = 1 if quick else 2
    iters = 1 if quick else 5
    out_dir = Path(os.environ.get("EVAL_OUT", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = select_device_and_dtype()
    if device.type == "cuda":
        device = _freest_cuda_device()
        major, minor = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
    else:
        major, minor, name = 0, 0, device.type
    print(f"device={device} dtype={dtype} gpu={name} cap={major}.{minor}", flush=True)
    if device.type != "cuda" or major < 8:
        raise SystemExit("нужна GPU с compute capability 8+ (bf16)")

    tokenizer, baseline = load_causal_lm(MODEL_ID, dtype, device)
    vocab_size = baseline.get_input_embeddings().num_embeddings
    base_mem = weight_footprint(baseline)
    print("baseline", base_mem, flush=True)
    base_reply = generate_reply(baseline, tokenizer, PROMPT, device)
    print("baseline reply:", base_reply, flush=True)
    base_speed = benchmark_forward(
        baseline,
        seq_lens=seq_lens,
        vocab_size=vocab_size,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    print("baseline speed", base_speed, flush=True)
    input_ids = load_wikitext2(tokenizer, max_tokens=max_tokens)
    print("tokens", int(input_ids.shape[1]), flush=True)
    base_ppl = perplexity(
        baseline, input_ids, max_length=max_length, stride=stride, log_every=10
    )
    print("baseline ppl", base_ppl, flush=True)

    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    _tokenizer, quantized = load_causal_lm(MODEL_ID, dtype, device)
    replaced = replace_linear_layers(quantized, backend="reference")
    quant_mem = weight_footprint(quantized)
    print(f"replaced {len(replaced)} expected {LLAMA_1B_LINEAR_LAYERS}", flush=True)
    print("quantized", quant_mem, flush=True)
    quant_reply = generate_reply(quantized, _tokenizer, PROMPT, device)
    print("int4 reply:", quant_reply, flush=True)
    quant_speed = benchmark_forward(
        quantized,
        seq_lens=seq_lens,
        vocab_size=vocab_size,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    print("int4 speed", quant_speed, flush=True)
    quant_ppl = perplexity(
        quantized, input_ids, max_length=max_length, stride=stride, log_every=10
    )
    print("int4 ppl", quant_ppl, flush=True)

    triton_check = _triton_check(device)
    print("triton", triton_check, flush=True)

    report = {
        "gpu": name,
        "capability": [major, minor],
        "dtype": str(dtype),
        "quick": quick,
        "tokens": int(input_ids.shape[1]),
        "baseline_memory": base_mem,
        "quantized_memory": quant_mem,
        "replaced_layers": len(replaced),
        "baseline_reply": base_reply,
        "quantized_reply": quant_reply,
        "baseline_speed": base_speed,
        "quantized_speed": quant_speed,
        "baseline_perplexity": base_ppl,
        "quantized_perplexity": quant_ppl,
        "triton": triton_check,
    }
    path = out_dir / ("quick.json" if quick else "a6000.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote", path, flush=True)


def _freest_cuda_device() -> torch.device:
    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        if free > best_free:
            best_index = index
            best_free = free
    torch.cuda.set_device(best_index)
    print(f"using cuda:{best_index} free_mib={best_free // 2**20}", flush=True)
    return torch.device(f"cuda:{best_index}")


def _triton_check(device: torch.device) -> dict:
    def handle(_signum, _frame):
        raise TimeoutError("triton check timed out")

    previous = signal.signal(signal.SIGALRM, handle)
    signal.alarm(300)
    try:
        return check_triton_against_reference(device)
    except Exception as error:
        return {"status": "error", "message": f"{type(error).__name__}: {error}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
