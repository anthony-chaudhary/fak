#!/usr/bin/env python
"""HF int8 (dynamic quantization) latency baseline — the SAME-RUNG peer for fak's Q8_0.

bench_hf.py measures HuggingFace at f32 (fak's *correctness* peer). This measures HF at
int8 via torch.ao.quantization.quantize_dynamic: every nn.Linear (q/k/v/o, gate/up/down,
the tied LM head) is replaced by a dynamically-quantized int8 Linear — weights stored
int8, activations quantized per-call, executed on CPU through torch's fbgemm/oneDNN int8
GEMM. The embedding LOOKUP stays f32 (nn.Embedding is not a Linear), exactly as fak keeps
the embedding f32 and quantizes only its use as the head. This is the closest HF analogue
to fak's Q8_0 weight-only path and to llama.cpp's Q8_0, so the int8 row makes "beat HF on
the int8 rung" a measured *same-precision* comparison instead of f32-vs-int8.

Same SmolLM2-135M, same deterministic LCG token-ids (bit-for-bit the recurrence in
cmd/modelbench/main.go and bench_hf.py), same machine, same timing protocol as bench_hf.py
— so this JSON drops straight into compare.py's table next to the f32 HF row.

Usage:
  python internal/model/bench_hf_quant.py --out experiments/model-baseline/hf-int8.json
"""
import argparse, json, os, sys, time, statistics
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM
try:
    from transformers import DynamicCache
except Exception:
    DynamicCache = None

PREFILL_SIZES = [16, 64, 256]
DECODE_PROMPT = 16
DECODE_STEPS = 32


def lcg_ids(n, vocab):
    """Identical recurrence to cmd/modelbench/main.go lcgIDs() and bench_hf.py."""
    ids, state = [], 2463534242
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7fffffff
        ids.append(state % vocab)
    return ids


def new_cache():
    return DynamicCache() if DynamicCache is not None else None


def prefill_ms(model, ids, reps):
    input_ids = torch.tensor([ids], dtype=torch.long)
    cache_position = torch.arange(len(ids))
    times = []
    with torch.inference_mode():
        for _ in range(reps):
            cache = new_cache()
            t = time.perf_counter()
            # logits_to_keep=1: head on the last position only — apples-to-apples with
            # fak Prefill (post-R15) and the f32 HF bench.
            model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                  cache_position=cache_position, logits_to_keep=1)
            times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def decode_per_token_ms(model, prompt_ids, steps, vocab, reps):
    per_tok = []
    with torch.inference_mode():
        for r in range(reps):
            cache = new_cache()
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            cp = torch.arange(len(prompt_ids))
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=cp)
            cache = out.past_key_values
            nid = (r * 131 + 7) % vocab
            pos = len(prompt_ids)
            t = time.perf_counter()
            for _ in range(steps):
                ii = torch.tensor([[nid]], dtype=torch.long)
                cpos = torch.tensor([pos])
                out = model(input_ids=ii, past_key_values=cache, use_cache=True, cache_position=cpos)
                cache = out.past_key_values
                nid = (nid * 48271 + 1) % vocab
                pos += 1
            per_tok.append((time.perf_counter() - t) / steps)
    return statistics.median(per_tok) * 1e3


def run_config(model, name, threads, vocab, reps):
    torch.set_num_threads(threads)
    with torch.inference_mode():
        warm = new_cache()
        model(input_ids=torch.tensor([lcg_ids(8, vocab)]), past_key_values=warm,
              use_cache=True, cache_position=torch.arange(8))
    cfg = {"config": name, "torch_threads": torch.get_num_threads(), "prefill": []}
    for p in PREFILL_SIZES:
        ms = prefill_ms(model, lcg_ids(p, vocab), reps)
        cfg["prefill"].append({"tokens": p, "reps": reps, "median_ms": ms, "tok_per_sec": p / (ms / 1e3)})
        sys.stderr.write(f"[hf-int8:{name}] prefill P={p}: {ms:.1f} ms ({p/(ms/1e3):.1f} tok/s)\n"); sys.stderr.flush()
    dms = decode_per_token_ms(model, lcg_ids(DECODE_PROMPT, vocab), DECODE_STEPS, vocab, reps)
    cfg["decode"] = {"prompt_tokens": DECODE_PROMPT, "decode_steps": DECODE_STEPS, "reps": reps,
                     "per_token_median_ms": dms, "tok_per_sec": 1.0 / (dms / 1e3)}
    sys.stderr.write(f"[hf-int8:{name}] decode: {dms:.1f} ms/tok ({1.0/(dms/1e3):.1f} tok/s)\n"); sys.stderr.flush()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ncpu = os.cpu_count() or 1

    sys.stderr.write("[hf-int8] loading f32, then quantize_dynamic(Linear -> qint8)...\n"); sys.stderr.flush()
    t = time.perf_counter()
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager")
    m.eval()
    qm = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    load_ms = (time.perf_counter() - t) * 1e3
    vocab = m.config.vocab_size

    report = {"engine": "huggingface-transformers (dynamic int8, qint8 weight-only Linear)",
              "model": "SmolLM2-135M (int8 dynamic)", "torch_version": torch.__version__,
              "cpu_count": ncpu, "quant": "torch.ao.quantization.quantize_dynamic Linear->qint8",
              "vocab_size": vocab, "load_quant_ms": load_ms, "configs": []}
    report["configs"].append(run_config(qm, "dynint8-1thread", 1, vocab, a.reps))
    report["configs"].append(run_config(qm, f"dynint8-{ncpu}thread", ncpu, vocab, a.reps))

    blob = json.dumps(report, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(blob)
        sys.stderr.write(f"wrote {a.out}\n")
    else:
        print(blob)


if __name__ == "__main__":
    main()
