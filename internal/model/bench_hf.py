#!/usr/bin/env python
"""HF transformers latency baseline for the in-kernel-model fusion lane.

This is the WITNESS side of the throughput comparison: the same SmolLM2-135M, the
same f32 weights, the same deterministic token-id sequences (an LCG reproduced
bit-for-bit from cmd/modelbench/main.go), measured on the SAME machine — so the
"fak pure-Go core is correct but not fast" disclaimer in IN-KERNEL-MODEL-RESULTS.md
becomes a measured tax instead of a hand-wave.

We feed token IDS directly (no tokenizer), exactly as the oracle export does, so the
comparison never depends on a Go tokenizer and token VALUES (which don't affect
matmul/attention cost) can't skew it.

Three configs decompose the gap:
  eager-1thread  : same algorithm class as fak (textbook eager attn), 1 core ->
                   isolates the naive-Go-loop vs MKL-BLAS gap at equal parallelism.
  eager-Nthread  : eager attn, all cores -> adds the "fak has no thread parallelism" gap.
  sdpa-Nthread   : fused-SDPA attn, all cores -> what a practitioner actually runs (the
                   real next-best CPU baseline for this model in this stack).

Usage:
  python internal/model/bench_hf.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
         --out experiments/model-baseline/hf.json
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
    """Identical recurrence to cmd/modelbench/main.go lcgIDs()."""
    ids = []
    state = 2463534242
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
            # logits_to_keep=1 -> compute the LM head ONLY on the last position, exactly
            # like fak Prefill (post-R15) and llama.cpp (logits_all=False). Without this,
            # HF computes the 49152-vocab head at all P positions (~17% extra work at
            # P=256), which would unfairly FLATTER fak's prefill ratio. Apples-to-apples.
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
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                        cache_position=cp)
            cache = out.past_key_values
            nid = (r * 131 + 7) % vocab
            pos = len(prompt_ids)
            t = time.perf_counter()
            for _ in range(steps):
                ii = torch.tensor([[nid]], dtype=torch.long)
                cpos = torch.tensor([pos])
                out = model(input_ids=ii, past_key_values=cache, use_cache=True,
                            cache_position=cpos)
                cache = out.past_key_values
                nid = (nid * 48271 + 1) % vocab  # value-irrelevant, matches Go side
                pos += 1
            per_tok.append((time.perf_counter() - t) / steps)
    return statistics.median(per_tok) * 1e3


def run_config(model, name, threads, vocab, reps):
    torch.set_num_threads(threads)
    # warm up after setting threads: triggers MKL init + lazy allocs, like the Go warmup.
    with torch.inference_mode():
        warm = new_cache()
        model(input_ids=torch.tensor([lcg_ids(8, vocab)]), past_key_values=warm,
              use_cache=True, cache_position=torch.arange(8))
    cfg = {"config": name, "torch_threads": torch.get_num_threads(), "prefill": []}
    for p in PREFILL_SIZES:
        ms = prefill_ms(model, lcg_ids(p, vocab), reps)
        cfg["prefill"].append({"tokens": p, "reps": reps, "median_ms": ms,
                               "tok_per_sec": p / (ms / 1e3)})
        sys.stderr.write(f"[hf:{name}] prefill P={p}: {ms:.1f} ms ({p/(ms/1e3):.1f} tok/s)\n")
        sys.stderr.flush()
    dms = decode_per_token_ms(model, lcg_ids(DECODE_PROMPT, vocab), DECODE_STEPS, vocab, reps)
    cfg["decode"] = {"prompt_tokens": DECODE_PROMPT, "decode_steps": DECODE_STEPS, "reps": reps,
                     "per_token_median_ms": dms, "tok_per_sec": 1.0 / (dms / 1e3)}
    sys.stderr.write(f"[hf:{name}] decode: {dms:.1f} ms/tok ({1.0/(dms/1e3):.1f} tok/s)\n")
    sys.stderr.flush()
    return cfg


def load(model_name, attn):
    sys.stderr.write(f"[hf] loading {model_name} (cpu, f32, attn={attn})...\n"); sys.stderr.flush()
    t = time.perf_counter()
    m = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation=attn)
    m.eval()
    return m, (time.perf_counter() - t) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ncpu = os.cpu_count() or 1
    vocab = None

    report = {"engine": "huggingface-transformers", "model": "SmolLM2-135M (f32)",
              "torch_version": torch.__version__, "cpu_count": ncpu, "configs": []}

    eager, load_eager_ms = load(a.model, "eager")
    vocab = eager.config.vocab_size
    report["vocab_size"] = vocab
    report["load_eager_ms"] = load_eager_ms
    report["configs"].append(run_config(eager, "eager-1thread", 1, vocab, a.reps))
    report["configs"].append(run_config(eager, f"eager-{ncpu}thread", ncpu, vocab, a.reps))
    del eager

    sdpa, load_sdpa_ms = load(a.model, "sdpa")
    report["load_sdpa_ms"] = load_sdpa_ms
    report["configs"].append(run_config(sdpa, f"sdpa-{ncpu}thread", ncpu, vocab, a.reps))

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
