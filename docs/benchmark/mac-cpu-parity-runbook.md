# Runbook — single-stream CPU Q8 parity: fak vs llama.cpp (Apple M3 Pro)

Reproduces [`experiments/model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json`](../../experiments/model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json) —
the canonical SOTA-parity baseline cited by [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
It is the **honest fence**: fak trails llama.cpp on single-stream raw throughput by design.
fak's value is the cross-agent **fleet-reuse** layer on top (4.1× vs a tuned warm-KV cache),
measured separately — never cross-compare the two regimes.

## What it measures

Both engines on **CPU only**, same machine, same Q8 weights, single stream (no batching, no
reuse) — apples-to-apples:

| Metric | fak (Q8) | llama.cpp CPU (Q8) | fak/llama.cpp |
|---|---:|---:|---:|
| Decode | 38.3 tok/s | 53.3 tok/s | **0.72×** |
| Prefill @256 | 242.0 tok/s | 418.5 tok/s | **0.58×** |

*(Apple M3 Pro, Qwen2.5-1.5B-Instruct Q8_0; fak `0.30.0`; llama.cpp build `541bf3762`.
Read the JSON's `metrics.*.fak_over_llamacpp` fields — do not hardcode the ratios.)*

The gap is the hand-tuned arm64 NEON GEMM llama.cpp has and fak's pure-Go kernel does not yet.
It *narrows* with model size (decode 0.39× → 0.53× across the 1.5B→7B ladder) and closes as the
arm64 register-blocked GEMM tile lands.

## Prerequisites (the bench box)

- Apple Silicon Mac, Go 1.26+ (`GOTOOLCHAIN=auto` self-fetches), `llama-bench` on `PATH`
  (Homebrew `llama.cpp`).
- Qwen2.5-1.5B-Instruct in two forms: HF safetensors (fak arm) and a matching `q8_0` GGUF
  (llama.cpp arm). Any equivalent local copies work; the ratio is what matters.

## Run it

```bash
# Build the fak bench harness from the public repo
git clone https://github.com/anthony-chaudhary/fak.git && cd fak
GOTOOLCHAIN=auto go build -o /tmp/modelbench ./cmd/modelbench

# fak arm — in-kernel Q8 (quantize-at-load), CPU, decode + prefill@256
/tmp/modelbench -hf <qwen2.5-1.5b-instruct-hf-dir> -lean \
  -decode-reps 5 -decode-steps 32 -prefill-reps 3 -prefill-sizes 256 \
  -out /tmp/fak-parity-1p5b.json

# llama.cpp arm — CPU (-ngl 0), same thread count, same Q8 GGUF
llama-bench -m <qwen2.5-1.5b-instruct-q8_0.gguf> -ngl 0 -t 12 -p 256 -n 32 -r 3 -o json \
  > /tmp/llama-parity-1p5b.json
```

The ratio is `fak.decode.tok_per_sec / llamacpp[n_gen>0].avg_ts` (decode) and the same over the
`n_prompt>0` row (prefill). The committed artifact embeds both raw arm outputs under `arms.*` for
provenance.

## Public-safety note

`modelbench`'s `source` field and `llama-bench`'s `model_filename` echo the local weight path
(e.g. `/Users/<you>/...`). **Scrub those to `~/...` and rewrite any personal hostname to
`node-macos-a` before committing** — a direct copy of raw bench output into the public tree
leaks operator paths the scrub audit's public needle-list cannot see. See the assembly note in
the artifact's `_doc`.
