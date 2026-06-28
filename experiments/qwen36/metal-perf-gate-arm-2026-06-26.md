# Metal Q4_K perf-gate arm (#64, epic #300)

The fak-Metal vs llama.cpp-Metal perf gate for Qwen3.6-27B on Apple Silicon.
Greedy/first-token parity for the Metal lane **already holds** (the
`TestMetalQ4KDecodeMatchesCPU` microbench: greedy token seq == CPU
`[433 92 166 106]`, oracle `248068` path). This is the **perf half** of #64.

## Run it

```sh
# #300 acceptance is "within 2x llama.cpp Metal" -> fak/llama >= 0.5
python tools/qwen36_perf_gate.py --metal --min-ratio 0.5 \
  --out      experiments/qwen36/qwen36-perf-gate-metal-20260626.json \
  --markdown experiments/qwen36/qwen36-perf-gate-metal-20260626.md
```

Exit `1` (fail-closed) is the **expected** current state: it records the open
GPU gap, and flips green automatically when fak-Metal reaches the bar.

## Recorded verdict (M3 Pro / node-macos-a, 2026-06-26)

| metric | fak-Metal Q4_K | llama.cpp-Metal bar | ratio | within-2x? |
|---|---:|---:|---:|---|
| decode | 1.2 tok/s | 7.29 tok/s | 0.16x | FAIL |

- **Witness:** `experiments/benchmark/runs/by-machine/node-macos-a/20260626T055239Z-q4k-metal-decode-27b/score.json`
  (`-tags fakmetal`, `FAK_Q4K=1 FAK_METAL=1`, llama booted out for a clean 36 GB GPU).
- **Artifacts:** `metal-fak-q4k-27b-20260626.json` (fak) and
  `llamacpp-metal-qwen36-bar-20260626.json` (the bar).

## Honesty caveats

- **The bar is observed-external, not a fak witness.** The llama.cpp-Metal
  numbers (51.55 prefill / 7.29 decode) are provenance-caveated per **#459**
  (build/version + exact flags) and **#452** (measurement conditions). The gate
  carries the caveat through into its report.
- **Prefill is not yet asserted.** The decode metric is length-agnostic and
  asserted here. The prefill bar (51.55 @ pp22) is recorded for provenance but
  scored only once a *matched-length* fak-Metal prefill point exists. Per #64
  comment#2 a prefill gate must **fix the prompt length** (P=256/512), not P~22:
  the witnessed fak-Metal prefill is 0.6 tok/s @ P=29 (a tiny-prompt,
  weight-read-dominated artifact) vs 4.5 tok/s @ P=421 (the honest agentic-length
  rate) -- an order of magnitude apart. Producing the matched-length pair needs a
  live run on the Metal verify node (M3 Pro, `-tags fakmetal`).

## What closes the gap

The decode wall is ~336 separate per-token command-buffer GEMVs at ~360us fixed
launch overhead each, plus low in-kernel utilization. The lever is a
one-command-buffer-per-token GPU-resident decode forward (#67), with the
quantized device GEMM / resident-weights work tracked in #69/#70/#71.
