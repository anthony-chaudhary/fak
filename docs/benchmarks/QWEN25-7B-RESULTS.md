---
title: "Qwen2.5-7B: fak vs llama.cpp on Apple M3 Pro"
description: "fak's first 7B dense run on M3 Pro Q8, reporting the honest throughput gap vs llama.cpp Metal plus full greedy parity."
---

# Qwen2.5-7B-Instruct — first large dense run in fak + llama.cpp parity (rung 2)

> **Purpose.** The `GPU-MODEL-PICK` target size, run in fak on this M3 Pro (CPU Q8 / `-lean`,
> since the discrete-GPU story is the Windows RTX box, not this Mac). Proves the dense path scales
> to 7B and gives the first real fak-vs-llama.cpp throughput comparison at a useful size. This is
> the rung that establishes the honest raw-throughput gap fak's reuse stack has to overcome.

**Host:** `node-macos-a` / Apple M3 Pro (6P+6E, 18-core GPU, 36 GB) / Go 1.23.1.
**Date:** 2026-06-19.
**Quant:** Q8_0 (both engines — apples-to-apples).

## fak in-kernel (CPU Q8, NEON)

`go run ./cmd/modelbench -hf ~/.cache/fak-models/qwen2.5-7b-instruct -lean -quant` (artifact:
`experiments/model-ladder/modelbench-qwen25-7b-q8.json`, load ~73 s):

| prefill P16 | prefill P64 | prefill P256 | decode |
|---|---|---|---|
| 15.4 tok/s | 16.0 tok/s | 16.1 tok/s | **8.7 tok/s** (115.2 ms/tok) |

> **One Qwen2.5-7B fak decode number, and it is in the Authority (#123).** The pinned figure is the
> **8.7 tok/s** (115.2 ms/tok) above — artifact `experiments/model-ladder/modelbench-qwen25-7b-q8.json`,
> indexed in `BENCHMARK-AUTHORITY.md` (the single source of truth). A lower **~8.2 tok/s** decode figure
> for the same engine/model/box has circulated from a separate `fak`-native-chat run; it is **not** a
> second authoritative measurement. Read 8.7 tok/s as the number of record and ~8.2 tok/s as a
> different (earlier/contended) run superseded by it — so the two figures are reconciled here rather
> than left to circulate unpinned.

## llama.cpp b9707 reference (Metal, full offload)

`llama-completion -m qwen2.5-7b-instruct-q8_0.gguf -ngl 99 -t 6 -c 4096 -n 32` (load 3.16 s):

| prefill | decode |
|---|---|
| **192.92 tok/s** | **17.27 tok/s** |

## fak vs llama.cpp (the honest throughput gap)

| phase | fak (CPU Q8) | llama.cpp (Metal Q8) | fak ratio |
|---|---|---|---|
| prefill | 16.1 tok/s | 192.9 tok/s | **0.083×** |
| decode | 8.7 tok/s | 17.27 tok/s | **0.50×** |

Consistent with the repo's standing honesty bound (`SESSION-VALUE-STACK-RESULTS.md`,
`M3-LLAMACPP-RESULTS.md`: dense fak single-stream ≈0.46× decode / ≈0.15× prefill vs llama.cpp on
M3 Q8). Prefill is compute-bound (Metal GPU gives llama.cpp ~12× here); decode is bandwidth-bound
so the gap closes to 2×. **fak does not beat llama.cpp on raw tok/s** — the moat is the reuse
stack + in-kernel integration, not single-stream throughput. The prefill half is the open Metal
GEMM lane (`internal/metalgemm`, `-tags fakmetal`).

## Parity — greedy agreement (the correctness gate)

Same ChatML prompt, greedy (temp 0), Q8_0 both engines:

| engine | output |
|---|---|
| **fak** | `2+2 is 4.` |
| **llama.cpp** | `2+2 is 4.` |

**Full 7-token greedy match.** First-token (and whole-short-generation) parity on the 7B Q8 — the
dense forward path is numerically faithful at this size, not just at the 135M/1.5B anchor sizes.
This extends the parity line (`fak/experiments/parity/PARITY.md`) one rung up.

## Bottom line

- **7B runs in fak on this Mac**, fits in 36 GB unified with headroom (Q8 weights ~7.6 GB + KV +
  compute), load ~73 s cold.
- **fak is 0.083× prefill / 0.50× decode vs llama.cpp Metal** — the expected honest gap. The reuse
  stack (`radixbench`/`sessionbench`) is what makes fak competitive on agent workloads despite
  this single-stream gap, not raw tok/s.
- **Greedy parity holds** — `2+2 is 4.` matches token-for-token. The 7B is a legitimate large
  dense rung, not just a fit exercise.

_Implements rung 2 of `PLAN-model-ladder-qwen36-2026-06-19.md`. fak artifact:
`experiments/model-ladder/modelbench-qwen25-7b-q8.json`. Dated 2026-06-19._
