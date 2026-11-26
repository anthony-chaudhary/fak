# Retiring the vendored Inkling Metal runtime

**Audience:** maintainers deciding what runs Inkling-2-Mini on an Apple GPU, and when the vendored runtime can go. The code map is in [`inkling-metal-runtime.md`](inkling-metal-runtime.md); this page is the plan for leaving it.

**Evidence labels.** *Measured (contended)* means a hardware run on the M3 Pro 36 GiB reference machine while other sessions loaded it (load average far above 4); the ratio between paths is meaningful, the absolute number is not a release figure. *Author-measured* means the change's author ran it and no independent reviewer reran it. *Read from code* means it comes from the sources and has not been profiled. All rows use `Inkling-2-Mini-UD-Q3_K_XL.gguf`, greedy (`--temp 0`).

## Role of the vendored code

The C++/Metal runtime in `third_party/inklingmetal` (from `incoai/splash`, Apache-2.0), driven through `internal/inklingmetal`, is **not the intended hot path**. It is kept for two reasons only:

1. **Comparison bar.** It is the fastest correct Inkling implementation on Apple GPUs that fak can run today. Every fak-native Metal change is measured against it on the same file, prompt and machine.
2. **Learning source.** Its techniques are studied, cited and re-implemented in fak's own Metal stack (`internal/metalgemm`, `internal/compute`, `internal/ggufload`). Copying is allowed; each borrowed technique is tracked by provenance and understood by a fak maintainer before it lands.

Once the fak-native path carries the same work at parity, the vendored tree is removed. It is a scaffold, not a dependency to maintain indefinitely.

## What to learn from it first

Ordered by the share of the measured gap each closes. "Lands in" is where the fak-native version belongs.

| Technique | Vendored location | Lands in | Why first |
|---|---|---|---|
| DFlash2 speculative decode: draft model, tree-shaped verify, accept loop | `model/DFlashDraft.cpp`, `model/VerifyBudget.cpp`, `model/StepCosts.cpp` | `internal/compute` (`Inkling2VerifyDevice`, a draft device) over `internal/metalgemm` kernels; acceptance stays in fak's spec-decode code | Worth about 2.3x decode (42.5 vs 18.4 tok/s). The largest single gap. |
| Verify width schedule | `model/VerifyBudget.cpp`, `model/StepCosts.cpp` | `internal/compute` next to the verify device | Must be timing-independent: a width chosen from measured GPU time made vendored greedy output non-reproducible. fak's version prices width from a fixed table, and its verify device is width-invariant per row. |
| Load-time repack of quantized formats into tiled plane images | `model/GgufPreparation.cpp`, `metal/kernels/shared/gguf_repack.metal` | `internal/ggufload` (repack at load) feeding `internal/metalgemm` packed GEMV | Big GEMVs already stream near the memory roof, so the gain is in layout for MoE and the q8 rows, not in raw bandwidth. |
| Paged int8 KV with split attention | `ops/PagedKv.hpp`, `ops/PageStorage.cpp`, `ops/PagedAttention.cpp` | `internal/metalgemm` (device KV, q8 long-context attention) | Decode cost at depth and long-context memory. fak already has q8 device KV for Qwen; Inkling needs it wired. |
| Prefill kernels (batched rows per dispatch) | `ops/LinearGguf.cpp` multi-row paths, prefill arenas in `model/RuntimeArenas.cpp` | `internal/metalgemm` kernels driven by `internal/compute` (`Inkling2PrefillDevice`) | Prefill is the widest gap (about 425 vs 61–64 tok/s). |
| Lanes (several sequences per pass) | upstream engine batching (not vendored); fak's `StepBatch` bridge over the vendored step | `internal/compute` batch device and the `fak serve` batcher | Needed for serve throughput; the vendored bridge was single-lane until fak widened it. |

## Stages and exit gates

Each stage is entered only when its gate passes on recorded receipts, not on a claim.

### Measured rows (the bar)

Every gate below compares fak-native against vendored on these rows, one run per target serially, GPU lease held, machine load average below 4, same binary:

1. Decode tok/s, short prompt (31 tokens), 256 new tokens, speculation on for both.
2. Decode tok/s, short prompt, speculation off for both.
3. Decode tok/s at 8k and 32k context.
4. Prefill tok/s at 537 and 8k prompt tokens.
5. Time to first token on a repeated 4k-token prefix (prefix reuse).
6. Aggregate decode tok/s at 2 and 4 concurrent sequences.
7. Correctness: greedy token ids identical across runs of the same target, and identical to the fak host forward on the first 16 ids.

### Stage 1: fak-native resident becomes the default

- **Gate:** native decode (rows 1–3) and prefill (row 4) are each at or above vendored; row 7 passes. Rows 5–6 may trail but must be recorded.
- **Change:** `FAK_INKLING_METAL_TARGET` default flips from `vendored` to the resident target. `vendored` stays selectable.

### Stage 2: vendored is opt-in and labelled as comparison

- **Gate:** stage 1 has held for one release with no correctness regression reported against the resident target.
- **Change:** the vendored target builds only behind a build tag (not in default builds); when selected, the stderr target line and the `fak serve` startup line say it is a comparison target. Default builds no longer compile `third_party/inklingmetal`.

### Stage 3: removal

- **Gate:** fak-native carries speculation, prefill and serve batching (rows 1–6) at parity for two consecutive releases. (Two is the default; change it here if a release cadence argues otherwise.)
- **Change:** delete `third_party/inklingmetal` and `internal/inklingmetal`, their notices entries and the provenance pin.
- **What replaces it as the bar:** a recorded receipt set, checked in with the removal commit, holding the last vendored numbers for rows 1–7 with the fak revision, upstream revision, machine and load; the native hardware tests (greedy-id identity against the host forward, determinism across processes, width invariance of verify); and the bench that produced the receipts, rerun against the native path each release. After removal the bar is the recorded vendored receipt plus "no regression against the previous native receipt".

## Current scoreboard

Decode at 31-token prompt unless noted. Rows marked *unlanded* come from reviewed lanes that are not yet on `main`.

| Path | Decode tok/s | Prefill tok/s | Evidence |
|---|---|---|---|
| Vendored, DFlash2 draft on (default) | 42.5 (38.9–44.6) | about 425 (165–497 across prompt lengths) | Decode: *measured (contended)*, load average 16–191. Prefill: *measured* baseline on fak 0.57.0. |
| Vendored, draft off | 18.4 | — | *Measured (contended)*, same session as above. |
| fak resident device (`compute.Inkling2Device` on `metalgemm`) | 23.4 warm; 22.3 in a later profile | 22–23 per-token; 61–64 batched at 537 tokens (opt-in) | *Author-measured*, load average about 150–200; 16/16 greedy ids identical to the host forward. *Unlanded.* |
| fak host-driven Metal forward (`native` target today) | 3.6–4.4 (best 5.4) | 2037-token prefill did not finish in 330 s | *Measured (contended)*. Host attention and about 240 submit-and-wait cycles per token. |

Reading the table: without speculation the resident device already passes the vendored path (23.4 vs 18.4). The remaining decode gap is speculation; the prefill gap is about 7x and is the widest.

Other recorded rows, for the gates: vendored aggregate decode without draft 40.6 tok/s at 2 exact lanes and 58.2 at 4 wide lanes (not bit-reproducible); vendored prefix reuse cuts time to first token from 8.95 s to 0.50 s at 4250 tokens; vendored decode 37.8 tok/s at 8k and 35.9 at 32k context. All *measured (contended)* or *author-measured*; rerun quiet before any gate decision.

## Known defects that affect the gates

- Vendored speculative greedy output was non-reproducible when verify width was chosen from measured GPU time; the default now uses a fixed cost table (about 2–4% slower on the short prompt, measured under load).
- Every vendored request re-prefilled from scratch before prefix reuse; re-prefill costs about 79 s at 32k tokens.
- The resident verify device has unit-test width invariance but no quiet hardware speculative run yet; an earlier run under heavy load showed superlinear verify cost and an out-of-memory at 5 rows.
- None of the timings above were taken on a quiet machine. The first quiet run of rows 1–7 is the first real scoreboard.
