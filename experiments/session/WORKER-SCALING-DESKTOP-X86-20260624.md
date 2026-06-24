# Decode vs prefill worker-count scaling — x86_64 32-core desktop (2026-06-24)

**Machine:** desktop agent-host (HARDWARE-CATALOG node `desktop`, x86_64, 32 logical cores,
windows, go1.26.3). This is the live agent-host — **run-on-bench-nodes-by-default excludes
it as a timing target.** Captured during an overnight dogf/bench session while the box was
the active agent host (lightly contended).

**Artifact:** [`worker-scaling-desktop-x86-20260624.json`](worker-scaling-desktop-x86-20260624.json)
**Engine:** fak in-kernel Q8_0 (pure-Go). **Harness:** `cmd/modelbench -gguf <Q8> -lean`, `FAK_WORKERS` pinned per row.

## The finding

Decode (single-token GEMV, memory/overhead-bound) and prefill (batched GEMM, compute-bound)
have **opposite worker-count optima.** The single global `FAK_WORKERS` / `GOMAXPROCS` default
(all 32 cores) lands in decode's **worst** regime on small models — and the penalty shrinks
as the model grows.

| model | decode peak | decode @ 32w (default) | **default penalty** | prefill peak |
|---|---|---|---|---|
| Qwen2.5-1.5B Q8 | 18.1 tok/s @ 8w | 7.3 tok/s | **2.5×** | 434 tok/s @ 24w |
| Qwen2.5-3B Q8 | 8.6 tok/s @ 8w | 4.0 tok/s | **2.1×** | 201 tok/s @ 16w |
| Qwen2.5-7B Q8 | 3.75 tok/s @ 16w | 3.28 tok/s | **1.14×** | 109 tok/s @ 32w (still scaling) |

The decode over-threading penalty is **largest on small models and shrinks as the model
grows** — the CPU-threading analogue of the repo's documented GPU *launch-bound is a
small-model artifact* finding (`CROSSOVER-1P5B-RX7600`, `VULKAN-Q8-RX7600`). On small models
the per-token decode work is tiny, so the many-thread barrier cost dominates when you
over-subscribe; on 7B the per-token GEMV is large enough to amortize the barrier across all
cores. Prefill's compute-bound GEMM scales with cores on every size (increasingly so as the
model grows).

## Honest fences

- **Within-run RATIO is the claim, not the absolute tok/s.** This box is the contended
  agent-host; its absolute numbers are **not** comparable to the uncontended Apple M3 Pro
  authority rows (canonical M3 Pro 1.5B Q8 decode = **38.1 tok/s**). Governance blesses
  within-run ratios; the worker A/B is measured same-box, same-run.
- **The decline is intrinsic, not pure contention.** It already appears at 16w and 24w
  (1.5B: 18.1 → 16.7 → 14.7 tok/s), both of which leave cores free. The sharp 32w cliff is
  part many-thread barrier cost, part contention amplification on a fully-subscribed box.
- **Single global worker count can't optimize both phases.** Phase-aware (or
  model-size-aware) worker selection would recover up to ~2.5× decode on small models at
  **zero** cost to prefill or to large-model decode. `internal/model` already optimized the
  decode worker pool (`parallel.go` spin-then-park barrier) and ships
  `internal/model/bench_scaling.sh` to measure exactly this tension — this is the x86_64
  32-core data point for that tool's question. The default-selection follow-up is left as a
  core change (not made here: `internal/model` was under active edit).

## Reproduce

```bash
for W in 4 8 16 24 32; do
  FAK_WORKERS=$W go run ./cmd/modelbench \
    -gguf ~/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct.Q8_0.gguf \
    -lean -decode-steps 32 -decode-reps 5
done
```
