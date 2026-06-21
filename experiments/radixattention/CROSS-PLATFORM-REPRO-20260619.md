# Cross-Platform Reproduction — RadixAttention deterministic metrics (2026-06-19)

> **Claim under test (from [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md)):**
> *"The deterministic metrics (token speedup, hit rate) are hardware-independent and
> reproduce the committed JSON exactly; only the live wall-clocks are single-box."*
>
> This file is the **second-architecture witness** for that claim. The committed
> `radixbench-*-agents-fresh-20260619.json` ladder was measured on a **Mac M3 Pro
> (darwin/arm64)**. The run below re-runs the same `agents` workload on a **Windows 11
> / AMD x86_64, 32-thread** box and shows the deterministic fields land bit-for-bit
> identical, while the live wall-clock ratio moves (as the thesis predicts).

## Workload

`agents` — 5 concurrent agents share a 128-token system prefix, each with a private
growing context, ReAct-shaped interleaved arrivals. `sys=128 step=24 agents=5 turns=6`,
SmolLM2-135M Q8_0, `internal/radixkv` RadixAttention-style prefix cache.

## Result: deterministic fields match exactly across architectures

| Field | Mac M3 (arm64) — committed | Windows x86_64 (this box) | Match |
|---|---|---|---|
| `cache_hit_rate` | 0.8666666666666667 | 0.8666666666666667 | ✓ |
| `prefill_token_speedup` | 7.5 | 7.5 | ✓ |
| `radix_reused_tokens` | 5512 | 5512 | ✓ |
| `radix_computed_tokens` | 848 | 848 | ✓ |
| `bounded_fcfs_hit_rate` | 0.6214 | 0.6214 | ✓ |
| `bounded_cacheaware_hit_rate` | 0.8667 (100% of optimal) | 0.8667 (100% of optimal) | ✓ |
| policy-eviction witness | freed 8, benign sibling kept | freed 8, benign sibling kept | ✓ |

The deterministic scheduling/reuse metrics are functions of the trace and the cache
policy, not of the floating-point hardware — so they are identical bit-for-bit. The
`internal/radixkv` split-reuse == recompute gate (max|Δ|=0) is what licenses this.

## Live wall-clock: hardware-dependent, as expected

| Box | live baseline → radix | live ratio |
|---|---|---|
| Mac M3 Pro (arm64) | committed | **4.58×** |
| Windows x86_64, 32-thread (this box) | 3009 → 1157 ms | **2.60×** |

The lower live ratio on the 32-thread x86 box is **the thesis, not a contradiction**:
its parallel baseline prefill at 135M is relatively cheaper, so the fixed
clone/memcpy cost is a larger fraction of the saved work — exactly the
"clone-overhead is a small-model artifact" regime documented in
[RADIXATTENTION-RESULTS.md](../../RADIXATTENTION-RESULTS.md). On both boxes the live
ratio sits below the deterministic 7.50× token ceiling and would climb toward it as
the model grows (the Mac ladder shows 135M 4.58× → 1.5B 6.95×).

## Reproduce

```powershell
# from fak/ — deterministic-only (instant, hardware-independent):
go build -o radixbench.exe ./cmd/radixbench
.\radixbench.exe -dir internal\model\.cache\smollm2-135m -quant -only agents -live=false

# with the live wall-clock arm (reps=3, min reported):
.\radixbench.exe -dir internal\model\.cache\smollm2-135m -quant -only agents -reps 3
```

## Witnesses

- `radixbench-smollm2-135m-q8-agents-x86win-20260619.json` — live arm (this box).
- `radixbench-smollm2-135m-q8-agents-x86win-deterministic-20260619.json` — `-live=false`.
- Mac committed baseline: `radixbench-smollm2-135m-q8-agents-fresh-20260619.json`.

## What licenses the bit-for-bit claim

The determinism is not luck: `go test ./internal/radixkv/` proves split-reuse ==
recompute with `max|Δ|=0` (green, uncached, 2026-06-19). The scheduling/reuse metrics
are pure functions of the trace and cache policy, so they cannot vary by FP hardware.

> Provenance note: the data above was first committed under a `test(...)` subject
> (`20f74ef`); the diff is benchmark **data + this witness doc**, not new `*_test.go`
> code, so `dos commit-audit` correctly flagged the subject *type* as over-claimed.
> The reproduction itself is real and is backed by the `internal/radixkv` gate named
> here. This `docs` follow-up corrects the record.
