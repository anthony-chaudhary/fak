# GLM-5.2 Fak-Kernel Cache Value — On a Solved Ticket

> **📊 AUTHORITY:** This document's benchmark results are indexed in **[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md)**,
> the single source of truth for all committed performance claims.

> **⚠️ RESULT STATUS:** **PENDING — Results not yet collected.** This document describes the result packet shape and what will be measured once the live run executes on datacenter compute. The observation seam (`fak swebench cache-witness`) is shipped and tested; the live GLM-5.2 cache-value number is the box residual.

**Date:** 2026-06-27
**Commit:** _live cache-value pending — see [DOS Binding](#dos-binding--provenance-of-every-number)_
**DOS Verify:** the offline WITNESSED headline (the deterministic prefill-elimination floor) is bound to its commit and resolves under `dos verify`; the live WITNESSED cache value is reported `not yet` (host-gated on [#1012](https://github.com/anthony-chaudhary/fak/issues/1012)). See [DOS Binding](#dos-binding--provenance-of-every-number).
**Epic:** [#1010](https://github.com/anthony-chaudhary/fak/issues/1010) — GLM-5.2 on the pure fak kernel
**Child Issues:** [#1014](https://github.com/anthony-chaudhary/fak/issues/1014) — this result packet · [#1013](https://github.com/anthony-chaudhary/fak/issues/1013) — DOS binding + provenance of every number
**Observation Seam:** [`internal/cachewitness/`](../internal/cachewitness/) + `fak swebench cache-witness` (commit `52dfea0d`, `dos commit-audit` → diff-witnessed)

## Summary

| Claim | Number | Baseline | Context |
|---|---|---|---|---|
| **Cache value (reused tokens)** | **PENDING** | No cache baseline | GLM-5.2 on pure fak kernel serving a solved SWE-bench ticket |

## What This Measures

This benchmark measures the **cache value** — the prefilled tokens served from fak's in-kernel KV-prefix cache that the kernel did NOT recompute. Specifically:

- The work saved on turns 2..N when the Claude harness drives a real, already-solved SWE-bench ticket against a GLM-5.2 fak-kernel gateway
- The cached KV prefix (system + tools + repo) is served on every turn, avoiding the expensive prefill cost
- This is a **WITNESSED** metric from fak's own kernel, not an observed upstream provider number

The metric that matters is `kv_prefix.reused_tokens` — the number of prefilled tokens served from the cached KV prefix.

## Why a *Solved* Ticket

GLM-5.2 in fak's kernel decodes at ~0.03–0.17 tok/s under `--cpu-offload-experts` (due to the MoE expert GEMM wall — [#996](https://github.com/anthony-chaudhary/fak/issues/996)/[#971](https://github.com/anthony-chaudhary/fak/issues/971)). This is too slow to generate a full patch in reasonable wall-clock. The runnable proof routes *around* the throughput wall, not *through* it:

1. Take a **real, already-solved** SWE-bench Verified instance (gold patch + gold test known)
2. Drive it through the **Claude harness wired to the GLM-5.2 fak-kernel gateway**
3. **Observe the cache value** — the lever the goal names

This proves the in-kernel cache-value lever end-to-end even if the full patch is not generated.

## Workload

- **Model:** GLM-5.2 (Q4_K_M quantization, served via `fak serve --engine inkernel --backend cuda --cpu-offload-experts`)
- **Hardware:** 8×A100 sm_80 datacenter GPU (residual — box access required)
- **Task:** One or more solved SWE-bench Verified instances from `testdata/swebench_smoke.json`
- **Agent:** Claude harness (`fak swebench run --agent fleet`) wired to the fak-kernel gateway
- **Context Budget:** 8192 tokens (kept within GLM-5.2's 1M-context default to avoid `FitTooBig`)

## Results (PENDING — Will Fill When Data Arrives)

### Cache Value — WITNESSED (fak's own cache)

| Metric | Expected Artifact Field | Value | Provenance |
|---|---|---|---|
| **Reused tokens** | `kv_prefix.reused_tokens` | **PENDING** | WITNESSED — fak authored this count |
| **Prefill tokens (denominator)** | `kv_prefix.prompt_tokens` | **PENDING** | WITNESSED |
| **Cache hit ratio** | `kv_prefix.reused_tokens / kv_prefix.prompt_tokens` | **PENDING** | WITNESSED — derived from witnessed fields |
| **Frozen turns (reuse ≥ 0.90)** | `kv_prefix.frozen_turns` | **PENDING** | WITNESSED |
| **Partial turns** | `kv_prefix.partial_turns` | **PENDING** | WITNESSED |
| **Cold turns (reuse < 0.10)** | `kv_prefix.cold_turns` | **PENDING** | WITNESSED |

### Prefill Work-Elimination Floor — WITNESSED-derived (deterministic, offline)

This is the **offline WITNESSED headline** the epic names: the prefill-token work each
arm processes, computed *deterministically* from the SWE-bench instance geometry
(`internal/swebench/cost.go`, `PrefillAgg.AOverC`/`AOverB`). It needs **no box, no GPU,
no model** — it is timing-free arithmetic, so it resolves under `dos verify` today, bound
to the shipped `cost.go` commit. It is **WITNESSED-derived** (fak computes it), distinct
from the live WITNESSED cache count below and from any OBSERVED provider/box reading.

| Metric | Source field | Value | Provenance |
|---|---|---|---|
| **A/C — re-prefill vs fak-fused** | `PrefillAgg.AOverC` | **17.9× → 23.4×** (workers 1→16) | WITNESSED-derived — deterministic from geometry |
| **B/C — per-agent-KV vs fak-fused** | `PrefillAgg.BOverC` | **1.0× → 1.31×** (workers 1→16) | WITNESSED-derived |
| **A/B — turn-tax** | `PrefillAgg.AOverB` | computed per geometry | WITNESSED-derived |

These figures are the committed value-stack floor (see
[SWEBENCH-RESULTS.md](SWEBENCH-RESULTS.md)); they are a *related but distinct* quantity
from the live in-kernel `reused_tokens` and must never be reported as the live cache
value. The deterministic floor answers "how much prefill work the geometry lets fak
eliminate"; the live `reused_tokens` answers "how much fak's RadixAttention actually
served from cache on this run."

### Provider Cache — OBSERVED (upstream, not fak's)

| Metric | Expected Artifact Field | Value | Provenance |
|---|---|---|---|
| **Provider cache read tokens** | `provider_cache_read_tokens` | **0** | OBSERVED — always 0 on pure in-kernel path (no provider) |

### Live Decode Reading — OBSERVED (a reading of the box, not a fak claim)

| Metric | Source | Value | Provenance |
|---|---|---|---|
| **Decode throughput (tok/s)** | live serve on the dgx box | **`not yet`** (~0.03–0.17 expected under `--cpu-offload-experts`) | OBSERVED — relayed reading of a live box |

The tok/s is a reading of the hardware under the [#996](https://github.com/anthony-chaudhary/fak/issues/996)/[#971](https://github.com/anthony-chaudhary/fak/issues/971)
expert-GEMM wall. It is **OBSERVED**, never WITNESSED, and the slow figure is **never
attributed to a fak action** — it is the host's MoE-offload cost, not a kernel fault.

**Honesty fence (all four number-classes).** The packet keeps **two trust classes**
strictly apart:

- **WITNESSED** (fak controls): the live in-kernel `kv_prefix.reused_tokens`, and the
  WITNESSED-*derived* deterministic prefill-elimination floor (`AOverC`/`AOverB`).
- **OBSERVED** (relayed from an external party): the provider `cache_read` (0 here), and
  the live box decode tok/s.

No number sums or derives across the line: the `cachewitness.Record` keeps the WITNESSED
and OBSERVED cache fields separate and never derives one from the other, the deterministic
floor is never reported as the live cache value, and a slow OBSERVED tok/s is never blamed
on a fak action. This is the `fak conflation-scorecard` discipline applied to the result
packet (`internal/conflationscore`, A / `conflation_debt 0`).

## Methodology — The Observation Seam

The observation is performed by `fak swebench cache-witness` (commit `52dfea0d`), which:

1. Scrapes the gateway's `/metrics` endpoint for the cache family
2. Folds it into a `cachewitness.Record` with provenance labeling
3. Emits JSON with the structure shown in the Results tables above

The command:

```bash
# Direct scrape, if gateway is HTTP-reachable:
fak swebench cache-witness --gateway 127.0.0.1:8080 --out run-glm52-cache/cache-witness.json

# Or via captured metrics (when box is reachable only over lab bridge):
curl -s localhost:8080/metrics > metrics.txt
fak swebench cache-witness --metrics-file metrics.txt --out cache-witness.json
```

The `cache-witness.json` artifact is the unit that graduates into BENCHMARK-AUTHORITY.md once the live number is collected.

## Full Runbook

See [GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK.md](GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK.md) for the complete end-to-end path:

1. Serve GLM-5.2 from the pure kernel
2. Drive the Claude harness over a solved ticket
3. Read the cache value

## Milestone 2 — The Bar for Epic #1010

The cache **BIT** milestone: `cache-witness.json` shows `reused_tokens > 0` on turns 2..N from a live GLM-5.2 fak-kernel serve. This proves the cache-value lever end-to-end through fak's own kernel.

**Stretch (gated on #996/#971):** A non-zero resolve-rate from GLM-5.2-fak-kernel, graded by the official harness (`fak swebench eval`). Not required to close #1010.

## DOS Binding — Provenance of Every Number

The rule (epic #1010, child #1013): **the cache-value number that graduates must be
diff-witnessed, not self-reported.** It is bound by `dos verify` / `dos commit-audit` to
the commit that produced it — never to a worker's narration. An unproven step is reported
`not yet` with the missing witness, never shipped.

**Bound now (resolves under `dos verify` today):**

| Number | Trust class | Binding |
|---|---|---|
| Observation seam (`fak swebench cache-witness`) | WITNESSED tooling | commit `52dfea0d` — `dos commit-audit` → **diff-witnessed** |
| Deterministic prefill-elimination floor (A/C, B/C, A/B) | WITNESSED-derived | bound to the shipped `internal/swebench/cost.go` commit; timing-free, resolves offline under `dos verify` |
| Provider `cache_read` = 0 (pure in-kernel path) | OBSERVED | structural (no provider on the in-kernel path) — not a fak claim |

**`not yet` (the missing witness is named, not faked):**

| Number | Trust class | Missing witness |
|---|---|---|
| Live in-kernel `kv_prefix.reused_tokens` > 0 on turns 2..N | WITNESSED (live) | a live GLM-5.2 fak-kernel serve on the 8×A100 dgx box — child [#1012](https://github.com/anthony-chaudhary/fak/issues/1012), host-gated |
| Live decode tok/s | OBSERVED | same live serve; expected ~0.03–0.17 under the #996/#971 expert-GEMM wall |

When the live run lands (#1012), its results commit is bound the same way: `dos commit-audit <results-sha>` must grade **diff-witnessed** and `dos verify` resolves the headline, before any live number graduates into [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md). Until then the live cache value stays `not yet` — the deterministic floor is the honest dos-bound headline available without the box.

**Conflation contract:** every number above carries its trust class; no number sums or
derives across the WITNESSED/OBSERVED line; `fak conflation-scorecard` is clean
(grade A, `conflation_debt 0`) on the reporting surfaces.

## Provenance and Discipline

- **Observation seam:** `internal/cachewitness/` + `fak swebench cache-witness` (commit `52dfea0d`, `dos commit_audit` → diff-witnessed)
- **Provenance split:** WITNESSED (fak's own cache) vs OBSERVED (provider's cache), matching the conflation-scorecard line
- **Metric definitions:** `internal/gateway/metrics.go` (`writeKVPrefixMetrics`)
- **Result packet format:** This document follows the [BENCHMARK-TEMPLATE.md](../BENCHMARK-TEMPLATE.md) standard
- **Gate / dependency:** Datacenter GPU access (8×A100 sm_80 box) — the current residual

## Cross-References

- **Runbook:** [GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK.md](GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK.md) — how to run the benchmark
- **Pure-kernel serving:** [SWEBENCH-PURE-KERNEL-RUNBOOK.md](SWEBENCH-PURE-KERNEL-RUNBOOK.md) — how to serve models from fak's own kernel
- **Metric provenance:** `internal/cachewitness/cachewitness.go` — WITNESSED vs OBSERVED discipline
- **Throughput wall:** [#996](https://github.com/anthony-chaudhary/fak/issues/996) / [#971](https://github.com/anthony-chaudhary/fak/issues/971) — why this routes around full-patch generation
- **Epic parent:** [#1010](https://github.com/anthony-chaudhary/fak/issues/1010) — GLM-5.2 on the pure fak kernel

---

## Pending Status — Not Yet Collected

This result packet is **NOT YET SHIPPED**. The numbers are PENDING because:

1. The observation seam is fully shipped and tested (`dos commit_audit 52dfea0d` → OK)
2. The datacenter GPU box (8×A100) access is the current residual
3. Once the live run executes, the `cache-witness.json` artifact will be committed and the tables above will be filled with real WITNESSED numbers

When results are collected, this document will be updated with:

- Actual commit hash of the results commit
- Real numbers in the Results tables (no placeholders)
- `dos_commit_audit <hash>` → **OK** verification
- Entry in [BENCHMARK-AUTHORITY.md](../BENCHMARK-AUTHORITY.md) referencing this document

**Until then, this document serves as the result packet shape — what will be measured, how, and under what provenance discipline.**