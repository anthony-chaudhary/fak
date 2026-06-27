# GLM-5.2 Fak-Kernel Cache Value — On a Solved Ticket

> **📊 AUTHORITY:** This document's benchmark results are indexed in **[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md)**,
> the single source of truth for all committed performance claims.

> **⚠️ RESULT STATUS:** **PENDING — Results not yet collected.** This document describes the result packet shape and what will be measured once the live run executes on datacenter compute. The observation seam (`fak swebench cache-witness`) is shipped and tested; the live GLM-5.2 cache-value number is the box residual.

**Date:** 2026-06-27
**Commit:** _pending — results not yet shipped_
**DOS Verify:** N/A (no results to audit yet)
**Epic:** [#1010](https://github.com/anthony-chaudhary/fak/issues/1010) — GLM-5.2 on the pure fak kernel
**Child Issue:** [#1014](https://github.com/anthony-chaudhary/fak/issues/1014) — this result packet
**Observation Seam:** [`internal/cachewitness/`](../internal/cachewitness/) + `fak swebench cache-witness` (commit `52dfea0d`, diff-witnessed)

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

### Provider Cache — OBSERVED (upstream, not fak's)

| Metric | Expected Artifact Field | Value | Provenance |
|---|---|---|---|
| **Provider cache read tokens** | `provider_cache_read_tokens` | **0** | OBSERVED — always 0 on pure in-kernel path (no provider) |

**Honesty fence:** The two numbers are DISTINCT signals over distinct caches. A record that summed them would conflate trust classes (WITNESSED vs OBSERVED). The `cachewitness.Record` keeps them in separate fields and never derives one from the other.

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