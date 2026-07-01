---
title: "GLM-5.2 on the pure fak kernel — the cache-value-on-a-solved-ticket runbook"
description: "The end-to-end path to observe fak's OWN in-kernel KV-prefix cache value while the Claude harness drives a real, already-solved GitHub ticket against a GLM-5.2 fak-kernel gateway on our datacenter compute — with the WITNESSED-vs-OBSERVED provenance the number is reported under."
---

# GLM-5.2 fak-kernel cache value, on a solved ticket — runbook

> **What this is.** The executable path for epic [#1010](https://github.com/anthony-chaudhary/fak/issues/1010):
> serve GLM-5.2 from fak's **own** CUDA forward pass on our sm_80 8-GPU datacenter server box, drive the
> **Claude harness** against it over a **real, already-solved** SWE-bench Verified instance,
> and **observe the cache value** — the repeated system+tools+repo prefix fak's RadixAttention
> serves from cached KV when reuse is available, reported as aggregate prefill the kernel did
> **not** redo. That reused-token count, reported as WITNESSED, is this runbook's headline datum.
>
> **Status: the observation seam is SHIPPED and tested; the live GLM-5.2 number is the box
> residual.** `fak swebench cache-witness` (commit `52dfea0d`, child #1011) reads the cache
> value off a live gateway. The number itself comes from a run on the box (child #1012).
> Nothing here invents a tok/s or a reuse figure.

---

## 1. Why a *solved* ticket and the *cache value*, not a throughput race

GLM-5.2 in fak's kernel decodes at ~0.03–0.17 tok/s under `--cpu-offload-experts` (the
[#996](https://github.com/anthony-chaudhary/fak/issues/996) / #971 host expert-GEMM wall) —
too slow to *generate* a full patch in reasonable wall-clock today. So the runnable proof is
not "GLM-5.2 writes the whole patch fast." It is:

1. Take a **real, already-solved** instance (gold patch + gold test known), so correctness is
   checkable from evidence rather than dependent on a slow full generation.
2. Drive it through the **Claude harness wired to the GLM-5.2 fak-kernel gateway**.
3. **Observe the cache value** — the lever the goal names. This routes *around* the throughput
   wall (#996), not *through* it: aggregate KV-prefix reuse during the solved-ticket run proves
   the in-kernel cache-value lever end-to-end even if the full patch is not generated.

## 2. The data observation, in DOS terms (two numbers, two trust classes)

The codebase draws the provenance line the [conflation scorecard](../CONFLATION-SCORECARD.md)
requires. `fak swebench cache-witness` folds it into one record:

| Field | Metric | Provenance | Meaning |
|---|---|---|---|
| `kv_prefix.reused_tokens` | `fak_gateway_kv_prefix_reused_tokens_total` | **WITNESSED** | fak's OWN cache: the RadixAttention prefix match the kernel did not re-prefill. fak authored it. |
| `kv_prefix.prompt_tokens` | `fak_gateway_kv_prefix_prompt_tokens_total` | **WITNESSED** | the prefill-token denominator of the realized cache-hit. |
| `kv_prefix.{frozen,partial,cold}_turns` | `..._turns_by_regime_total` | **WITNESSED** | the cliff distribution from the live `cacheobs.FrozenFloor` / `cacheobs.ColdCeil` thresholds; frozen is the append-only regime the value comes from. |
| `provider_cache_read_tokens` | `fak_gateway_inference_cached_prompt_tokens_total` | **OBSERVED** | the upstream provider's `cache_read`, relayed verbatim. **0 on the pure in-kernel path** (no provider). Never proof fak preserved anything. |

The record **never sums** the two — they are distinct caches over distinct paths. `CacheBit()`
reports honestly whether fak's own cache engaged in the aggregate run/window
(`reused_tokens > 0`) versus an all-cold run; the /metrics family does not attribute reuse to a
specific solved-ticket turn.

## 3. Serve GLM-5.2 from the pure kernel (on the box)

```bash
fak serve \
  --gguf /mnt/.../GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf \
  --engine inkernel --backend cuda --cpu-offload-experts \
  --context-budget-tokens 8192 \
  --addr 127.0.0.1:8080 --model glm-5.2
```

(`--cpu-offload-experts` puts the MoE experts on host RAM; `--context-budget-tokens 8192`
keeps the KV plan off GLM-5.2's 1M-context default that otherwise trips `FitTooBig`.)

## 4. Drive the Claude harness over the solved ticket, then read the cache value

Drive the fak coding agent (the harness) against the gateway on the solved instance:

```bash
fak swebench run --agent fleet \
  --gateway 127.0.0.1:8080 --model glm-5.2 \
  --filter smoke --difficulty testdata/swebench_smoke.json \
  --output run-glm52-cache
```

Then fold the cache value the run realized:

```bash
# direct, if the box is HTTP-reachable:
fak swebench cache-witness --gateway 127.0.0.1:8080 --out run-glm52-cache/cache-witness.json
# or, when the box is reachable only over the lab bridge, capture and relay /metrics:
#   curl -s localhost:8080/metrics > metrics.txt   (on the box)
#   fak swebench cache-witness --metrics-file metrics.txt --out cache-witness.json
```

`cache-witness.json` is the raw evidence: `kv_prefix.reused_tokens` (WITNESSED) beside
`provider_cache_read_tokens` (OBSERVED), with the human summary stating whether the cache
**BIT** and at what reuse fraction.

## 5. The honest result fence

- **Milestone 2 (this runbook's bar):** the cache **bites** during the solved-ticket run —
  `cache-witness.json` shows aggregate `reused_tokens > 0` from a live GLM-5.2 fak-kernel serve,
  with `cache_bit_scope: "aggregate-run-kv-prefix-reuse"`. That proves the cache-value lever
  end-to-end through fak's own kernel without claiming per-turn solved-ticket attribution.
- **Stretch (gated on #996/#971):** a non-zero resolve-rate from GLM-5.2-fak-kernel, graded by
  the official harness (`fak swebench eval`). Not required to close #1010.
- **`not yet`, not a failure:** if the box is RAM-blocked (one GLM-5.2 serve fits at a time) or
  decode stalls, report `not yet` with the missing witness, never a shipped number.

## 6. Provenance

- Observation seam: `internal/cachewitness/` + `fak swebench cache-witness` (`cmd/fak/swebench_cachewitness.go`), commit `52dfea0d`, `dos commit-audit` = diff-witnessed.
- Pure-kernel serve flags: [`SWEBENCH-PURE-KERNEL-RUNBOOK.md`](SWEBENCH-PURE-KERNEL-RUNBOOK.md).
- Metric definitions + the WITNESSED/OBSERVED split: `internal/gateway/metrics.go` (`writeKVPrefixMetrics`).
- The throughput wall this routes around: #996 / #971.
