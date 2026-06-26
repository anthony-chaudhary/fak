# fak ↔ bench — SWE-bench Verified comparison

_generated 2026-06-26T02:31:12Z_

**Instances:** 20  ·  **worker sweep:** 1,2,4,8  ·  **difficulty:** 1-4 hours=3 15 min - 1 hour=13 <15 min fix=4

Geometry provenance: default=20. Σ agent steps: 400.

## The four metric families (fak, keyed to bench's vocabulary)

| family | vs bench | provenance | headline |
|---|---|---|---|
| prefill / KV-reuse work-elimination (deterministic) | fak-native | computed (deterministic) | prefill_work_elimination_A_over_C@workers=1 = 16.24x |
| turns + tokens | comparable | computed (deterministic) | turn_tax_A_over_B = 16.24x |
| in-process adjudication cost | fak-native | live (measured) | fusion_speedup = 4050.69x |
| resolve-rate + safety | comparable | gated (GPU server-only) | pass_rate_pct = 0% |

### prefill / KV-reuse work-elimination (deterministic)
_bench analog: related to (NOT the same quantity as) bench's server-stream cache-hit/KV-reuse; live TTFT/wall-clock is the gated GPU server number, not produced here — fak-native, computed (deterministic)_

- `prefill_work_elimination_A_over_C@workers=1` = **16.24x**  — fak's naive-re-prefill arm (A) vs fak-fused arm (C) — a fak-vs-fak ablation, not a fair-comparator win
- `cross_worker_prefix_reuse_B_over_C@workers=1` = **1x**  — fak's per-agent-KV arm (B) vs fak-fused shared-prefix arm (C) — a deterministic work-ratio, NOT bench's bounded token_hit_ratio_pct and NOT a head-to-head
- `prefill_work_elimination_A_over_C@workers=2` = **18.53x**  — fak's naive-re-prefill arm (A) vs fak-fused arm (C) — a fak-vs-fak ablation, not a fair-comparator win
- `cross_worker_prefix_reuse_B_over_C@workers=2` = **1.14x**  — fak's per-agent-KV arm (B) vs fak-fused shared-prefix arm (C) — a deterministic work-ratio, NOT bench's bounded token_hit_ratio_pct and NOT a head-to-head
- `prefill_work_elimination_A_over_C@workers=4` = **19.94x**  — fak's naive-re-prefill arm (A) vs fak-fused arm (C) — a fak-vs-fak ablation, not a fair-comparator win
- `cross_worker_prefix_reuse_B_over_C@workers=4` = **1.23x**  — fak's per-agent-KV arm (B) vs fak-fused shared-prefix arm (C) — a deterministic work-ratio, NOT bench's bounded token_hit_ratio_pct and NOT a head-to-head
- `prefill_work_elimination_A_over_C@workers=8` = **20.73x**  — fak's naive-re-prefill arm (A) vs fak-fused arm (C) — a fak-vs-fak ablation, not a fair-comparator win
- `cross_worker_prefix_reuse_B_over_C@workers=8` = **1.28x**  — fak's per-agent-KV arm (B) vs fak-fused shared-prefix arm (C) — a deterministic work-ratio, NOT bench's bounded token_hit_ratio_pct and NOT a head-to-head

### turns + tokens
_bench analog: agent-stream actual_agent_steps = (len(messages)-2)//2; prompt+completion tokens — comparable, computed (deterministic)_

- `total_agent_steps` = **400steps**  — Σ geometry turns ≈ Σ actual_agent_steps over the set
- `median_agent_steps` = **20steps**
- `turn_tax_A_over_B` = **16.24x**  — extra prefill work a re-prefill harness pays per turn vs KV persistence (worker-independent)

### in-process adjudication cost
_bench analog: none — bench measures a served endpoint, not the tool-call adjudication boundary — fak-native, live (measured)_

- `in_process_adjudication_p50` = **2888ns**
- `spawn_per_hook_p50` = **11698400ns**
- `fusion_speedup` = **4050.69x**  — in-process vs the deployed process-per-hook status quo

### resolve-rate + safety
_bench analog: agent-stream resolved_count / pass_rate_pct (report.json); safety is fak-native — comparable, gated (GPU server-only)_

- `pass_rate_pct` = **0%**  — no predictions graded — resolve-rate needs a capable model + Docker (GPU server); ~0 with the local 135M model

## Honesty

- prefill work-elimination is a deterministic floor (exact token arithmetic), NOT a measured wall-clock — live timing is the GPU server headline
- cross-worker reuse (B/C) is the value-stack lever; the A/C and turn-tax (A/B) numbers are fak-vs-harness-arms, reported as such, not as a tuned-SGLang head-to-head
- resolve-rate is ~0 with the local 135M model; the real resolve number comes from a Qwen3.6-27B-class model on GPU server via the same harness
- to be scraped by bench like SGLang, fak serve needs a Prometheus /metrics route exposing kernel.Counters() — gap noted, not yet shipped
