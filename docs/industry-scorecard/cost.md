---
title: "fak industry scorecard — cost"
description: "The cost dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# cost — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Cost & efficiency (`cost-efficiency`)

### ○ Goodput under an SLA (requests/s meeting TTFT and TPOT SLOs simultaneously) — fak: **no-claim**

*Why it matters:* Raw throughput overcounts: a request that violates its latency SLO is wasted work. Goodput, the max request rate at which a target fraction of requests meet BOTH TTFT and TPOT SLOs, is the metric a buyer running an interactive product actually pays against. It exposes systems that win on aggregate tokens/s but collapse under tail-latency constraints.

- **SOTA bar:** DistServe serves up to 7.4x more requests or holds a 12.6x tighter SLO than prior SOTA while keeping >90% of requests within latency bounds; MuxWise reports 1.3x (Llama-8B) to 1.62x (Llama-70B) higher goodput than SGLang-PD at the 99th-percentile SLO.
- **Leading systems:** DistServe (introduced goodput, P/D disaggregation), SGLang-PD, MuxWise (P/D multiplexing)
- **Source:** [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670) (2024-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. fak ships no goodput-under-SLA number (no TTFT/TPOT SLO attainment curve). The DGX run measured raw peak throughput at fixed concurrency, not requests meeting joint latency bounds. DistServe/MuxWise-style goodput is a standard serving-systems axis fak has zero evidence on; named as a gap, not parity.
- **Trace:** No goodput/SLA-attainment number exists in BENCHMARK-AUTHORITY.md or CLAIMS.md. The only concurrent-serving head-to-head (data.json 'served-throughput-vs-sglang') reports peak tok/s, not SLO-bounded goodput.

### ○ SLO-constrained goodput (per-GPU) — fak: **no-claim**

*Why it matters:* Raw throughput (tokens/s) is gameable by sacrificing latency. Goodput - the max request rate served while still meeting the TTFT and TPOT SLO for >=X% (e.g. 90%) of requests, normalized per GPU - is the metric that actually maps to cost-per-served-user. It is the central buyer's KPI because it unifies latency and throughput into one defensible number.

- **SOTA bar:** DistServe reports up to 4.48x higher goodput (or 7.4x higher sustainable request rate / 12.6x tighter SLO at 90% attainment) over prior SOTA serving systems by disaggregating prefill and decode. GenAI-Perf can directly measure goodput against user-set TTFT/TPOT constraints.
- **Leading systems:** DistServe, DynaServe, NVIDIA Dynamo, GenAI-Perf goodput mode
- **Source:** [https://arxiv.org/html/2401.09670v1](https://arxiv.org/html/2401.09670v1) (2024-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. SLO-constrained goodput is the metric of a full PD-disaggregating serving STACK (DistServe 4.48x goodput, GenAI-Perf goodput mode); fak is a kernel/adjudication plane that FRONTS such a stack (the DGX run is SGLang-serves + fak-adjudicates) and does not own the scheduler that goodput measures. no-claim.
- **Trace:** No goodput-under-SLO number exists. CLAIMS.md labels continuous-batching [SIMULATED] (read-only telemetry, not on the live serving path) and the polymodel decode lane is explicitly SERIAL/off-mainline, so fak has no SLO-constrained per-GPU goodput measurement.

### ○ End-to-end effective request capacity / cost gain from the KV system under SLO — fak: **no-claim**

*Why it matters:* All the mechanisms above only matter if they raise the requests an operator can serve within latency SLOs on fixed hardware. The integrated, trace-driven gain (effective capacity, cost per token) is the bottom-line metric a buyer uses to compare whole KV-centric architectures rather than individual features. It captures the 'trade storage for compute' thesis at the system level.

- **SOTA bar:** Mooncake's KVCache-centric disaggregated architecture raises effective request capacity 59%-498% on real Kimi traces while meeting SLOs; runs across thousands of nodes serving 100B+ tokens/day
- **Leading systems:** Mooncake, LMCache + vLLM, NVIDIA Dynamo
- **Source:** [https://www.usenix.org/conference/fast25/presentation/qin](https://www.usenix.org/conference/fast25/presentation/qin) (2025-02)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP / honestly a non-win. fak has no Mooncake-style 'effective request capacity under SLO' number, and its one real concurrent-serving head-to-head (fak-gateway in front of SGLang on 8xA100) TRAILS raw SGLang (0.75x at peak) because the value added there is the adjudication/coherence/measurement plane, not throughput. So fak cannot claim the 59-498% capacity gain axis; conflating its reuse-work multipliers with SLO capacity would be dishonest.
- **Trace:** No SLO-bound effective-capacity figure exists. The fleet multipliers (60.3x naive / 4.1x tuned, headline-qwen-50x5.json) are work-ELIMINATION on fak's own kernel held constant, not SLO-bounded request capacity; the only live concurrent-serving head-to-head (data.json served-throughput-vs-sglang, compare.json) shows fak at 0.60x->0.75x->~0.97x of raw SGLang, i.e. a gateway TAX, not a capacity gain.

### ○ Inference unit economics ($ per 1M tokens) at realistic utilization — fak: **no-claim**

*Why it matters:* Cost-per-million-tokens at the utilization you can actually sustain is the metric finance signs off on. It folds throughput, batching, quantization, and utilization into one number and exposes the gap between marketing throughput and the bill.

- **SOTA bar:** GPT-4-equivalent quality now serves at ~$0.40 / 1M tokens vs ~$20 in late 2022 (≈10x/yr decline). Real production GPU utilization is only 30-60%, so realistic cost-per-token runs 2-3x above spreadsheet estimates; FP8/FP4 quantization roughly halves cost-per-token on H100/H200. Self-hosting a 7B breaks even only above ~50% utilization.
- **Leading systems:** Hosted frontier APIs, vLLM/SGLang self-host on H100/H200, FP8/FP4 quantization
- **Source:** [https://introl.com/blog/inference-unit-economics-true-cost-per-million-tokens-guide](https://introl.com/blog/inference-unit-economics-true-cost-per-million-tokens-guide) (2025-09-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE as a measured number for a reuse kernel, though it is the axis fak's reuse story IMPLIES. fak's headline is work eliminated (50x5 fleet 4.1x vs tuned / 60.3x vs naive; ultra-long-context ~4x vs warm cache vs ~40x vs naive) which translates to lower infra cost ON TOP of any model at the same quality, but fak has committed NO $/1M-token figure, no utilization model, and no quantization-cost number. Stating a dollar SOTA-relative position would invent a number. Honest no-claim.
- **Trace:** No $/token row in BENCHMARK-AUTHORITY.md or CLAIMS.md; fak measures work-elimination ratios (tokens/FLOPs), never a dollar figure or utilization-adjusted cost

### ○ Energy efficiency (Wh per token / tokens-per-watt) and power-capped throughput — fak: **no-claim**

*Why it matters:* Power, not GPUs, is becoming the binding constraint on fleet scale; tokens-per-watt sets the ceiling on how much serving a datacenter can host and increasingly drives both cost and siting decisions. Buyers with power budgets evaluate on it directly.

- **SOTA bar:** Empirical inference energy is ~0.0001-0.002 Wh per output token depending on model size; a ~500-token GPT-4o-class query is ~0.3 Wh (~3e-4 Wh/token). FP4/FP8 and Blackwell-class hardware are the main levers. No neutral audited tokens-per-watt leaderboard yet exists, so the bar is a measured range, not a single SOTA number.
- **Leading systems:** NVIDIA Blackwell (GB200/GB300, FP4), AMD MI355X (FP4)
- **Source:** [https://arxiv.org/pdf/2603.06630](https://arxiv.org/pdf/2603.06630) (2026-03-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP, named not hidden. CLAIMS explicitly flags token-per-watt as [SIMULATED] because there is no power-telemetry source on the build box; the seam is real, the numbers are illustrative. Zero measured evidence against the empirical ~1e-4 to 2e-3 Wh/token band or any Blackwell/MI355X FP4 reference. fak can never be read as claiming efficiency parity; honest no-claim.
- **Trace:** CLAIMS.md Engine: token-per-watt is [SIMULATED] read-only telemetry, 'no watt source on the box'; tokens-per-watt scorecard row competitor_value=null

