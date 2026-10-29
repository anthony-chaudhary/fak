# Inference scaling goals: comparison envelope before a hardware ranking

Research checked 2026-10-28. This is a decision inventory, not a measured hardware
result or a recommendation to buy, rent, deploy, or change an active server.

## Objective and existing seams

Compare configurations on the **same offered workload**, with jointly satisfied
time-to-first-token (TTFT) and mean time-per-output-token (TPOT) limits. Separate
measured service-SLO goodput from an independently supplied quality qualification.
Keep source, model, configuration, warmup policy, offered count, common wall window,
and optional cost unit attached. Unknown quality is not measured zero.

`fak model-observe report` already reads per-request observations; the optional
service-goal evaluator belongs there. Existing `fak-dev inference-topology`
inventory/validate/analyze/render and `model-observe` bandwidth, roofline, cache,
topology economics, and time-burndown surfaces are complementary. Do not create a
second optimizer, registry, or universal hardware score.

For an observation window W and offered count N, measured SLO goodput is the
number of successful requests with sufficient timing evidence satisfying both
limits divided by W. Token goodput counts those requests' completion tokens/W;
it is not a task-quality score. SLO attainment divides by **N**, not just completed
rows. Missing, rejected, failed, non-streaming, and incomplete requests remain
visible. Marginal p95 TTFT and TPOT, each with coverage, do not prove that the
same requests passed both limits. TPOT is undefined for one output token; the
goal must declare TTFT-only eligibility or leave that request unqualified.

## Using the existing report

Both entry points use the same evaluator:

```text
fak model-observe report --input observations.jsonl --goal goal.json --format json
modelperfobs report --input observations.jsonl --goal goal.json --format md
```

Without `--goal`, the existing summary JSON and Markdown remain unchanged. With
it, the report uses `fak-service-goal-report/1` and retains the full supplied goal:

```json
{
  "schema": "fak-service-goal/1",
  "id": "interactive-local",
  "workload_ref": "trace:fixed-offered-requests",
  "ttft_limit_ms": 500,
  "tpot_limit_ms": 50,
  "warmup_policy": "exclude",
  "single_token_tpot_policy": "ttft-only",
  "run": {
    "source_ref": "observations:content-digest",
    "model_ref": "model:artifact-digest-and-tokenizer",
    "config_ref": "config:engine-device-software-sampling",
    "wall_seconds": 60,
    "offered_requests": 100,
    "quality": "unknown",
    "quality_ref": "",
    "cost": null,
    "cost_unit": ""
  }
}
```

Run evidence must describe one common measurement window **including drain**, a
positive offered count, and its observation/model/configuration references. These
references and the external run-wide quality verdict are caller attestations;
the evaluator does not fetch artifacts, verify their digests, or prove quality.
`qualified` and `failed` require a nonempty external `quality_ref`. Qualified
goodput is null for `unknown`, known zero for `failed`, and measured SLO goodput
for `qualified`. Cost per qualified SLO request is null without a stated cost and
at least one qualified pass; no exchange rates or cost categories are inferred.

The explicit supported warmup policy is `exclude`: mark each warmup observation
with `warmup: true` and exclude it from the declared offered count. All other
rows, including failures, count against that offered count. Duplicate/missing
request IDs, invalid numeric/token counts, overflow and offered counts below
observed non-warmups are rejected. Unknown JSON fields and trailing JSON values
in the goal are rejected rather than silently ignored.

One-token TPOT policies are `ttft-only` (a successful, adequately timed streaming
request can pass on TTFT) or `unknown` (it cannot establish a joint pass).
Insufficient timing remains unknown, not a successful zero-latency observation.
Known violations are distinct from unknowns and HTTP/request errors. P95 uses
the existing report's empirical `round((n-1)*0.95)` index, with measured count and
offered-request coverage; there is no percentile imputation for missing timings.
This mean TPOT contract does not establish tail inter-token latency, per-token
jitter, or end-to-end agent-task correctness. CPU fixtures verify software
accounting only, never physical inference speed or a deployment conclusion.

## Steelman the spectrum

| Axis | Strong case for the first choice | Strong case for the alternative | Evidence to collect / reversal condition |
|---|---|---|---|
| Memory bandwidth vs peak arithmetic | Low-batch decode can be dominated by moving weights/KV; sustainable bandwidth and locality can beat a larger arithmetic headline. | Larger batches, prefills, and different operators can move the bottleneck to compute; lower precision can reduce both bytes and operations. | Phase- and shape-specific arithmetic intensity, actual memory traffic/cache hits and achieved bandwidth; do not infer device utilization from token rate. |
| Capacity vs speed | More VRAM preserves weights, long-context KV, concurrency, and headroom without host transfers. | A smaller faster device may win if the complete working set fits, or selective offload/recomputation costs less than buying capacity. | Weights + live KV + scratch + graphs + fragmentation + reserve, at the target concurrency and output lengths; include eviction/recompute and OOMs. |
| One-device replicas vs sharding | A fully resident single-device replica avoids collectives and often simplifies latency and reliability; independent requests can scale across replicas. | Tensor/pipeline parallelism can supply capacity, reduce compute time, or free KV space even when weights alone fit. | TP collectives, PP bubbles/latency, load balance, per-stage memory, real topology, and joint-SLO goodput at the offered load. Fit is an initial condition, not a universal no-split theorem. |
| Colocated vs prefill/decode split | Colocation avoids KV transfers, duplicated resources and a second queue, and chunked prefill can reduce phase interference. | Phase separation permits independent sizing, parallelism, hardware and power choices, and can reduce tail interference under strict SLOs. | Transfer bytes/latency and overlap, queueing and imbalance, prompt/output distributions, arrival burstiness, connector support and failures. Compare against tuned colocation, not a straw baseline. |
| Local KV vs pooled/disaggregated KV | Local state minimizes movement and coordination; short or unique prompts may not amortize a shared cache. | Prefix-heavy long contexts can justify shared DRAM/SSD and cache-aware routing, avoiding repeated prefills and stranded capacity. | Legal exact-prefix hit rate, invalidation/security boundaries, cache working set, transfer/recompute tradeoff, cache misses and overload rejection. |
| Dense replication vs MoE expert parallelism | Keeping experts resident locally avoids all-to-all and routing skew when capacity allows. | Expert sharding can unlock otherwise infeasible capacity and useful compute utilization. | Active vs total parameters, expert skew/hotspots, all-to-all fabric, redundant experts and overlap; EP is not the same mechanism as PD or PP. |
| Low-precision TOPS/FLOPS vs realized utility | Native low-precision tensor operations can offer substantial compute and bandwidth leverage. | Unsupported kernels, conversion/dequantization, accumulation, sparsity assumptions and quality loss can erase headline gains. | Exact datatype/accumulator, dense vs structured sparse counting, operator shape, kernels actually selected, memory footprint, external quality gate and full wall time. |
| Purchase price vs lifecycle cost | A cheap used accelerator or already-owned host can supply valuable bandwidth/capacity quickly. | Newer hardware may cost less per qualified result after energy, integration, support, downtime and resale. | Dated delivered price, tax/shipping, host/PSU/cooling/fabric, idle+load energy, measured utilization, engineering time, repairs and conservative residual value. Cost unit and accounting boundary must match. |
| Available now vs wait for next generation | Existing compatible stock earns useful work immediately and avoids schedule risk. | A known deadline, larger capacity requirement, or substantial supported improvement can make waiting rational. | Real quotes/stock, shipping/qualification dates, foregone useful work, uncertainty and a deadline-specific scenario; do not present speculation as availability. |
| Workstation/package vs server | Small systems can be quieter, accessible, portable, and use already-owned power/networking. | Servers may offer better cooling, PCIe lanes, memory channels, fabric, redundant power, serviceability and sustained duty cycle. | Socket/rank/channel topology, slot spacing, PCIe routing, thermals, acoustics, wall power, sleep/recovery behavior and sustained clocks. |
| TB4 vs TB5 vs direct PCIe/fabric | Existing TB4/USB4 gear may suffice for infrequent transfers with resident work. | Higher negotiated TB5 PCIe bandwidth or direct attachment can reduce frequent movement costs. | Host controller, PCIe tunneling, enclosure/device/cable and display contention, measured payload throughput/latency; port signaling is not VRAM bandwidth. |
| vPro/manageability vs arithmetic | Out-of-band recovery and remote fleet support can improve service availability and reduce operator effort. | A simpler already-manageable host may avoid platform premium and integration requirements. | Exact enabled SKU/firmware/OS/network prerequisites, remote recovery witness, incident labor/downtime; vPro is not a tensor-performance multiplier. |
| NVIDIA vs AMD engineering time | A mature supported kernel/tool path can shorten qualification and lower operational risk. | An open, compatible AMD stack or existing expertise can reduce acquisition constraints and enable tailored optimizations. | Exact architecture/OS/driver/runtime support matrices, required operations, kernel coverage, correctness, debug/profiling time and ongoing maintenance. No brand-wide inference-cost conclusion. |
| OSS/default fusion vs custom/agent fusion | Upstream fused kernels, batching, graph capture and supported defaults can amortize broad community effort. | Shape-specific custom fusion or agent-assisted engineering can remove a real measured bottleneck that upstream does not cover. | Verify selected code path and supported defaults; include code generation/review, compilation, warmup, launch savings, register pressure, portability and quality/regression evidence. Proposed agent code is not a validated kernel. |
| Popular ecosystem vs niche advantage | Popular devices inherit integration, examples, fixes and some server-side software investment; shared work can lift many users. | A less popular platform may have a compelling supported workload/capacity advantage and less contention. | Demonstrated model/operator support and release cadence, not popularity as proof; server optimizations may depend on absent tensor formats, memory/fabric or licenses. |
| Resale/salvage vs keeping assets | Standard reusable components can retain value or move to another workload. | Resale prices are uncertain; specialized parts may require host/cooling/support that limits salvage. | Condition, warranty/transfer terms, market date, removal/sale costs, realistic buyer pool and conservative sensitivity ranges; never assume an appreciation rate. |

Additional axes: task correctness rather than only token acceptance; context/output
length distribution; concurrency and burstiness; admission and retries; cold start
and model switching; tokenizer/sampling/prompt-template equality; reproducibility;
failure domains; privacy/data locality; host NUMA placement; network contention;
observability overhead; software licensing/support; repairability and supply risk.
Hold these fixed or explicitly declare their change before comparing results.

## Primary-source anchors and their limits

- [DistServe v3](https://arxiv.org/html/2401.09670v3) analyzes phase interference,
  resource allocation and placement under joint TTFT/TPOT constraints. Its measured
  cluster envelopes do not establish a universal benefit on slower fabrics.
- [Splitwise v2](https://arxiv.org/html/2311.18677v2) characterizes differing phase
  compute, memory and power requirements and studies heterogeneous phase pools.
  Benefits depend on the workload, provisioning objective and state-transfer path.
- [Mooncake v5](https://arxiv.org/html/2407.00079v5) studies KV-centric caching,
  scheduling and overload rejection. Keep refused offered requests in the service
  denominator; its heterogeneous-accelerator discussion includes prospective work.
- [vLLM PD documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
  distinguishes independent latency tuning and tail control from raw throughput;
  [chunked prefill](https://docs.vllm.ai/en/latest/configuration/optimization/)
  supplies a colocated alternative, with tuning and resource tradeoffs.
- [Prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
  saves repeated prefill rather than the work of generating new tokens.
  [Parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) and
  [EP deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
  document distinct capacity/communication choices, not interchangeable switches.
- [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
  provides roofline and memory-workload tools for distinguishing actual bottlenecks.
  [RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
  are device specifications, not inference benchmarks. The proposed **5090 ~40x
  older MI40** comparison remains unverified: exact older chip identity, precision,
  dense/sparse convention, workload and quality must be supplied before a ratio
  has meaning. Do not silently substitute M40, MI50 or another product.
- [Intel Thunderbolt overview](https://www.intel.com/content/www/us/en/architecture-and-technology/thunderbolt/overview.html)
  distinguishes TB5's 80 Gbps bidirectional signaling, asymmetric display boost,
  and PCIe requirements (64 vs TB4's 32 Gbps). These are not measured GPU payload
  rates. [vPro](https://www.intel.com/content/www/us/en/architecture-and-technology/vpro/overview.html)
  concerns manageability with platform prerequisites, not inference arithmetic.
- [AMD ROCm compatibility](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
  and [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/)
  must be checked for the actual device/software tuple; nominal vendor support is
  not a workload qualification. Mutable latest pages were checked on the date above.
- [Linux NUMA memory policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)
  distinguishes memory policies, policy scopes and cpuset restrictions. Record
  actual CPU/memory/NIC placement; aggregate host memory is not uniformly local.

## Next experiment

On one supported native model/configuration, capture a fixed offered trace and
common wall window, including errors/rejections and explicit warmups. Run a paired
colocated/split or replica/sharded comparison only after end-to-end correctness,
quality and source/config provenance are established. Sweep one bottleneck variable
at a time and publish coverage/unknowns with goodput; preserve useful negative
results instead of optimizing the metric denominator.
