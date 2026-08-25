# Performance borrow map: current agentic and model-runtime practice

- **Date:** 2026-08-25
- **Status:** source queue refreshed; live issue read-back succeeded; live issue creation still blocked by invalid `gh` credentials
- **Companions:** [related-system inventory contract](CONCEPT-RELATED-SYSTEM-INVENTORY-2026-08-25.md), [monitored repository registry](../research/monitored-repositories.json), [mini-sglang study](CONCEPT-STUDY-MINI-SGLANG-2026-08-22.md), [InferenceX study](BORROW-BENCHMARK-SERVING-METRICS-INFERENCEX-STUDY-2026-07-13.md)

## Verdict

The current field says the next performance work should be evidence-first and source-pinned, not another local micro-optimization hunt. Four live GitHub tickets already carry the near-term path:

- [#8923](https://github.com/anthony-chaudhary/fak/issues/8923) forces the next Qwen3.8 baseline wave through a measured borrowed hot path instead of unprofiled custom kernels.
- [#8819](https://github.com/anthony-chaudhary/fak/issues/8819) and [#8848](https://github.com/anthony-chaudhary/fak/issues/8848) provide the profile and cross-node experiment matrix that should choose the hot path.
- [#8918](https://github.com/anthony-chaudhary/fak/issues/8918), [#8924](https://github.com/anthony-chaudhary/fak/issues/8924), and [#8925](https://github.com/anthony-chaudhary/fak/issues/8925) make repository discovery itself measurable, so agents find the right prior art without repeated broad scans.
- [#8774](https://github.com/anthony-chaudhary/fak/issues/8774) and [#8775](https://github.com/anthony-chaudhary/fak/issues/8775) carry the agentic-serving metric borrows from InferenceX: backend lifecycle joins and full-response interactivity.

The source registry now adds current exact revisions for the fast-moving model-serving and agentic-process repos that should feed those issues. None of these rows proves a FAK gap by itself. Each row is a queue item for a `study-repo` pass with inventory map, self-query witness, candidate matrix, license disposition, and issue dedupe.

## Source Ledger

Observed at `2026-08-25T20:40:56Z` via GitHub REST API and commit read-back.

| Source | Pinned revision | Source event | License | Why it matters for FAK |
|---|---|---:|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm/commit/80771bbbddf9e5153eea3aca8055049ee5aaaed1) | `80771bbbddf9e5153eea3aca8055049ee5aaaed1` | 2026-08-25 | Apache-2.0 | Scheduler, prefix-cache, memory allocator, speculative-decode, and determinism tradeoffs. Current commit disables a non-deterministic fused path under a batch-invariant flag, reinforcing that speedups need invariant fences. |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/commit/edff717ef0106b4413b371bf7a05a5193ffeee85) | `edff717ef0106b4413b371bf7a05a5193ffeee85` | 2026-08-25 | Apache-2.0 | RadixAttention, HiCache/disaggregation, scheduler policy, and request-state concurrency. Current commit snapshots mutable rooms before iterating outside a lock, a useful reminder that throughput work must not outrun state correctness. |
| [LMCache/LMCache](https://github.com/LMCache/LMCache/commit/23cca67908e17b193eb8fab08ba1beb0115881cd) | `23cca67908e17b193eb8fab08ba1beb0115881cd` | 2026-08-25 | Apache-2.0 | KV transfer registration, external KV lifecycle, and vLLM/Dynamo integration. Current commit fixes missing-registration GPU transfers, which maps to FAK's route-by-reference and KV ownership honesty. |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer/commit/73d778422388f0e849ecb180375d34ac445711ca) | `73d778422388f0e849ecb180375d34ac445711ca` | 2026-08-24 | Apache-2.0 | Quantization, distillation, pruning, and speculative-decoding export paths into serving engines. Candidate support evidence for Qwen3.8 quant and borrowed-hot-path planning under #8923. |
| [vllm-project/speculators](https://github.com/vllm-project/speculators/commit/0faffeb3bd547b4451a978d7aaf26a2f01b83d62) | `0faffeb3bd547b4451a978d7aaf26a2f01b83d62` | 2026-08-25 | Apache-2.0 | Speculative-decoding algorithm library for vLLM. Relevant to DFlash/DSpark/MTP candidate selection and acceptance telemetry. No direct FAK issue matched this exact source identity in the bounded dedupe scan. |
| [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress/commit/161705a68f64df329c88a6da5f20300a89aa7542) | `161705a68f64df329c88a6da5f20300a89aa7542` | 2026-08-18 | Apache-2.0 | KV-cache compression with an entropy-gated chunk compressor. Existing lead: [#5328](https://github.com/anthony-chaudhary/fak/issues/5328). |
| [novitalabs/pegaflow](https://github.com/novitalabs/pegaflow/commit/d3ed31a7a4ee9bc7cdc5f112f0a1eb26efe63675) | `d3ed31a7a4ee9bc7cdc5f112f0a1eb26efe63675` | 2026-08-17 | Apache-2.0 | KV storage with GPU offload, SSD caching, and cross-node sharing for vLLM/SGLang. Candidate for external-engine KV ownership and cache-tier comparisons. |
| [jundot/omlx](https://github.com/jundot/omlx/commit/90ecf1c26dbed875e6ced82c4faa6e9250037f2d) | `90ecf1c26dbed875e6ced82c4faa6e9250037f2d` | 2026-08-25 | Apache-2.0 | Apple-Silicon continuous batching and SSD caching. Candidate for Mac/Qwen3.8 resident-decode refresh after deduping against the DFlash2 oMLX fork note. |
| [lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed/commit/1468fe0cbe92d389015ce73842fbaf349ce052d1) | `1468fe0cbe92d389015ce73842fbaf349ce052d1` | 2026-08-25 | MIT | Speed-of-light inference engine and performance workflow hygiene. Candidate for lower-bound measurement discipline before #8923 picks a hot path. |
| [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox/commit/c994209a2c03c1bf3426924aba52b5edc05501ff) | `c994209a2c03c1bf3426924aba52b5edc05501ff` | 2026-08-25 | Apache-2.0 | Speculative inference server for consumer and heterogeneous hardware. Candidate for the supported-envelope boundary around speculative serving knobs. |
| [apache/maka](https://github.com/apache/maka/commit/ee7da140ddf3ee522fb6eb5e0d9c215422e4e1fd) | `ee7da140ddf3ee522fb6eb5e0d9c215422e4e1fd` | 2026-08-25 | Apache-2.0 | Local-first agent workspace with append-only logs for messages, tools, permission decisions, and termination events. Relevant because replayable agent state reduces repeated orientation work and makes performance/process causality auditable. |
| [superplanehq/superplane](https://github.com/superplanehq/superplane/commit/541a64b87aa534d5cbbbccb369d069e75a06bce9) | `541a64b87aa534d5cbbbccb369d069e75a06bce9` | 2026-08-25 | Apache-2.0 | Agentic engineering control plane with current issue-intake work. Candidate for queue/reconciler planning under #8873 and #8909. |
| [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox/commit/3444bfb35a42e933752d3c6624a24e00a3e53d6b) | `3444bfb35a42e933752d3c6624a24e00a3e53d6b` | 2026-08-25 | Apache-2.0 | Isolated, stateful singleton workloads for agent runtimes and RL. Dedupe found related governance/supervision issues before any new sandbox ticket. |
| [ARahim3/mlx-dspark](https://github.com/ARahim3/mlx-dspark/commit/89f4bb1629230833bb2d05653fa83052d5802b01) | `89f4bb1629230833bb2d05653fa83052d5802b01` | 2026-08-25 | MIT | Prior study filed [#4199](https://github.com/anthony-chaudhary/fak/issues/4199). Current source has moved enough that further speculative-decoding borrowing needs a refresh. |

## Best-Practice Map

| Practice | Field evidence | FAK action |
|---|---|---|
| Profile first, then borrow one hot path. | vLLM and TokenSpeed both expose performance work tied to concrete runtime invariants or perf workflows. | Run #8819/#8848 before implementing #8393/#8394/#8923. |
| Preserve correctness invariants while optimizing. | vLLM disables a fused all-reduce/RMS path when it violates a batch invariant; SGLang snapshots mutable state before lock-free iteration. | Require quality, determinism, engine identity, and fallback-denial fields in every native-performance receipt. |
| Treat KV movement as a lifecycle protocol, not a byte cache. | LMCache and Pegaflow both center registration, transfer, offload, and cross-node sharing rather than bare lookup. | Keep native-owned KV and external-engine KV separate in receipts; use #8774-style backend lifecycle joins for causality. |
| Compress or speculate only behind a verifier. | KVPress uses gated compression; vLLM Speculators and mlx-dspark-style DSpark paths need acceptance/losslessness accounting. | Reuse FAK's verify/rollback discipline; file only source-pinned refinements after self-query witnesses, not another generic "add spec decode" ticket. |
| Optimize the agent loop as a system, not only the model. | Maka, Superplane, and agent-sandbox point at append-only event logs, queue intake, stateful isolation, and restart semantics. | Feed #8873/#8909 and the discovery issues so fewer turns are spent rediscovering state or babysitting workers. |
| Benchmark interactive usefulness, not just tok/s. | InferenceX already produced #8774/#8775 for backend-log lifecycle and full-response interactivity. | Keep agentic performance receipts quality-constrained and source-digested. |
| Make research reusable. | The registry and Ruflo map prove one source can be indexed before the full deep study finishes. | Use `fak study-inventory` first for every new source row; the inventory gate should stay red until non-tree evidence exists. |

## Dedupe And Ticket State

Live issue read-back through the public GitHub REST API found these relevant tickets:

- Discovery/process: [#8918](https://github.com/anthony-chaudhary/fak/issues/8918), [#8924](https://github.com/anthony-chaudhary/fak/issues/8924), [#8925](https://github.com/anthony-chaudhary/fak/issues/8925).
- Qwen/native performance: [#8923](https://github.com/anthony-chaudhary/fak/issues/8923), [#8819](https://github.com/anthony-chaudhary/fak/issues/8819), [#8848](https://github.com/anthony-chaudhary/fak/issues/8848).
- Agentic-serving metrics: [#8774](https://github.com/anthony-chaudhary/fak/issues/8774), [#8775](https://github.com/anthony-chaudhary/fak/issues/8775).
- Existing source-specific leads: [#5328](https://github.com/anthony-chaudhary/fak/issues/5328) for KVPress and [#4199](https://github.com/anthony-chaudhary/fak/issues/4199) for the previous mlx-dspark controller borrow.

No live issue creation was attempted after `gh auth status` reported an invalid keyring token. The issue-creation path should be retried with `gh` only after authentication is repaired. Until then, the registry rows plus this note are the issue-ready source queue, and the existing live tickets above are the dispatchable work.

## Next Study Queue

1. Start with `vllm-project/speculators` under #8923 if #8819/#8848 show speculative decode or acceptance control as the dominant lever.
2. Study `NVIDIA/Model-Optimizer` only after the chosen Qwen3.8 hot path needs quant/export evidence; otherwise keep it as a watch row.
3. Study `novitalabs/pegaflow` and refresh `LMCache/LMCache` together only if the next KV work requires external-engine transfer/offload ownership, not for native-kernel work.
4. Study `jundot/omlx` or refresh `ARahim3/mlx-dspark` only for Apple-Silicon Qwen3.8 or consumer-device speculative serving.
5. Study `apache/maka` or `superplanehq/superplane` when #8873/#8909 need current agent-process evidence for append-only logs, queue intake, or restart semantics.

## Fences

- GitHub stars and freshness are ranking signals, not quality proof.
- Open issues and recent commits prove direction and churn, not shipped behavior.
- A registry row is not a filed gap. A gap still needs pinned code `path:line@sha`, FAK self-query witness, exact FAK seam, license disposition, and issue read-back.
- Performance claims remain invalid without matched model/artifact/runtime/hardware/workload identity and quality gates.
