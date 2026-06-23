---
title: "fak industry scorecard — distributed"
description: "The distributed dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# distributed — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Parallelism & disaggregation (`parallelism`)

### ○ Prefill-decode disaggregation: independent scaling of compute-bound prefill and memory-bound decode — fak: **no-claim**

*Why it matters:* Prefill is compute-bound and decode is memory-bandwidth-bound; co-locating them forces one hardware/parallelism config to serve both and couples their scaling. Disaggregating onto separate instances (over a KV-cache transport) lets each phase scale and tune independently and removes interference entirely. It is now the default architecture for large-scale, SLO-tight deployments.

- **SOTA bar:** P/D disaggregation with SGLang serves up to 6.9x more goodput requests in chatbot scenarios and 2.23x more in heavy-decode scenarios within SLO; llm-d reports ~3x lower mean TTFT at 4 QPS and ~2x baseline QPS under SLO on H100 nodes.
- **Leading systems:** DistServe, Splitwise, SGLang-PD / vLLM disaggregated, llm-d (vLLM + NIXL KV connector)
- **Source:** [https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html](https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html) (2025)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for the reuse kernel: fak does not disaggregate prefill from decode into independently-scaled pools; its differentiator is cross-agent prefix fusion on one kernel, not P/D separation. DistServe/Splitwise/SGLang-PD/llm-d numbers have no fak counterpart; named as out-of-scope rather than a measured gap.
- **Trace:** No P/D-disaggregation goodput/TTFT number in BENCHMARK-AUTHORITY.md or CLAIMS.md. fak runs a single fused forward (prefill+decode together); no disaggregated prefill/decode pool is built or measured.

### ○ Prefill-decode disaggregation & per-phase SLO isolation — fak: **no-claim**

*Why it matters:* Prefill is compute-bound (TTFT-sensitive) and decode is memory-bandwidth-bound (TPOT-sensitive); colocating them couples their tails. Disaggregating onto separate GPU pools lets each phase scale and meet its own SLO independently, which has become the production standard for hitting tight TTFT AND tight TPOT simultaneously at scale.

- **SOTA bar:** DistServe: up to 7.4x higher request rate / 12.6x tighter SLO at >90% attainment. NVIDIA Dynamo Planner monitors GPU capacity vs TTFT/ITL SLOs to decide per-request whether to serve disaggregated. PD-disaggregation is now supported across vLLM, SGLang, TensorRT-LLM, LMDeploy, Dynamo and deployed at production scale.
- **Leading systems:** NVIDIA Dynamo, DistServe, llm-d, DeepSeek (production PD-disagg)
- **Source:** [https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) (2025-03)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. PD-disaggregation with per-phase SLO isolation (DistServe 7.4x request rate / 12.6x tighter SLO; NVIDIA Dynamo Planner) is a serving-stack architecture; fak owns neither the prefill/decode scheduler nor the per-phase GPU pools. It is an adjudication/coherence plane that rides on whatever engine does this. no-claim.
- **Trace:** No prefill-decode disaggregation is shipped or claimed. fak's own engine runs a single co-located forward; on the GPU server it fronts SGLang TP=8 (which itself is not run in a disaggregated config in the committed GPU server artifacts).

### ○ Large-scale expert parallelism (EP) for giant MoE models — fak: **no-claim**

*Why it matters:* Sparse MoE models (DeepSeek-V3/R1 671B, Qwen MoE, Mixtral) activate only a few experts per token, so naive tensor parallelism replicates expert weights and wastes memory bandwidth. Scaling EP across dozens of GPUs (one or few experts per device) is the only way to serve 600B+ MoE economically, and demands all-to-all expert dispatch, expert-load balancing, and overlap of comms with compute. EP scale and balance is the differentiating capability for frontier-MoE serving.

- **SOTA bar:** SGLang runs DeepSeek-V3 671B at EP32 for prefill (4 nodes) and EP72 for decode (9 nodes) across 96 H100s with DP attention to eliminate KV duplication; first open-source implementation to nearly match DeepSeek's official large-scale-EP throughput.
- **Leading systems:** SGLang, DeepSeek (DeepEP), vLLM
- **Source:** [https://www.lmsys.org/blog/2025-05-05-large-scale-ep/](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) (2025-05-05)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a single-box reuse kernel. fak DOES run GLM-5.2's MoE/FFN experts + router on its own pure k_q8_gemm GPU kernel (cosine=1.0, datacenter GPU, commit 498a4ab) — but on a SINGLE device, with no expert-parallel sharding across nodes and the DSA sparse-attention still host-side. fak's own-engine ceiling is ~7B (model-size-ceiling row); giant-MoE EP is delegated to SGLang/DeepEP, which fak fronts. No fak EP number exists; verdict no-claim.
- **Trace:** none — fak runs GLM-5.2 MoE experts on ONE datacenter GPU (CLAIMS Engine, commit 498a4ab), never EP across nodes

### ○ Data-parallel (DP) attention / hybrid attention-FFN parallelism — fak: **no-claim**

*Why it matters:* When EP fans the FFN out across many GPUs, replicating the KV cache under tensor parallelism for attention duplicates memory and caps batch size. DP attention runs attention data-parallel (each rank owns distinct sequences, no KV duplication) while the MoE FFN stays expert-parallel. This is what lets a deployment hold large batches and long contexts at high EP without exploding KV memory, and it is a distinct, non-obvious axis from raw TP/PP/EP degrees.

- **SOTA bar:** SGLang's DP Attention is the production technique that eliminates KV-cache duplication across devices for DeepSeek-V3, enabling the EP32/EP72 deployment to scale memory-efficiently; combined with PD disaggregation and MTP it delivers up to ~60% higher output throughput.
- **Leading systems:** SGLang, vLLM
- **Source:** [https://www.lmsys.org/blog/2025-07-17-mtp/](https://www.lmsys.org/blog/2025-07-17-mtp/) (2025-07-17)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a single-device reuse kernel. DP attention is a multi-device memory-deduplication technique (eliminate KV duplication across GPUs); fak runs one in-kernel KV cache on one device and has no hybrid attention/FFN parallelism. fak's KV story is prefix-REUSE across agents on one cache (RadixAttention parity, kv-hit-rate row), an orthogonal axis from DP-attention's cross-device dedup. No fak number; verdict no-claim.
- **Trace:** none — no fak number exists

### ○ Multi-node scale-out (combined TP x PP x EP x DP across hosts) — fak: **no-claim**

*Why it matters:* Serving a frontier model means composing tensor, pipeline, expert, and data parallelism across many hosts and an InfiniBand/RoCE fabric. The relevant operator question is not 'does it run on 8 GPUs' but how many GPUs/nodes a single coherent deployment spans, what interconnect it requires, and how cleanly the parallelism dimensions compose. Datacenter-scale orchestration (rack-scale NVLink domains, cross-node fabric) is the table-stakes axis for frontier serving.

- **SOTA bar:** Production open-source deployments span 96 H100 GPUs (12 nodes) for a single DeepSeek-V3 instance (SGLang); rack-scale GB200/GB300 NVL72 presents 72 Blackwell GPUs as one NVLink domain, and Dynamo is positioned as a datacenter-scale distributed inference framework orchestrating disaggregated pools across the cluster.
- **Leading systems:** SGLang, NVIDIA Dynamo, TensorRT-LLM, NVIDIA GB200/GB300 NVL72
- **Source:** [https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving) (2025-03-18)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for the reuse/adjudication kernel. fak's OWN engine runs on a single device (CPU, or one RTX 4070 / RX 7600 / M3 Pro / datacenter GPU). The only multi-GPU run in the evidence (8-GPU datacenter server, Qwen3.6-27B) is SGLang doing the TP scale-out with fak as the in-front gateway/adjudication plane — and there fak TRAILS raw SGLang on throughput (served-throughput-vs-sglang row). fak has no own TPxPPxEPxDP scale-out number; verdict no-claim.
- **Trace:** none — fak's own engine is single-box (CLAIMS GPU backends are all one device); the 8-GPU datacenter server run is SGLang-serves + fak-adjudicates

