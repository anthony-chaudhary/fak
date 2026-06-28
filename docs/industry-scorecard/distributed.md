---
title: "fak industry scorecard — distributed"
description: "The distributed dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# distributed — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Parallelism & disaggregation (`parallelism`)

### ○ Prefill-decode disaggregation: independent scaling of compute-bound prefill and memory-bound decode — fak: **no-claim**

*Why it matters:* Prefill is compute-bound and decode is memory-bandwidth-bound; co-locating them forces one hardware/parallelism config to serve both and couples their scaling. Disaggregating onto separate instances (over a KV-cache transport) lets each phase scale and tune independently and removes interference entirely. It is now the default architecture for large-scale, SLO-tight deployments.

- **SOTA bar:** SGLang PD disaggregation achieves up to 6.9x more goodput (Chatbot 3200in/800out, TTFT<=1000ms, TPOT<=25ms) on MI300X; this remains the headline open-framework PD-disaggregation figure, now corroborated by NVIDIA Dynamo's 2026 production results (38% goodput over best aggregated config; 7x with wide expert parallel on GB200 NVL72).
- **Leading systems:** SGLang PD (MI300X), NVIDIA Dynamo 1.0 (38% over best aggregated config; 7x w/ wide-EP on GB200), vLLM disaggregated prefilling (NIXL)
- **Source:** [https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html](https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html) (2025-08)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for the reuse kernel: fak does not disaggregate prefill from decode into independently-scaled pools; its differentiator is cross-agent prefix fusion on one kernel, not P/D separation. DistServe/Splitwise/SGLang-PD/llm-d numbers have no fak counterpart; named as out-of-scope rather than a measured gap.
- **Trace:** No P/D-disaggregation goodput/TTFT number in BENCHMARK-AUTHORITY.md or CLAIMS.md. fak runs a single fused forward (prefill+decode together); no disaggregated prefill/decode pool is built or measured.

### ○ Prefill-decode disaggregation & per-phase SLO isolation — fak: **no-claim**

*Why it matters:* Prefill is compute-bound (TTFT-sensitive) and decode is memory-bandwidth-bound (TPOT-sensitive); colocating them couples their tails. Disaggregating onto separate GPU pools lets each phase scale and meet its own SLO independently, which has become the production standard for hitting tight TTFT AND tight TPOT simultaneously at scale.

- **SOTA bar:** Phase disaggregation for SLO isolation (separating compute-bound prefill from memory-bound decode so each phase's SLO is met independently) is now production-standard via NVIDIA Dynamo, which reports 38% goodput over the best aggregated config and 7x with wide expert parallel on GB200 NVL72; DistServe's 7.4x remains the founding goodput figure.
- **Leading systems:** NVIDIA Dynamo 1.0, DistServe, TaiChi (unifies agg+disagg to fix both TTFT and TPOT violations, 2025-08)
- **Source:** [https://developer.nvidia.com/blog/removing-the-guesswork-from-disaggregated-serving/](https://developer.nvidia.com/blog/removing-the-guesswork-from-disaggregated-serving/) (2026-03)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. PD-disaggregation with per-phase SLO isolation (DistServe 7.4x request rate / 12.6x tighter SLO; NVIDIA Dynamo Planner) is a serving-stack architecture; fak owns neither the prefill/decode scheduler nor the per-phase GPU pools. It is an adjudication/coherence plane that rides on whatever engine does this. no-claim.
- **Trace:** No prefill-decode disaggregation is shipped or claimed. fak's own engine runs a single co-located forward; on the GPU server it fronts SGLang TP=8 (which itself is not run in a disaggregated config in the committed GPU server artifacts).

### ○ Large-scale expert parallelism (EP) for giant MoE models — fak: **no-claim**

*Why it matters:* Sparse MoE models (DeepSeek-V3/R1 671B, Qwen MoE, Mixtral) activate only a few experts per token, so naive tensor parallelism replicates expert weights and wastes memory bandwidth. Scaling EP across dozens of GPUs (one or few experts per device) is the only way to serve 600B+ MoE economically, and demands all-to-all expert dispatch, expert-load balancing, and overlap of comms with compute. EP scale and balance is the differentiating capability for frontier-MoE serving.

- **SOTA bar:** EP320: Huawei CloudMatrix-Infer runs DeepSeek-R1 with 320-way expert parallelism (one expert per NPU die: 32 shared + 256 router + 32 redundant experts) over the UB network, superseding the 96-GPU large-scale EP (effective EP~72) demonstrated by LMSYS/SGLang.
- **Leading systems:** Huawei CloudMatrix-Infer (CloudMatrix384, DeepSeek-R1), SGLang + DeepEP (96-GPU, prior bar), vLLM expert-parallel deployment
- **Source:** [https://arxiv.org/abs/2506.12708](https://arxiv.org/abs/2506.12708) (2025-06)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a single-box reuse kernel. fak DOES run GLM-5.2's MoE/FFN experts + router on its own pure k_q8_gemm GPU kernel (the in-kernel GLM-5.2 / DeepSeek-class path, #1010 / #1026; cosine=1.0, datacenter GPU, commit 498a4ab) — but on a SINGLE device, with no expert-parallel sharding across nodes and the DSA sparse-attention still host-side. fak's own-engine ceiling is ~7B (see max-model-and-moe-coverage); giant-MoE EP (the MLPerf v6.0 / DeepSeek-R1 671B reference regime) is delegated to SGLang/DeepEP, which fak fronts. No fak EP number exists; verdict no-claim.
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

- **SOTA bar:** A single GB300 NVL72 is 72 Blackwell-Ultra GPUs in one NVLink domain; MLPerf Inference v6.0 (2026-03-30) composed FOUR of them -- 288 GPUs over Quantum-X800 InfiniBand, NVIDIA's largest-ever submission -- into one coherent DeepSeek-R1 671B MoE serving run at ~2.494M tok/s aggregate offline. Disaggregated serving via NVIDIA Dynamo composes the TPxPPxEPxDP dimensions across the fabric (up to 1.5x on Llama-3.1-405B interactive).
- **Leading systems:** NVIDIA GB300 NVL72 x4 (288 GPUs over Quantum-X800 InfiniBand, MLPerf v6.0), SGLang/TensorRT-LLM + NVIDIA Dynamo multi-node, GB200 NVL72 (prior generation)
- **Source:** [https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform](https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform) (2026-04-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for the reuse/adjudication kernel. fak's OWN engine runs on a single device (CPU, or one RTX 4070 / RX 7600 / M3 Pro / datacenter GPU). The only multi-GPU run in the evidence (8-GPU datacenter server, Qwen3.6-27B) is SGLang doing the TP scale-out with fak as the in-front gateway/adjudication plane — and there fak TRAILS raw SGLang on throughput (served-throughput-vs-sglang row). fak has no own TPxPPxEPxDP scale-out number; verdict no-claim.
- **Trace:** none — fak's own engine is single-box (CLAIMS GPU backends are all one device); the 8-GPU datacenter server run is SGLang-serves + fak-adjudicates

