# Frontier-lab and regional census

**As of:** 2026-08-26
**Parent issues:** [#9300](https://github.com/anthony-chaudhary/fak/issues/9300), [#9313](https://github.com/anthony-chaudhary/fak/issues/9313)
**Authority:** [`../index.json`](../index.json)

## Result first

The current index contains **59 frontier-lab entries** within a **207-entry**
corpus. This pass adds primary-source coverage for major U.S./Canadian labs and for
China, Israel, the Middle East, India, Japan, Korea, and Southeast Asia. It reduces a
strong OpenAI/Anthropic bias, but it does **not** make the global census complete.

The new evidence strengthens four expectations:

1. **One endpoint is often several workload classes.** Reasoning controls, modality,
   language, and agent/tool modes change output length, latency, memory, and batchability.
2. **Active parameters are not a deployment receipt.** MoE expert count/routing,
   quantization, total parameters, host boundary, KV compression, and network topology
   all matter.
3. **Regional and sovereign labs optimize different envelopes.** Data locality,
   language/tokenization, edge/private deployment, and national compute programs can be
   first-order constraints.
4. **Training disclosure is uneven.** Ai2 gives a physical 1,280-H100 cluster and MFU;
   many other labs publish only model architecture or benchmark claims. Unknown stays
   unknown.

## Extraction contract

A checked row means the index contains at least one dated source with the fields below.
It does not mean complete organizational coverage.

| Field | What to capture |
|---|---|
| Identity | Lab, parent, headquarters/region, and relevant cloud or sovereign program. |
| Model/workload | Model family, modality, reasoning/tool mode, context, and target product workload. |
| Physical envelope | Accelerator SKU/count, node/host boundary, topology, memory/quantization, power, and region when disclosed. |
| Scale | Total/active parameters, experts, training tokens, users, requests, or production traffic—never substitute one for another. |
| Lifecycle | Research, preview, released, API, on-device, private-cloud, in production, or planned. |
| Evidence | Primary source first; retain official statement, vendor claim, measurement, report, or rumor labels. |
| Limits | Missing denominator, undisclosed cluster, author benchmark, maximum-versus-prevalence, and traffic uncertainty. |
| fak use | Concrete benchmark, scheduler, routing, cache, hardware, or trust-boundary implication. |

## Checked labs and current evidence

### United States and Canada

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | OpenAI | Stargate/site, Oracle/NVIDIA, user-scale, and capacity announcements. | Physical installed/healthy fleet, workload mix, tenant concentration, and plan-to-goodput conversion. |
| Checked | Anthropic | AWS/Google/Azure partnerships, multicloud failure evidence, U.S. infrastructure investment, and usage research. | Provider traffic shares, physical fleet, cache/batch distributions, and regional failover. |
| Checked | xAI | Colossus/Grok model and financing claims. | Physical SKU census behind H100-equivalent totals, healthy capacity, utilization, and request/token distributions. |
| Checked | Google DeepMind | Gemini 3.1 Flash-Lite: TPU/JAX/Pathways training, 1M input, 64K output, high-volume/latency-sensitive positioning, and five distribution channels. | TPU generation/count, pod topology, production model mix, batch/concurrency, and neutral goodput. |
| Checked | Meta | Llama 4 MoE active/total parameter and host-fit envelope; existing datacenter and capex evidence. | Production expert skew, fleet share, Meta AI request/token distribution, and quality-constrained serving economics. |
| Checked | Amazon Nova | Nova 2 family, configurable extended thinking, multimodality, and up-to-1M context. | Training/serving hardware, mode-selection shares, Bedrock traffic, and output-length tails. |
| Checked | Apple | Roughly 3B on-device model, 2-bit QAT, KV sharing, server Parallel-Track MoE, and Private Cloud Compute split. | Server size/fleet, request routing share, concurrency, and production latency. |
| Checked | Cohere | Command A enterprise/agentic and 256K-context positioning. | Neutral efficiency reproduction, private-deployment fleet, customer/tenant traffic, and regional demand. |
| Checked | Ai2 | OLMo 2 32B trained to 6T tokens on 160 × 8 H100 nodes; >1,800 tokens/s/GPU and ~38% MFU. | Net wall-clock availability/cost and inference demand; training throughput is not serving goodput. |
| Checked | Microsoft Research Phi | Phi-4-reasoning 14B inference-time-compute model; compact open-weight reasoning envelope. | Physical training/serving cluster, first-party traffic, reasoning-length distribution, and neutral goodput. |
| Unchecked | NVIDIA model research | NVIDIA appears as platform/partner, not as a separately audited model lab. | Nemotron model training and deployed inference envelope. |
| Unchecked | Databricks/MosaicML, Essential AI, Reflection AI, Safe Superintelligence, Thinking Machines, World Labs | No dedicated primary-source entries. | Model status, cluster/funding-to-capacity conversion, product workload, and deployment state. |

### China

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | DeepSeek | V3 training cluster/economics and hardware/architecture reflections. | Current fleet, export-control response, production traffic, and post-release capacity. |
| Checked | Alibaba Qwen | Qwen3 reasoning-budget evidence plus Alibaba Cloud/cache entries. | Physical Qwen training/serving cluster, product traffic, and reasoning-mode shares. |
| Checked | Moonshot AI | Kimi K2 model/cluster evidence. | Production request distribution, utilization, cache behavior, and financing-to-capacity lifecycle. |
| Checked | MiniMax | M3 1M-context coding/multimodal release. | Context prevalence, cluster hardware, traffic, compaction, and serving cost. |
| Checked | ByteDance Seed | Seed1.5-VL 532M vision encoder + 20B-active MoE, GUI/game agent tasks, and Volcano Engine availability. | Total parameters, cluster hardware, Doubao/Volcano traffic, modality share, and production routing. |
| Checked | Baidu | ERNIE 5.0 PaddlePaddle hybrid parallelism, FP8 activation storage, adaptive offload, and separated tokenizer/backbone nodes. | Cluster size/SKU, net training efficiency, serving topology, and ERNIE traffic. |
| Checked | Tencent | Hunyuan-Large 389B total/52B active, 256K context, mixed routing, and KV compression. | Expert-load distribution, cluster, deployed fleet, and service economics. |
| Checked | Z.ai / Zhipu AI | GLM-4.5 355B total/32B active, 23T training tokens, and hybrid thinking/direct modes. | Cluster hardware, mode-selection rates, API traffic, and independent benchmark reproduction. |
| Checked | Huawei Cloud Pangu | Pangu 5.0 tiers above 1B/10B/100B/1T parameters and >400 scenarios across >30 industries; the separate Atlas 900 A3 SuperPoD vendor specification names 384 Ascend NPUs, 300 TB HBM, 48 PFLOPS dense BF16, 16.1 PB/s HBM bandwidth, 7.2 PB/s scale-up bandwidth, and 96 cabinets. | Built/installed/healthy/schedulable deployment, power/cooling, model-tier traffic, requests/users, batch/cache behavior, and neutral goodput. |
| Partial | 01.AI | Yi-34B-200K model card: 34B parameters, 200K advertised context, 5B-token long-context continuation. | Physical compute, neutral long-context serving results, production traffic, and lifecycle evidence. |
| Partial | StepFun | Step3 release: 321B total / 38B active multimodal MoE and attention/FFN disaggregation design. | Training cluster, neutral matched efficiency, shipped serving topology, adoption, and traffic. |
| Unchecked | Baichuan, iFlytek, Meituan | No dedicated current entries. | Model releases, physical compute, adoption, deployment, and lifecycle evidence. |
| Partial | Shanghai AI Lab / InternLM | InternLM3-8B release: 4T training tokens and first-party training/deployment toolchain. | Physical compute, cost accounting, production deployment, and traffic. |
| Partial | SenseTime | SenseNova 5.0 vendor release: MoE, >10TB token data, ~200K effective context, cloud-device-edge matrix. | Neutral benchmark, physical compute, production workload, and lifecycle evidence. |
| Partial | Xiaomi MiMo | MiMo-V2-Pro release: >1T total / 42B active parameters, 1M context, tiered API pricing above 256K. | Physical compute, traffic/context distribution, neutral serving performance, and adoption. |

### Europe

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | Mistral AI | Mistral Compute, model training, environmental LCA, and cross-cloud distribution. | Physical cluster and utilization, API/private mix, tenant concentration, and sovereign-cloud economics. |
| Unchecked | Aleph Alpha, Black Forest Labs, Helsing, Kyutai, LightOn, Poolside, Silo AI/AMD | No dedicated primary-source model/infrastructure entries. | Training/serving envelopes, acquisitions/ownership effects, and production traffic. |
| Unchecked | UK labs beyond DeepMind | No dedicated census for Stability AI, Synthesia, Wayve, or university/sovereign programs. | Model scope, compute, financing, and deployment state. |

### Israel

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | AI21 Labs | Jamba 1.5 hybrid SSM-Transformer MoE, 256K effective context, single-GPU/one-node fit claims, and quantized deployment envelope. | Production concurrency, traffic, hardware mix, and neutral long-context throughput. |
| Unchecked | AI research/product labs at NVIDIA Israel and other private firms | No dedicated entries. | Lab boundaries, model programs, and physical deployment. |

### Middle East

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | Technology Innovation Institute (UAE) | Falcon Arabic and Falcon-H1 sovereign-language, hybrid Transformer/Mamba, multilingual, edge-to-enterprise positioning. | Training cluster, production traffic, per-tier throughput/energy, and independent performance. |
| Partial | G42 / Inception / MBZUAI (UAE) | Stargate UAE plus MBZUAI annual-review evidence for NANDA, LLM360, and K2-65B. | Dedicated Jais evidence, model training/serving envelope, national cluster allocation, and product demand. |
| Unchecked | Saudi SDAIA/ALLaM and other sovereign programs | No dedicated entry. | Primary model sources, compute allocation, language traffic, and deployment. |

### India

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | Sarvam AI | 30B/105B 128-expert models trained in India on IndiaAI compute, 16T/12T tokens, with named production products. | Physical cluster/allocation, request volumes, traffic mix, goodput, and independent performance. |
| Unchecked | Krutrim, BharatGen, Soket AI, Gnani, CoRover, IIT/research consortia | No dedicated entries. | Model families, IndiaAI allocation, language/token distributions, and production state. |

### Japan

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | Sakana AI | Composition/model-merging, automated AI science, edge-efficient model strategy, and enterprise deployment claims. | Model and cluster scale, measured energy, customer traffic, and repeatable production outcomes. |
| Checked | NTT tsuzumi | 0.6B CPU and 7B single-GPU Japanese enterprise deployment tiers. | Hardware SKU, quantization, context, production latency/concurrency, and demand. |
| Unchecked | Preferred Networks, Fujitsu, SoftBank/SAKURA, AIST and university programs | No dedicated model-lab entries. | Domestic accelerator/cloud use, training runs, sovereign programs, and service traffic. |

### Korea

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | NAVER Cloud | HyperCLOVA X THINK: 6T Korean/English tokens, 128K context, reasoning/length-control training, and multimodal/agentic targets. | Parameter/cluster size, production demand, language mix, and serving economics. |
| Checked | LG AI Research | EXAONE 4.0 32B professional + 1.2B on-device hybrid reasoning/direct family with tool use. | Cluster, traffic, mode selection, device latency, and neutral efficiency. |
| Partial | Samsung Electronics | Gauss2 variants, MoE Supreme tier, internal developer and call-center use. | Model sizes, physical cluster, request/token distribution, latency, and external product deployment. |
| Partial | SK Telecom | A.X 3.1/4.0 dual strategy, 72B/7B variants, and A. call-summarization application. | Training/serving hardware, traffic, latency, batching, and external adoption. |
| Partial | Kakao | Kanana technical report and open 2.1B Nano release across a 2.1B-32.5B model family. | Physical compute, production deployment, service traffic, and lifecycle. |

### Southeast Asia

| Status | Lab | Evidence now indexed | What remains unknown |
|---|---|---|---|
| Checked | Sea AI Lab | Sailor2: stated production preference for 8B/20B tiers, 1B speculative-decoding/research tier, ~500B continued-pretraining tokens, and 15 languages. | Request counts, tenant/language shares, speculative acceptance, hardware, and deployed goodput. |
| Checked | AI Singapore SEA-LION | v4.5 multilingual/multimodal/agentic family for >11 SEA languages with custom speculative decoder. | Model size, hardware, acceptance distribution, traffic, and language/user shares. |
| Unchecked | Grab, GoTo, SCB10X, VinAI, regional sovereign programs | SCB10X appears only as a Sailor2 collaborator; no dedicated lab entries. | Model programs, compute, product demand, and country/language distributions. |

## Cross-lab parameter observations

| Dimension | Evidence in this batch | Safe expectation |
|---|---|---|
| Reasoning mode | Amazon Nova 2, GLM-4.5, HyperCLOVA X THINK | Treat reasoning selection as a workload variable with separate output/SLO distributions. |
| MoE | Llama 4, Seed1.5-VL, Hunyuan-Large, GLM-4.5, Sarvam | Capture total/active parameters, expert count, routing skew, memory, and topology; active parameters alone are insufficient. |
| Long context | Gemini 1M/64K output; Nova up to 1M; Command A/Jamba/Hunyuan/HyperCLOVA 128K–256K | Capability ceiling is not prevalence. Benchmark token-time, KV occupancy, compaction, and multi-tenant concurrency. |
| Device/private cloud | Apple on-device + Private Cloud Compute; TII/Sakana edge claims | Route by trust and resource boundary; do not collapse device and datacenter economics. |
| Regional language | Falcon Arabic, Sarvam, HyperCLOVA, Sailor2 | Measure tokenizer expansion, language mix, data locality, and locale-specific SLOs. |
| Physical training receipt | Ai2 OLMo: 1,280 H100s, GPUDirect-TCPXO, >1,800 tokens/s/GPU, ~38% MFU | Prefer physical SKU/topology/utilization receipts over “equivalent” counts. Training evidence still does not prove serving demand. |
| Speculative decoding | Sailor2 1B tier named for speculative decoding | Measure acceptance, verifier cost, draft placement, and net goodput under regional-language workloads. |

## Remaining priority queue

1. NVIDIA Nemotron plus Microsoft first-party deployment/traffic evidence.
2. Huawei Ascend deployment and power/cooling receipts, 01.AI, StepFun, Baichuan,
   Shanghai AI Lab, SenseTime, iFlytek, Meituan, and Xiaomi.
3. G42/Inception Jais/Nanda deployment evidence and Saudi ALLaM/SDAIA sovereign programs.
4. Samsung Gauss, SKT A.X, and Kakao-related Korean programs.
5. Preferred Networks, Fujitsu, SoftBank/SAKURA, and Japanese public compute.
6. Grab/GoTo, SCB10X, VinAI, and regional sovereign initiatives.
7. Europe beyond Mistral and DeepMind, plus private U.S./Canadian labs.
8. For every checked lab: a second pass on failures, cancellations, users/requests,
   physical serving hardware, geography, and lifecycle from release to active goodput.

The census remains **partial** until that queue and the broader
[`../coverage-audit.md`](../coverage-audit.md) requirements are closed with primary
sources and explicit unknowns.
