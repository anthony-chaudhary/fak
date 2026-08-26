# Startup and adjacent-provider landscape

**As of:** 2026-08-26. This is a taxonomy and watchlist, not an endorsement, market-share
claim, or proof that every company remains independent. Company state changes quickly;
material claims belong in [`index.json`](index.json) with a dated source.

| Layer | Representative companies/projects to monitor | Expectation exposed |
|---|---|---|
| Neocloud / GPU capacity | CoreWeave, Nebius, Crusoe, Lambda, Together AI, Voltage Park, Fluidstack, Applied Digital, Northern Data/Taiga, Nscale | Capital-intensive capacity, customer concentration, financing, power access, cluster goodput, and managed Kubernetes/bare metal matter together. |
| Specialized inference silicon/cloud | Groq, Cerebras, SambaNova, d-Matrix, Etched, Tenstorrent, Positron, FuriosaAI, Rebellions, SiMa.ai | Low latency, deterministic execution, memory bandwidth, and phase specialization compete with general GPU ecosystem breadth. |
| Serving/runtime/orchestration | vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, llm-d, Ray Serve/Anyscale, BentoML, Baseten, Modal, Fireworks AI, Replicate, OctoAI lineage | Continuous batching, KV state, disaggregation, routing, autoscaling, multi-tenancy, and observability are becoming a control plane. |
| Model/API routing and gateways | OpenRouter, Portkey, Martian, Not Diamond, Cloudflare AI Gateway, Kong/Envoy ecosystem | Providers, models, regions, cache state, price, quality, and policy vary per request; fallback can damage cache locality and provenance. |
| KV cache / memory / storage | LMCache, Mooncake, DeepSeek 3FS, Alibaba Tair KVCache, Alluxio, Weka, VAST Data, Hammerspace | Inference state is becoming a distributed storage/network problem, not only local HBM. |
| Networking/interconnect | Broadcom, Marvell, Astera Labs, Enfabrica, Lightmatter, Celestial AI, Ayar Labs, Cornelis Networks, Xsight Labs | Scale-up/scale-out fabrics, optics, congestion, collectives, and composability can gate useful accelerator output. |
| HBM, packaging, chiplets | SK hynix, Samsung, Micron, TSMC/CoWoS, ASE, Amkor, Eliyan, UCIe ecosystem | Memory and advanced packaging lead times can bind before nominal compute supply. |
| Optical components/CPO | Coherent, Lumentum, Innolight, Fabrinet, POET, DustPhotonics, Ranovus | Lasers, fiber, transceivers, substrates, and co-packaged optics become rack/fleet delivery constraints. |
| Cooling / heat reuse | Vertiv, Schneider Electric, CoolIT, Submer, LiquidStack, Iceotope, ZutaCore, JetCool, Accelsius | Rack density shifts cooling from facility overhead to platform design; water and heat rejection alter site feasibility. |
| Power generation / grid / storage | Oklo, Kairos, X-energy, TerraPower, Last Energy, Bloom Energy, Crusoe, Fervo, Form Energy, Redwood Materials | Firm generation, turbines, nuclear/geothermal timelines, batteries, and behind-the-meter systems determine delivery dates and social license. |
| Datacenter development / modular build | Vantage, QTS, Digital Realty, Equinix, Aligned, DataBank, STACK, EdgeConneX, Crusoe, Lancium, Tract, PowerHouse, T5 | Land, substations, transmission, construction labor, modules, financing, permitting, and tenant contracts form one pipeline. |
| Sovereign AI / regional clouds | G42/Core42, TII, HUMAIN, Nscale, Scaleway, OVHcloud, Jio, Yotta, sovereign programs in Europe/Asia/Middle East | Jurisdiction, residency, national capacity, export controls, and local language/product needs constrain placement. |
| Datacenter observability / digital twins | NVIDIA DSX/Omniverse, Cadence, Siemens, Schneider, Vertiv, specialized DCIM vendors | Power, thermal, network, workload, and maintenance policy increasingly need joint simulation and read-back. |

## What to record for each company or project

- legal/operating state: independent, acquired, merged, licensed, failed, paused, or renamed;
- latest dated release and actual availability state;
- funding, debt, leases, purchase obligations, backlog, revenue, losses, and customer concentration;
- installed/healthy capacity, power lifecycle state, sites/regions, accelerator and network mix;
- workload/channel mix, active tenants/users, request/token distribution, SLO and goodput;
- claimed advantage with workload, baseline, quality, topology, energy, and cost boundaries;
- partnerships, licensing, acquisitions, cancellations, export controls, and supply dependencies;
- source class and confidence; rumor origin, independent corroboration, expiry, and outcome.

## Failure modes in startup coverage

1. Treating funding or valuation as technical/product evidence.
2. Counting announced GPUs/MW as online healthy capacity.
3. Repeating vendor benchmark maxima without workload and quality constraints.
4. Missing acquisitions, licensing deals, shutdowns, or customer-concentration risk.
5. Calling every GPU lessor a differentiated cloud or every chip startup deployed.
6. Treating multiple articles copied from one leak as independent rumor corroboration.
7. Ignoring private debt, take-or-pay contracts, and power-delivery risk.

## Lifecycle outcomes beyond funding and launches

| Company / asset | Initial category | Observed outcome | What survives | What failed or changed |
|---|---|---|---|---|
| Untether AI | In-memory/RISC-V AI accelerator | Shutdown, engineering-team transfer to AMD, later bankruptcy | Engineering talent may continue inside AMD. | Standalone company and product support ended; reported liabilities exceeded assets. |
| Replicate | Hosted model execution/API | Joined Cloudflare | Brand and API were promised to continue; model execution primitives gain network/platform integration. | Independent ownership; transaction economics and long-term product boundary are undisclosed. |
| SchedMD / Slurm | Cluster scheduling/workload management | Acquired by NVIDIA | Slurm is promised to remain open source and vendor-neutral. | Independent vendor governance; neutrality must be observed, not assumed. |
| Groq IP/team | Inference accelerator/cloud | Non-exclusive NVIDIA license plus reported key-talent transfer | Groq remained operational; licensing was not described as a full acquisition. | Independent roadmap, staffing, and competitive boundary may change. |

### Failure and acquisition extraction contract

For each future event, record separately:

```text
legal entity / team / IP / product / customer contracts / support / brand
announced / signed / closed / integrated / discontinued / bankrupt
transaction value / assets / liabilities / runway if public
customer migration and compatibility path
open-source governance and hardware neutrality
later outcome check
```

Do not label an acquihire, IP license, minority investment, product acquisition, or
bankruptcy as a full-company acquisition unless the source says so.
