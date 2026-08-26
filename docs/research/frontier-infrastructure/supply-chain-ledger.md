# Datacenter delivery and supply-chain ledger

**As of:** 2026-08-26. **Tracker:** #9304. This ledger treats an AI datacenter as a
multi-stage industrial system. A GPU purchase or gigawatt announcement becomes useful capacity
only after every binding input is delivered, integrated, commissioned, healthy, scheduled,
and matched to a real workload.

## Capacity lifecycle

| State | Minimum evidence |
|---|---|
| Announced / intended | Public plan with actor, location or scope, and date. |
| Optioned / reserved | Land, components, generation, or capacity rights held but not fully contracted. |
| Contracted | Binding purchase, lease, utility, generation, or customer agreement, with conditions. |
| Permitted / interconnected | Material land-use, environmental, grid/interconnection, and construction approvals. |
| Under construction | Site and infrastructure work has begun; supply and completion risk remain. |
| Powered | Electricity and cooling can support the stated IT load under operating constraints. |
| Installed | Hardware is physically present; integration, firmware, network, and acceptance remain. |
| Accepted | System passes customer/operator acceptance and availability tests. |
| Healthy / schedulable | Capacity is visible to the control plane and not reserved, failed, or degraded. |
| Active | Workloads are running; utilization and SLO may still be poor. |
| Useful goodput | Quality-constrained work is completed within SLO after failures, retries, and overhead. |

Any aggregate “GW,” “GPUs,” or “capacity” figure must name its state and boundary.

## Power, grid, and generation

- **Grid connection:** the IEA estimates grid constraints could delay about 20% of globally
  planned datacenter capacity through 2030. Interconnection queues, transmission upgrades,
  substation buildout, and utility planning can dominate chip delivery.
- **Demand growth:** the IEA reported datacenter electricity consumption rising 17% in 2025,
  from roughly 485 TWh in 2025 toward about 950 TWh in 2030 in its outlook.
- **Firm generation:** public projects increasingly pair datacenters with gas generation,
  nuclear uprates/new builds, batteries, geothermal, fuel cells, or behind-the-meter systems.
  Nameplate generation is not guaranteed delivered IT power; fuel, outages, transmission,
  emissions, and permitting remain.
- **Demand response:** Google disclosed shifting or reducing machine-learning load during grid
  events, evidence that deadline and mobility classes can convert some compute into a grid
  resource.
- **Power electronics:** AI load density and rapid current transients are pushing interest in
  higher-voltage DC distribution, medium-voltage solid-state transformers, and rack/platform
  co-design. These remain emerging architectures, not universal deployed defaults.
- **Ratepayer allocation:** Anthropic, OpenAI, Meta, and local reporting increasingly address
  who pays for generation, substations, and transmission. Social license depends on the answer.

### Required power fields

Utility/region, interconnection request and queue date, contracted MW, expected phase dates,
firm vs interruptible supply, generation type, redundancy, storage, transmission/substation
scope, upgrade payer, tariff, demand-response rights, metered IT and facility load, PUE, outage
and curtailment history, and uncertainty.

## Cooling, water, and heat rejection

- Liquid cooling is now a platform assumption for many rack-scale AI systems, but “liquid
  cooled” can mean direct-to-chip loops, rear-door heat exchangers, immersion, or hybrid paths.
- Closed-loop language does not mean zero water use: heat rejection, humidification, electricity
  generation, construction, and maintenance can create direct or indirect water demand.
- The IEA places cooling near 7% of efficient hyperscale electricity use but above 30% in some
  less-efficient enterprise facilities; climate, load, and design change the ratio.
- Public water disclosure remains weak and inconsistent. Report source water, consumptive use,
  withdrawals, recycling, discharge, seasonality, drought restrictions, WUE, and boundary.
- Rack density affects coolant distribution units, pumps, piping, serviceability, leak domains,
  heat exchangers, chillers/dry coolers, and recovery after failure—not only nominal capacity.

## Electrical and construction equipment

The long-lead bill of materials includes transformers, switchgear, breakers, busways, UPS,
generators/turbines, batteries, substations, transmission equipment, chillers, pumps, cooling
units, racks, cabling, fire suppression, and controls. OpenAI’s domestic-manufacturing RFP and
reported project delays are evidence that broad industrial capacity, not just semiconductors,
sets delivery time.

Record vendor, factory/region, order date, promised and actual delivery, capacity rights,
qualification/acceptance, substitutions, cancellation terms, and single-source exposure.

## HBM, semiconductors, and advanced packaging

- Accelerator supply depends on foundry wafers, HBM stacks, silicon interposers, substrates,
  advanced packaging such as CoWoS, testing, and board/system integration.
- Normalized “H100-equivalent” counts erase physical SKU, memory, network, power, and delivery
  differences. Keep physical inventory and normalization methodology separate.
- Memory capacity/bandwidth can bind training, long context, KV cache, and decode independently
  of nominal FLOPS.
- A reported August 2026 rumor claimed NVIDIA warned major customers of >15% AI-server price
  increases due to memory pressure. It remains explicitly unconfirmed in `index.json`; use it as
  a sensitivity scenario, not a fact.

Required fields: accelerator/CPU/NIC/HBM SKU and revision, quantity by lifecycle state, foundry,
packaging, HBM capacity/bandwidth, board/system integrator, yield/acceptance, firmware/software,
price boundary, order/delivery, and substitutions.

## Networking and optics

- Rack-scale systems use scale-up fabrics while pods/sites use scale-out Ethernet/RoCE,
  InfiniBand, or proprietary interconnect. MoE collectives, disaggregation, checkpointing, and
  storage traffic create different bottlenecks.
- NVIDIA’s 2026 $2B investments in both Lumentum and Coherent plus multibillion-dollar purchase
  commitments are direct evidence that lasers, optical components, packaging, and domestic
  manufacturing capacity are strategic supply inputs.
- Financial Times reporting described pressure across fiber, lasers, and indium-phosphide
  substrates as the field moves toward co-packaged optics.
- Peak link bandwidth is not collective goodput. Record topology, oversubscription, congestion,
  cable/optics failures, retransmission, collectives, tail latency, and serviceability.

## Storage and data movement

Training checkpoints, datasets, model weights, container images, observability, and inference KV
state compete for network and storage bandwidth. Emerging systems such as multi-tier KV stores,
Mooncake/3FS-style disaggregation, and cache offload move inference state beyond local HBM.

Record bytes and objects by phase, read/write amplification, checkpoint interval, recovery time,
cache tiers, transfer/recompute choice, consistency, encryption, region, failure domain, and
shared-network interference.

## Land, permitting, labor, and community

- Local opposition now affects elections and state/local rules around electricity prices,
  pollution, tax incentives, land, noise, water, and generation.
- Data Center Watch estimates $64B of U.S. projects blocked or delayed by opposition, but its
  advocacy methodology requires project-level verification.
- A widely repeated report that 30–50% of planned 2026 U.S. AI datacenters could be delayed or
  canceled was challenged by SemiAnalysis as a denominator/state-conflation error. Keep both the
  claim and rebuttal; never quote one percentage without the underlying project roster.
- Skilled electrical, mechanical, construction, commissioning, and operations labor can gate
  delivery. Public “jobs created” claims do not establish staffing availability or retention.

Required fields: parcel/site, zoning, permits, environmental review, appeals/litigation, tax and
rate agreements, water/noise/emissions commitments, workforce, construction state, community
benefits/costs, opposition, and decision dates.

## Contradiction case: “half of 2026 capacity delayed”

| Claim | Counterevidence | Correct treatment |
|---|---|---|
| Secondary reporting citing Bloomberg said 30–50% of planned 2026 U.S. projects might be delayed/canceled due to power and electrical shortages. | SemiAnalysis argued the headline conflated project states and traced repetition to one framing rather than a common project denominator. | Retain both sources. Build a named project roster with original expected date, current state, MW boundary, reason, and confidence before computing a percentage. |

This case is the template for all market totals: identify the denominator, capacity state,
geography, original event date, and whether later articles are independent evidence.

## Delivery-risk register

| Risk | Leading indicator | Falsifiable witness |
|---|---|---|
| Power interconnection delay | Queue position, study milestones, utility construction | Energization and metered delivery date |
| Generation delay | Permits, equipment order, fuel/interconnect | Commercial operation and firm available MW |
| Transformer/switchgear shortage | Factory slot and promised ship date | Accepted equipment on site |
| Cooling/water constraint | Design PUE/WUE, permits, drought rules | Full-load thermal test and operating consumption |
| HBM/packaging constraint | supplier allocation, wafer/package starts | Accepted accelerator systems by SKU |
| Optics/network constraint | laser/substrate/transceiver allocation | Fabric acceptance and sustained collective goodput |
| Construction/labor delay | milestone slip, contractor staffing | Commissioned phase against baseline schedule |
| Community/permitting block | hearing, appeal, ballot, litigation | final permits and unresolved-condition closure |
| Customer/finance risk | take-or-pay, credit, funding, concentration | paid capacity and sustained utilization |
| Software/RAS failure | burn-in, incident, unhealthy node rate | healthy/schedulable fraction and interruption-adjusted goodput |

## Outstanding evidence gaps

- Project-level U.S. and global site census with lifecycle, MW boundary, original/current dates,
  grid/interconnect status, and delivery confidence.
- Utility interconnection queues and tariff/ratepayer agreements joined to named AI sites.
- Transformer, turbine, switchgear, cooling, HBM, packaging, optics, and construction capacity
  ledgers by vendor/factory and committed delivery.
- Metered PUE/WUE, installed-to-healthy conversion, failure/interruption, and cluster goodput.
- China, Middle East, India, and Europe supply chains and export-control substitutions.
- Cancellation/refutation history instead of announcement-only accumulation.
