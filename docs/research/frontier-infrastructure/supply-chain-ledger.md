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

## Named vendor and site lifecycle additions

| Evidence date | Entity / system | Observed lifecycle state | Quantified evidence | What it does **not** prove |
|---|---|---|---|---|
| 2026-02-11 | Vertiv power and thermal infrastructure | Supplier backlog / ordered | Q4 organic orders +252%; ~2.9× book-to-bill; $15.0B backlog, +109% YoY. | Product mix, shipment date, site installation, commissioning, or online MW. |
| 2026-07-29 | SK hynix HBM4 | Mass shipments begun | Supplier says HBM4 mass shipments started in Q2 2026. | Units, yield, customer allocation, accelerator integration, or accepted systems. |
| 2026-06-18 | SK hynix 12-layer HBM4E | Customer samples / qualification | Up to 16 Gbps/pin; >20% claimed power-efficiency gain; 17% claimed heat-resistance reduction. | Qualification success, mass-production timing, yield, or fleet availability. |
| 2026-04-23 | TSMC CoWoS | 5.5-reticle production; larger package roadmap | 14-reticle package with ~10 compute dies and 20 HBM stacks planned for 2028. | Wafer-equivalent capacity, yield, allocation, shipment volume, or delivered accelerators. |
| 2025-07-22 | GE Vernova / Crusoe onsite generation | Ordered | 29 turbine packages, expected nearly 1 GW nameplate; 19 booked June 2025 after 10 in December 2024. | Delivery/commissioning, continuous output, IT load, PUE, fuel, emissions permits, or compute acceptance. |
| 2026 case study | Start Campus Sines / Siemens Energy | Phase 1 equipment delivered for early operations; future phases planned | Blue GIS, transformers, and electrical infrastructure in Phase 1; future substations and seawater cooling. | Accepted MW, PUE, water flow, commissioning dates, equipment lead time, or independent uptime. |

### Lifecycle joins these rows require

```text
supplier order/backlog
  -> factory slot
  -> sample/qualification (for semiconductors)
  -> shipped
  -> site delivered
  -> installed
  -> commissioned
  -> accepted/healthy
  -> schedulable
  -> active
  -> quality-constrained goodput
```

No row above may skip from backlog, sample, roadmap, or equipment order directly to
usable AI capacity.

## Power, cooling, and modular-delivery additions

| Evidence date | Entity / system | Observed state | Quantified envelope | Boundary that remains |
|---|---|---|---|---|
| 2025-09-18 | Schneider Electric / NVIDIA GB300 reference design | Validated, documented design | 3 clusters; up to 1,152 GPUs; up to 142 kW/rack; liquid-to-liquid CDUs and integrated controls | Blueprint is not built, commissioned, loaded, or production-measured infrastructure. |
| 2026-01-22 | Trane DCDA CDU family | Product launched; first China orders expected Q1 | 400/800/1,350 kW models; custom to 1,700 kW; coordinated control up to 16 units | Expected order delivery is not installed or operating cooling; PUE/space claims are site-dependent. |
| 2026-05-21 | LiquidStack/Trane GigaModular CDU | Commercially available after full-load integration testing and certification | Modular cooling up to 14 MW; early orders | Cooling MW is not IT MW; orders/testing are not shipment, commissioning, uptime, or accepted compute. |
| 2026-07-29 | Johnson Controls | Supplier orders/backlog | $21.0B total backlog, +32%; Americas $15.9B, +40%, supported by datacenter and mission-critical demand | Company backlog spans products/services and non-datacenter work; no direct online-MW conversion. |

### Design, product, backlog, and site are different evidence

```text
reference design
  -> product validated/certified
  -> commercially available
  -> ordered/backlog
  -> shipped
  -> installed
  -> commissioned/accepted
  -> cooling available at design conditions
  -> customer IT installed
  -> live critical IT load
  -> quality-constrained goodput
```

Reference designs answer **what must fit together**. Product releases answer **what a
vendor offers**. Backlog answers **what customers have ordered under the supplier's
definition**. None alone answers how much AI capacity is operating.

## Grid and electrical manufacturing additions

| Evidence date | Supplier / footprint | Observed state | Quantified evidence | Missing delivery join |
|---|---|---|---|---|
| 2025-02-12 | Eaton, Jonesville SC | New three-phase transformer factory announced; production expected 2027 | $340M investment; third U.S. three-phase transformer factory | Factory completion, annual unit/MVA output, allocation, shipment, substation install, energization. |
| 2025-09-04 | Hitachi Energy, U.S. | Multi-site manufacturing investment announced | $1B U.S. program; $457M new large-power-transformer factory in South Boston VA | Construction/output date, MVA throughput, yield, orders, allocation, site commissioning. |
| 2026-02-03 | Siemens Energy, U.S. | Brownfield expansion and new factory program announced | $1B across transformers/service, gas turbines, and grid components | Product split, factory start, output, lead-time reduction, shipment, commissioning. |
| 2025-05-14 | GE Vernova T&D India | New manufacturing/testing capacity planned in Chennai and Noida | ~$16M; Electrification backlog reported >3× YoY | Product-specific throughput, domestic/export allocation, shipped/installed equipment, live grid capacity. |
| 2025-03-25 | Schneider Electric, U.S. | Multi-site electrical manufacturing and test-lab investment announced through 2027 | >$700M planned; >1,000 jobs; medium-voltage products, circuit breakers, switchgear, power distribution, and AI-datacenter test labs | Facility-by-facility completion, unit/MVA output, factory slots, customer allocation, shipment, acceptance, energization. |
| 2026-04-08 | Eaton, Bellevue NE | 370,000-square-foot medium-voltage switchgear factory announced; production targeted for H1 2027 | >$30M investment; >200 jobs; air- and gas-insulated switchgear | Qualified production, annual units/ratings, backlog allocation, shipment, site acceptance, live datacenter MW. |
| 2025-10-17 | PJM large-load forecasting | RTO filing documents utility-specific request/verification processes rather than one PJM load-interconnection queue | 32 GW peak growth forecast for 2024-2030, about 30 GW from data centers; disclosed state/utility thresholds include 70 MW, 100 MW, and 25 MW | Customer-level request census, duplicate-screening, financial commitment, probability adjustment, upgrade contract, construction, and energization. |

### Large-load request and forecast discipline

PJM's 2025 filing shows why a “queue MW” headline is not a clean denominator: large-load
requests originate with utilities and load-serving entities under different thresholds,
validation rules, probability factors, and duplicate-handling practices. Record forecast
MW, requested MW, probability-adjusted MW, contracted MW, upgrade-ready MW, energized MW,
and metered load separately. A forecast is evidence of planning pressure, not evidence that
the datacenter, substation, generation, or transmission upgrade exists.

## Cooling and prefabricated delivery additions

| Evidence date | Supplier / footprint | Observed state | Quantified evidence | Missing delivery join |
|---|---|---|---|---|
| 2025-11-17 | Modine / Airedale, Franklin WI | 155,000-square-foot cooling factory opened within a four-site U.S. expansion | $100M multi-site program; >300 jobs targeted by March 2026; about 430 employees within three years | Qualified unit/thermal-MW output, yield, backlog allocation, shipment, site acceptance, commissioning, supported live IT MW |
| 2026-05-26 | Modine / Airedale, undisclosed strategic customer | Long-term capacity reservation with upfront customer funding | >$4B of products reserved for 2027-2029; $165M upfront cash | Customer identity, product mix, annual units/thermal MW, factory allocation, shipment, site, commissioning, live load |
| 2025-11-06 | Schneider Electric prefabricated pod | Shipping pre-designed and pre-assembled modular product | Supports high-density pods rated to 1MW+ with liquid-cooling and electrical distribution options | Shipped units, customer/site, accepted rating, assembly/commissioning date, measured thermal performance, live IT MW |

The lifecycle is intentionally asymmetric: an opened factory outranks an investment plan; a
capacity reservation outranks a non-binding forecast; a shipping module outranks a reference
design. None equals accepted equipment, commissioned cooling, or live critical IT load.

### Electrical delivery chain

```text
investment / factory announcement
  -> factory construction / line qualification
  -> annual product-specific output
  -> customer order / backlog
  -> testing / shipment
  -> site/substation installation
  -> utility acceptance / energization
  -> continuous deliverable facility power
  -> live critical IT load
```

Transformer MVA, switchgear ratings, HVDC/FACTS equipment, generation MW, facility MW,
and critical IT MW remain distinct physical units.

## Storage and network delivery additions

| Evidence date | Entity / system | Observed state | Quantified envelope | Missing production join |
|---|---|---|---|---|
| 2024-03-12 | VAST / NVIDIA / CoreWeave BlueField architecture | Vendor said testing and first deployment were underway | Claimed 70% VAST-infrastructure footprint/power reduction and >5% net energy savings; target hundreds of thousands of GPUs | Installed prevalence, storage load, DPU resource use, failures, cost, and neutral application goodput. |
| 2026-04-22 | VAST Series F/business scale | Financing and company-reported commercial scale | $30B valuation; ~$1B transaction; >$4B bookings; >$500M CARR; claimed millions-of-GPUs environments | Audited customer/workload mix, storage capacity/traffic, concentration, and platform-to-GPU goodput. |
| 2026-03-16 | WEKA NeuralMesh | General availability | Enterprise storage/memory platform for training, inference, and agents | Shipments/deployments, bytes/IOPS/bandwidth, memory hit, failures, and application goodput. |
| 2026-06-09 | Arista 7060XE7 | Product portfolio announced; availability Q4 2026–Q1 2027 | ~100 Tbps/system; 1.6T ports; air/liquid variants; claimed ~60% LPO power reduction | Shipment, multi-vendor interoperability, field reliability, topology, congestion, and collective goodput. |
| 2026-03-12 | Arista XPO MSA | Specification/MSA plus demonstrations | 12.8 Tbps/module; 204.8 Tbps/OCP RU; 4× density; up to 400 W module cooling | Qualification, yield, shipment, interoperability, field service, and deployed application performance. |

### Data-movement lifecycle

```text
architecture / standard
  -> reference implementation
  -> product announcement / GA
  -> qualification and interoperability
  -> shipment / deployment
  -> workload integration
  -> bytes + IOPS + bandwidth + latency + error evidence
  -> checkpoint / training / inference / agent goodput
```

Peak bandwidth, storage bookings, supported-GPU claims, and general availability remain
separate from end-to-end data-plane performance.
