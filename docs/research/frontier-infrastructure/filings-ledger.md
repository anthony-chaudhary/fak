---
title: "Hyperscaler filings and obligations ledger"
description: "As of: 2026-08-26. Tracker: #9302. This ledger records the latest disclosed."
---

# Hyperscaler filings and obligations ledger

**As of:** 2026-08-26. **Tracker:** #9302. This ledger records the latest disclosed
capital, asset-mix, finance, backlog, and capacity facts without pretending that the four
companies use identical accounting boundaries.

## Latest disclosed 2026 position

| Company | Period/source | Capital and asset mix | Demand/obligation signal | Boundary and implication |
|---|---|---|---|---|
| Alphabet | Q2 2026 earnings call, 2026-07-22 | Q2 capex **$44.9B**; technical infrastructure roughly **60% servers / 40% datacenters and networking**; 2026 guidance raised to **$195–205B** | Google Cloud backlog **$514B**, up >$50B sequentially; just over half expected as revenue within 24 months; third-party capacity used as a bridge while internal capacity is built | Alphabet capex includes non-AI and non-cloud assets, though it says the vast majority supports AI technical infrastructure. Server/data-center mix is spend, not installed useful capacity. |
| Microsoft | FY2026 Q4 earnings call, 2026-07-29 | Quarterly capex **$41B**; roughly **two-thirds short-lived CPUs/GPUs**, remainder long-lived; **$5.6B finance leases** for large datacenter sites; cash PP&E **$35.8B** | Commercial RPO **$678B**, weighted duration 2.3 years; Azure demand still exceeded available capacity; incoming supply is shared among Azure, first-party apps, R&D, and replacement | Microsoft total capex includes AI and non-AI infrastructure. Finance-lease commencement creates quarter volatility; the full lease value and cash paid differ. |
| Amazon | Q2 2026 release plus reported earnings-call guidance, 2026-07-30 | 2026 capex plan reported at about **$220B**, up from Amazon’s official Q4 2025 expectation of about **$200B**; trailing capex/free-cash-flow pressure is material | AWS Q2 revenue **$42.2B**, up **37%**, $169B annualized; management says substantial future AWS capex is already supported by customer commitments | Amazon does not isolate AWS/AI cash capex in the release. The $220B update is earnings-call reporting and should remain distinguished from the earlier official $200B release until a primary transcript/filing is indexed. |
| Meta | Q2 2026 release, 2026-07-29 | Q2 capex including finance-lease principal **$31.08B**; 2026 guidance **$130–145B** | Q2 operating cash flow **$31.86B** and free cash flow **$0.784B** after the investment surge | Meta’s capex definition explicitly includes principal payments on finance leases and supports AI plus core business; it is not directly comparable to cash PP&E at Microsoft or Alphabet’s technical-infrastructure mix. |

The midpoint of these disclosed/planned 2026 totals is roughly **$747.5B**
(Alphabet $200B, Microsoft $190B calendar-year expectation, Amazon $220B, Meta $137.5B),
but this is an orientation number, not a clean AI-capex sum. It mixes guidance and reported
call updates, fiscal/calendar conventions, leases and cash purchases, AI and non-AI assets,
and different consolidation boundaries.

## Accounting dimensions to normalize

| Dimension | Why it changes the number |
|---|---|
| Cash PP&E vs capex | Goods can be received before payment; cash flow and property additions diverge. |
| Finance leases | Full asset value can enter capex at commencement while cash principal is paid later. |
| Operating leases / take-or-pay | Large capacity obligations can sit outside a simple capex number. |
| Short-lived vs long-lived assets | Accelerators/CPUs cycle much faster than land, buildings, power, and network. |
| Customer-supplied/prepaid hardware | A provider can operate capacity whose hardware economics sit with a partner/customer. |
| Inventory/system sales | Selling TPU or other systems turns part of infrastructure build into inventory and customer-site capacity. |
| AI vs non-AI | Cloud, search, ads, retail, satellites, core apps, replacement cycles, and offices can share the total. |
| Announced vs recognized | Planned annual capex, quarterly additions, cash paid, depreciation, and accepted capacity are separate states. |
| Backlog/RPO | Contracted revenue indicates demand but differs in duration, cancellation, service mix, and compute intensity. |
| Customer concentration | A large frontier-lab contract can dominate bookings without representing broad user demand. |

## What the latest disclosures imply

1. **Supply remains binding despite record spend.** Alphabet and Microsoft explicitly describe
   capacity constraints or bridging with third-party capacity; Amazon’s reported increase also
   accompanies demand beyond near-term supply.
2. **Hardware and facilities are both large.** Alphabet’s 60/40 technical mix and Microsoft’s
   two-thirds/one-third short/long-lived split reject a “chips are the whole cost” model.
3. **Leases make headline comparisons dangerous.** Microsoft identifies quarterly finance-lease
   volatility; Meta includes finance-lease principal in its capex guidance.
4. **Future demand is contracted over years.** Backlog/RPO and customer commitments help fund
   buildout but create counterparty, duration, pricing, and delivery risk.
5. **Useful capacity trails capital commitment.** Capex must pass through component delivery,
   construction, power, commissioning, software, health, scheduling, and workload acceptance.
6. **The denominator is moving.** Component and memory-price changes can raise spend without a
   proportional increase in delivered compute.

## Required quarterly refresh fields

For each hyperscaler/cloud, capture:

- publication date, fiscal quarter, calendar-equivalent period, and source type;
- reported capex, cash PP&E, finance leases, operating-lease commitments, purchase obligations,
  prepayments, customer-supplied assets, and depreciation;
- short-lived/long-lived and server/network/building mix where disclosed;
- annual guidance and change from the prior guide, with explicit reasons;
- backlog/RPO, duration, customer concentration, cloud growth, AI product usage, and capacity
  constraint language;
- free cash flow, debt/funding, asset useful-life changes, impairment, and cancellation risk;
- whether numbers cover AI only, technical infrastructure, cloud, or the entire company;
- capacity lifecycle evidence: contracted, delivered, commissioned, healthy, and monetized.

## Current gaps

- Official Amazon Q2 2026 call transcript/10-Q extraction for the $220B update, cash capex,
  backlog, useful-life assumptions, and customer commitments.
- Oracle, CoreWeave, Nebius, DigitalOcean, IBM, Alibaba, Tencent, Baidu, and sovereign-cloud
  filings normalized to the same fields.
- Hyperscaler purchase obligations, lease commitments, partner prepayments, and customer-supplied
  hardware reconciled to annual capex.
- AI/non-AI allocation, installed capacity, health, utilization, and goodput conversion.

## Additional issuer and contract-financing rows

| Issuer / period | Reported evidence | Accounting and delivery boundary | Capacity inference allowed |
|---|---|---|---|
| CoreWeave Q1 2026 | $2.078B quarterly revenue; $99.4B revenue backlog; >1 GW active power; >3.5 GW contracted power; $8.5B non-recourse delayed-draw financing. CoreWeave defines backlog as RPO plus other estimated future revenue under committed contracts, subject to delivery and service availability. | Backlog is broader than GAAP RPO; recognition still depends on delivering available service. | None from backlog alone. Join contracts to financed, active-power, installed, accepted, and healthy capacity. |
| Oracle FY2026 | $638B RPO; $75B of prepaid or customer-supplied hardware portions in large AI contracts; $18.1B FY cloud-infrastructure revenue; negative $23.7B FCF; $43B debt and $5B equity financing in FY2026. | Customer prepayment and customer-owned GPUs reduce Oracle-funded capex but do not erase delivery obligations. RPO is future contract value, not hardware. | Treat customer-owned hardware, prepaid purchases, Oracle-owned PP&E, and useful serving capacity as separate states. |
| Alibaba quarter ended June 2026 | Nearly $10B quarterly capex, +75% YoY; $1.8B AI-related product revenue; $69.9B cash/liquid investments. | Aggregate AI-related capex spans stack layers and does not identify accelerator, datacenter, network, or other asset shares. | Market-spend signal only until physical units and asset split are disclosed. |
| Nebius / Microsoft contract, September 2025 | Multi-year dedicated capacity from Vineland; associated capex expected to be funded by contract cash flow and contract-secured debt. | Contract value, MW, accelerator count, financing amount, and recognition schedule are undisclosed; dedicated capacity creates concentration risk. | Track contract-backed financing and customer concentration, not capacity from the announcement. |
| Baidu Q2 2026 | RMB7.3B AI Cloud Infra revenue, +50% YoY; GPU Cloud revenue +283% YoY; RMB12.5B core AI-powered business revenue. | AI-powered business fields are unaudited internal management data; GPU Cloud absolute revenue and capex/accelerator mix are absent. | Demand signal only; no installed-capacity or goodput conversion. |

## Cross-issuer financing patterns

- **Owner-funded PP&E:** classic capex/cash-flow path; asset mix still needs disclosure.
- **Finance leases and infrastructure obligations:** capacity may be controlled without
  appearing in cash capex in the same period.
- **Customer prepayment:** the customer advances funds for hardware; delivery and service
  obligations remain with the provider.
- **Customer-supplied hardware:** GPUs may be operated by the cloud but not purchased or
  owned by it.
- **Contract-secured project debt:** a committed customer improves financing terms but
  can concentrate demand and strand dedicated capacity if terms change.
- **Backlog/RPO:** contractual demand, not revenue, cash, installed hardware, availability,
  or quality-constrained goodput.
