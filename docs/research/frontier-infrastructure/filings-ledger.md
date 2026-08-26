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
