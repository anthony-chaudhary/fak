# Policy, sovereign compute, and standards ledger

**As of:** 2026-08-26
**Issue:** [#9315](https://github.com/anthony-chaudhary/fak/issues/9315)
**Authority:** [`index.json`](index.json)

## Result first

Regional infrastructure is not a uniform pool. Export controls, sovereign procurement,
capacity-allocation rules, model regulation, and fabric standards change what hardware
can be acquired, who can use it, where workloads may run, and which evidence a deployed
service must retain.

This ledger separates **proposal, rule, enforcement posture, tender, allocation,
award, delivery, and operating capacity**. It is not legal advice.

## Dated ledger

| Date / jurisdiction | Instrument or program | Current observed state | Quantified boundary | Operational implication |
|---|---|---|---|---|
| 2025-05-13 / United States | BIS AI Diffusion Rule rescission and chip guidance | BIS said the January 15 rule would be rescinded before its May 15 compliance date, told enforcement not to enforce it, and issued PRC-chip/model/diversion guidance. | Two rule dates; no capacity number. | Bind hardware eligibility to effective/enforcement state, end user, end use, destination, and exact control text. |
| 2025-01-31 / India | IndiaAI Compute Capacity | Multi-provider, subsidized sovereign compute marketplace announced and allocations published. | >18,000 heterogeneous GPU compute units. | Track provider, SKU, allocation, subsidy, project window, and start-by date; do not treat units as one cluster. |
| 2026-07-30 / European Union | EuroHPC AI Gigafactories call | Tender call, not delivered sites. | Up to 7 gigafactories; up to €10B public support; ≥€20B expected private investment; adjacent network of 19 AI Factories. | Track tender → award → financing → site → installed/accepted capacity → access allocation. |
| 2026-08-02 / European Union | AI Act application/enforcement | General application and enforcement began; GPAI obligations had applied since August 2, 2025. | Dates, not compute capacity. | Deployment receipts may need jurisdiction, provider/deployer role, model/system classification, documentation, transparency, and audit evidence. |
| 2024-05-30 / Singapore | Green Data Centre Roadmap | Policy roadmap and later allocation calls condition growth on efficiency and green energy. | ≥300 MW near-term additional capacity; potentially ≥200 MW more through green energy. | Roadmap MW is not operating AI IT load; retain award, energy-source, efficiency, construction, and commissioning states. |
| 2026-07-16 / global consortium | Ultra Ethernet Specification 1.0.3 | Current published version after 1.0, 1.0.1, and 1.0.2 corrections. | Version/date lifecycle. | Record deployed version and compliance evidence; a standard does not prove interoperability or goodput. |

## State machines

### Export-control state

```text
proposal -> published rule -> effective -> enforceable
         -> amended / stayed / rescinded / replaced
transaction = item + performance + destination + end user + end use + date + license
```

A country label alone is insufficient. A policy announcement alone is insufficient.

### Sovereign-compute state

```text
budget/announcement -> procurement/tender -> provider empanelment -> award
-> financed -> built/available -> allocated -> started -> healthy usage -> useful output
```

Public investment, procurement ceiling, GPU-unit count, and active allocation are not
interchangeable.

### Standard state

```text
draft -> published specification -> correction/revision -> implementation
-> compliance test -> multi-vendor interoperability -> production deployment
```

IETF Internet-Drafts remain drafts. Consortium specifications remain distinct from
certified deployed behavior.

## Architecture consequences

1. **Regional scheduling:** route only after checking data locality, hardware eligibility,
   provider/deployer obligations, and sovereign-access rules.
2. **Hardware receipts:** include ownership, provider, SKU, physical location, export or
   license condition, and the policy version in force at acquisition/use time.
3. **Capacity accounting:** keep public funding, tender value, allocated GPU units,
   roadmap MW, and useful goodput in separate fields.
4. **Network receipts:** name fabric implementation, specification revision, congestion
   control, and observed compliance/interoperability.
5. **Change handling:** policy and standards must be append-only events with effective
   dates; silent replacement destroys reproducibility.

## Remaining gaps

- The operative U.S. replacement export-control rule and transaction-level country,
  entity, end-use, and license matrices.
- EU member-state site awards, construction, processor mix, access allocation, and
  gigafactory delivery.
- IndiaAI provider/SKU inventory over time, utilization, subsidy economics, and
  installed-to-active conversion.
- China domestic accelerator controls/subsidies, semiconductor policy, and regional
  supply substitution.
- UAE, Saudi, Japan, Korea, Canada, and other sovereign compute procurement ledgers.
- Datacenter energy, water, permitting, carbon, and reporting rules by jurisdiction.
- Adopted international serving/KV-cache/benchmark standards; current IETF work remains
  draft-level.
- Evidence of UEC implementation, compliance, multi-vendor interoperability, and
  production prevalence.
