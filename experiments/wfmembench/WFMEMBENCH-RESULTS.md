# Workflow-memory benchmark — virtual views vs. stale/poison replay (#434)

Memory as an **agent-workflow substrate**, not a chatbot recall layer: three
memory policies scored over one finished session whose tool results carry the
workflow hazards that matter for `fak` — provenance, stale world witnesses,
sealed/poisoned pages, tombstones, multi-agent handoff, and effect claims that
require evidence.

Reproduce:

```
go run ./cmd/wfmembench -out experiments/wfmembench
```

## Fixture

8 pages — 6 benign, 2 sealed, 1 tombstoned — 980 raw bytes, 577 resident bytes.

Hazards encoded:

- clean tool result (account record, order status)
- stale mutable source (order world_epoch A, superseded preference)
- poisoned/sealed result (prompt injection)
- poisoned/sealed result (secret exfil)
- tombstoned page (agent-suppressed stale preference)
- multi-agent handoff (audit-bot sub-agent)
- verified effect claim (refund settled=true, receipt_sha)
- unverified effect claim (note: no confirmation checked)

## Arms

| arm | kind | resident bytes | resident tokens | view fault | source coverage | stale reuse | poison leak | task ok | fallback→raw |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| full-transcript | modeled | 980 | 245 | 1.00 | 1.00 | 1 | 2 | yes | 1.00 |
| naive-global-summary | modeled | 384 | 96 | 0.00 | 0.00 | 1 | 2 | yes | 0.00 |
| provenance-bound-virtual-views | measured | 577 | 145 | 0.50 | 0.83 | 0 | 0 | yes | 0.50 |

- **full-transcript** (modeled) — baseline: carries the whole flat transcript; sealed and tombstoned pages leak because nothing is filtered.
- **naive-global-summary** (modeled) — baseline: one lossy blob; cheapest in bytes but loses provenance, leaks sealed content, and cannot fall back to raw.
- **provenance-bound-virtual-views** (measured) — fak: demand-paged provenance-bound views; sealed pages refused, stale views recomputed across the policy boundary — fail-closed.

## Stale replay (acceptance #4)

a goal view built under policy epoch A is served again under epoch B; the stale view is rejected and RECOMPUTEd, never served as a HIT.

- recomputes: 5
- stale reuse (must be 0): 0
- old view rejected: yes

## Poison replay (acceptance #5)

a query targeting a sealed source's descriptor is REFUSED; no sealed or tombstoned page enters any derived view.

- sealed refused: 6
- poison leakage (must be 0): 0
- sealed contained: yes

## Verdict

The full transcript is correct but carries the most bytes and leaks every sealed
and tombstoned page. The naive global summary is cheapest in bytes but destroys
provenance, leaks sealed content, and cannot fall back to a source. Only the
provenance-bound virtual views are both lean and fail-closed: zero stale reuse,
zero poison leakage, the goal still answered, and a measured raw fallback rate.

The two modeled baselines are closed-form reductions over the page table; the
virtual-views arm is measured by driving the real derived-view substrate.
