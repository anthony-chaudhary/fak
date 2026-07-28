---
title: "Project-work backlog census (2026-07-15)"
description: "Verdict OBSERVE: only 26 of 1,419 open issues carry valid project-work metadata, so strict enforcement stays unpromoted. Includes the reproduction witness."
---

# Live project-work backlog census — 2026-07-15

Verdict: **OBSERVE**. Strict project-work enforcement must not yet be promoted: only 26 of 1,419 open issues (1.83%) carry valid metadata, while 1,393 (98.17%) have unknown or invalid production denominators.

## Witness

- Observed at 2026-07-15T05:34:00-07:00 against all 1,419 open GitHub issues.
- Source command: `gh issue list --state open --limit 2000 --json number,title,body,labels,state,milestone,updatedAt`.
- Source snapshot SHA-256: `561daa4d516e9b61eda3c50e833f76ba8dd24c7975930528ce0442cd81eba7bf`.
- Review command: `fak issue contract --from-issues <snapshot> --strict-project-work --json`.
- Review exited 3, as expected for a corpus containing strict-project-work refusals.
- Review SHA-256: `d2d7de0eee9dd9501fdcd4cdf443de4e983e29a335bd69c1f020b32d1eac2c4f`.
- Machine-readable scrubbed census and decision queue (private GPU host aliases are normalized to `GPU-server`; issue numbers and all measured fields are unchanged): [`project-work-backlog-census-2026-07-15.json`](../project-work-backlog-census-2026-07-15.json), SHA-256 `e746757433ca2cd8639de068a6c67ff0110575592b2d067d265a58e702219041`.

## Coverage

| Issue class | Open | Valid | Invalid | Undeclared |
|---|---:|---:|---:|---:|
| bug | 40 | 0 | 0 | 40 |
| documentation | 16 | 0 | 0 | 16 |
| enhancement | 488 | 2 | 0 | 486 |
| epic | 190 | 0 | 0 | 190 |
| other | 685 | 24 | 5 | 656 |
| **Total** | **1,419** | **26** | **5** | **1,388** |

All 17 open milestones currently have zero valid records; the 26 valid records are unmilestoned. The largest unknown groups are unmilestoned (1,016 undeclared), KV-cache milestone (50), G0 (49), G1 (45), and agentic coding (42). The JSON artifact contains one row per open issue, including milestone, class, parser status, invalid reasons, safe repair proposal, and required human decision.

## Repair and decision policy

No live issue was edited. This is deliberate: the contract permits only lossless derivation, and estimate, parent denominator, contribution, and maturity cannot be inferred honestly from a title, label, open/closed state, or generic parser default.

The five partially declared invalid records are #4800, #4801, #4803, #4806, and #4845. Four need an owner-confirmed completion standard; #4845 needs an explicit numbered parent binding. Their parser proposals are retained in the JSON queue but were not auto-applied.

The remaining 1,388 rows are an explicit visible decision queue with action owner_supply_estimate_parent_contribution_and_completion_standard. This records unknown effort instead of fabricating a denominator or silently hiding the work.

## Enforcement rollout decision

- **Now — observe:** keep strict review in census/dry-run mode; promotion now would strand 98.17% of open work.
- **Warn threshold:** a fresh full census is at least 95% valid, has zero invalid rows, and preserves every remaining undeclared row in the decision queue.
- **Deny threshold:** two consecutive fresh censuses are at least 99% valid, have zero invalid rows, and show no more than a two-percentage-point dispatchable-share regression.
- **Rollback:** return to observe on any silent metadata guess, non-metadata body mutation, or dispatchable-share regression above two percentage points.

This rollout makes denominator risk measurable and reversible. It does not represent legacy work as production-ready and does not claim the backlog migration is complete merely because a parser can suggest text.
