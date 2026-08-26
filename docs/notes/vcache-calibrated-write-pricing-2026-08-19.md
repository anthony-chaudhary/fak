---
title: "Calibrated 5m/1h cache-write pricing — 2026-08-19"
description: "Verdict: provider probe evidence can now replace the static cache-write price independently for the 5-minute and 1-hour tiers,"
---
# Calibrated 5m/1h cache-write pricing — 2026-08-19

**Verdict:** provider probe evidence can now replace the static cache-write price independently for the 5-minute and 1-hour tiers, and the live served-session spend meter records when that measured pricing is active.

- Probe rows accept itemized `write_cost_equiv` and `write_ttl`; the fitter derives each multiplier as billed token-equivalent write cost divided by prefix tokens, averaged only within that tier.
- The durable provider/model calibration ledger carries separate value and measured bits for 5m and 1h writes.
- Runtime selection keeps the existing freshness and exact-model gates. Missing, stale, unmeasured, invalid, or mismatched evidence leaves the static 1.25×/2.0× defaults unchanged.
- `TestMeasuredWriteMultipliersChangeServedSpendByTier` proves fresh values change cache-creation spend independently, while an unmeasured value cannot steer accounting.
- The served-spend pricing source gains `+vcache-calibrated-write`, making the live override diagnosable rather than silent.

Structured artifact: [`vcache-calibrated-write-pricing-2026-08-19.json`](../_witnesses/vcache-calibrated-write-pricing-2026-08-19.json).

This retires the calibrated 5m/1h write-multiplier acceptance item in #8090. Executable heartbeat actions, mismatch demotion, real-session prediction error, and provider-wide live probes remain in that issue.
