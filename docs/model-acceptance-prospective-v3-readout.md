---
title: "Prospective exact-model v3: capability PASS"
description: "The preregistered post-reset campaign completed all 18 fixed attempts and witnessed the exact production IDs at their declared tiers."
---

# Prospective exact-model v3: capability PASS

Issue: [#4633](https://github.com/anthony-chaudhary/fak/issues/4633)  
Declaration issue: [#4845](https://github.com/anthony-chaudhary/fak/issues/4845)  
Production-readiness parent: [#4632](https://github.com/anthony-chaudhary/fak/issues/4632)

## Verdict

**PASS.** The declaration was committed before provider observation in
`aee4633baa089b564cafe6bf38da038c9418c3e3` and its corrected timestamp was
published in `695626994a`. On 2026-08-05 the fixed campaign completed exactly
18 authenticated attempts with no replacements. Every attempt used the declared
exact model ID, satisfied the complete-line sentinel contract, made two valid tool
calls, and stayed within the predeclared correctness, provider-error, invalid-tool,
latency, token, and cost thresholds.

This is a bounded capability witness for the declared task classes and tiers. It is
not evidence for unrelated protocol/cache, pricing, canary, or rollback gates.

## Preregistered envelope

- Corpus: `top3-prospective-sentinel-v3`.
- Exact IDs and allowed tiers: `claude-opus-4-8` / T0,
  `claude-sonnet-4-6` / T1, and `claude-haiku-4-5-20251001` / T2.
- Task classes: read-only multi-record synthesis and typed transient retry recovery.
- Three repetitions per class per model: 18 fixed attempts total.
- Thresholds declared before observation: success 100%; provider errors 0%; invalid
  tools 0%; p95 latency <=30,000 ms; average input tokens <=30,000; average cost
  <=$1.00.
- Stop/replacement contract: stop after 18 attempts; replace none.

## Observed result

| Exact model | Tier | Samples | Success | Provider errors | Invalid tools | p95 latency | Avg input tokens | Avg cost | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `claude-opus-4-8` | T0 | 6 | 100% | 0% | 0% | 9,725 ms | 5.0 | $0.024027 | PASS |
| `claude-sonnet-4-6` | T1 | 6 | 100% | 0% | 0% | 13,943 ms | 4.5 | $0.021247 | PASS |
| `claude-haiku-4-5-20251001` | T2 | 6 | 100% | 0% | 0% | 6,260 ms | 22.0 | $0.013846 | PASS |

The completed machine-readable artifact is
[`examples/model-acceptance-prospective-v3-report.json`](../examples/model-acceptance-prospective-v3-report.json).
It retains every successful or failed scheduled attempt; this run had no failures to
classify. Runtime enforcement remains the fail-closed dispatch seam shipped by #4799:
above-tier, absent, stale, malformed, alias, or HOLD evidence is refused before the
provider continuation unless an explicit audited operator override is supplied.

## Immutable evidence

Raw streams remain in operator scratch because they contain provider/session
metadata. The public artifact contains the scrubbed fold and these bindings make the
private streams independently detectable if changed:

- Declaration SHA-256: `a4db44ffe6064c11f7578190868cef9217bc1fd9a5f9b2c258f54df57cade7fa`.
- Canonical raw-manifest SHA-256 (18 JSONL entries; sorted
  `name<TAB>bytes<TAB>sha256`):
  `f473e8b41e00244c21839f5d60ca553e4f204cedcad561d1ab0ef53efa18c5e7`.
- Completed report SHA-256:
  `6e71177d75bba80680fef17127827d693581a5f014fdcd6bef47fb3e54bbfb35`.
- Acceptance decision SHA-256:
  `9ab66155dee7d7d1ca1e66f021e812dfc961a3ff2308c9fdd2050959b1e72843`.
- Runner exit: `0` (`PASS`).
- Observation window: 2026-08-05 15:26:54 through 15:29:37
  America/Los_Angeles.

The v2 weekly-limit HOLD remains immutable infrastructure history and is not counted
as a capability sample in this campaign.
