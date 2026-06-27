// Package vcacheobserve is the vCache per-sub-concept OBSERVABILITY lens over real
// provider-cache telemetry — the "10x observability into all the sub-concepts"
// surface behind `fak vcache observe`.
//
// The base provider prefix cache obviously works (any session with cache_read>0
// proves that). What this package makes visible is everything the vCache milestones
// add ON TOP of that base — and, crucially, what is true for THIS account's real
// traffic versus the scorecard's synthetic defaults. It groups a run's turns by
// prefix family (one Claude session = one shared system prefix), then runs the
// SHIPPED decision leaves over that real data and assembles one panel per
// sub-concept:
//
//   - base cache (OBSERVED): hit rate + cached tokens served, straight from the
//     provider's own counters.
//   - M2 star anchors (vcachestar): per-family realized economics + first-positive
//     turn — the within-family reuse win.
//   - M1 concentration (vcachecal §5.2): the Zipf exponent s MEASURED from the
//     account's family distribution, and the flat-workload (s<=1) "defeated" flag.
//   - M1 warmth belief (vcachecal §7): the false-warm / false-cold rates from
//     running the shipped Belief estimator across each family's turns — the safety
//     signal (Law A1: a 0% false-warm rate means we never book a save we did not get).
//   - M3 dedicated warming (vcachewarm): whether natural-first warming already pays
//     (first-positive turn), so no dedicated warm is warranted.
//   - M4 chains & recall (vcachechain §11.0): the single-unit recall cost gate at
//     the account's real mean prefix size — refused, with the break-even sibling count.
//   - M5 governor (vcachegov §5.4): the pin/lazy/evict verdict per family from the
//     OBSERVED arrival rate λT, via the shipped Classify.
//   - score composite (vcachescore): the measured grade vs the synthetic-default
//     grade — the headline contrast (same realized economics, different concentration
//     assumption).
//   - cachemeta (tier-1): canonicalization holding, inferred from a 0% false-warm rate.
//
// Every panel is labeled OBSERVED (relayed from the provider's own counters — fak
// does not control it) or DECISION (fak's deterministic verdict over those counters),
// so the report never conflates a provider effect with a fak action. Correctness
// never depends on a cache hit (Law A2): a hit is a realized rebate, never a trust
// claim.
//
// Tier: mechanism (2) — see internal/architest. It composes the tier-2 decision
// leaves (vcachecal, vcachechain, vcachegov, vcachescore) plus cachemeta (tier 1)
// and the standard library. It is pure, deterministic, clock-free (the caller injects
// per-turn millis), and not registered into the kernel.
package vcacheobserve
