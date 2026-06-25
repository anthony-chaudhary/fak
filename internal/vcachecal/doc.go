// Package vcachecal is the vCache observe & calibrate engine — milestone M1 of the
// vCache epic (issue #716). The full design lives in
// docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md (§5 the economics, §7 the warmth
// belief model, §11.4 Law D calibrate-don't-assume, §13-M1). Where the M5 Governor
// (internal/vcachegov) decides the STEADY STATE of a warm set and M4 (internal/
// vcachechain) decides how to RECALL a unit, M1 is the foundation every later
// milestone's economics stands on: it OBSERVES the provider cache and CALIBRATES the
// real constants, without warming anything.
//
// This package implements the four M1 acceptance criteria as a PURE, deterministic,
// off-path decision layer — the same posture as vcachegov and vcachechain:
//
//   - Warmth-belief estimator (§7, scope 1). Each tracked prefix carries a
//     cachemeta.Lifecycle at TierProvider, reused UNCHANGED as the clock/state
//     substrate. Advance decays belief on the clock (resident→expiring→expired, since
//     we cannot see eviction); a real call that read cache revives it (Touch) and
//     resets the TTL clock; a believed-warm call that reads cache_read=0 DEMOTES to
//     cold at once and records the divergence (Rule A1). It is an open-loop estimator
//     with feedback, NOT a known-state machine.
//   - Probe harness (scope 2, Law D2). FitCalibration fits the TTL T, the minimum
//     cacheable prefix M_min, and the read discount r per (provider, model, endpoint)
//     from real replay samples, falling back to the §5 hypothesis for any constant no
//     probe observed — calibrate-don't-assume, made mechanical (measured vs assumed).
//     ProbeBudget charges probe traffic against an LRU budget so MEASURING warmth does
//     not evict what you believe warm (the observer-perturbs-state guard).
//   - Concentration measurement (scope 3, §5.2). FitConcentration fits the Zipf
//     exponent s from the ranked vBlocks and flags a flat workload (s ≤ 1) as
//     structurally defeated — the actionable gate "measure s before trusting vCache".
//   - Prediction-error report (scope 4). PredictionError folds the estimator's
//     per-call predictions against the real cache_read counters and reports the
//     false-warm and false-cold RATES, not assumed zero.
//
// OBSERVE-ONLY (issue acceptance): no warming request is issued in M1. This package
// issues no network call, writes no payload, and exposes no warm/write primitive — it
// only predicts, reconciles, and fits. The live provider loop that flips estimation
// into control is not here; like the M5 Governor, this is an off-path engine the
// future M2/M3 warming leaves wire into, so it adds zero rungs to the request path.
//
// Correctness never depends on the outcome (Law A2): whatever the estimator believes,
// the caller must always be able to re-send the full prefix; a hit is only ever a
// cost/latency win, never a license to elide resent context.
//
// Tier: mechanism (2) — see internal/architest. This package imports only cachemeta
// (tier 1) and the standard library; an upward import fails the architest gate. It is
// deliberately NOT registered into the kernel (internal/registrations).
package vcachecal
