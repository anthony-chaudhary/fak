package gateway

// drain_time_load.go — the expected-drain-time term for the fleet residency
// scorer (issue #5275, epic #2236, borrow: lmdeploy request-distribution proxy,
// INSPIRE / clean-room Go over an Apache-2.0 source).
//
// CacheAwarePolicy.score (residency_router.go) discounts a worker by its raw
// live load through an unweighted 1/(1+load) term, so two workers at equal
// in-flight work are treated as equal even when one drains twice as fast. On a
// heterogeneous fleet (a mix of fast and slow boxes whose per-worker service
// rate differs several-fold) that mis-routes: a fast worker with more in-flight
// work can still be the better pick, because it will empty its queue sooner.
//
// This file supplies ONLY the pure term: given a worker's live in-flight work
// (queued or active token-or-request units) and its measured per-worker
// throughput (units per second, an INJECTED measured rate — never read from a
// live timer), it computes the expected drain time = in-flight / throughput,
// and a comparison that prefers the worker whose queue drains SOONEST. It
// composes additively with the other scorer terms: this term ranks a worker by
// TIME-to-serve, orthogonal to the prefix-overlap axis, and the two multiply.
// Deterministic and wall-clock free — no network, no GPU, no time source.

import "math"

// DrainLoad carries one worker's live-load signals the drain-time term reads.
//
//   - InFlight: the worker's live in-flight work — queued plus active
//     token-or-request units. Non-positive means idle (nothing to drain).
//   - Throughput: the worker's measured per-worker service rate, in the same
//     units per second. This is an injected measured value, not a reading off a
//     live clock, so the term stays deterministic. A non-positive (or non-finite)
//     rate fails closed: the worker is treated as infinitely slow and is never
//     preferred, rather than dividing by zero.
type DrainLoad struct {
	InFlight   float64
	Throughput float64
}

// ExpectedDrainTime returns how long the worker is expected to take to empty its
// live in-flight work: in-flight / throughput. Properties the tests pin:
//
//   - Zero base: a worker with no in-flight work drains in zero time (an idle
//     worker is maximally preferred), regardless of its throughput.
//   - Monotone in load: at equal throughput, more in-flight work yields a larger
//     drain time, so a busier worker is less preferred.
//   - Monotone in speed: at equal in-flight work, a higher throughput yields a
//     smaller drain time, so a faster worker is more preferred — and a fast
//     worker with more in-flight can still drain sooner than a slow idler-than-it
//     peer, which is the whole point of the term.
//   - Fail closed: a non-positive or non-finite throughput (or a non-finite
//     in-flight) makes the worker infinitely slow (+Inf drain), never preferred,
//     instead of a divide-by-zero.
func ExpectedDrainTime(in DrainLoad) float64 {
	// Idle: nothing in flight drains in zero time, whatever the rate.
	if in.InFlight <= 0 {
		return 0
	}
	if math.IsNaN(in.InFlight) || math.IsInf(in.InFlight, 0) {
		return math.Inf(1)
	}
	// A dead or unknown rate fails closed to infinitely slow.
	if in.Throughput <= 0 || math.IsNaN(in.Throughput) || math.IsInf(in.Throughput, 0) {
		return math.Inf(1)
	}
	d := in.InFlight / in.Throughput
	if math.IsNaN(d) || math.IsInf(d, 0) {
		return math.Inf(1)
	}
	return d
}

// PreferByExpectedDrain reports whether worker a is preferred over worker b under
// the drain-time axis: a is preferred exactly when its expected drain time is
// STRICTLY smaller (it will empty its queue sooner). A tie (equal drain time,
// including two idle workers) is not a strict preference, so this stays a clean
// less-than and a caller can break ties on a lower-order axis. Two infinitely
// slow workers tie (neither preferred), so a stalled fleet never flaps.
func PreferByExpectedDrain(a, b DrainLoad) bool {
	return ExpectedDrainTime(a) < ExpectedDrainTime(b)
}

// DrainPreferenceScore folds the expected drain time into the same inverse-shape
// the residency scorer already uses, so it drops into a term product cleanly:
// 1/(1 + drain), which is 1 for an idle worker (zero drain, maximally preferred),
// shrinks toward 0 as the drain time grows, and is exactly 0 for an infinitely
// slow (fail-closed) worker. A higher score means a sooner drain, so this stands
// in for the raw 1/(1+load) term rather than adding on top of it — folding it in
// never double-counts the worker's load.
func DrainPreferenceScore(in DrainLoad) float64 {
	d := ExpectedDrainTime(in)
	if math.IsInf(d, 1) {
		return 0
	}
	return 1 / (1 + d)
}
