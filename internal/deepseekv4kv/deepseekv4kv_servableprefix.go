package deepseekv4kv

import (
	"fmt"
	"strings"
)

// Invariant: DeepSeek-V4 KV servable prefix calculation is fail-closed and window-bounded.
// Any non-positive sequence length, empty kinds slice, or unrecognized cache kind must
// fail closed to 0 hittable prefix tokens, while sliding-window attention (SWA) bounds reach.
//
// Guard: The tightest sub-cache group binds cross-kind prefix reuse; SWA never serves tokens
// outside its trailing window (min(seq, SWAWindow)).
//
// ServablePrefixUnits answers a single cross-kind question the per-kind storage
// accounting in this fixture does NOT: given a request of seq leading tokens over a
// set of typed sub-caches (kinds), how many leading tokens are actually hittable in
// EVERY group at once. A model-wide prefix hit of length L is only valid if each group
// can serve length L under its own token-dependency rule, so the answer is the MIN of
// the per-kind hittable prefix lengths — the tightest group binds.
//
//   - dense (tail), CSA and HCA are full-reach: every leading token is hittable, so
//     their per-kind bound is seq.
//   - SWA only ever attended its last SWAWindow tokens, so its hittable prefix is
//     bounded to min(seq, SWAWindow) — a long request cannot claim a hit on early
//     tokens the sliding window never reached.
//
// This is the intersection (fold) the study note borrows: a naive prefix length
// over-counts the hit whenever a windowed group can't serve the early tokens.
//
// It fails closed: an empty kinds slice (no group can agree), a non-positive seq, or an
// unknown kind all yield 0 (nothing servable) rather than an over-counted bound.
func ServablePrefixUnits(seq int, kinds []Kind) int {
	if seq <= 0 || len(kinds) == 0 {
		return 0
	}
	bound := seq
	for _, k := range kinds {
		reach := hittablePrefixLen(k, seq)
		if reach < bound {
			bound = reach
		}
	}
	if bound < 0 {
		bound = 0
	}
	return bound
}

// hittablePrefixLen is the per-kind servable prefix length for a request of seq leading
// tokens: how many of the leading tokens this one group can serve on a hit. Full-reach
// groups serve all seq; the sliding window serves only its last SWAWindow tokens. An
// unknown kind fails closed to 0 so a bad group can never inflate the intersection.
func hittablePrefixLen(k Kind, seq int) int {
	if seq <= 0 {
		return 0
	}
	switch k {
	case KindCSA, KindHCA, KindTail:
		return seq
	case KindSWA:
		return minInt(seq, SWAWindow)
	default:
		return 0
	}
}

// FormatServablePrefix renders the per-kind hittable prefix lengths and the binding
// cross-kind bound for one request, as a short line for a runbook or CI log.
func FormatServablePrefix(seq int, kinds []Kind) string {
	var b strings.Builder
	fmt.Fprintf(&b, "servable prefix over %d tokens: ", seq)
	for i, k := range kinds {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s=%d", k, hittablePrefixLen(k, seq))
	}
	fmt.Fprintf(&b, " -> bound=%d", ServablePrefixUnits(seq, kinds))
	return b.String()
}
