package radixkv

// lookupobs.go — the prefix-cache LIFECYCLE event counters (#5804): the demand-side
// bookkeeping a warm/cold rerun needs to explain a hit rate instead of merely reporting it.
//
// The tree already booked the SUPPLY side — evictions (LRU, cost-aware, policy) and
// structural splits. What it never booked is the DEMAND side: how many times the cache was
// asked, how often the answer was yes, and how many entries were filled. Without those, a
// hit-rate number is unfalsifiable — a 0% rate on an empty cache (nothing COULD have
// matched) and a 0% rate on a full cache (the request diverged from everything resident)
// are the same figure but opposite diagnoses. The first is a warmup artifact and self-heals;
// the second means the workload's prefixes genuinely do not overlap and no budget increase
// will help. This file makes that distinction a counted, typed fact.
//
// Mechanism — OBSERVE only, never steer. countLookup is called by LookupNS on EVERY demand
// probe, after the walk but before any lease or clock mutation, and classifies the outcome
// into exactly one of three buckets; attachLeaf books one fill per attached leaf. Nothing
// here reads the wall clock, allocates, or touches victim choice, so the hot path pays only
// an integer increment. MatchLenNS is deliberately NOT counted: it is documented as a
// read-only accounting probe, so counting it would double-book the workloads that use it to
// measure the very hit rate these counters explain.
//
// The three lookup buckets are mutually exclusive and exhaustive, which is the invariant
// worth testing (TestLookupLifecycleCountersPartition):
//
//	Lookups == LookupHits + LookupMissCold + LookupMissDivergent
//
//   - HIT       — the walk matched at least one leading token (matched > 0).
//   - COLD      — a miss where nothing could have matched: the namespace root had no
//     children, or the request itself was empty. Warmup, not a workload signal.
//   - DIVERGENT — a miss against a POPULATED namespace with a non-empty request: real
//     demand that shared no leading token with anything resident. This is
//     the actionable bucket — a rising divergent share means prefix reuse is
//     absent, not merely evicted.
//
// Namespacing (#3889): the cold/divergent split is judged per NAMESPACE, from the root
// LookupNS actually walked. A probe under ns "A" against an empty "A" is COLD even when a
// busy "B" holds thousands of nodes — otherwise a single hot namespace would mask every
// other namespace's warmup as divergence.
//
// Counters are monotonic for the tree's lifetime and are never reset by eviction: they
// count EVENTS, not residency. Reading them is Stats().

// countLookup books exactly one demand probe. matched is the leading-token count the walk
// returned; couldMatch reports whether a miss was even possible — the namespace root held
// children AND the request was non-empty — which is what separates a cold miss from a
// divergent one. Callers pass the pre-walk population state, since the walk itself can
// split a node and change the shape.
func (t *Tree) countLookup(matched int, couldMatch bool) {
	t.lookups++
	switch {
	case matched > 0:
		t.lookupHits++
	case couldMatch:
		t.lookupMissDivergent++
	default:
		t.lookupMissCold++
	}
}
