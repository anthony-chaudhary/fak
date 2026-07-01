package radixkv

import "github.com/anthony-chaudhary/fak/internal/model"

// prewarm.go — the SELF-HOSTED serving-path warm for tool-latency KV-prefix PREWARM, the
// "WHERE" half of issue #810 (parent epic #809). The companion "WHEN" already ships in
// internal/compute: DecidePrewarmAdmission folds the tool-call boundary into one of
// WarmNow / WarmDefer / WarmSkip by pure integer policy. On a WarmNow verdict the scheduler
// knows the next `/v1/messages` request's prefix up to the tool-result slot is
// BYTE-DETERMINED now (the current context plus the result about to slot in), and the tool's
// I/O latency window is idle. WarmInsert lands that byte-known prefix into the
// RadixAttention tree DURING that window so the real turn matches it HOT instead of
// cold-starting prefill.
//
// The defensible refinement over the closest published prior — KVFlow (arXiv 2507.07400),
// which background-prefetches a PROBABILISTICALLY predicted next prefix via an Agent Step
// Graph — is DETERMINISM: the tool-call boundary gives the EXACT continuation, so this is a
// pure prefetch, never a speculation. radixkv does not re-implement that policy. WarmInsert
// is the MECHANISM a WarmNow caller drives; the byte-known trigger and the timeliness /
// pollution gates stay in tier-1 compute — the same split #807 (TurnIntent) and #808
// (discard_admit) take: the decision in compute, the mechanism in the layer that owns the
// cache.
//
// FENCE 2 — the LOWEST-PRIORITY EVICTION CLASS — is realized HERE, and it is the piece a
// plain Insert structurally could not give. Insert stamps a new leaf with the FRESHEST
// recency, so a warmed prefix would outlive demand residency — exactly backwards. A warm is
// an opportunistic bet that must be reclaimable the instant any demand session needs the
// room, and must NEVER displace a real prefix to place itself. WarmInsert therefore stamps
// the warm leaf with the LOWEST possible recency (warmRecency) and takes NO lease, so:
//
//   - under a token budget it is always the FIRST LRU victim: every demand leaf carries a
//     recency >= 1 (Lookup/Insert advance the logical clock before they stamp), warmRecency
//     is 0, so evictToBudget reclaims a warm before any demand prefix;
//   - a warm placed into an already-full tree is reclaimed by its OWN evictToBudget pass
//     before it can cost a single demand token — a fail-safe that holds even if the
//     decision-layer pollution gate (compute's WarmPoolFree) were bypassed;
//   - when a real demand request later REUSES the warmed prefix, Lookup freshens the matched
//     path's recency, PROMOTING the prefix out of the opportunistic class into normal demand
//     residency. That promotion is the closed loop: a warm that paid off stops being a
//     reclaim-first bet and becomes a kept prefix.
//
// Correctness never depends on a warm landing or surviving: a skipped, evicted, or
// never-reused warm only ever costs a cold prefill (the status quo), never a wrong answer —
// the prefix it caches is byte-identical to what a demand prefill would have built (kv ==
// the caller's full-prefix cache, or nil in the model-free accounting mode this package
// measures hit-rate in). Because the trigger is byte-known the warm cannot be wrong, so
// unlike the riskier siblings #809(b)/(c) it needs no effect-witnessed invalidation.
//
// HONEST SCOPE — what this is NOT. This is the in-tree warm plus its lowest-priority
// eviction semantics, proven host-free in accounting mode: the RadixAttention hit-rate seam
// (MatchLen) is model- and hardware-independent, the metric SGLang's own paper headlines.
// The LIVE serve-loop call-site that fires WarmInsert off a compute.WarmNow during a real
// tool dispatch, and the wall-clock RadixAttention-hit-rate / `cache_read_input_tokens`
// readback on a running model, remain host-gated and are deliberately not simulated here.

// warmRecency is the recency stamp a freshly warmed (un-promoted) leaf carries: the lowest
// possible value, so the LRU eviction pass treats an opportunistic warm as the first victim.
// Every demand-path stamp (tick / Lookup) is >= 1 because the logical clock is advanced
// before the stamp, so a warm leaf is strictly older than any demand leaf until a demand
// reuse freshens (promotes) it. The root also carries 0, but it is never an eviction or
// accounting candidate, so 0 unambiguously marks "un-promoted opportunistic warm".
const warmRecency uint64 = 0

// WarmInsert lands the byte-known continuation prefix `tokens` into the tree at the lowest
// eviction priority (fence 2), returning the number of suffix tokens NEWLY warmed — 0 if the
// prefix was already cached (already hot, by demand or an earlier warm) or `tokens` is empty.
// It is the self-hosted serving-path warm a caller drives on a compute.WarmNow verdict during
// a tool's I/O latency window. Unlike Insert it takes NO lease and stamps the warm leaf with
// warmRecency, so the warm stays immediately reclaimable and never pins or displaces demand
// residency. kv is the kernel-owned full-prefix cache for the warm (nil in accounting mode).
// The warm reuses the proven longest-prefix walk and edge split, so any prefix it shares with
// already-cached spans is matched, never duplicated.
func (t *Tree) WarmInsert(tokens []int, kv *model.KVCache) int {
	if len(tokens) == 0 {
		return 0
	}
	// walk then split-if-mid-edge to a real node boundary, exactly as Lookup does, then hang
	// the warm suffix off it (shared boundaryFor preamble).
	boundary, matched := t.boundaryFor(tokens)
	suffix := tokens[matched:]
	if len(suffix) == 0 {
		return 0 // the whole byte-known prefix is already cached — already hot, nothing to warm
	}
	s := append([]int(nil), suffix...)
	leaf := &node{
		key:      s,
		parent:   boundary,
		children: map[int]*node{},
		kv:       kv,
		plen:     boundary.plen + len(s),
		lastUsed: warmRecency, // lowest priority: first LRU victim, never displaces demand
	}
	boundary.children[s[0]] = leaf
	t.tokens += len(s)
	// Reclaim to budget. Because the warm leaf carries warmRecency it is the LRU victim of
	// its own pass when the tree is already full, so a warm into a saturated pool is dropped
	// before it can cost a demand token.
	t.evictToBudget()
	return len(s)
}

// WarmTokens returns the total cached tokens currently held as OPPORTUNISTIC warm — nodes
// stamped with warmRecency that no demand request has reused yet. It drops the instant a
// demand Lookup promotes a warm (freshening its recency) or eviction reclaims it, so it
// measures live un-promoted prefetch residency: the self-hosted analogue of vcachewarm's
// dedicated-warm accounting, and the seam an observability surface reads to see how much of
// the cache is opportunistic bet vs. demand residency. Read-only — no mutation, no lock.
func (t *Tree) WarmTokens() int {
	total := 0
	var visit func(n *node)
	visit = func(n *node) {
		if n != t.root && n.lastUsed == warmRecency {
			total += len(n.key)
		}
		for _, c := range n.children {
			visit(c)
		}
	}
	visit(t.root)
	return total
}
