package ctxplan

import "sort"

// index.go — the candidate INDEX: the planner's access path, the piece that bounds the
// per-turn planning COMPUTE the way the budget bounds the per-turn resident TOKENS.
//
// # Why this exists (the one cost "O(1) resident" did not bound)
//
// PlanCells/Candidates score EVERY span in the store each turn. The store grows one span a
// turn, so at turn i the planner scores i candidates, and Σ i = N·(N+1)/2 — a Θ(N²)
// cumulative re-planning cost (scaling.go's PlannerComputeCum). "O(1) resident" scopes the
// RESIDENT SET; it says nothing about the planner's own work, which a full re-scan leaves
// quadratic. scaling.go names the fix exactly: "unless the candidate set is index-bounded,
// which would flatten this term." This is that index.
//
// The relational reading the design leans on (the package doc's Postgres-planner table)
// completes here. A query planner does not seq-scan the whole heap to answer a selective
// query — it consults an INDEX (a B-tree, a GIN inverted index) that returns only the rows
// an access path can reach, and the cost model picks the index scan over the seq scan when
// it is cheaper. ctxplan's "table" is the history store; its "query" is the Forecast; this
// is the index the planner probes so it scores O(c) candidates per turn (c bounded by the
// query's selectivity + a recency window + the durable set), not O(N). The cumulative
// planning cost flattens from Θ(N²) to Θ(c·N) — LINEAR in the turn horizon, the same shape
// the budget gives the resident-token curve.
//
// # The three access paths (a union, deduped, capped)
//
// A Probe unions three bounded candidate sources, each the analogue of one index scan, and
// tiers them so a hard cap drops the least-promising first:
//
//	tier 0  pins            the spans a turn cannot proceed without (Forecast.Pins) — always probed.
//	tier 1  relevance       the INVERTED INDEX: forecast intent tokens -> posting lists of the spans
//	                        whose role+descriptor contain them. This is the GIN/text-index analogue and
//	                        the selective one — for a fixed query it returns only spans with matching
//	                        CONTENT, whose count grows with relevant history, not with the turn count.
//	tier 2  recency         the most-recent RecencyWindow spans (a tail scan) — the spans the recency
//	                        prior favors and that a turn most often references next, regardless of match.
//	tier 3  durability      the bounded/durable spans (a stated preference, an identity, the system
//	                        prompt) — kept candidate across the whole session by the durability prior.
//
// The union is deterministic and bounded by RecencyWindow + |durable set| + |relevance
// matches| + |pins|, then capped at MaxCandidates. Anything past the cap (or in none of the
// four paths) is PRUNED from this turn's candidate set — not destroyed.
//
// # The honesty fence (pruning is a forecast miss, never a lost fact)
//
// Index pruning has the SAME posture as the forecast itself: a pruned span the turn turns
// out to need is a demand-page FAULT (ctxplan.DemandPage), not a lost fact, because the
// store stays lossless. The index never deletes a span, never launders a sealed one (a
// sealed/tombstoned span that IS probed still scores 0 in Benefit and is elided
// sealed/tombstoned; one that is pruned simply stays in the cold store, where the trust
// gate still guards it on any page-in). So the index changes the planner's COST, not its
// correctness or its faithfulness: Audit's partition is over the probed candidate set, and
// every span outside it is one demand-page away. When the relevant + recent + durable spans
// suffice to fill the budget — the common case — the index-bounded plan's Selected set is
// IDENTICAL to the full-scan plan's (index_test.go's behavior-preservation witness); when
// they do not, the difference is a bounded efficiency miss with the exact posture of a
// forecast miss, never a correctness one.
//
// The Index holds SAFE span metadata only (the same Span the planner already reasons over —
// never sealed bytes), so it is a pure foundation-tier structure like the rest of the leaf;
// a caller materializes the probed selection through the Store's trust gate exactly as
// before.

// Index is the planner's candidate access path over a history store: an inverted token
// index (the selective relevance scan), the append order (the recency tail), and the
// durable set, maintained incrementally so a Probe returns a BOUNDED candidate set without
// re-scanning all N spans. It is metadata-only (no bytes), append-only (Add), and
// deterministic (a Probe over a fixed (index, forecast, options) yields the same span set).
type Index struct {
	spans   []Span           // append order — the "heap"; the recency tail is its suffix
	byID    map[string]int   // span id -> position in spans (pin resolution, dedup)
	posting map[string][]int // content token -> its posting list of span indices (the inverted index)
	durable []int            // span indices whose durability rank is >= bounded (kept candidate session-wide)
}

// NewIndex returns an empty candidate index.
func NewIndex() *Index {
	return &Index{byID: map[string]int{}, posting: map[string][]int{}}
}

// BuildIndex bulk-builds an index over an existing span set (one Add per span, in order).
// It is the bridge from a full-scan caller (PlanCells over store.Spans()) to the bounded
// planner: build once, then Probe each turn. Incremental callers use NewIndex + Add.
func BuildIndex(spans []Span) *Index {
	ix := NewIndex()
	for _, s := range spans {
		ix.Add(s)
	}
	return ix
}

// Len reports the number of indexed spans (the "table" size, N).
func (ix *Index) Len() int { return len(ix.spans) }

// Add indexes one span in O(tokens(span)) time: it appends to the heap, records the id,
// appends the span's index to the posting list of every distinct content token in its
// role+descriptor (the same extractive tokenization the relevance ranker uses), and adds it
// to the durable set if its durability rank is >= bounded. A span belongs to MANY token
// posting lists (one per distinct content token), so each token keeps its OWN list of span
// indices — appended in chronological (Add) order, which is deterministic across builds; a
// Probe re-sorts the union into tier/recency order, so the per-list order is not relied on.
//
// # The unique-id addressing contract
//
// Add ADDRESSES the span by its id: ix.byID maps the id to this position, and Probe's pin
// resolution + SetSealed/SetTombstoned (maintain.go) all key on it. An id is therefore a
// span's stable, UNIQUE address — every shipped store assigns one (MemStore's "span:<n>",
// recall's "page:<step>"). Adding a DUPLICATE id OVERWRITES the address (byID is last-wins),
// so the index assumes ids are unique; a store that reuses an id is outside the addressing
// contract (a colliding id makes a span unaddressable for mutation and ambiguous for
// recovery, which StoreAudit detects and refuses to certify, storeaudit.go).
//
// The span is stored by value, but its one reference-type field (Attrs) is CLONED so the
// index owns it: a caller that mutates its own Attrs map after Add cannot reach into the
// index's stored metadata (Attrs["utility"] feeds Benefit, so a shared map would let a
// post-Add mutation silently change a span's score without any flag flip). With Attrs cloned
// and every other field a value type, the only mutation a recorded span undergoes through the
// index API is a trust/suppression flag flip (SetSealed/SetTombstoned) — the structural
// reason the maintenance surface is small and the incremental==batch equivalence holds.
func (ix *Index) Add(s Span) {
	i := len(ix.spans)
	s.Attrs = cloneAttrs(s.Attrs)
	ix.spans = append(ix.spans, s)
	ix.byID[s.ID] = i

	for t := range tokenSet(s.Role + " " + s.Descriptor) {
		ix.posting[t] = append(ix.posting[t], i)
	}
	if durabilityRank[NormDurability(s.Durability)] >= durabilityRank[DurabilityBounded] {
		ix.durable = append(ix.durable, i)
	}
}

// ProbeOptions tunes the bounded probe. The zero value is valid (orDefaults fills it), so a
// caller can pass ProbeOptions{} for sensible defaults.
type ProbeOptions struct {
	// RecencyWindow (R) is how many of the most-recent spans are ALWAYS candidates,
	// regardless of relevance — the recency access path. The spans a turn most often
	// references next are the recent ones, so a window keeps them probed even when the
	// forecast intents miss them. Defaults to DefaultRecencyWindow.
	RecencyWindow int
	// MaxCandidates (the hard bound) caps the probed set so the per-turn planning cost is
	// bounded no matter how the access paths union out. When the union exceeds it, the
	// lowest-priority candidates (durable, then recency, then relevance) are dropped first;
	// pins are never dropped. Defaults to DefaultMaxCandidates.
	MaxCandidates int
	// IncludeDurability is the set of durability classes the durability access path admits
	// as always-candidate. Defaults to {durable, bounded} — the classes the durability
	// prior keeps worth holding resident across turns. A turn/session span is reached only
	// via relevance or recency, not held candidate session-wide.
	IncludeDurability []string
}

// DefaultRecencyWindow is the seed recency tail: the last few dozen spans are always
// candidates. Conservative (a few turns of headroom), not tuned — the same posture
// DefaultWeights takes.
const DefaultRecencyWindow = 32

// DefaultMaxCandidates is the seed hard bound on the probed candidate set. It is the
// constant the per-turn planning cost is bounded by; a session that runs a million turns
// still scores at most this many candidates a turn (the flatten this file exists to give).
const DefaultMaxCandidates = 128

func (o ProbeOptions) orDefaults() ProbeOptions {
	if o.RecencyWindow <= 0 {
		o.RecencyWindow = DefaultRecencyWindow
	}
	if o.MaxCandidates <= 0 {
		o.MaxCandidates = DefaultMaxCandidates
	}
	if o.IncludeDurability == nil {
		o.IncludeDurability = []string{DurabilityDurable, DurabilityBounded}
	}
	return o
}

// Probe returns the bounded candidate span set for a forecast: the union of the pin,
// relevance, recency, and durability access paths, deduped by id, tiered by priority, and
// capped at MaxCandidates. It runs in O(R + matches + |durable| + cap·log cap), independent
// of N for a bounded query — the index scan that replaces the Θ(N) seq scan. The returned
// spans are in render order (step asc, then id), the order Candidates/Optimize expect.
func (ix *Index) Probe(f Forecast, opts ProbeOptions) []Span {
	opts = opts.orDefaults()

	const (
		tierPin = iota
		tierRelevance
		tierRecency
		tierDurable
		tierNone
	)
	// best[i] = the highest-priority (lowest) tier span i was reached by; tierNone = unseen.
	best := make([]int, len(ix.spans))
	for i := range best {
		best[i] = tierNone
	}
	mark := func(i, tier int) {
		if i >= 0 && i < len(best) && tier < best[i] {
			best[i] = tier
		}
	}

	// tier 0 — pins: resolve each pinned id to its span index.
	for _, id := range f.Pins {
		if i, ok := ix.byID[id]; ok {
			mark(i, tierPin)
		}
	}
	// tier 1 — relevance: walk each intent token's posting list (the inverted index).
	for t := range tokenSet(joinIntents(f.Intents)) {
		for _, j := range ix.posting[t] {
			mark(j, tierRelevance)
		}
	}
	// tier 2 — recency: the most-recent RecencyWindow spans (the append-order tail).
	lo := len(ix.spans) - opts.RecencyWindow
	if lo < 0 {
		lo = 0
	}
	for i := len(ix.spans) - 1; i >= lo; i-- {
		mark(i, tierRecency)
	}
	// tier 3 — durability: the bounded/durable set, filtered to the admitted classes.
	admit := map[string]bool{}
	for _, c := range opts.IncludeDurability {
		admit[NormDurability(c)] = true
	}
	for _, i := range ix.durable {
		if admit[NormDurability(ix.spans[i].Durability)] {
			mark(i, tierDurable)
		}
	}

	// Collect every reached span, then order by (tier asc, step DESC, id asc): a hard cap
	// keeps the highest-priority, then the most recent within a tier — deterministic, and
	// it never drops a pin. Step-descending so that when the cap bites it keeps the freshest
	// of a tier; the final slice is re-sorted into render order below.
	type hit struct {
		idx, tier, step int
		id              string
	}
	hits := make([]hit, 0, len(ix.spans))
	for i, tier := range best {
		if tier != tierNone {
			hits = append(hits, hit{idx: i, tier: tier, step: ix.spans[i].Step, id: ix.spans[i].ID})
		}
	}
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].tier != hits[b].tier {
			return hits[a].tier < hits[b].tier
		}
		if hits[a].step != hits[b].step {
			return hits[a].step > hits[b].step // freshest first within a tier
		}
		return hits[a].id < hits[b].id
	})
	if len(hits) > opts.MaxCandidates {
		hits = hits[:opts.MaxCandidates]
	}

	out := make([]Span, len(hits))
	for k, h := range hits {
		out[k] = ix.spans[h.idx]
	}
	// Render order: step asc, then id — the order Candidates/Optimize present a plan in.
	sort.Slice(out, func(a, b int) bool {
		if out[a].Step != out[b].Step {
			return out[a].Step < out[b].Step
		}
		return out[a].ID < out[b].ID
	})
	return out
}

// PlanCells is the index-bounded planning entry point — the bounded-compute peer of the
// package-level PlanCells (which scans all spans). It probes the bounded candidate set for
// the forecast, scores ONLY that set against the forecast, and optimizes the resident view
// under the budget greedily. The result is a Plan over the probed candidates: its
// Candidates count is the bounded probe size, not N, so the planner's per-turn work is
// bounded — yet the plan is over real spans and is faithful and materializable exactly like
// a full-scan plan (the elided set partitions the PROBED candidates; pruned spans stay
// demand-pageable in the lossless store).
func (ix *Index) PlanCells(f Forecast, b Budget, cost CostModel, opts ProbeOptions) Plan {
	spans := ix.Probe(f, opts)
	cands := Candidates(spans, f, cost)
	p := Optimize(cands, b, pinSet(f.Pins), ObjGreedy)
	p.Horizon = f.Horizon
	return p
}

// IndexBoundedPlannerCompute is the cumulative re-planning WORK through turn n when the
// planner scores only an index-bounded candidate set of size c each turn, in
// candidate-scoring operations: Σ_{i=1..n} c = c·n = Θ(c·N) — LINEAR in the turn horizon.
// It is the flatten cumPlannerCompute (scaling.go) names: a full re-scan scores i
// candidates at turn i for Σ i = N·(N+1)/2 = Θ(N²); bounding the candidate set at c turns
// that quadratic into a per-turn surcharge of a CONSTANT c, the same shape the working-set
// cap gives the resident-token curve. Zero for a non-positive bound or horizon.
//
// This is the compute analogue of cumCapped: the budget caps resident TOKENS at W for a
// Θ(W·N) resident cost; the index caps scored CANDIDATES at c for a Θ(c·N) planning cost.
// Both replace a Θ(N²) growth with a linear one, which is what makes "1 current turn + a
// flexible history" hold for the planner's own work and not just its output.
func IndexBoundedPlannerCompute(candidateBound, n int) int64 {
	if candidateBound <= 0 || n <= 0 {
		return 0
	}
	return int64(candidateBound) * int64(n)
}

// joinIntents concatenates a forecast's intents into one string for tokenization (the same
// thing Forecast.relevance does inline); kept here so Probe shares the exact intent
// vocabulary the benefit scorer matches against.
func joinIntents(intents []string) string {
	switch len(intents) {
	case 0:
		return ""
	case 1:
		return intents[0]
	}
	n := len(intents) - 1
	for _, s := range intents {
		n += len(s)
	}
	b := make([]byte, 0, n)
	for k, s := range intents {
		if k > 0 {
			b = append(b, ' ')
		}
		b = append(b, s...)
	}
	return string(b)
}
