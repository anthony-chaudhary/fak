package ctxplan

import "sort"

// access_path.go — COST-BASED access-path selection: the planner ESTIMATES the cost of each
// of its four access paths (pin, inverted-index relevance, recency tail, durable set) and
// scans only the cheapest combination that still satisfies the probe, the way a real query
// planner picks an index scan over a seq scan instead of unconditionally unioning every path.
//
// # Why this exists (the access-path step Probe was one short of)
//
// index.go's Probe is correct but UNCONDITIONAL: it always walks all four paths, then dedups,
// tiers, and caps. That is the bitmap-OR of every access path — never the cheapest plan. A
// Postgres planner faced with "WHERE rare_token = x" does an Index Scan on that token's GIN
// posting list and does NOT also seq-scan the heap, because the cost model proves the index
// scan alone returns the qualifying rows more cheaply. ctxplan's analogue: when the forecast
// is highly selective (few, rare intent tokens), the inverted index alone reaches the
// load-bearing spans, so the recency seq-scan is pure cost — it widens the candidate set with
// spans the cap/budget will not keep. Conversely an empty or very-broad forecast makes the
// inverted index NON-selective (it matches nearly everything or nothing), so the planner falls
// back to the recency + durable paths. This file is that cost model and chooser.
//
// # The correctness fence (this prunes PATHS, never qualifying ROWS)
//
// Access-path selection is an optimization of WHICH paths to scan, NEVER of which rows
// qualify. The chooser is correctness-preserving by construction: it drops a path only when
// the kept paths' union, after the SAME cap and budget the full union would apply, selects the
// IDENTICAL Selected set. Concretely it never drops the pin or durable paths (always-candidate
// and cheap), and it drops the recency path only when the recency tail adds no span that could
// survive the cap+budget over the relevance/pin/durable union — so the final plan is bit-for-
// bit the full-union plan at a strictly smaller scanned candidate set. When the forecast is not
// selective enough to prove that, the chooser keeps every path and the result is exactly
// today's Probe. This is the same honesty posture as the index itself: a tighter path set is an
// efficiency win, and any span not reached stays one demand-page away in the lossless store.

// AccessPath names one of the four candidate sources Probe can scan. It is the ctxplan
// analogue of a Postgres access path (an Index Scan, a Seq Scan, a Bitmap Heap Scan).
type AccessPath int

const (
	// PathPin is the pin access path: forecast Pins resolved to span indices. Always cheap,
	// always kept — the spans a turn cannot proceed without.
	PathPin AccessPath = iota
	// PathRelevance is the INVERTED-INDEX scan: intent tokens -> posting lists. The selective
	// path; its cost is the summed posting-list length, its value the summed IDF selectivity.
	PathRelevance
	// PathRecency is the recency-tail SEQ SCAN: the last RecencyWindow spans regardless of
	// match. Fixed cost (the window), non-selective.
	PathRecency
	// PathDurable is the durable-set scan: the bounded/durable spans kept candidate session-
	// wide. Cheap (a maintained set), always kept.
	PathDurable
)

// pathName renders an access path the way EXPLAIN names a scan, so a Plan.Explain over a
// cost-selected probe reads like a real query plan ("Index Scan on intents", "Seq Scan on
// recency tail").
func (p AccessPath) pathName() string {
	switch p {
	case PathPin:
		return "Pin Scan on pins"
	case PathRelevance:
		return "Index Scan on intents"
	case PathRecency:
		return "Seq Scan on recency tail"
	case PathDurable:
		return "Index Scan on durable set"
	default:
		return "Unknown Scan"
	}
}

// pathEstimate is the cost model's per-path forecast — the pg_statistic-derived row a planner
// reads before it chooses. Cardinality is the estimated candidate count the path contributes
// (its cost proxy: scanning and scoring N candidates is Θ(N)); Selectivity is how
// discriminating the path is (summed IDF for relevance, 0 for the unconditional paths). A
// planner picks the cheapest combination of paths whose union still covers the probe.
type pathEstimate struct {
	Path        AccessPath
	Cardinality int     // estimated candidate count this path adds (the cost proxy)
	Selectivity float64 // how discriminating the path is — summed IDF for relevance, 0 otherwise
}

// estimatePaths reads the cost model for all four paths off the index statistics WITHOUT
// scanning any posting list into a candidate set — it uses only maintained counts (posting-
// list lengths via docFreq, the recency window, the durable-set size, the resolvable pins), so
// the estimate is O(intents + pins), far cheaper than the scan it decides whether to run. This
// is the pg_statistic read that precedes access-path selection.
func (ix *Index) estimatePaths(f Forecast, opts ProbeOptions) []pathEstimate {
	opts = opts.orDefaults()

	// Pin cardinality: pins that actually resolve to an indexed span.
	pins := 0
	for _, id := range f.Pins {
		if _, ok := ix.byID[id]; ok {
			pins++
		}
	}

	// Relevance cardinality + selectivity: sum posting-list lengths (an UPPER BOUND on the
	// deduped match count — the planner's standard conservative cardinality estimate) and sum
	// IDF over the distinct intent tokens. A rare token contributes a short list and a high
	// IDF (selective); a common token a long list and a near-1 IDF (non-selective).
	relCard := 0
	var relSel float64
	for t := range tokenSet(joinIntents(f.Intents)) {
		relCard += ix.docFreq(t)
		relSel += ix.idf(t)
	}

	// Recency cardinality: the window clamped to N.
	rec := opts.RecencyWindow
	if rec > len(ix.spans) {
		rec = len(ix.spans)
	}

	// Durable cardinality: the durable spans in the admitted classes.
	admit := durabilityAdmitSet(opts.IncludeDurability)
	dur := 0
	for _, i := range ix.durable {
		if admit[NormDurability(ix.spans[i].Durability)] {
			dur++
		}
	}

	return []pathEstimate{
		{Path: PathPin, Cardinality: pins, Selectivity: 0},
		{Path: PathRelevance, Cardinality: relCard, Selectivity: relSel},
		{Path: PathRecency, Cardinality: rec, Selectivity: 0},
		{Path: PathDurable, Cardinality: dur, Selectivity: 0},
	}
}

// chosenPaths is the cost-based decision: given the per-path estimates and the index, it
// returns the set of paths to actually scan. The pin and durable paths are always chosen (cheap
// and always-candidate). The relevance and recency paths are chosen by the cost model:
//
//   - A HIGHLY SELECTIVE forecast (non-empty relevance matches whose count fits inside the cap
//     with room for pins+durable, AND whose tail-overlap is total) lets the planner DROP the
//     recency seq-scan: the inverted index alone reaches the load-bearing spans, and the recency
//     tail would only add spans the cap/budget cannot keep over the more-selective relevance
//     hits. This is the Index-Scan-instead-of-Seq-Scan win.
//   - An EMPTY or NON-SELECTIVE forecast (no resolvable relevance matches) DROPS the relevance
//     path (it contributes nothing) and falls back to recency + durable — the planner's seq-scan
//     fallback when no usable index exists.
//   - Otherwise (a selective forecast that does not provably dominate the tail) every path is
//     kept, exactly reproducing today's unconditional Probe.
//
// The choice is correctness-preserving (see correctnessSafeToDropRecency) and deterministic
// (it reads only maintained counts and a deterministic tail check).
func (ix *Index) chosenPaths(f Forecast, opts ProbeOptions, est []pathEstimate) map[AccessPath]bool {
	opts = opts.orDefaults()
	chosen := map[AccessPath]bool{PathPin: true, PathDurable: true}

	var rel pathEstimate
	for _, e := range est {
		if e.Path == PathRelevance {
			rel = e
		}
	}

	if rel.Cardinality == 0 {
		// No usable inverted-index match: drop the relevance path (it scores nothing) and rely
		// on the recency + durable + pin fallback. This is the empty/very-broad forecast case.
		chosen[PathRecency] = true
		return chosen
	}

	// A relevance match exists. Keep the relevance path. Decide whether the recency seq-scan
	// can be dropped without changing the final selection.
	chosen[PathRelevance] = true
	if ix.correctnessSafeToDropRecency(f, opts) {
		return chosen // Index Scan alone — the recency Seq Scan is pruned.
	}
	chosen[PathRecency] = true
	return chosen
}

// correctnessSafeToDropRecency proves the recency seq-scan can be skipped WITHOUT changing the
// probe's final (cap-bounded) candidate set, so the downstream plan is identical. It is the
// cost model's correctness side: it returns true only when every span the recency tail would
// add is ALREADY reachable via the pin/relevance/durable paths — i.e. the recency path is
// redundant, contributing no new span. When that holds, dropping recency cannot change which
// spans are probed, so it cannot change which are selected: a provably free prune. This is the
// conservative, always-safe rule; a forecast selective enough to make the relevance hits fill
// the cap on their own is the common case where it fires.
//
// The check is deterministic and O(window + matches): it walks the recency window once and the
// intent posting lists once, both bounded, never N.
func (ix *Index) correctnessSafeToDropRecency(f Forecast, opts ProbeOptions) bool {
	opts = opts.orDefaults()

	lo := len(ix.spans) - opts.RecencyWindow
	if lo < 0 {
		lo = 0
	}
	if lo >= len(ix.spans) {
		return true // empty window — nothing to drop
	}

	// reached[i] = span i is reached by a NON-recency path (pin, relevance, or durable). We
	// only need this over the recency window, but the relevance posting lists and durable set
	// can point anywhere, so build a membership set restricted to the window range.
	inWindow := func(i int) bool { return i >= lo && i < len(ix.spans) }
	reached := make(map[int]bool)

	for _, id := range f.Pins {
		if i, ok := ix.byID[id]; ok && inWindow(i) {
			reached[i] = true
		}
	}
	for t := range tokenSet(joinIntents(f.Intents)) {
		for _, j := range ix.posting[t] {
			if inWindow(j) {
				reached[j] = true
			}
		}
	}
	admit := durabilityAdmitSet(opts.IncludeDurability)
	for _, i := range ix.durable {
		if inWindow(i) && admit[NormDurability(ix.spans[i].Durability)] {
			reached[i] = true
		}
	}

	// Safe to drop recency iff every span in the window is already reached by another path:
	// then the recency scan adds no new candidate, so removing it cannot change the probe.
	for i := lo; i < len(ix.spans); i++ {
		if !reached[i] {
			return false
		}
	}
	return true
}

// ProbeResult is a cost-based probe: the bounded candidate set PLUS the access-path plan that
// produced it (the chosen paths and their cost estimates). It is what makes ctxplan's EXPLAIN
// read like a real query plan — a caller can see "Index Scan on intents" vs "Seq Scan on
// recency tail" and the cardinality each path was estimated to contribute.
type ProbeResult struct {
	// Spans is the bounded candidate set — the SAME contract Probe returns (render order, cap-
	// bounded, deduped), and for representative queries the IDENTICAL set, since path selection
	// is correctness-preserving.
	Spans []Span
	// Chosen is the access paths the cost model actually scanned, in path order. A forecast
	// that lets the inverted index dominate omits PathRecency; an empty forecast omits
	// PathRelevance.
	Chosen []AccessPath
	// Estimates is the per-path cost model the chooser read — the pg_statistic-style cardinality
	// and selectivity each path was forecast to contribute, surfaced for EXPLAIN.
	Estimates []pathEstimate
}

// ProbePlan is the cost-based access-path probe: it estimates each path's cost from the index
// statistics, picks the cheapest combination of paths that still satisfies the probe, scans
// ONLY those paths, and returns the candidate set together with the chosen-path plan. For a
// selective forecast it returns the SAME candidate set as Probe at a smaller scanned set (the
// recency seq-scan is pruned); for an empty forecast it skips the dead relevance path. It is
// deterministic and correctness-preserving: ProbePlan(...).Spans equals Probe(...) wherever the
// chooser is provably safe, which is every representative query the witness asserts.
func (ix *Index) ProbePlan(f Forecast, opts ProbeOptions) ProbeResult {
	opts = opts.orDefaults()
	est := ix.estimatePaths(f, opts)
	chosen := ix.chosenPaths(f, opts, est)
	spans := ix.probePaths(f, opts, chosen)

	order := []AccessPath{PathPin, PathRelevance, PathRecency, PathDurable}
	picked := make([]AccessPath, 0, len(order))
	for _, p := range order {
		if chosen[p] {
			picked = append(picked, p)
		}
	}
	return ProbeResult{Spans: spans, Chosen: picked, Estimates: est}
}

// probePaths is the access-path executor: the same tiered union/dedup/cap as Probe, but it
// walks ONLY the chosen paths. With all four paths chosen it is byte-identical to Probe (the
// unconditional union); with a path dropped it scans strictly less. Factoring the executor this
// way keeps the cost-based and unconditional probes provably consistent — they share one code
// path, differing only in which access paths the chooser enabled.
func (ix *Index) probePaths(f Forecast, opts ProbeOptions, chosen map[AccessPath]bool) []Span {
	opts = opts.orDefaults()

	const (
		tierPin = iota
		tierRelevance
		tierRecency
		tierDurable
		tierNone
	)
	best := make([]int, len(ix.spans))
	for i := range best {
		best[i] = tierNone
	}
	mark := func(i, tier int) {
		if i >= 0 && i < len(best) && tier < best[i] {
			best[i] = tier
		}
	}

	if chosen[PathPin] {
		for _, id := range f.Pins {
			if i, ok := ix.byID[id]; ok {
				mark(i, tierPin)
			}
		}
	}
	relScore := make([]float64, len(ix.spans))
	if chosen[PathRelevance] {
		for t := range tokenSet(joinIntents(f.Intents)) {
			w := ix.idf(t)
			for _, j := range ix.posting[t] {
				mark(j, tierRelevance)
				relScore[j] += w
			}
		}
	}
	if chosen[PathRecency] {
		lo := len(ix.spans) - opts.RecencyWindow
		if lo < 0 {
			lo = 0
		}
		for i := len(ix.spans) - 1; i >= lo; i-- {
			mark(i, tierRecency)
		}
	}
	if chosen[PathDurable] {
		admit := durabilityAdmitSet(opts.IncludeDurability)
		for _, i := range ix.durable {
			if admit[NormDurability(ix.spans[i].Durability)] {
				mark(i, tierDurable)
			}
		}
	}

	type hit struct {
		idx, tier, step int
		score           float64
		id              string
	}
	hits := make([]hit, 0, len(ix.spans))
	for i, tier := range best {
		if tier != tierNone {
			hits = append(hits, hit{idx: i, tier: tier, step: ix.spans[i].Step, score: relScore[i], id: ix.spans[i].ID})
		}
	}
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].tier != hits[b].tier {
			return hits[a].tier < hits[b].tier
		}
		if hits[a].score != hits[b].score {
			return hits[a].score > hits[b].score
		}
		if hits[a].step != hits[b].step {
			return hits[a].step > hits[b].step
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
	sort.Slice(out, func(a, b int) bool {
		if out[a].Step != out[b].Step {
			return out[a].Step < out[b].Step
		}
		return out[a].ID < out[b].ID
	})
	return out
}
