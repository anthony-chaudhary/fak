package memq

import "sort"

// SpanGroupAttr is the OPEN attribute-bag key that scopes neighborhood pooling
// (RankRelevanceSpan, #4014): cells sharing Attrs["thread"] pool only against each
// other, so two interleaved conversations never smear relevance across the boundary.
// Cells without the attr share one default group — the common single-thread case.
const SpanGroupAttr = "thread"

// poolSpanScores is the neighborhood-pooling pass behind RankRelevanceSpan (#4014) —
// the SnapKV idea (pyramidkv_utils.py's avg/max_pool1d over the score vector, @94255b6)
// reprojected onto memq's ordinal Step axis: a contiguous informative span should
// outrank an isolated high-score spike, and a lone spike should not drag weak
// neighbors in.
//
// For each cell in the working set it replaces the raw intent-overlap score with the
// SUM of the raw scores in a centered, odd-width window of its step-adjacent
// neighbors, zero-padded at the group edges. Because the window width is constant,
// ranking by this zero-padded windowed sum equals ranking by the upstream's
// count_include_pad average (same denominator everywhere) — kept integral so no float
// creeps into the audit contract. window==1 is the identity kernel: the returned
// scores equal the raw ones, so the ranking reproduces plain RankRelevance exactly
// (the equivalence witness pinned in spanpool_test.go).
//
// Neighborhood = the nearest SURVIVING cells: pooling runs over the working set as
// the rank op sees it (after filters), so a sealed/tombstoned cell that never reached
// the ranker neither lends nor borrows score. Cells are grouped by
// Attrs[SpanGroupAttr] (absent → one default group) and ordered by (Step, ID) within
// the group; a cell with Step < 0 has no ordinal neighborhood and keeps its raw
// score. The whole pass is deterministic: no RNG, group keys are visited in sorted
// order, and every cell's pooled value depends only on its own group's total order.
func poolSpanScores(work []Cell, score map[string]int, window int) map[string]int {
	pooled := make(map[string]int, len(work))
	for _, c := range work {
		pooled[c.ID] = score[c.ID] // floor: unordered cells keep their raw score
	}
	if window <= 1 {
		return pooled // identity kernel — plain relevance, byte-for-byte
	}
	groups := map[string][]Cell{}
	var keys []string
	for _, c := range work {
		if c.Step < 0 {
			continue // no ordinal position, no neighborhood
		}
		k := c.Attrs[SpanGroupAttr]
		if _, ok := groups[k]; !ok {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], c)
	}
	sort.Strings(keys) // deterministic visit order (values are per-group, so this is audit hygiene, not correctness)
	half := (window - 1) / 2
	for _, k := range keys {
		g := groups[k]
		sort.Slice(g, func(i, j int) bool {
			if g[i].Step != g[j].Step {
				return g[i].Step < g[j].Step
			}
			return g[i].ID < g[j].ID
		})
		for i, c := range g {
			sum := 0
			for j := i - half; j <= i+half; j++ {
				if j < 0 || j >= len(g) {
					continue // zero padding: a missing edge neighbor contributes 0
				}
				sum += score[g[j].ID]
			}
			pooled[c.ID] = sum
		}
	}
	return pooled
}
