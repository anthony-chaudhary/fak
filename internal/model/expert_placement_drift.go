package model

import "sort"

// expert_placement_drift.go — a pure gauge for how well a PLANNED expert placement (which experts a
// residency planner decided to keep host-resident) matches the OBSERVED routing a real forward window
// actually exercised. It answers two orthogonal questions from the same trace:
//
//   - Coverage: what fraction of observed expert TOUCHES were served by a planned-resident expert.
//     served/total. A high coverage means the plan kept the experts the workload actually hit.
//   - Drift: how far the plan's resident SET has drifted from the observed top-k hot set. It is the
//     set-overlap complement 1 - |planned ∩ observed_topk| / k, so the plan resident set equal to the
//     observed top-k scores 0 (no drift) and a plan disjoint from the hot set scores 1 (total drift).
//
// Coverage weights by touch volume (a resident expert that fires 1000× counts more than one that
// fires once); Drift is unweighted set membership over the k hottest experts. They diverge exactly
// when the plan covers most touches but has swapped a couple of the marginal top-k members — the
// signal a "coverage looks fine" scalar hides. Both are pure functions of (freq, mask, k): no I/O,
// no model state, deterministic for a given input.

// ExpertPlacementScore is the (coverage, drift) verdict ScoreExpertPlacement returns. Both are in
// [0,1]. Coverage is served/total observed touches; Drift is 1 - |planned ∩ observed_topk|/k.
type ExpertPlacementScore struct {
	Coverage float64 `json:"coverage"`
	Drift    float64 `json:"drift"`
}

// ScoreExpertPlacement scores a planned expert placement against an observed per-expert access
// histogram. freq[e] is the observed touch count of expert e; residentMask[e] reports whether the
// plan keeps expert e host-resident (planned); k is the top-k hot-set width the drift is measured
// over. It is pure and allocation-light.
//
//   - Coverage = (Σ freq[e] over resident e) / (Σ freq[e]). The share of observed touches a
//     planned-resident expert served. Total zero (no observed touches) yields Coverage 0.
//   - Drift = 1 - |planned ∩ observed_topk| / k, where planned is the resident set and observed_topk
//     is the k experts with the largest freq (ties broken by lower expert id, matching torch.topk).
//     residentMask exactly equal to the observed top-k ⇒ Drift 0; a resident set disjoint from the
//     top-k ⇒ Drift 1. k <= 0 has no top-k to compare and yields Drift 0.
//
// freq and residentMask are indexed by expert id; a residentMask shorter than freq treats the missing
// tail as not-resident, and a mask entry past the end of freq is ignored (its expert has no touches).
func ScoreExpertPlacement(freq []int64, residentMask []bool, k int) ExpertPlacementScore {
	var served, total int64
	for e, f := range freq {
		if f < 0 {
			f = 0
		}
		total += f
		if e < len(residentMask) && residentMask[e] {
			served += f
		}
	}
	var score ExpertPlacementScore
	if total > 0 {
		score.Coverage = float64(served) / float64(total)
	}

	if k <= 0 {
		return score // no top-k window to compare against — drift is undefined, reported as 0.
	}
	topk := observedTopK(freq, k)
	overlap := 0
	for _, e := range topk {
		if e < len(residentMask) && residentMask[e] {
			overlap++
		}
	}
	score.Drift = 1 - float64(overlap)/float64(k)
	return score
}

// observedTopK returns the expert ids of the k largest freq values, largest first, ties broken by the
// lower expert id (torch.topk's stable order). Fewer than k experts returns all of them. Only expert
// ids present in freq are eligible, so the returned slice has at most min(k, len(freq)) entries.
func observedTopK(freq []int64, k int) []int {
	idx := make([]int, len(freq))
	for e := range freq {
		idx[e] = e
	}
	sort.SliceStable(idx, func(a, b int) bool {
		fa, fb := freq[idx[a]], freq[idx[b]]
		if fa != fb {
			return fa > fb // larger frequency first
		}
		return idx[a] < idx[b] // stable tie-break: lower expert id wins
	})
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}

// ExpertAccessHistogram folds an access trace's per-touch events into a per-expert access-count
// histogram indexed by expert id, width numExperts. Each ExpertAccessTraceEvent is one routed-expert
// touch, so hist[e] is the number of times expert e was selected across all layers and positions in
// the trace; an expert never touched keeps its zero slot so the histogram aligns with a residentMask
// of the same width for ScoreExpertPlacement. Events whose Expert id falls outside [0,numExperts) are
// skipped rather than panicking. A non-positive numExperts yields an empty histogram.
func ExpertAccessHistogram(events []ExpertAccessTraceEvent, numExperts int) []int64 {
	if numExperts <= 0 {
		return nil
	}
	hist := make([]int64, numExperts)
	for _, ev := range events {
		if ev.Expert < 0 || ev.Expert >= numExperts {
			continue
		}
		hist[ev.Expert]++
	}
	return hist
}
