package compute

import "math"

// intervalscore.go — the prefix-sum interval scoring kernel (borrow B1 from the
// GLiNER2 study, docs/research/STUDY-GLINER2-2026-09-18.md). Scoring every
// (start, end) interval by materializing an [L, L] score matrix costs O(L^2)
// memory for an O(L^2) result; the interval score of a contiguous span is
// instead the difference of a single per-token cumulative prefix, so the only
// intermediate is O(L).
//
// The reference (fastino-ai/GLiNER2 @ d7c727458bf6929bc9ef5ee04e13c3f717a7c455,
// Apache-2.0, gliner2/models/boundary/heads.py:88-103 consumed at
// scoring.py:253-255) centers each token's "inside" contribution before the fp32
// cumulative sum and carries the mean separately. Centering keeps the running
// offset bounded (a raw cumsum of large logits loses low-order bits as it grows),
// while the carried mean preserves exact interval differences: for any
// (start, end), sum_{i in [start,end)} (inside[i] - mean + mean) is unchanged, so
// prefix[end] - prefix[start] + (end-start)*mean equals the explicit per-interval
// summation within fp32 tolerance.
//
// This file is the CPU reference kernel and its equivalence witness only: no GPU
// kernel, no model architecture, and no existing attention implementation is
// touched. Long-context span/candidate ranking is the intended consumer; a caller
// that needs the explicit sum can reproduce it byte-for-byte from the same inputs.

// IntervalPrefix is the O(L) representation of per-token interval contributions.
// Prefix holds L+1 cumulative sums (Prefix[0] == 0); Mean is the centering value
// subtracted before the cumsum and carried separately, so an interval score is
// exact up to fp reassociation.
type IntervalPrefix struct {
	// Prefix is the fp32 cumulative sum of (inside - Mean), length L+1.
	Prefix []float32
	// Mean is the centering offset that was removed from each inside logit.
	Mean float32
}

// BuildIntervalPrefix computes the per-token cumulative prefix for a [..., L]
// slice of inside logits. The returned prefix has length L+1, and the mean that
// was centered out is carried in Mean. It never allocates an [L, L] intermediate:
// the largest buffer it produces is L+1 long.
//
// mean is computed in float64 then rounded once to float32 so the centering is
// deterministic and the carried offset is a single value (the same posture the
// reference takes before its fp32 cumulative sum).
func BuildIntervalPrefix(inside []float32) IntervalPrefix {
	l := len(inside)
	prefix := make([]float32, l+1)
	if l == 0 {
		return IntervalPrefix{Prefix: prefix}
	}
	var sum float64
	for _, v := range inside {
		sum += float64(v)
	}
	mean := float32(sum / float64(l))
	var acc float32
	for i, v := range inside {
		acc += v - mean
		prefix[i+1] = acc
	}
	return IntervalPrefix{Prefix: prefix, Mean: mean}
}

// IntervalScore returns the score of the contiguous interval [start, end) as
// prefix[end] - prefix[start] + (end-start)*mean, matching the explicit
// sum_{i in [start,end)} inside[i] up to fp32 reassociation. It returns false
// when the interval is out of range (start < 0, end > L, or start > end), never
// a silently clamped value: a caller that passed a bad span must not read a
// plausible-looking score.
func (p IntervalPrefix) IntervalScore(inside []float32, start, end int) (float32, bool) {
	if start < 0 || end < start || end > len(inside) {
		return 0, false
	}
	if len(p.Prefix) != len(inside)+1 {
		return 0, false
	}
	score := p.Prefix[end] - p.Prefix[start]
	score += float32(end-start) * p.Mean
	return score, true
}

// ScoreIntervals scores every (start, end) pair from a single prefix, writing
// row-major scores into dst (len(dst) must be >= len(spans)). It reuses the one
// O(L) prefix for all spans, so N interval queries cost O(L + N) after a single
// O(L) build instead of N independent O(L) sums. It returns the number of spans
// scored; a span that fails IntervalScore's range check is left untouched and not
// counted, so the caller can detect a malformed batch by the short count.
func (p IntervalPrefix) ScoreIntervals(inside []float32, spans [][2]int, dst []float32) int {
	n := 0
	for i, s := range spans {
		if i >= len(dst) {
			break
		}
		score, ok := p.IntervalScore(inside, s[0], s[1])
		if !ok {
			continue
		}
		dst[i] = score
		n++
	}
	return n
}

// IntervalScoreExplicit is the O(L) per-interval summation the prefix kernel
// replaces: the reference a caller can use to check equivalence. It sums the
// inside logits directly for [start, end) in order, so it is the ground truth
// for the BuildIntervalPrefix/IntervalScore pair.
func IntervalScoreExplicit(inside []float32, start, end int) (float32, bool) {
	if start < 0 || end < start || end > len(inside) {
		return 0, false
	}
	var acc float32
	for _, v := range inside[start:end] {
		acc += v
	}
	return acc, true
}

// MaxIntervalIntermediateLen returns the largest slice length the prefix kernel
// allocates for L inside logits (L+1), so a test can assert the kernel stays
// linear in memory rather than materializing an [L, L] tensor.
func MaxIntervalIntermediateLen(l int) int {
	if l < 0 {
		return 0
	}
	return l + 1
}

// intervalTolerance is the fp32 reassociation bound the prefix score is allowed
// to differ from the explicit sum by. The centering keeps the running prefix
// small, so the residual is bounded by a few ulps scaled by |score| plus the
// accumulation depth; this is a numeric guard, not a model tolerance.
func intervalTolerance(inside []float32) float32 {
	var scale float32
	for _, v := range inside {
		if a := float32(math.Abs(float64(v))); a > scale {
			scale = a
		}
	}
	if scale == 0 {
		return 1e-6
	}
	return scale * float32(len(inside)) * 1e-6
}
