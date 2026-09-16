package compute

import (
	"math"
	"sort"
)

// deepselect_topk.go — DeepSeek "DeepSelect" threshold-scan + radix-select top-k.
//
// The shape is DeepSeek's: stream the input in blocks, keep only items that clear a
// running threshold, and whenever the survivor buffer grows past k+slack, collapse it
// to exactly k with a radix select and lower the threshold to the k-th value. The
// buffer therefore never grows past k+slack+B, and the final radix select produces the
// exact top-k.
//
// EXACTNESS (why this is a selection, not an approximation): the survivors are ordered
// by the STRICT TOTAL ORDER (value desc, position asc) — the same order dsaTopKIndices
// and sortDSAIndexCands use. After a collapse the survivor set is exactly the top-k of
// everything seen so far, so its minimum value m is a valid rank bound: there are
// already k survivors with value >= m, hence any future item with value < m can never
// enter the global top-k. The total order is what makes that bound hold under ties: two
// items with equal value are still distinguished by position, so "k survivors at value
// m" is a complete adversarial answer to "can a later value-m item displace one".
//
// NOTE on the pruning comparison. The pruning test here is `value >= threshold`, not the
// strict `value > threshold` the original scan uses. With a POSITION-ORDERED scan the two
// are equivalent: positions increase monotonically, so a later item with value == m always
// has a HIGHER position and loses the tie-break to the k survivors, and strict `>` is safe.
// This engine deliberately scans blocks in a RANDOMIZED order (for load/early-exit
// behaviour), so a later-scanned value-m item may carry a LOWER position and legitimately
// out-rank a survivor — dropping it with `>` would change the selected SET. Keeping ties
// costs only buffer space (removed by the next radix select) and is what preserves set
// equality with sortDSAIndexCands for every input, including engineered near-ties.

// Candidate pairs a selectable score with the item's ORIGINAL position. The position is
// the tie-break key (lower wins) and is preserved through every compaction, so the emitted
// set is directly comparable to the sort-based reference.
type Candidate struct {
	Value    float64
	Position int
}

// deepSelectSeed is a fixed constant (no global math/rand state) so a run — and a test —
// reproduces the same block permutation deterministically.
const deepSelectSeed uint64 = 0x9E3779B97F4A7C15

// splitMix64 is the standard SplitMix64 finalizer: a small, dependency-free, seedable
// mixer for the block shuffle. Deriving every shuffle step from one fixed seed keeps the
// randomized scan reproducible without touching process-global random state.
func splitMix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	z := x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// blockPermutation returns a deterministic Fisher-Yates permutation of [0, nBlocks).
func blockPermutation(nBlocks int, seed uint64) []int {
	order := make([]int, nBlocks)
	for i := range order {
		order[i] = i
	}
	s := seed
	for i := nBlocks - 1; i > 0; i-- {
		s = splitMix64(s)
		j := int(s % uint64(i+1))
		order[i], order[j] = order[j], order[i]
	}
	return order
}

// candidateKey maps a float64 to a uint64 whose unsigned order equals the value's numeric
// order (the standard signed-float radix transform): flip the sign bit for non-negative
// values, flip every bit for negative values. Two equal values map to the same key, which
// is exactly what lets the radix select isolate the boundary value; the position tie-break
// is then resolved over the equal-key group. NaN is out of contract here (DSA scores are
// relu'd and finite, and NaN is never admitted by the threshold test).
func candidateKey(v float64) uint64 {
	b := math.Float64bits(v)
	if b&(1<<63) != 0 {
		return ^b
	}
	return b | (1 << 63)
}

// sortCandidates orders by value descending, ties by lower position — byte-identical to
// the dsaIndexCandLess / dsaTopKIndices total order.
func sortCandidates(c []Candidate) {
	sort.Slice(c, func(i, j int) bool {
		if c[i].Value != c[j].Value {
			return c[i].Value > c[j].Value
		}
		return c[i].Position < c[j].Position
	})
}

// sortByPositionAsc is the equal-key tie-break: among candidates whose value is identical,
// the lower original position wins.
func sortByPositionAsc(c []Candidate) {
	sort.SliceStable(c, func(i, j int) bool { return c[i].Position < c[j].Position })
}

// radixSelectTopK returns the EXACT top-k of items under (value desc, position asc) using
// an MSB-first byte-histogram radix select over the monotone key. It is exact, not a
// quantile estimate: at each byte it keeps every candidate strictly above the boundary
// bucket as selected, narrows the active set to the boundary bucket, and when the key is
// exhausted it resolves the remaining equal-value slots by position. Ties therefore land on
// the same lower-position items sortDSAIndexCands would keep.
func radixSelectTopK(items []Candidate, k int) []Candidate {
	n := len(items)
	if k <= 0 || n == 0 {
		return nil
	}
	if k >= n {
		out := append([]Candidate(nil), items...)
		sortCandidates(out)
		return out
	}

	selected := make([]Candidate, 0, k)
	active := append([]Candidate(nil), items...)

	for shift := 56; shift >= 0; shift -= 8 {
		remaining := k - len(selected)
		if remaining <= 0 {
			break
		}
		if remaining >= len(active) {
			selected = append(selected, active...)
			break
		}

		var hist [256]int
		for i := range active {
			hist[byte(candidateKey(active[i].Value)>>uint(shift))]++
		}

		// Descending scan: the boundary bucket is the one that carries the k-th key.
		bucket, above := 255, 0
		for ; bucket >= 0; bucket-- {
			if above+hist[bucket] >= remaining {
				break
			}
			above += hist[bucket]
		}
		if bucket < 0 {
			// Defensive: remaining exceeds the active set; take everything.
			selected = append(selected, active...)
			break
		}

		// Partition in place: keys above the bucket are selected, keys in the bucket stay
		// active for the next (lower) byte, keys below are discarded (rank-proven out).
		next := active[:0]
		for _, c := range active {
			switch b := byte(candidateKey(c.Value) >> uint(shift)); {
			case b > byte(bucket):
				selected = append(selected, c)
			case b == byte(bucket):
				next = append(next, c)
			}
		}
		active = next
	}

	// Any slots still open are inside a group whose full keys are identical — resolve them
	// by the position tie-break alone.
	if remaining := k - len(selected); remaining > 0 {
		if remaining >= len(active) {
			selected = append(selected, active...)
		} else {
			sortByPositionAsc(active)
			selected = append(selected, active[:remaining]...)
		}
	}

	sortCandidates(selected)
	return selected
}

// minCandidateValue is the rank bound carried between blocks: the smallest value still in
// the top-k survivor set (any item strictly below it is rank-proven out).
func minCandidateValue(c []Candidate) float64 {
	m := math.Inf(1)
	for i := range c {
		if c[i].Value < m {
			m = c[i].Value
		}
	}
	return m
}

// DeepSelectTopk runs DeepSeek's threshold-scan + radix-select top-k and returns the exact
// top-k under (value desc, position asc). B is the input block size; B2 the compaction slack
// (the survivor buffer collapses once it reaches k+B2). B and B2 must be sized for the
// working set — B=B2=1024 is reasonable for k≈512 — but the result is EXACT for any k, and
// for B2=1 (collapse every time the buffer exceeds k+1) as well. The cost of a small B2 is
// extra radix passes, never a different selected set.
func DeepSelectTopk(items []Candidate, k, B, B2 int) []Candidate {
	n := len(items)
	if k <= 0 || n == 0 {
		return nil
	}
	if k >= n {
		out := append([]Candidate(nil), items...)
		sortCandidates(out)
		return out
	}
	if B <= 0 {
		B = 1
	}
	if B2 < 0 {
		B2 = 0
	}

	nBlocks := (n + B - 1) / B
	order := blockPermutation(nBlocks, deepSelectSeed)

	candidates := make([]Candidate, 0, k+B2+B)
	threshold := math.Inf(-1)
	for _, bi := range order {
		lo := bi * B
		hi := lo + B
		if hi > n {
			hi = n
		}
		for i := lo; i < hi; i++ {
			if c := items[i]; c.Value >= threshold {
				candidates = append(candidates, c)
			}
		}
		if len(candidates) >= k+B2 {
			candidates = radixSelectTopK(candidates, k)
			threshold = minCandidateValue(candidates)
		}
	}
	return radixSelectTopK(candidates, k)
}
