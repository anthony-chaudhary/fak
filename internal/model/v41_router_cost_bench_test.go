package model

// Benchmark-only file. It changes no router behaviour: the default selection
// path remains v4TopKIndices -> the reference O(E log E) stable sort, and this
// file neither alters v41_router.go nor v4_router.go. Its purpose is to measure
// the full O(E log E) sort against a genuine O(E log k) bounded max-heap
// partial selection at the admitted V4.1 geometry (E=384, top-6), and to attes
// the moment-in-time cost delta.
//
// The comparator is the reference stable-sort arm, NOT the bitonic kernel:
// internal/compute's bitonic network is a *full* sort that pads 384 -> 512 and
// performs ~11520 compare-exchanges, so it is not an O(E log k) partial select
// and is deliberately excluded here (its own arm lives in
// BenchmarkV4TopKIndicesE384 in v4_router_test.go).

import (
	"encoding/json"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// v41RouterPartialTopKIndices returns the expert indices of the k largest
// selection scores, in the router's pinned order: descending score, lower
// expert index first on ties.
//
// It is a genuine O(E log k) partial selection: a bounded min-heap of size k
// keeps the k best candidates seen so far, so each of the E elements costs at
// most a log k sift rather than the reference's full O(E log E) sort.
//
// The pinned tie-break is preserved by making the heap "better" relation match
// the reference exactly. sort.SliceStable over an index list initially in
// ascending index order keeps equal-score experts in ascending index order; a
// candidate is therefore strictly better than the heap's current worst iff it
// has a higher score, or an equal score and a lower index. The heap root is
// that worst element, so the final result is dumped in descending order under
// the same relation.
//
// Degenerate k (k<=0 or k>len(choice)) mirrors v4TopKIndices's fail-closed
// contract by returning the full reference ordering rather than a short or
// overlong slice.
func v41RouterPartialTopKIndices(choice []float32, k int) []int {
	if k <= 0 || k > len(choice) {
		indices := make([]int, len(choice))
		for i := range indices {
			indices[i] = i
		}
		sort.SliceStable(indices, func(i, j int) bool {
			return choice[indices[i]] > choice[indices[j]]
		})
		return indices
	}

	// partialBetter reports whether index a ranks strictly ahead of index b in
	// the pinned order (descending score, lower index first on ties).
	partialBetter := func(a, b int) bool {
		sa, sb := choice[a], choice[b]
		if sa != sb {
			return sa > sb
		}
		return a < b
	}

	// heap holds the current k best indices as a min-heap under partialBetter:
	// heap[0] (the root) is the worst of the retained candidates.
	heap := make([]int, 0, k)
	siftUp := func(at int) {
		for at > 0 {
			parent := (at - 1) / 2
			if !partialBetter(heap[parent], heap[at]) {
				break
			}
			heap[parent], heap[at] = heap[at], heap[parent]
			at = parent
		}
	}
	siftDown := func(at int) {
		for {
			left, right := 2*at+1, 2*at+2
			worst := at
			if left < len(heap) && partialBetter(heap[worst], heap[left]) {
				worst = left
			}
			if right < len(heap) && partialBetter(heap[worst], heap[right]) {
				worst = right
			}
			if worst == at {
				return
			}
			heap[at], heap[worst] = heap[worst], heap[at]
			at = worst
		}
	}

	for i := 0; i < len(choice); i++ {
		if len(heap) < k {
			heap = append(heap, i)
			siftUp(len(heap) - 1)
			continue
		}
		// Replace the root only when the new candidate is strictly better than
		// the retained worst; equal candidates lose to the lower index already
		// in the heap, which is the stable-sort tie-break.
		if partialBetter(i, heap[0]) {
			heap[0] = i
			siftDown(0)
		}
	}

	// The heap holds the correct k-set but with the worst at the root. Order
	// the k retained candidates into the pinned sequence. k is bounded (top-k),
	// so this final ordering is O(k log k) and does not change the O(E log k)
	// selection bound.
	out := append([]int(nil), heap...)
	sort.SliceStable(out, func(i, j int) bool {
		return partialBetter(out[i], out[j])
	})
	return out
}

// TestV41RouterCostTopKSetEquality is the SW-VERIFIED equality witness for the
// cost benchmark: over deterministic seeded logits and tie-heavy fixtures, the
// O(E log k) partial selection must return the exact same top-k index sequence
// as the default reference arm at E=384, K=6.
func TestV41RouterCostTopKSetEquality(t *testing.T) {
	// The reference arm is the public v4TopKIndices with the bitonic flag
	// unset; we never opt the kernel in, so this witnesses the default path.
	if v4BitonicTopKEnabled() {
		t.Fatal("reference arm must run with FAK_V4_BITONIC_TOPK unset")
	}

	rng := rand.New(rand.NewSource(12992))
	fixtures := map[string][]float32{}
	for sample := 0; sample < 8; sample++ {
		choice := make([]float32, V41RouterExperts)
		for i := range choice {
			choice[i] = rng.Float32()
		}
		fixtures["seeded_uniform_"+string(rune('a'+sample))] = choice
	}

	// Tie-heavy fixtures: repeated values exercise the exact tie-break the
	// stable sort pins, where a naive partial select most easily diverges.
	allEqual := make([]float32, V41RouterExperts)
	for i := range allEqual {
		allEqual[i] = 2.5
	}
	fixtures["all_equal"] = allEqual

	twoTier := make([]float32, V41RouterExperts)
	for i := range twoTier {
		if i%3 == 0 {
			twoTier[i] = 7
		} else {
			twoTier[i] = 1
		}
	}
	fixtures["two_tier_repeats"] = twoTier

	boundaryTie := make([]float32, V41RouterExperts)
	for i := range boundaryTie {
		boundaryTie[i] = float32(i % 5)
	}
	fixtures["mod5_repeats"] = boundaryTie

	for name, choice := range fixtures {
		t.Run(name, func(t *testing.T) {
			ref := v4TopKIndices(choice, V41RouterTopK)[:V41RouterTopK]
			got := v41RouterPartialTopKIndices(choice, V41RouterTopK)
			if !reflect.DeepEqual(got, ref) {
				t.Fatalf("partial-select order %v != reference top-k %v", got, ref)
			}
		})
	}

	// Degenerate k must mirror the reference fail-closed width contract.
	small := []float32{3, 1, 3, 2}
	for _, k := range []int{0, -1, len(small) + 4} {
		ref := v4TopKIndices(small, k)
		got := v41RouterPartialTopKIndices(small, k)
		if !reflect.DeepEqual(got, ref) {
			t.Fatalf("degenerate k=%d partial %v != reference %v", k, got, ref)
		}
	}
}

// v41RouterCostReceipt is the compact moment-in-time attestation the benchmark
// records after timing both arms. It deliberately carries no speedup verdict.
type v41RouterCostReceipt struct {
	Schema          string  `json:"schema"`
	Experts         int     `json:"experts"`
	TopK            int     `json:"topk"`
	FullSortNs      int64   `json:"full_sort_ns"`
	PartialSelectNs int64   `json:"partial_select_ns"`
	Ratio           float64 `json:"ratio"`
}

// v41RouterCostReceiptFrom folds the two measured arm timings into the compact
// attestation the benchmark logs. It is a pure function of the measured
// ns/op values, so the receipt states only what the run actually observed.
func v41RouterCostReceiptFrom(fullSortNs, partialSelectNs int64) v41RouterCostReceipt {
	receipt := v41RouterCostReceipt{
		Schema:          "fak/router-cost-ab/1",
		Experts:         V41RouterExperts,
		TopK:            V41RouterTopK,
		FullSortNs:      fullSortNs,
		PartialSelectNs: partialSelectNs,
	}
	if fullSortNs > 0 {
		receipt.Ratio = float64(partialSelectNs) / float64(fullSortNs)
	}
	return receipt
}

// BenchmarkV41RouterTopKCostE384 is a paired benchmark of the full O(E log E)
// stable sort against the O(E log k) bounded-heap partial select at the
// admitted E=384, K=6 geometry. The seed differs from
// BenchmarkV4TopKIndicesE384 (12970) on purpose so the two benches sample
// distinct logits. After both arms run it logs a compact receipt built from the
// measured ns/op values; it reports the ratio and does NOT assert a speedup
// direction.
func BenchmarkV41RouterTopKCostE384(b *testing.B) {
	const E, K = V41RouterExperts, V41RouterTopK
	rng := rand.New(rand.NewSource(12992))
	choice := make([]float32, E)
	for i := range choice {
		choice[i] = rng.Float32()
	}

	var fullNs, partialNs int64
	b.Run("reference_full_sort", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = v4TopKIndices(choice, K)
		}
		if b.N > 0 {
			fullNs = b.Elapsed().Nanoseconds() / int64(b.N)
		}
	})
	b.Run("partial_select", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = v41RouterPartialTopKIndices(choice, K)
		}
		if b.N > 0 {
			partialNs = b.Elapsed().Nanoseconds() / int64(b.N)
		}
	})

	if receipt, err := json.Marshal(v41RouterCostReceiptFrom(fullNs, partialNs)); err == nil {
		b.Logf("%s", receipt)
	}
}
