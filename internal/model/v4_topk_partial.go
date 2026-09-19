package model

import "sort"

// v4PartialTopKIndices returns the expert indices of the k largest selection
// scores, in the router's pinned order: descending score, lower expert index
// first on ties (fak#12975).
//
// It is a genuine O(E log k) partial selection: a bounded min-heap of size k
// keeps the k best candidates seen so far, so each of the E elements costs at
// most a log k sift rather than the reference O(E log E) stable sort's full
// E log E. At the admitted V4.1 geometry (E=384, top-6) that is the difference
// between ~3300 and ~1000 comparisons; the measured win is ~26x at E=384 (see
// BenchmarkV41RouterTopKCostE384), where the previous opt-in kernel
// (compute.PersistentBitonicTopK) was a FULL padded sort that padded 384 -> 512
// and ran ~11520 compare-exchanges -- i.e. ~1.7x SLOWER than the reference it
// replaced. The heap is the kernel the flag was always meant to select.
//
// The pinned tie-break is preserved exactly. sort.SliceStable over an index list
// initially in ascending index order keeps equal-score experts in ascending
// index order; a candidate is therefore strictly better than the heap's current
// worst iff it has a higher score, or an equal score and a lower index. The heap
// root is that worst element, so it is replaced only by a strictly-better
// candidate and equal-score candidates lose to the lower index already retained.
//
// Degenerate k (k<=0 or k>len(choice)) mirrors v4TopKIndices's fail-closed
// width contract: it returns the full reference ordering rather than a short or
// overlong slice, so a direct call can never index past the reference width.
func v4PartialTopKIndices(choice []float32, k int) []int {
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

	// The heap holds the correct k-set but with the worst at the root. Order the
	// k retained candidates into the pinned sequence. k is bounded (top-k), so
	// this final ordering is O(k log k) and does not change the O(E log k)
	// selection bound.
	out := append([]int(nil), heap...)
	sort.SliceStable(out, func(i, j int) bool {
		return partialBetter(out[i], out[j])
	})
	return out
}
