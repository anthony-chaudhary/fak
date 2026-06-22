// Tests for the pure, model-independent accounting helpers in radixbench:
// longest-common-prefix discovery, the declare-one-prefix reuse accounting,
// lexicographic (DFS) ordering, and the simple token tallies. These functions
// are deterministic and depend on no model, GPU, file, or network, so the
// expected values below are computed by hand from the request token streams.
package main

import "testing"

func TestLongestCommonPrefix(t *testing.T) {
	tests := []struct {
		name string
		reqs [][]int
		want int
	}{
		{"empty", nil, 0},
		{"empty slice", [][]int{}, 0},
		// Single request: the loop over reqs[1:] is empty, so lcp == len(reqs[0]).
		{"single", [][]int{{1, 2, 3, 4}}, 4},
		{"single empty", [][]int{{}}, 0},
		// All three share [9, 8, 7] then diverge.
		{"three-share-3", [][]int{{9, 8, 7, 1}, {9, 8, 7, 2, 2}, {9, 8, 7}}, 3},
		// First token already differs => no shared prefix.
		{"no-share", [][]int{{1, 2, 3}, {4, 5, 6}}, 0},
		// One request is a strict prefix of the other; lcp is bounded by the shorter.
		{"prefix-bound", [][]int{{1, 2, 3, 4, 5}, {1, 2}}, 2},
		// Identical requests share their full (equal) length.
		{"identical", [][]int{{5, 6, 7}, {5, 6, 7}}, 3},
		// One empty request forces lcp to 0 regardless of the others.
		{"one-empty", [][]int{{1, 2, 3}, {}}, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := longestCommonPrefix(tc.reqs); got != tc.want {
				t.Errorf("longestCommonPrefix(%v) = %d, want %d", tc.reqs, got, tc.want)
			}
		})
	}
}

func TestDeclaredMatched(t *testing.T) {
	tests := []struct {
		name        string
		reqs        [][]int
		wantLCP     int
		wantMatched int
	}{
		// matched = lcp * (len(reqs)-1): the prefix is reused by every request after the first.
		{"empty", nil, 0, 0},
		{"single", [][]int{{1, 2, 3}}, 3, 0}, // lcp=3 but (1-1)=0 reuses
		{"three-share-3", [][]int{{9, 8, 7, 1}, {9, 8, 7, 2}, {9, 8, 7}}, 3, 6}, // 3*(3-1)
		{"no-share", [][]int{{1}, {2}, {3}}, 0, 0},
		{"identical-pair", [][]int{{5, 6, 7}, {5, 6, 7}}, 3, 3}, // 3*(2-1)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			lcp, matched := declaredMatched(tc.reqs)
			if lcp != tc.wantLCP || matched != tc.wantMatched {
				t.Errorf("declaredMatched(%v) = (lcp=%d, matched=%d), want (lcp=%d, matched=%d)",
					tc.reqs, lcp, matched, tc.wantLCP, tc.wantMatched)
			}
		})
	}
}

func TestLexLess(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want bool
	}{
		{"first-elem-less", []int{1, 9}, []int{2, 0}, true},
		{"first-elem-greater", []int{2, 0}, []int{1, 9}, false},
		// Equal shared portion, a is shorter => a is less.
		{"shorter-is-less", []int{1, 2}, []int{1, 2, 3}, true},
		{"longer-is-not-less", []int{1, 2, 3}, []int{1, 2}, false},
		// Fully equal sequences => not strictly less (len(a) < len(b) is false).
		{"equal-not-less", []int{1, 2, 3}, []int{1, 2, 3}, false},
		// Divergence at a later index decides it.
		{"diverge-late", []int{1, 2, 3}, []int{1, 2, 4}, true},
		// Empty is less than any non-empty.
		{"empty-less", []int{}, []int{0}, true},
		{"empty-vs-empty", []int{}, []int{}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := lexLess(tc.a, tc.b); got != tc.want {
				t.Errorf("lexLess(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestMaxReqLenAndTotalTokens(t *testing.T) {
	tests := []struct {
		name      string
		reqs      [][]int
		wantMax   int
		wantTotal int
	}{
		{"empty", nil, 0, 0},
		{"single", [][]int{{1, 2, 3}}, 3, 3},
		{"mixed", [][]int{{1, 2}, {1, 2, 3, 4}, {1}}, 4, 7},
		{"with-empty", [][]int{{}, {7, 7}, {}}, 2, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := maxReqLen(tc.reqs); got != tc.wantMax {
				t.Errorf("maxReqLen(%v) = %d, want %d", tc.reqs, got, tc.wantMax)
			}
			if got := totalTokens(tc.reqs); got != tc.wantTotal {
				t.Errorf("totalTokens(%v) = %d, want %d", tc.reqs, got, tc.wantTotal)
			}
		})
	}
}
