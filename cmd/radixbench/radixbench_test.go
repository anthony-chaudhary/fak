// Tests for the pure, model-independent accounting helpers in radixbench:
// longest-common-prefix discovery, the declare-one-prefix reuse accounting,
// lexicographic (DFS) ordering, and the simple token tallies. These functions
// are deterministic and depend on no model, GPU, file, or network, so the
// expected values below are computed by hand from the request token streams.
package main

import (
	"os"
	"path/filepath"
	"testing"
)

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

// TestLoadWorkload is the witness for the #322 adoption path: an operator's own
// token-id prompt set, loaded from JSON, becomes a Workload the same accounting runs
// over. It writes a temp file, loads it, and checks the requests + metadata + the
// name-from-filename fallback survive the round trip — no model, file, or network.
func TestLoadWorkload(t *testing.T) {
	dir := t.TempDir()

	// A workload with explicit metadata and a shared 3-token prefix.
	named := filepath.Join(dir, "few-shot.json")
	if err := os.WriteFile(named, []byte(
		`{"name":"few-shot","desc":"shared preamble","sglang_published":"50-99% band",`+
			`"requests":[[1,2,3,10],[1,2,3,11],[1,2,3,12]]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := loadWorkload(named)
	if err != nil {
		t.Fatalf("loadWorkload: %v", err)
	}
	if w.Name != "few-shot" || w.Desc != "shared preamble" || w.SGLang != "50-99% band" {
		t.Errorf("metadata not preserved: %+v", w)
	}
	if len(w.Requests) != 3 || totalTokens(w.Requests) != 12 {
		t.Errorf("requests not loaded verbatim: %v", w.Requests)
	}
	// The shared [1,2,3] prefix must be discoverable by the same radix accounting.
	if matched, _, _ := radixMatched(w.Requests, 0); matched != 6 { // 3 reused by reqs 2 and 3
		t.Errorf("radix reuse over loaded workload = %d, want 6", matched)
	}

	// Name defaults to the base filename when the JSON omits one.
	noName := filepath.Join(dir, "agents.json")
	if err := os.WriteFile(noName, []byte(`{"requests":[[7,8],[7,9]]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	w2, err := loadWorkload(noName)
	if err != nil {
		t.Fatalf("loadWorkload (no name): %v", err)
	}
	if w2.Name != "agents" {
		t.Errorf("name fallback = %q, want %q", w2.Name, "agents")
	}

	// An empty / malformed workload is a clear error, not a silent zero-request run.
	empty := filepath.Join(dir, "empty.json")
	if err := os.WriteFile(empty, []byte(`{"requests":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadWorkload(empty); err == nil {
		t.Error("loadWorkload accepted a zero-request workload, want error")
	}
	if _, err := loadWorkload(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("loadWorkload accepted a missing file, want error")
	}
}

func TestMaxTokenID(t *testing.T) {
	ws := []Workload{
		{Requests: [][]int{{1, 2}, {3, 255}}},
		{Requests: [][]int{{4}, {300, 5}}},
	}
	if got := maxTokenID(ws); got != 300 {
		t.Errorf("maxTokenID = %d, want 300", got)
	}
	if got := maxTokenID(nil); got != -1 {
		t.Errorf("maxTokenID(nil) = %d, want -1 (no ids)", got)
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
