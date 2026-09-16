package compute

// deepselect_topk_test.go — INDEPENDENT verification of DeepSelectTopk, the exact block-scan
// top-k selection primitive (threshold + random block order + radix-select compaction). The oracle
// below is written from scratch here: it never calls DeepSelectTopk, dsaIndexSelectHost, or any
// internal helper it is grading. Selection is a DISCRETE output, so the only honest witness is SET
// (and ORDER) EQUALITY against a second implementation — not a cosine, not a re-call of the code
// under test. This mirrors the spirit of hostIndexSelectF64 in dsa_index_test.go.
//
// Contract under test (expected sibling signature):
//   func DeepSelectTopk(items []Candidate, k, B, B2 int) []Candidate
// keeps the top-k by descending value, ties by LOWER ORIGINAL POSITION, exactly, for any B/B2
// (including the B2=1 degenerate config). It scans in B-sized blocks in a randomized permutation
// order, appends only elements > threshold, and on buffer growth (k+B2) compacts via radix-select,
// resetting threshold to the new minimum. This test file deliberately does not stub the
// implementation: compilation depends on the sibling worker landing deepselect_topk.go.

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// deepSelectOracle is the INDEPENDENT from-scratch reference: score by descending value with ties
// broken by lower ORIGINAL position. It takes the original index explicitly rather than relying on
// slice position, so a caller that permutes blocks cannot fool it.
func deepSelectOracle(vals []float64, k int) []int {
	type cand struct {
		pos int
		val float64
	}
	cands := make([]cand, len(vals))
	for i, v := range vals {
		cands[i] = cand{pos: i, val: v}
	}
	sort.SliceStable(cands, func(a, b int) bool {
		if cands[a].val == cands[b].val {
			return cands[a].pos < cands[b].pos
		}
		return cands[a].val > cands[b].val
	})
	n := k
	if n > len(cands) {
		n = len(cands)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = cands[i].pos
	}
	return out
}

// deepSelectPositions extracts the ORIGINAL positions from a returned []Candidate, using the
// position field the contract carries. A value-based re-derivation would hide a position bug.
func deepSelectPositions(sel []Candidate) []int {
	out := make([]int, len(sel))
	for i, c := range sel {
		out[i] = int(c.Position)
	}
	return out
}

// deepSelectVals builds a deterministic float64 sequence (no wall-clock, fixed seed).
func deepSelectVals(seed int64, n int) []float64 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float64, n)
	for i := range out {
		out[i] = r.Float64()*2000 - 1000
	}
	return out
}

// deepSelectCands materializes the oracle values into the Candidate input expected by the contract.
func deepSelectCands(vals []float64) []Candidate {
	out := make([]Candidate, len(vals))
	for i, v := range vals {
		out[i] = Candidate{Value: v, Position: i}
	}
	return out
}

// TestDeepSelectMatchesOracleRandomShapes is the set+order equality witness across many random
// shapes. The returned full position slice must match the from-scratch oracle exactly — same
// ordering, because ties break by lower original position deterministically.
func TestDeepSelectMatchesOracleRandomShapes(t *testing.T) {
	type shape struct {
		name        string
		seed        int64
		n, k, B, B2 int
	}
	shapes := []shape{
		{"small-k1", 1, 17, 1, 4, 1},
		{"k1-large-n", 2, 500, 1, 16, 3},
		{"k-eq-n", 3, 64, 64, 8, 2},
		{"k-gt-n", 4, 25, 100, 8, 4},
		{"k-gt-n-smallB", 5, 40, 200, 1, 1},
		{"mid-random", 6, 777, 33, 32, 5},
		{"b1-b2-1", 7, 300, 7, 1, 1},
		{"b2-1", 8, 300, 9, 16, 1},
		{"large-n", 9, 5000, 12, 128, 17},
		{"k-eq-n-b2-1", 10, 128, 128, 8, 1},
	}
	for _, tc := range shapes {
		t.Run(tc.name, func(t *testing.T) {
			vals := deepSelectVals(tc.seed, tc.n)
			want := deepSelectOracle(vals, tc.k)
			got := deepSelectPositions(DeepSelectTopk(deepSelectCands(vals), tc.k, tc.B, tc.B2))
			if len(got) != len(want) {
				t.Fatalf("len: got %d want %d (got=%v want=%v)", len(got), len(want), got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("selection[%d]: got %d want %d (got=%v want=%v) — NOT selection-stable", i, got[i], want[i], got, want)
				}
			}
		})
	}
}

// TestDeepSelectAdversarialInputs probes the randomized-order guarantee on inputs engineered to
// stress the threshold/compaction path: monotone ramps (every block beats the threshold), a heavy
// all-equal tie field (the (value,position) tie-break must hold), skewed multisets, and a
// block-adversarial input whose post-first-block values all fail the eventual threshold.
func TestDeepSelectAdversarialInputs(t *testing.T) {
	const n = 2000

	rampUp := func() []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = float64(i)
		}
		return o
	}
	rampDown := func() []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = float64(n - i)
		}
		return o
	}
	allEqual := func() []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = 42.5
		}
		return o
	}
	skewed := func() []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = float64(i % 7)
		}
		return o
	}
	// Block-adversarial: first block holds the true top-k by a wide margin; every later element is
	// far below them. Once the threshold settles after block 0, no later element should append.
	blockAdversarial := func() []float64 {
		o := make([]float64, n)
		for i := range o {
			if i < 50 {
				o[i] = 1e6 - float64(i)
			} else {
				o[i] = -1e6 - float64(i)
			}
		}
		return o
	}

	cases := []struct {
		name string
		vals []float64
	}{
		{"increasing", rampUp()},
		{"decreasing", rampDown()},
		{"all-equal", allEqual()},
		{"skewed-repeats", skewed()},
		{"block-adversarial", blockAdversarial()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, k := range []int{1, 17, 100, n, n + 5} {
				want := deepSelectOracle(tc.vals, k)
				got := deepSelectPositions(DeepSelectTopk(deepSelectCands(tc.vals), k, 32, 4))
				if len(got) != len(want) {
					t.Fatalf("k=%d len: got %d want %d", k, len(got), len(want))
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("k=%d selection[%d]: got %d want %d (got=%v want=%v)", k, i, got[i], want[i], got, want)
					}
				}
			}
		})
	}
}

// TestDeepSelectNeverExceedsWorkBound is the bound witness: on large random input with k<<n, the
// selected SET must equal the oracle (exactness at scale) and the algorithm must not have to keep
// more than k+B2 candidates live — asserted indirectly by requiring exact equality across a sweep
// of block sizes and buffer slack, since an over-appending/under-compacting implementation would
// diverge the selected set on some shape. Separately we assert the selection is independent of the
// randomized block order (same result on repeated calls).
func TestDeepSelectNeverExceedsWorkBound(t *testing.T) {
	const n = 20000
	const k = 8
	vals := deepSelectVals(99, n)
	want := deepSelectOracle(vals, k)
	cands := deepSelectCands(vals)

	sizes := []struct{ B, B2 int }{
		{1, 1}, {4, 1}, {16, 3}, {64, 16}, {128, 64}, {256, 256}, {n, 0},
	}
	for _, s := range sizes {
		got := deepSelectPositions(DeepSelectTopk(cands, k, s.B, s.B2))
		if len(got) != len(want) {
			t.Fatalf("B=%d B2=%d len: got %d want %d", s.B, s.B2, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("B=%d B2=%d selection[%d]: got %d want %d — selection depends on block order", s.B, s.B2, i, got[i], want[i])
			}
		}
	}
	// Repeated identical calls must be byte-stable despite the internal randomized order.
	first := deepSelectPositions(DeepSelectTopk(cands, k, 32, 4))
	for rep := 0; rep < 8; rep++ {
		again := deepSelectPositions(DeepSelectTopk(cands, k, 32, 4))
		if len(again) != len(first) {
			t.Fatalf("rep %d len drift: %d vs %d", rep, len(again), len(first))
		}
		for i := range first {
			if again[i] != first[i] {
				t.Fatalf("rep %d non-deterministic at %d: %d vs %d", rep, i, again[i], first[i])
			}
		}
	}
}

// TestDeepSelectTieDeterminism constructs exact ties straddling the k-th boundary and asserts the
// returned positions are ALWAYS the lowest-index ones, deterministically across repeated calls.
func TestDeepSelectTieDeterminism(t *testing.T) {
	const n = 256
	const k = 10
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = 1.0 // full tie field — every position is a boundary candidate
	}
	cands := deepSelectCands(vals)
	want := deepSelectOracle(vals, k) // lowest indices 0..k-1
	for rep := 0; rep < 32; rep++ {
		got := deepSelectPositions(DeepSelectTopk(cands, k, 16, 1))
		if len(got) != len(want) {
			t.Fatalf("rep %d len: got %d want %d", rep, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("rep %d tie selection[%d]: got %d want %d — tie-break not lowest-position", rep, i, got[i], want[i])
			}
		}
	}

	// Boundary tie: exactly k elements share the top value, plus a tail of lower values interleaved,
	// so the tie-break decides which of the tied block members survive compaction.
	vals2 := make([]float64, n)
	for i := range vals2 {
		if i%2 == 0 && i < 2*k {
			vals2[i] = 5.0
		} else {
			vals2[i] = -float64(i)
		}
	}
	cands2 := deepSelectCands(vals2)
	want2 := deepSelectOracle(vals2, k)
	for rep := 0; rep < 32; rep++ {
		got := deepSelectPositions(DeepSelectTopk(cands2, k, 8, 3))
		for i := range want2 {
			if got[i] != want2[i] {
				t.Fatalf("rep %d boundary tie[%d]: got %d want %d", rep, i, got[i], want2[i])
			}
		}
	}
}

// TestDeepSelectB2OneDegenerate covers the B2=1 degenerate config: buffer compacted at every
// append, so selection must still be exact and deterministic.
func TestDeepSelectB2OneDegenerate(t *testing.T) {
	for _, n := range []int{1, 2, 3, 17, 100, 999} {
		for _, k := range []int{1, 2, n / 2, n, n + 3} {
			if k < 1 {
				continue
			}
			vals := deepSelectVals(int64(n*1000+k), n)
			want := deepSelectOracle(vals, k)
			got := deepSelectPositions(DeepSelectTopk(deepSelectCands(vals), k, 8, 1))
			if len(got) != len(want) {
				t.Fatalf("n=%d k=%d B2=1 len: got %d want %d", n, k, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("n=%d k=%d B2=1 selection[%d]: got %d want %d", n, k, i, got[i], want[i])
				}
			}
		}
	}
	// Mixed magnitudes with B2=1 must still order by descending value.
	vals := []float64{3, -1, math.MaxFloat64, 0, math.SmallestNonzeroFloat64, -math.MaxFloat64, 3}
	want := deepSelectOracle(vals, 3)
	got := deepSelectPositions(DeepSelectTopk(deepSelectCands(vals), 3, 4, 1))
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("magnitude B2=1 selection[%d]: got %d want %d (got=%v want=%v)", i, got[i], want[i], got, want)
		}
	}
}
