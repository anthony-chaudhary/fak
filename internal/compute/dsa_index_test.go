package compute

import (
	"math"
	"sort"
	"testing"
)

// hostIndexSelectF64 is an INDEPENDENT reference (not the package's dsaIndexSelectHost) for the
// GLM-MoE-DSA indexer score + top-k: score(k) = Σ_h weights[h]·relu(scale·dot(q_h, k)) in f64, then
// the top-k positions by score descending, ties by lower position. Written from scratch here so the
// selection-equality test does not just compare a function to itself — it pins the cpu-ref
// DSAIndexSelect against a second, differently-shaped implementation of the same math.
func hostIndexSelectF64(indexQ, indexK, weights []float32, nKeys, nH, indexDim, queryPos, topK int, scale float32) []int {
	type sk struct {
		pos int
		sc  float64
	}
	var cands []sk
	for k := 0; k <= queryPos && k < nKeys; k++ {
		var score float64
		for h := 0; h < nH; h++ {
			var dot float64
			for d := 0; d < indexDim; d++ {
				dot += float64(indexQ[h*indexDim+d]) * float64(indexK[k*indexDim+d])
			}
			hs := dot * float64(scale)
			if hs < 0 {
				hs = 0
			}
			score += float64(weights[h]) * hs
		}
		cands = append(cands, sk{pos: k, sc: score})
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].sc == cands[j].sc {
			return cands[i].pos < cands[j].pos
		}
		return cands[i].sc > cands[j].sc
	})
	n := topK
	if n > len(cands) {
		n = len(cands)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = cands[i].pos
	}
	return out
}

// TestCPURefDSAIndexSelectMatchesHostF64 is the selection-stability witness on the cpu-ref: the
// optional DSAIndexBackend.DSAIndexSelect returns EXACTLY the top-k key positions an independent f64
// reference selects, across a sweep of dims/positions including engineered near-ties. This is the
// floor the device kernel (k_dsa_index_score + k_dsa_index_topk) is held to: not a cosine, but SET
// EQUALITY — the indexer drives a discrete selection, so a single flipped position would diverge the
// forward. The cuda backend reproduces this exact selection on real hardware (the device accumulates
// the score dot in f64, so the selected set is bit-identical); that on-device run is the dgx witness.
func TestCPURefDSAIndexSelectMatchesHostF64(t *testing.T) {
	be := Default()
	ib, ok := be.(DSAIndexBackend)
	if !ok {
		t.Fatalf("cpu-ref backend %q does not implement DSAIndexBackend", be.Name())
	}
	scale := float32(1.0 / math.Sqrt(8))
	cases := []struct {
		name            string
		nH, indexDim    int
		nKeys, queryPos int
		topK            int
		tie             bool // engineer two keys to the same score (exercises the tie-break)
	}{
		{"small", 4, 8, 6, 5, 2, false},
		{"topk-ge-keys", 4, 8, 3, 2, 8, false},
		{"causal-mask", 4, 8, 10, 4, 3, false},
		{"near-tie", 2, 4, 8, 7, 3, true},
		{"single-head", 1, 16, 12, 11, 5, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Deterministic pseudo-random operands (no rand: keep the test reproducible).
			indexQ := make([]float32, tc.nH*tc.indexDim)
			for i := range indexQ {
				indexQ[i] = float32(math.Sin(float64(i)*0.7+1.3)) * 0.5
			}
			indexK := make([]float32, tc.nKeys*tc.indexDim)
			for i := range indexK {
				indexK[i] = float32(math.Cos(float64(i)*0.37+0.2)) * 0.5
			}
			weights := make([]float32, tc.nH)
			for h := range weights {
				weights[h] = float32(0.3 + 0.1*float64(h))
			}
			if tc.tie {
				// Make key 2 a byte-for-byte copy of key 1 so they score identically — the
				// selection must keep the LOWER position first (the dsaTopKIndices tie-break).
				copy(indexK[2*tc.indexDim:3*tc.indexDim], indexK[1*tc.indexDim:2*tc.indexDim])
			}

			want := hostIndexSelectF64(indexQ, indexK, weights, tc.nKeys, tc.nH, tc.indexDim, tc.queryPos, tc.topK, scale)

			qt := be.Upload(NewF32(be, []int{tc.nH * tc.indexDim}, indexQ), F32)
			kt := be.Upload(NewF32(be, []int{tc.nKeys * tc.indexDim}, indexK), F32)
			wt := be.Upload(NewF32(be, []int{tc.nH}, weights), F32)
			got := ib.DSAIndexSelect(qt, kt, wt, tc.nKeys, tc.nH, tc.indexDim, tc.queryPos, tc.topK, scale)

			if len(got) != len(want) {
				t.Fatalf("selection length: got %d want %d (got=%v want=%v)", len(got), len(want), got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("selection[%d]: got %d want %d (full got=%v want=%v) — NOT selection-stable", i, got[i], want[i], got, want)
				}
			}
			// Every selected position must be causal and unique (defense in depth).
			seen := map[int]bool{}
			for _, p := range got {
				if p < 0 || p > tc.queryPos {
					t.Fatalf("selected non-causal position %d (queryPos=%d)", p, tc.queryPos)
				}
				if seen[p] {
					t.Fatalf("selected duplicate position %d", p)
				}
				seen[p] = true
			}
			t.Logf("cpu-ref DSAIndexSelect %s: selection==host-f64 exactly: %v", tc.name, got)
		})
	}
}
