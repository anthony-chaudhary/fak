package compute

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

// dsaRadixKeyF64 mirrors k_dsa_index_topk's dsaRadixKey (cuda/spec_graph.cuh): the monotone uint64
// key of an f64 score — flip the sign bit for non-negative values, flip every bit for negative
// values — so the key's unsigned order equals the score's numeric order and equal scores share a
// key. Finite scores only: the scoring lane seeds +0.0 and accumulates finite f32 products, so
// NaN/-0.0 never reach the key (the same assumption the device kernel carries).
func dsaRadixKeyF64(v float64) uint64 {
	b := math.Float64bits(v)
	if b&(1<<63) != 0 {
		return ^b
	}
	return b | (1 << 63)
}

// dsaRadixSelectMirror transcribes the k_dsa_index_topk selection pass (cuda/spec_graph.cuh) for
// the CPU: (1) MSB-first 8-round window narrowing — 256-bucket byte histogram, suffix-scan for the
// boundary bucket, narrow the active key window; (2) gather the strictly-above-boundary items plus
// the lowest-position boundary items (the position-ascending scan fills the tie slots first — the
// dsaTopKIndices tie-break); (3) the in-place bitonic ordering over the <= topK selected pairs
// under the total order (value desc, position asc). It returns the selected POSITIONS in that exact
// order. The transcription collapses the kernel's atomic slot allocation to a sequential scan (same
// phases, same decisions, same closed contract); the dgx witness
// (TestCUDAGLMMoeDsaIndexSelectMatches) holds the on-device kernel to this same order. This is the
// software-verifiable half of the cudaDsaIndexSelectionExact=true contract.
func dsaRadixSelectMirror(scores []float64, nValid, topK int) []int {
	want := topK
	if want > nValid {
		want = nValid
	}
	if want <= 0 {
		return nil
	}
	// Phase 1: MSB-first window narrowing. `remaining` counts how many keys are still owed FROM the
	// current window [lo, lo+wsize) (wsize = 2^(s+8); wsize == 0 denotes the full key range at the
	// first round, since 2^64 is unrepresentable). Invariant: the window holds at least `remaining`
	// keys and every key above the window is already accounted for as selected. Each round narrows to
	// the boundary bucket that carries the next-owed key and credits the buckets above it.
	remaining := want
	var lo uint64
	for s := 56; s >= 0; s -= 8 {
		var hist [256]int
		var wsize uint64
		if s != 56 {
			wsize = uint64(1) << (s + 8)
		}
		for i := 0; i < nValid; i++ {
			k := dsaRadixKeyF64(scores[i])
			if wsize == 0 || k-lo < wsize {
				hist[k>>uint(s)&0xFF]++
			}
		}
		running, bb := 0, 0
		for b := 255; b >= 0; b-- {
			running += hist[b]
			if running >= remaining {
				bb = b
				break
			}
		}
		lo += uint64(bb) << uint(s)
		remaining -= running - hist[bb] // keys in the buckets above the boundary are all selected
	}
	kStar := lo
	// Phase 2: gather — above-boundary items plus the lowest-position boundary items.
	cAbove := 0
	for i := 0; i < nValid; i++ {
		if dsaRadixKeyF64(scores[i]) > kStar {
			cAbove++
		}
	}
	type pair struct {
		v float64
		p int
	}
	P := 1
	for P < want {
		P <<= 1
	}
	// Two slot counters, exactly as the kernel's sScalars[2]/sScalars[3]: above-boundary items own
	// slots [0, cAbove); the position-ascending scan fills the boundary items into
	// [cAbove, cAbove+fillCap) — the lowest positions among ties. One shared counter would let an
	// early tie steal a slot a later above-boundary item still needs.
	buf := make([]pair, P)
	fillCap := want - cAbove
	aboveSlot, eqSlot := 0, 0
	for i := 0; i < nValid; i++ {
		k := dsaRadixKeyF64(scores[i])
		switch {
		case k > kStar:
			buf[aboveSlot] = pair{scores[i], i}
			aboveSlot++
		case k == kStar:
			if eqSlot < fillCap {
				buf[cAbove+eqSlot] = pair{scores[i], i}
				eqSlot++
			}
		}
	}
	for i := want; i < P; i++ { // rank-last sentinels sink below every real pair
		buf[i] = pair{math.Inf(-1), 0}
	}
	// Phase 3: in-place bitonic ordering (value desc, position asc); index 0 holds the best pair.
	for kk := 2; kk <= P; kk <<= 1 {
		for j := kk >> 1; j > 0; j >>= 1 {
			for p := 0; p < P; p++ {
				if p&j != 0 {
					continue
				}
				b := p ^ j
				va, vb := buf[p].v, buf[b].v
				aBeforeB := va > vb || (va == vb && buf[p].p < buf[b].p)
				asc := (p/kk)&1 == 0
				if (asc && !aBeforeB) || (!asc && aBeforeB) {
					buf[p], buf[b] = buf[b], buf[p]
				}
			}
		}
	}
	out := make([]int, want)
	for i := 0; i < want; i++ {
		out[i] = buf[i].p
	}
	return out
}

// dsaRadixOracle is the INDEPENDENT reference for the pinned shapes: a plain stable sort over
// (score descending, position ascending) — the closed total order the selection must realize. It
// shares no machinery with dsaRadixSelectMirror (no radix key, no histogram, no bitonic network),
// so agreement between the two pins the borrowed behavior rather than re-calling it.
func dsaRadixOracle(scores []float64, nValid, topK int) []int {
	want := topK
	if want > nValid {
		want = nValid
	}
	if want <= 0 {
		return nil
	}
	idx := make([]int, nValid)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		sa, sb := scores[idx[a]], scores[idx[b]]
		if sa != sb {
			return sa > sb
		}
		return idx[a] < idx[b]
	})
	return idx[:want]
}

// TestCUDADSAIndexSelectRadixMatchesHostF64 pins the borrowed single-pass radix/bucket selection
// pass (k_dsa_index_topk, ADAPTED from vLLM's persistent_topk_kernel / FilteredTopKUnifiedKernel,
// csrc/persistent_topk.cuh:868/:1295 @ ae065918640a198a56f6f17eb45e251c3229fe41) that replaced the
// repeated-max topK-times selection capped at DSA_TOPK_MAX_KEYS=4096. Two halves, both
// software-verifiable (CPU/in-memory):
//
//   - STRUCTURAL closed contract on the production artifact (cuda/spec_graph.cuh): the shared
//     taken[4096] mask and the DSA_TOPK_MAX_KEYS nKeys cap are gone; the radix key helper is in
//     their place. Exact-string sentinels (the build-tag prefix checks in
//     cuda_graph_once_config_contract_test.go are the precedent). This half fails on the parent
//     commit, where the capped repeated-max pass is still the selection.
//   - BEHAVIORAL pin: the Go-side transcription of the pass (dsaRadixSelectMirror) realizes the
//     exact total order (score desc, ties by lower position — the dsaTopKIndices order) against the
//     independent f64 oracle on the three contract shapes — (a) nKeys=4096/topK=1024 at the old cap
//     boundary the device previously declined, (b) nKeys=65536/topK=1024 deep context, (c)
//     all-scores-equal adversarial tie -> positions exactly 0..topK-1 — plus causal-mask, heavy-tie
//     and magnitude-extreme extras.
//
// [SW-VERIFIED]: no GPU and no throughput claim here; the on-device run is the dgx witness
// (TestCUDAGLMMoeDsaIndexSelectMatches) held to this same closed order.
func TestCUDADSAIndexSelectRadixMatchesHostF64(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("cuda", "spec_graph.cuh"))
	if err != nil {
		t.Fatalf("read cuda/spec_graph.cuh: %v", err)
	}
	text := string(src)
	for _, present := range []string{"dsaRadixKey(", "k_dsa_index_topk"} {
		if !strings.Contains(text, present) {
			t.Fatalf("spec_graph.cuh lost the radix selection pass: missing %q", present)
		}
	}
	for _, absent := range []string{
		"#define DSA_TOPK_MAX_KEYS",
		"__shared__ char taken[4096]",
		"if (nKeys > DSA_TOPK_MAX_KEYS)",
	} {
		if strings.Contains(text, absent) {
			t.Fatalf("spec_graph.cuh still carries the capped repeated-max selection: %q", absent)
		}
	}

	randScores := func(seed float64, n int) []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = math.Sin(float64(i)*0.7+seed)*1e6 + math.Cos(float64(i)*0.31+seed)*1e3
		}
		return o
	}
	const topK = 1024
	allEqual := func(n int) []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = 42.5
		}
		return o
	}
	heavyTies := func(n int) []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = float64((i * 7919) % 13)
		}
		return o
	}
	extremes := func(n int) []float64 {
		o := make([]float64, n)
		for i := range o {
			switch i % 4 {
			case 0:
				o[i] = math.MaxFloat64
			case 1:
				o[i] = -math.MaxFloat64
			case 2:
				o[i] = math.SmallestNonzeroFloat64
			default:
				o[i] = -math.SmallestNonzeroFloat64 * float64(i%7+1)
			}
		}
		return o
	}
	cases := []struct {
		name            string
		nKeys, queryPos int
		topK            int
		scores          []float64
	}{
		{"cap-boundary-4096", 4096, 4095, topK, randScores(1.3, 4096)},
		{"deep-context-65536", 65536, 65535, topK, randScores(2.7, 65536)},
		{"all-equal-tie", 65536, 65535, topK, allEqual(65536)},
		{"causal-mask-half", 65536, 32767, topK, randScores(4.1, 65536)},
		{"heavy-ties-straddle", 4096, 4095, topK, heavyTies(4096)},
		{"topk-ge-nvalid", 512, 511, 600, randScores(5.9, 512)},
		{"magnitude-extremes", 4096, 4095, topK, extremes(4096)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			nValid := tc.queryPos + 1
			want := dsaRadixOracle(tc.scores, nValid, tc.topK)
			got := dsaRadixSelectMirror(tc.scores, nValid, tc.topK)
			if len(got) != len(want) {
				t.Fatalf("selection length: got %d want %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("selection[%d]: got %d want %d — NOT the (score desc, position asc) total order", i, got[i], want[i])
				}
			}
			// Defense in depth: every selected position is causal and unique.
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
		})
	}

	// Contract shape (c) asserted LITERALLY: the all-equal field must return positions 0..topK-1.
	eq := make([]float64, 65536)
	gotEq := dsaRadixSelectMirror(eq, 65536, topK)
	if len(gotEq) != topK {
		t.Fatalf("all-equal selection length: got %d want %d", len(gotEq), topK)
	}
	for i := 0; i < topK; i++ {
		if gotEq[i] != i {
			t.Fatalf("all-equal selection[%d]: got %d want %d (ties must land on the lowest positions)", i, gotEq[i], i)
		}
	}

	// Go-side seam: the host fallback behind the DSAIndexBackend capability agrees with the pinned
	// contract at the old cap boundary (nKeys=4096, topK=1024 — previously a device decline).
	be := Default()
	ib, ok := be.(DSAIndexBackend)
	if !ok {
		t.Fatalf("default backend %q does not implement DSAIndexBackend", be.Name())
	}
	scale := float32(1.0 / math.Sqrt(8))
	const nH, indexDim, nKeys, queryPos = 4, 8, 4096, 4095
	indexQ := make([]float32, nH*indexDim)
	for i := range indexQ {
		indexQ[i] = float32(math.Sin(float64(i)*0.7+1.3)) * 0.5
	}
	indexK := make([]float32, nKeys*indexDim)
	for i := range indexK {
		indexK[i] = float32(math.Cos(float64(i)*0.37+0.2)) * 0.5
	}
	weights := make([]float32, nH)
	for h := range weights {
		weights[h] = float32(0.3 + 0.1*float64(h))
	}
	wantSeam := hostIndexSelectF64(indexQ, indexK, weights, nKeys, nH, indexDim, queryPos, topK, scale)
	qt := be.Upload(NewF32(be, []int{nH * indexDim}, indexQ), F32)
	kt := be.Upload(NewF32(be, []int{nKeys * indexDim}, indexK), F32)
	wt := be.Upload(NewF32(be, []int{nH}, weights), F32)
	gotSeam := ib.DSAIndexSelect(qt, kt, wt, nKeys, nH, indexDim, queryPos, topK, scale)
	if len(gotSeam) != len(wantSeam) {
		t.Fatalf("seam selection length: got %d want %d", len(gotSeam), len(wantSeam))
	}
	for i := range wantSeam {
		if gotSeam[i] != wantSeam[i] {
			t.Fatalf("seam selection[%d]: got %d want %d — host fallback diverges at the old cap boundary", i, gotSeam[i], wantSeam[i])
		}
	}
}
