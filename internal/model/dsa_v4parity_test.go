package model

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// dsa_v4parity_test.go — the weight-free V4-SCALE parity witness on the REAL
// internal/model DSA selector seam.
//
// The DeepSeek-V4 native-kernel track (epic #3006) names its recommended first
// rung in docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md: a weight-free
// V4-shape fixture that drives dsaIndexScores / dsaTopKIndices and digests the
// selection — the parity oracle a native kernel must later match. The existing
// offline fixture (internal/dsparity, spec'd in docs/deepseek/v4-parity-harness.md
// / #3021) witnesses the invariance PROPERTIES, but it is stdlib-only with NO
// cross-package import (v4-parity-harness.md:123), so it exercises a SYNTHETIC
// reimplementation of the tie-break — never the actual internal/model seam. The
// in-package dsa_index_test.go tests the real seam only at tiny shapes (topK=2).
//
// This file closes that gap: it drives the ACTUAL dsaIndexScores / dsaTopKIndices
// / dsaIndexShare / dsaIndexDigest at V4 SCALE and asserts the harness's parity
// properties on the real code path — causal well-formedness, request-order
// invariance (bitwise), a total + seed-stable tie-break, digest drift-detection,
// and bitwise every-4-layer index sharing.
//
// Provenance discipline (this repo's fence): the ONE genuine DeepSeek-V4 fact used
// is the attention top-k of 1024 keys/query (V4 technical report, as recorded in
// v4-parity-harness.md). Every other dimension here — key count, head count, head
// dim, layer depth — is a SHAPE-STRESS synthetic value, NOT a V4 checkpoint dim,
// and no throughput/latency/quality number is claimed. This is a correctness
// witness for the selector seam, not a benchmark and not a serve.

const (
	// dsaV4TopK is the one genuine V4 fact: attention top-k = 1024 keys/query.
	dsaV4TopK = 1024
	// dsaV4Keys is a SHAPE-STRESS synthetic context slice (not a checkpoint dim),
	// chosen larger than the top-k so the selection is a strict subset.
	dsaV4Keys = 4096
)

// dsaV4PermByKeyPos builds a fixed pseudo-random permutation of [0..n) under a
// deterministic seed. Used as a distinct per-key-position score so every top-k
// selection is unambiguous (no unintended ties) and reproducible run to run.
func dsaV4PermByKeyPos(n int, seed int64) []int {
	return rand.New(rand.NewSource(seed)).Perm(n)
}

// dsaV4ScoreRow returns one query's score row where the score at each column is a
// distinct function of the KEY POSITION at that column (perm[keyPos]) — never of
// the column index. Tying the score to the key position is what makes the
// request-order-invariance witness meaningful: permuting the column order carries
// the (keyPos, score) pairs together, so the canonicalized selection cannot move.
func dsaV4ScoreRow(keyPositions, perm []int) []float64 {
	row := make([]float64, len(keyPositions))
	for c, kp := range keyPositions {
		row[c] = float64(perm[kp])
	}
	return row
}

// dsaV4IdentityKeyPositions returns keyPositions [0,1,..,n-1].
func dsaV4IdentityKeyPositions(n int) []int {
	kp := make([]int, n)
	for i := range kp {
		kp[i] = i
	}
	return kp
}

// TestDSAV4ScaleTopKIsCausalAndBounded witnesses that the real selector, at V4
// scale (top-k 1024 over 4096 keys), returns exactly min(topK, causalPrefix)
// keys, never a future key, never a duplicate, and in the canonical (score desc,
// key-pos asc) order that downstream digest-stability rests on.
func TestDSAV4ScaleTopKIsCausalAndBounded(t *testing.T) {
	perm := dsaV4PermByKeyPos(dsaV4Keys, 1)
	keyPositions := dsaV4IdentityKeyPositions(dsaV4Keys)

	// A spread of query positions straddling top-k: below it (short causal
	// prefix), at it, and well above it (full 1024 selection).
	for _, qp := range []int{0, 1, 500, dsaV4TopK - 1, dsaV4TopK, dsaV4TopK + 777, dsaV4Keys - 1} {
		row := dsaV4ScoreRow(keyPositions, perm)
		out, ok := dsaTopKIndices([][]float64{row}, []int{qp}, keyPositions, dsaV4TopK)
		if !ok {
			t.Fatalf("qp=%d: dsaTopKIndices returned not-ok", qp)
		}
		sel := out[0]

		wantLen := qp + 1
		if wantLen > dsaV4TopK {
			wantLen = dsaV4TopK
		}
		if len(sel) != wantLen {
			t.Fatalf("qp=%d: selected %d keys, want %d", qp, len(sel), wantLen)
		}

		seen := make(map[int]struct{}, len(sel))
		for _, kp := range sel {
			if kp > qp {
				t.Fatalf("qp=%d: selected future key %d", qp, kp)
			}
			if _, dup := seen[kp]; dup {
				t.Fatalf("qp=%d: duplicate key %d", qp, kp)
			}
			seen[kp] = struct{}{}
		}

		// Canonical order: strictly non-increasing score, ties broken by ascending
		// key position — the total order the harness's seed/request-order rows need.
		for i := 1; i < len(sel); i++ {
			a, b := sel[i-1], sel[i]
			sa, sb := float64(perm[a]), float64(perm[b])
			if sa < sb || (sa == sb && a > b) {
				t.Fatalf("qp=%d: non-canonical order at %d: (key %d, score %v) before (key %d, score %v)",
					qp, i, a, sa, b, sb)
			}
		}
	}
}

// TestDSAV4ScaleSelectionIsRequestOrderInvariant witnesses the harness's
// order/expert-routing-permutation + seed/topk-selection-fixed rows on the real
// seam: the selection DIGEST is bitwise identical no matter what order the keys
// are presented in. This is the concrete "top-k under different request ordering"
// property — a decode must not depend on how its keys happened to be batched.
func TestDSAV4ScaleSelectionIsRequestOrderInvariant(t *testing.T) {
	perm := dsaV4PermByKeyPos(dsaV4Keys, 2)
	keyPositions := dsaV4IdentityKeyPositions(dsaV4Keys)
	qp := dsaV4Keys - 1 // every key causal

	base := dsaV4ScoreRow(keyPositions, perm)
	baseOut, ok := dsaTopKIndices([][]float64{base}, []int{qp}, keyPositions, dsaV4TopK)
	if !ok {
		t.Fatal("baseline dsaTopKIndices returned not-ok")
	}
	baseDigest := dsaIndexDigest(baseOut)

	r := rand.New(rand.NewSource(20260709))
	for iter := 0; iter < 16; iter++ {
		order := r.Perm(dsaV4Keys)
		permKeyPos := make([]int, dsaV4Keys)
		permRow := make([]float64, dsaV4Keys)
		for newc, oldc := range order {
			// Move (keyPos, score) together so the SET of pairs is unchanged.
			permKeyPos[newc] = keyPositions[oldc]
			permRow[newc] = base[oldc]
		}
		out, ok := dsaTopKIndices([][]float64{permRow}, []int{qp}, permKeyPos, dsaV4TopK)
		if !ok {
			t.Fatalf("iter %d: dsaTopKIndices returned not-ok", iter)
		}
		if d := dsaIndexDigest(out); d != baseDigest {
			t.Fatalf("iter %d: digest %s != baseline %s — selection is request-order dependent", iter, d, baseDigest)
		}
	}
}

// TestDSAV4ScaleTieBreakIsTotalAndSeedStable witnesses that a FULL tie (every
// candidate scores identically) resolves by the total, position-ascending
// tie-break — the lowest 1024 key positions — and does so identically across
// runs. V4's 1024-key selector must have a total, seed-stable tie-break or its
// output wobbles under otherwise-identical inputs.
func TestDSAV4ScaleTieBreakIsTotalAndSeedStable(t *testing.T) {
	const nKeys = 2048
	keyPositions := dsaV4IdentityKeyPositions(nKeys)
	row := make([]float64, nKeys)
	for i := range row {
		row[i] = 1.0 // total tie
	}
	qp := nKeys - 1

	run := func() [][]int {
		out, ok := dsaTopKIndices([][]float64{row}, []int{qp}, keyPositions, dsaV4TopK)
		if !ok {
			t.Fatal("dsaTopKIndices returned not-ok")
		}
		return out
	}
	a, b := run(), run()
	if !reflect.DeepEqual(a, b) {
		t.Fatal("tie-break is not idempotent across runs")
	}
	if dsaIndexDigest(a) != dsaIndexDigest(b) {
		t.Fatal("digest is not seed-stable under a total tie")
	}

	want := dsaV4IdentityKeyPositions(dsaV4TopK) // lowest 1024 positions, ascending
	if !reflect.DeepEqual(a[0], want) {
		t.Fatalf("total tie did not resolve to lowest-position keys ascending: first=%d last=%d len=%d",
			a[0][0], a[0][len(a[0])-1], len(a[0]))
	}
}

// TestDSAV4ScaleDigestDetectsDrift witnesses dsaIndexDigest as the drift detector
// the batch/seed rows compare against: a single perturbed index must change the
// digest. Without this, a bit-diff comparison could silently miss a regression.
func TestDSAV4ScaleDigestDetectsDrift(t *testing.T) {
	perm := dsaV4PermByKeyPos(dsaV4Keys, 4)
	keyPositions := dsaV4IdentityKeyPositions(dsaV4Keys)
	qp := dsaV4Keys - 1

	out, ok := dsaTopKIndices([][]float64{dsaV4ScoreRow(keyPositions, perm)}, []int{qp}, keyPositions, dsaV4TopK)
	if !ok {
		t.Fatal("dsaTopKIndices returned not-ok")
	}
	base := dsaIndexDigest(out)

	drift := cloneIndexDecision(out)
	last := len(drift[0]) - 1
	drift[0][last] ^= 1 // flip one bit of one selected key
	if dsaIndexDigest(drift) == base {
		t.Fatal("digest unchanged after a single-index perturbation — drift would go undetected")
	}
}

// TestDSAV4ScaleIndexShareIsBitwiseAcrossLayers witnesses the every-4-layer
// index-share contract on the real seam at a V4-representative depth: each shared
// layer reuses its immediately-preceding full layer's decision BITWISE (identical
// digest), and full-layer decisions pass through unaltered. Seed-stability must
// hold end-to-end across shared layers, not just within one.
func TestDSAV4ScaleIndexShareIsBitwiseAcrossLayers(t *testing.T) {
	const depth = 60 // SHAPE-STRESS depth, not a checkpoint layer count
	perm := dsaV4PermByKeyPos(dsaV4Keys, 5)
	keyPositions := dsaV4IdentityKeyPositions(dsaV4Keys)
	qp := dsaV4Keys - 1

	layerTypes := make([]string, depth)
	fullByLayer := make(map[int][][]int)
	for l := 0; l < depth; l++ {
		if l%4 != 0 {
			layerTypes[l] = "shared"
			continue
		}
		layerTypes[l] = "full"
		// Rotate the permutation by the layer index so distinct full layers make
		// distinct decisions — the share must copy the RIGHT one, not any one.
		row := make([]float64, dsaV4Keys)
		for c, kp := range keyPositions {
			row[c] = float64(perm[(kp+l)%dsaV4Keys])
		}
		out, ok := dsaTopKIndices([][]float64{row}, []int{qp}, keyPositions, dsaV4TopK)
		if !ok {
			t.Fatalf("layer %d (full): dsaTopKIndices returned not-ok", l)
		}
		fullByLayer[l] = out
	}

	shared, ok := dsaIndexShare(layerTypes, fullByLayer)
	if !ok {
		t.Fatal("dsaIndexShare returned not-ok")
	}

	lastFullDigest := ""
	for l := 0; l < depth; l++ {
		got, ok := shared[l]
		if !ok {
			t.Fatalf("layer %d missing from share output", l)
		}
		d := dsaIndexDigest(got)
		if layerTypes[l] == "full" {
			if fd := dsaIndexDigest(fullByLayer[l]); d != fd {
				t.Fatalf("layer %d: full decision altered by share (%s != %s)", l, d, fd)
			}
			lastFullDigest = d
			continue
		}
		if d != lastFullDigest {
			t.Fatalf("layer %d (shared): digest %s != preceding full %s", l, d, lastFullDigest)
		}
	}
}

// TestDSAV4ScaleScoringPipelineIsDeterministic drives the FULL real pipeline —
// dsaIndexScores (the learned relu·weight·dot indexer formula) → dsaTopKIndices →
// dsaIndexDigest — at V4 scale (top-k 1024) over many-head synthetic projections,
// witnessing that the scoring seam produces finite scores and composes with the
// selector to a bitwise-reproducible digest.
func TestDSAV4ScaleScoringPipelineIsDeterministic(t *testing.T) {
	const (
		nQ     = 4
		nKeys  = 1536 // > top-k, so the selection is a strict subset
		nHeads = 16
		dim    = 64
		scale  = 0.125
	)
	r := rand.New(rand.NewSource(6))

	indexK := make([][]float64, nKeys)
	for k := range indexK {
		indexK[k] = make([]float64, dim)
		for d := range indexK[k] {
			indexK[k][d] = r.NormFloat64()
		}
	}
	indexQ := make([][][]float64, nQ)
	indexWeights := make([][]float64, nQ)
	for q := range indexQ {
		indexQ[q] = make([][]float64, nHeads)
		indexWeights[q] = make([]float64, nHeads)
		for h := 0; h < nHeads; h++ {
			indexQ[q][h] = make([]float64, dim)
			for d := range indexQ[q][h] {
				indexQ[q][h][d] = r.NormFloat64()
			}
			indexWeights[q][h] = math.Abs(r.NormFloat64()) // head gates are non-negative
		}
	}
	queryPositions := make([]int, nQ)
	for q := range queryPositions {
		queryPositions[q] = nKeys - 1 // every key causal
	}
	keyPositions := dsaV4IdentityKeyPositions(nKeys)

	pipeline := func() string {
		scores, ok := dsaIndexScores(indexQ, indexK, indexWeights, scale)
		if !ok {
			t.Fatal("dsaIndexScores returned not-ok")
		}
		for q := range scores {
			for k := range scores[q] {
				if math.IsNaN(scores[q][k]) || math.IsInf(scores[q][k], 0) {
					t.Fatalf("non-finite score at q=%d k=%d", q, k)
				}
			}
		}
		sel, ok := dsaTopKIndices(scores, queryPositions, keyPositions, dsaV4TopK)
		if !ok {
			t.Fatal("dsaTopKIndices returned not-ok")
		}
		for q := range sel {
			if len(sel[q]) != dsaV4TopK {
				t.Fatalf("q=%d: selected %d keys, want %d", q, len(sel[q]), dsaV4TopK)
			}
		}
		return dsaIndexDigest(sel)
	}

	if a, b := pipeline(), pipeline(); a != b {
		t.Fatalf("pipeline digest not deterministic: %s != %s", a, b)
	}
}
