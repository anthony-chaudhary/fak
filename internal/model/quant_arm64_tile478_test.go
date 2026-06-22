//go:build arm64

package model

import (
	"math"
	"testing"
)

// TestQGemm8Tile478Gate is the issue-#478 correctness gate for the register-blocked Q8_0 GEMM
// tile, authored as a NEW gate alongside the isolated-tile bit-identity pin
// (TestQGemm8TileNEONMatchesCell). Where that test exercises one detached 4×4 qgemm8tileNEON tile,
// this one drives the FULL dispatcher qGemm8TileInto — the 4×4 tiles PLUS the out%4 row-remainder
// and P%4 token-remainder cleanup through qgemm8cell(...,4) — across randomized, deliberately
// non-tile-aligned shapes, against the portable lane-4 scalar reference qGemm8scalar.
//
// Two postures, exactly as #478 specifies:
//
//	(a) PREFER + ASSERT bit-identical. The tile preserves qgemm8cell(...,4)'s deferred-reduction
//	    order (SDOT → exact SCVTF → broadcast-scale FMLA matching math.FMA, then the same pairwise
//	    lane fold), so every output float MUST be Float32bits-equal to the reference. This is the
//	    strong gate. It is INDEPENDENT of — and does not relax — the existing prefill bit-identity
//	    test TestQGemm8IntoMatchesScalarNEON; it adds a second, dispatcher-level pin.
//	(b) RECORD the argmax-exact + cosine floor vs the reference anyway. With bit-identity holding
//	    these are trivially exact / cosine 1.0, but recording them (i) documents the floor #478's
//	    acceptance asks for and (ii) makes the gate degrade gracefully to the quantized-oracle
//	    posture (argmax-exact + cosine ≥ threshold) IF a future deferred-reduction tile variant ever
//	    forfeits bit-identity — the same posture the #477 amortized kernel gate already takes.
//
// argmax/cosine are the shared package test helpers (oracle_test.go), the exact pair
// TestQdot8AmortArgmaxAndCosine uses, so this gate reads against the same reference notion.
func TestQGemm8Tile478Gate(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON tile inactive")
	}
	const cosFloor = 0.9999 // RECORDED argmax-exact + cosine floor (bit-identity makes it exact here)
	cases := []struct{ out, in, P int }{
		{4, 32, 4},     // exact tile, single block
		{8, 64, 8},     // 2×2 tiles
		{6, 576, 5},    // row + token remainder
		{13, 192, 7},   // odd rows and tokens
		{7, 64, 3},     // P < NR (all-remainder tokens)
		{1, 32, 1},     // degenerate 1×1
		{17, 256, 9},   // out%4 and P%4 both nonzero
		{64, 1536, 16}, // Qwen-ish projection, aligned tokens
		{66, 320, 11},  // both remainders at a larger row count
	}
	worstCos := 1.0
	argmaxMismatch := 0
	for _, c := range cases {
		w := mkVec(c.out*c.in, uint64(c.out*73+c.in*5+c.P*3+1))
		qt := quantizeQ8(w, c.out, c.in)
		X := mkVec(c.P*c.in, uint64(c.out*131+c.in*17+c.P*7+9))
		qp := quantizeBatchPanel(X, c.P, c.in)

		want := qGemm8scalar(qt, qp, 4)
		got := make([]float32, c.P*c.out)
		qGemm8TileInto(qt, qp, got)

		// (a) bit-identity — the strong assertion; reduction order is preserved.
		for k := range got {
			if math.Float32bits(got[k]) != math.Float32bits(want[k]) {
				t.Fatalf("out=%d in=%d P=%d idx=%d: tile %v (%08x) != scalar %v (%08x) — bit-identity broken",
					c.out, c.in, c.P, k, got[k], math.Float32bits(got[k]), want[k], math.Float32bits(want[k]))
			}
		}

		// (b) RECORD argmax-exact + cosine per token (over each token's `out` logits).
		for tok := 0; tok < c.P; tok++ {
			g := got[tok*c.out : tok*c.out+c.out]
			r := want[tok*c.out : tok*c.out+c.out]
			if argmax(g) != argmax(r) {
				argmaxMismatch++
				t.Fatalf("out=%d in=%d P=%d token=%d: argmax tile=%d != scalar=%d — greedy token would differ",
					c.out, c.in, c.P, tok, argmax(g), argmax(r))
			}
			if cos := cosine(g, r); cos < worstCos {
				worstCos = cos
			}
		}
	}
	if worstCos < cosFloor {
		t.Fatalf("#478 tile gate: worst cosine %.8f < recorded floor %.4f", worstCos, cosFloor)
	}
	t.Logf("#478 tile gate: qGemm8TileInto bit-identical to qGemm8scalar across %d shapes; argmax-exact (mismatches=%d); min cosine RECORDED = %.8f (floor %.4f)",
		len(cases), argmaxMismatch, worstCos, cosFloor)
}
