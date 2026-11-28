package compute

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestBitplaneHeadExactDecompositionAndIntervalSoundness(t *testing.T) {
	t.Parallel()
	const vocab = 32
	const hidden = 77 // non-multiple of 32 tests partial tail bit-plane words
	rng := rand.New(rand.NewSource(20261127))

	weights := make([]float32, vocab*hidden)
	for i := range weights {
		weights[i] = float32(rng.NormFloat64() * 0.5)
	}
	x := make([]float32, hidden)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.5)
	}

	head, err := NewBitplaneHead(weights, vocab, hidden, 12, 6)
	if err != nil {
		t.Fatalf("NewBitplaneHead: %v", err)
	}

	// Verify exact bit-plane reconstruction matches affine quantization within RowEps.
	for r := 0; r < vocab; r++ {
		for k := 0; k < hidden; k++ {
			deq, err := head.DequantWeight(r, k)
			if err != nil {
				t.Fatalf("DequantWeight(%d,%d): %v", r, k, err)
			}
			diff := math.Abs(float64(weights[r*hidden+k]) - float64(deq))
			if diff > float64(head.RowEps[r])+1e-6 {
				t.Fatalf("row=%d col=%d residual %.8g > RowEps %.8g", r, k, diff, head.RowEps[r])
			}
		}
	}

	wantTok, logits, err := ExactArgmaxF32(weights, x, vocab, hidden)
	if err != nil {
		t.Fatalf("ExactArgmaxF32: %v", err)
	}

	for _, topBits := range []int{1, 4, 6, 8, 12} {
		res, err := head.CertifyAndRefineArgmaxWithTopBits(x, topBits)
		if err != nil {
			t.Fatalf("topBits=%d: %v", topBits, err)
		}
		if res.Token != wantTok {
			t.Fatalf("topBits=%d: token=%d, want %d", topBits, res.Token, wantTok)
		}
		for r := 0; r < vocab; r++ {
			lo, hi := res.Bounds[r][0], res.Bounds[r][1]
			exact := float64(logits[r])
			if exact < lo || exact > hi {
				t.Fatalf("topBits=%d row=%d exact=%g outside bound [%g, %g]", topBits, r, exact, lo, hi)
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestBitplaneHeadCertificationRegimesAndZeroContenderEarlyExit(t *testing.T) {
	t.Parallel()
	const vocab = 64
	const hidden = 128

	t.Run("margin regime certifies zero contenders early exit", func(t *testing.T) {
		rng := rand.New(rand.NewSource(101))
		weights := make([]float32, vocab*hidden)
		for i := range weights {
			weights[i] = float32(rng.NormFloat64() * 0.05)
		}
		const winner = 23
		for k := 0; k < hidden; k++ {
			weights[winner*hidden+k] = 1.0
		}
		x := make([]float32, hidden)
		for k := 0; k < hidden; k++ {
			x[k] = 0.25
		}

		head, err := NewBitplaneHead(weights, vocab, hidden, 8, 4)
		if err != nil {
			t.Fatalf("NewBitplaneHead: %v", err)
		}
		// Corrupt RawWeights for winner row AFTER packing to prove zero-contender
		// early exit does not touch RawWeights at all.
		savedWinner := append([]float32(nil), head.RawWeights[winner*hidden:(winner+1)*hidden]...)
		for k := 0; k < hidden; k++ {
			head.RawWeights[winner*hidden+k] = float32(math.NaN())
		}
		res, err := head.CertifyAndRefineArgmax(x)
		if err != nil {
			t.Fatalf("CertifyAndRefineArgmax: %v", err)
		}
		if !res.Certified || res.RefinedRows != 0 || len(res.Contenders) != 0 || res.PrunedRows != vocab-1 {
			t.Fatalf("expected zero-contender early exit, got certified=%v refined=%d contenders=%v pruned=%d",
				res.Certified, res.RefinedRows, res.Contenders, res.PrunedRows)
		}
		if res.Token != winner {
			t.Fatalf("token=%d, want %d", res.Token, winner)
		}

		// Restore RawWeights and verify parity against full F32 oracle.
		copy(head.RawWeights[winner*hidden:(winner+1)*hidden], savedWinner)
		wantTok, _, err := ExactArgmaxF32(weights, x, vocab, hidden)
		if err != nil || wantTok != res.Token {
			t.Fatalf("oracle mismatch: want=%d err=%v, got=%d", wantTok, err, res.Token)
		}
	})

	t.Run("partial-prune regime refines only close contenders", func(t *testing.T) {
		rng := rand.New(rand.NewSource(202))
		weights := make([]float32, vocab*hidden)
		for i := range weights {
			weights[i] = float32(rng.NormFloat64() * 0.05)
		}
		const cA, cB = 11, 47
		for k := 0; k < hidden; k++ {
			ripple := 0.15 * math.Cos(float64(k)*(2.0*math.Pi/16.0))
			weights[cA*hidden+k] = float32(0.8 + ripple)
			weights[cB*hidden+k] = float32(0.802 + ripple)
		}
		x := make([]float32, hidden)
		for k := 0; k < hidden; k++ {
			x[k] = 0.25
		}

		head, err := NewBitplaneHead(weights, vocab, hidden, 12, 6)
		if err != nil {
			t.Fatalf("NewBitplaneHead: %v", err)
		}
		if !(head.RowScale[cA] > 0) || !(head.RowScale[cB] > 0) {
			t.Fatalf("contender rows must be non-constant (RowScale[cA]=%v RowScale[cB]=%v)",
				head.RowScale[cA], head.RowScale[cB])
		}
		res, err := head.CertifyAndRefineArgmax(x)
		if err != nil {
			t.Fatalf("CertifyAndRefineArgmax: %v", err)
		}
		if res.Bounds[cA][1] <= res.Bounds[cB][0] {
			t.Fatalf("expected overlapping top-bit bounds for contenders: cA=[%g,%g] cB=[%g,%g]",
				res.Bounds[cA][0], res.Bounds[cA][1], res.Bounds[cB][0], res.Bounds[cB][1])
		}
		wantTok, _, err := ExactArgmaxF32(weights, x, vocab, hidden)
		if err != nil {
			t.Fatalf("ExactArgmaxF32: %v", err)
		}
		if res.Token != wantTok || wantTok != cB {
			t.Fatalf("token=%d oracle=%d want=%d", res.Token, wantTok, cB)
		}
		if res.Certified || res.RefinedRows != 2 || res.PrunedRows != vocab-2 {
			t.Fatalf("expected selective 2-contender refinement, got certified=%v refined=%d pruned=%d contenders=%v",
				res.Certified, res.RefinedRows, res.PrunedRows, res.Contenders)
		}
		if res.Contenders[0] != cA || res.Contenders[1] != cB {
			t.Fatalf("contenders=%v, want [%d %d]", res.Contenders, cA, cB)
		}
	})

	t.Run("tie regime preserves lowest-index winner", func(t *testing.T) {
		weights := make([]float32, vocab*hidden)
		for r := 0; r < vocab; r++ {
			for k := 0; k < hidden; k++ {
				weights[r*hidden+k] = -1.0
			}
		}
		// Rows 7 and 39 tie strictly above all other rows; row 7 must win.
		for k := 0; k < hidden; k++ {
			val := float32(math.Sin(float64(k)+1.0) * 0.75)
			weights[7*hidden+k] = val
			weights[39*hidden+k] = val
		}
		x := make([]float32, hidden)
		for k := 0; k < hidden; k++ {
			x[k] = float32(math.Sin(float64(k) + 1.0))
		}

		head, err := NewBitplaneHead(weights, vocab, hidden, 8, 5)
		if err != nil {
			t.Fatalf("NewBitplaneHead: %v", err)
		}
		res, err := head.CertifyAndRefineArgmax(x)
		if err != nil {
			t.Fatalf("CertifyAndRefineArgmax: %v", err)
		}
		wantTok, _, err := ExactArgmaxF32(weights, x, vocab, hidden)
		if err != nil {
			t.Fatalf("ExactArgmaxF32: %v", err)
		}
		if res.Token != 7 || wantTok != 7 {
			t.Fatalf("tie-break failed: got=%d oracle=%d want=7", res.Token, wantTok)
		}
		if res.PrunedRows != vocab-2 || res.RefinedRows != 2 {
			t.Fatalf("unexpected tie contenders: refined=%d pruned=%d contenders=%v",
				res.RefinedRows, res.PrunedRows, res.Contenders)
		}
	})

	t.Run("unpruned regime with 1 top bit refines all rows and matches oracle", func(t *testing.T) {
		rng := rand.New(rand.NewSource(303))
		weights := make([]float32, vocab*hidden)
		for i := range weights {
			weights[i] = float32(rng.NormFloat64() * 0.25)
		}
		x := make([]float32, hidden)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 0.25)
		}

		head, err := NewBitplaneHead(weights, vocab, hidden, 8, 1)
		if err != nil {
			t.Fatalf("NewBitplaneHead: %v", err)
		}
		res, err := head.CertifyAndRefineArgmax(x)
		if err != nil {
			t.Fatalf("CertifyAndRefineArgmax: %v", err)
		}
		wantTok, _, err := ExactArgmaxF32(weights, x, vocab, hidden)
		if err != nil {
			t.Fatalf("ExactArgmaxF32: %v", err)
		}
		if res.Token != wantTok {
			t.Fatalf("token=%d, want %d", res.Token, wantTok)
		}
		if res.Certified || res.RefinedRows != vocab || res.PrunedRows != 0 {
			t.Fatalf("expected all rows unpruned at 1 top bit, got certified=%v refined=%d pruned=%d",
				res.Certified, res.RefinedRows, res.PrunedRows)
		}
	})

	t.Run("single vocab row exits early with zero contenders", func(t *testing.T) {
		head, err := NewBitplaneHead([]float32{1, -2, 3, 4}, 1, 4, 8, 4)
		if err != nil {
			t.Fatalf("NewBitplaneHead: %v", err)
		}
		res, err := head.CertifyAndRefineArgmax([]float32{0.5, -0.5, 1.0, 0.25})
		if err != nil {
			t.Fatalf("CertifyAndRefineArgmax: %v", err)
		}
		if !res.Certified || res.Token != 0 || res.RefinedRows != 0 || res.PrunedRows != 0 {
			t.Fatalf("unexpected single-row result: %+v", res)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestBitplaneHeadInvalidShapeAndNonFiniteRefusals(t *testing.T) {
	t.Parallel()

	badPacks := []struct {
		name              string
		w                 []float32
		vocab, hidden     int
		totalBits, topBit int
		want              error
	}{
		{"zero vocab", []float32{1}, 0, 1, 8, 4, ErrBitplaneInvalidShape},
		{"zero hidden", []float32{1}, 1, 0, 8, 4, ErrBitplaneInvalidShape},
		{"totalBits zero", []float32{1}, 1, 1, 0, 1, ErrBitplaneInvalidShape},
		{"totalBits too wide", []float32{1}, 1, 1, 17, 4, ErrBitplaneInvalidShape},
		{"topBits exceeds totalBits", []float32{1}, 1, 1, 8, 9, ErrBitplaneInvalidShape},
		{"weights length mismatch", []float32{1, 2, 3}, 2, 2, 8, 4, ErrBitplaneInvalidShape},
		{"nan weight", []float32{1, float32(math.NaN())}, 1, 2, 8, 4, ErrBitplaneNonFinite},
		{"inf weight", []float32{1, float32(math.Inf(1))}, 1, 2, 8, 4, ErrBitplaneNonFinite},
	}
	for _, tc := range badPacks {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewBitplaneHead(tc.w, tc.vocab, tc.hidden, tc.totalBits, tc.topBit); !errors.Is(err, tc.want) {
				t.Fatalf("got err=%v, want %v", err, tc.want)
			}
		})
	}

	head, err := NewBitplaneHead([]float32{1, 2, 3, 4}, 2, 2, 8, 4)
	if err != nil {
		t.Fatalf("NewBitplaneHead: %v", err)
	}
	if _, err := head.CertifyAndRefineArgmax([]float32{1}); !errors.Is(err, ErrBitplaneInvalidShape) {
		t.Fatalf("short x: got %v, want %v", err, ErrBitplaneInvalidShape)
	}
	if _, err := head.CertifyAndRefineArgmaxWithTopBits([]float32{1, 2}, 9); !errors.Is(err, ErrBitplaneInvalidShape) {
		t.Fatalf("bad topBits: got %v, want %v", err, ErrBitplaneInvalidShape)
	}
	if _, err := head.CertifyAndRefineArgmax([]float32{1, float32(math.NaN())}); !errors.Is(err, ErrBitplaneNonFinite) {
		t.Fatalf("nan x: got %v, want %v", err, ErrBitplaneNonFinite)
	}
	if _, err := head.CertifyAndRefineArgmax([]float32{1, float32(math.Inf(-1))}); !errors.Is(err, ErrBitplaneNonFinite) {
		t.Fatalf("inf x: got %v, want %v", err, ErrBitplaneNonFinite)
	}
	var nilHead *BitplaneHead
	if _, err := nilHead.CertifyAndRefineArgmax([]float32{1, 2}); !errors.Is(err, ErrBitplaneInvalidShape) {
		t.Fatalf("nil head: got %v, want %v", err, ErrBitplaneInvalidShape)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestBitplaneHeadCUDASourceContractUnwired(t *testing.T) {
	t.Parallel()
	b, err := os.ReadFile("sampling_topk.cu")
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	for _, want := range []string{
		"magnitudedev/magnitude",
		"readout.seismic:278@8d7dbcc1ba",
		"planes.cuh:64@814f72a",
		"fcuda_bitplane_certified_argmax_workspace_bytes",
		"fcuda_bitplane_certified_argmax",
		"k_bitplane_x_stats",
		"k_bitplane_bounds_stage",
		"k_bitplane_certify_contenders",
		"k_bitplane_refine_contenders",
		"k_bitplane_pick_winner",
		"smemX[BITPLANE_HEAD_TILE_K]",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("sampling_topk.cu missing contract fragment %q", want)
		}
	}
	for _, fname := range []string{"cuda_kernels.cu", "cuda_backend.h"} {
		body, err := os.ReadFile(fname)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(body), "fcuda_bitplane_certified_argmax") {
			t.Fatalf("%s must not wire unwired bitplane symbol yet", fname)
		}
	}
}
