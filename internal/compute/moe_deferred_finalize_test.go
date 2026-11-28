package compute

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func eagerInlineMoeDownAndReduce(
	interOut []float32,
	fc2Weights []float32,
	weights []float32,
	topkIDs []int32,
	numExperts, numTokens, topk, hidden, inter int,
) []float32 {
	out := make([]float32, numTokens*hidden)
	for t := 0; t < numTokens; t++ {
		for j := 0; j < hidden; j++ {
			var acc float32
			for k := 0; k < topk; k++ {
				row := t*topk + k
				id := topkIDs[row]
				if id < 0 || int(id) >= numExperts {
					continue
				}
				h := interOut[row*inter : (row+1)*inter]
				wRow := fc2Weights[(int(id)*hidden+j)*inter : (int(id)*hidden+j+1)*inter]
				var dot float32
				for i := 0; i < inter; i++ {
					dot += wRow[i] * h[i]
				}
				// Eager inline scaling multiplies router weight right at expert output.
				acc += weights[row] * dot
			}
			out[t*hidden+j] = acc
		}
	}
	return out
}

// fak-test:runtime fast est=1s lane=default
func TestMoeDeferredFinalizeMatchesEagerOracle(t *testing.T) {
	t.Parallel()

	t.Run("unweighted rows are router-weight invariant and single-rank finalize matches eager bit-for-bit", func(t *testing.T) {
		rng := rand.New(rand.NewSource(20261116))
		const numExperts, topk, hidden, inter = 6, 3, 16, 12

		for _, numTokens := range []int{1, 5, MoeDeferredFinalizeMaxM} {
			interOut := make([]float32, numTokens*topk*inter)
			for i := range interOut {
				interOut[i] = float32(rng.NormFloat64() * 0.5)
			}
			fc2 := make([]float32, numExperts*hidden*inter)
			for i := range fc2 {
				fc2[i] = float32(rng.NormFloat64() * 0.4)
			}
			topkIDs := make([]int32, numTokens*topk)
			weights := make([]float32, numTokens*topk)
			for tIdx := 0; tIdx < numTokens; tIdx++ {
				for k := 0; k < topk; k++ {
					idx := tIdx*topk + k
					if tIdx%4 == 0 && k == topk-1 {
						topkIDs[idx] = -1 // unroutable / pad slot
						weights[idx] = 9.0
					} else {
						topkIDs[idx] = int32((tIdx*3 + k*2) % numExperts)
						weights[idx] = float32(0.1 + 0.3*float64(k))
					}
				}
			}

			deferredRows, err := MoeDeferredExpertDown(interOut, fc2, topkIDs, numExperts, numTokens, topk, hidden, inter)
			if err != nil {
				t.Fatalf("tokens=%d MoeDeferredExpertDown: %v", numTokens, err)
			}
			// Unroutable slot must hold zeros.
			for j := 0; j < hidden; j++ {
				if v := deferredRows[(topk-1)*hidden+j]; v != 0 {
					t.Fatalf("tokens=%d unroutable pad slot j=%d has %v, want 0", numTokens, j, v)
				}
			}

			// Deferred rows are unweighted; verify explicit unit-weight equivalence vs eager.
			unitWeights := make([]float32, numTokens*topk)
			for i := range unitWeights {
				unitWeights[i] = 1.0
			}
			// Compare row 0 slot 0 directly against unweighted dot.
			eagerUnit := eagerInlineMoeDownAndReduce(interOut[:topk*inter], fc2, []float32{1, 0, 0}, topkIDs[:topk], numExperts, 1, topk, hidden, inter)
			for j := 0; j < hidden; j++ {
				if math.Float32bits(deferredRows[j]) != math.Float32bits(eagerUnit[j]) {
					t.Fatalf("tokens=%d deferred row 0 slot 0 [%d]=%v != unweighted %v", numTokens, j, deferredRows[j], eagerUnit[j])
				}
			}

			got, err := MoeDeferredFinalize(deferredRows, weights, topkIDs, numExperts, numTokens, topk, hidden)
			if err != nil {
				t.Fatalf("tokens=%d MoeDeferredFinalize: %v", numTokens, err)
			}
			want := eagerInlineMoeDownAndReduce(interOut, fc2, weights, topkIDs, numExperts, numTokens, topk, hidden, inter)
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("tokens=%d idx=%d got=%v want=%v (not bit-identical)", numTokens, i, got[i], want[i])
				}
			}
		}
	})

	t.Run("multi-rank deferred finalize all-reduce matches eager inline scaling + all-reduce", func(t *testing.T) {
		rng := rand.New(rand.NewSource(9911))
		const numExperts, numTokens, topk, hidden, inter, ranks = 4, 4, 2, 16, 24, 3
		interPerRank := inter / ranks

		fullInter := make([]float32, numTokens*topk*inter)
		for i := range fullInter {
			fullInter[i] = float32(rng.NormFloat64() * 0.35)
		}
		fullFC2 := make([]float32, numExperts*hidden*inter)
		for i := range fullFC2 {
			fullFC2[i] = float32(rng.NormFloat64() * 0.35)
		}
		topkIDs := []int32{0, 2, 1, 3, 2, 0, 3, 1}
		weights := []float32{0.6, 0.4, 0.75, 0.25, 0.55, 0.45, 0.8, 0.2}
		shared := make([]float32, numTokens*hidden)
		for i := range shared {
			shared[i] = float32(rng.NormFloat64() * 0.2)
		}

		// Eager inline scaling per rank followed by all-reduce + shared expert.
		eagerReduced := make([]float32, numTokens*hidden)
		rankDeferred := make([][]float32, ranks)
		for r := 0; r < ranks; r++ {
			rInter := make([]float32, numTokens*topk*interPerRank)
			for row := 0; row < numTokens*topk; row++ {
				srcOff := row*inter + r*interPerRank
				copy(rInter[row*interPerRank:(row+1)*interPerRank], fullInter[srcOff:srcOff+interPerRank])
			}
			rFC2 := make([]float32, numExperts*hidden*interPerRank)
			for e := 0; e < numExperts; e++ {
				for j := 0; j < hidden; j++ {
					srcOff := (e*hidden+j)*inter + r*interPerRank
					dstOff := (e*hidden + j) * interPerRank
					copy(rFC2[dstOff:dstOff+interPerRank], fullFC2[srcOff:srcOff+interPerRank])
				}
			}

			defRows, err := MoeDeferredExpertDown(rInter, rFC2, topkIDs, numExperts, numTokens, topk, hidden, interPerRank)
			if err != nil {
				t.Fatalf("rank %d MoeDeferredExpertDown: %v", r, err)
			}
			rankDeferred[r] = defRows

			eagerRank := eagerInlineMoeDownAndReduce(rInter, rFC2, weights, topkIDs, numExperts, numTokens, topk, hidden, interPerRank)
			for i := range eagerReduced {
				eagerReduced[i] += eagerRank[i]
			}
		}
		for i := range eagerReduced {
			eagerReduced[i] += shared[i]
		}

		got, err := MoeDeferredFinalizeAllReduce(rankDeferred, weights, topkIDs, shared, numExperts, numTokens, topk, hidden)
		if err != nil {
			t.Fatalf("MoeDeferredFinalizeAllReduce: %v", err)
		}
		for i := range got {
			if d := math.Abs(float64(got[i]) - float64(eagerReduced[i])); d > 1e-5 {
				t.Fatalf("all-reduce idx=%d got=%g eager=%g diff=%g > 1e-5", i, got[i], eagerReduced[i], d)
			}
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestMoeDeferredFinalizeRefusals(t *testing.T) {
	t.Parallel()

	const numExperts, numTokens, topk, hidden, inter = 4, 2, 2, 4, 4
	rows := make([]float32, numTokens*topk*hidden)
	weights := []float32{0.5, 0.5, 0.3, 0.7}
	ids := []int32{0, 1, 2, 3}
	interOut := make([]float32, numTokens*topk*inter)
	fc2 := make([]float32, numExperts*hidden*inter)

	// Geometry refusals
	if _, err := MoeDeferredFinalize(rows, weights, ids, numExperts, 0, topk, hidden); !errors.Is(err, ErrMoeDeferredFinalizeGeometry) {
		t.Errorf("zero tokens: err=%v, want ErrMoeDeferredFinalizeGeometry", err)
	}
	if _, err := MoeDeferredFinalize(make([]float32, (MoeDeferredFinalizeMaxM+1)*topk*hidden),
		make([]float32, (MoeDeferredFinalizeMaxM+1)*topk),
		make([]int32, (MoeDeferredFinalizeMaxM+1)*topk),
		numExperts, MoeDeferredFinalizeMaxM+1, topk, hidden); !errors.Is(err, ErrMoeDeferredFinalizeGeometry) {
		t.Errorf("tokens > kMaxM: err=%v, want ErrMoeDeferredFinalizeGeometry", err)
	}
	if _, err := MoeDeferredFinalize(rows[:len(rows)-1], weights, ids, numExperts, numTokens, topk, hidden); !errors.Is(err, ErrMoeDeferredFinalizeGeometry) {
		t.Errorf("short deferredRows: err=%v, want ErrMoeDeferredFinalizeGeometry", err)
	}
	if _, err := MoeDeferredExpertDown(interOut[:len(interOut)-1], fc2, ids, numExperts, numTokens, topk, hidden, inter); !errors.Is(err, ErrMoeDeferredFinalizeGeometry) {
		t.Errorf("short interOut: err=%v, want ErrMoeDeferredFinalizeGeometry", err)
	}
	if _, err := MoeDeferredFinalizeAllReduce(nil, weights, ids, nil, numExperts, numTokens, topk, hidden); !errors.Is(err, ErrMoeDeferredFinalizeGeometry) {
		t.Errorf("empty rankDeferredRows: err=%v, want ErrMoeDeferredFinalizeGeometry", err)
	}
	if _, err := MoeDeferredFinalizeAllReduce([][]float32{rows}, weights, ids, make([]float32, numTokens*hidden-1), numExperts, numTokens, topk, hidden); !errors.Is(err, ErrMoeDeferredFinalizeGeometry) {
		t.Errorf("bad sharedRows length: err=%v, want ErrMoeDeferredFinalizeGeometry", err)
	}

	// Weight refusals
	if _, err := MoeDeferredFinalize(rows, weights[:len(weights)-1], ids, numExperts, numTokens, topk, hidden); !errors.Is(err, ErrMoeDeferredFinalizeWeights) {
		t.Errorf("short weights: err=%v, want ErrMoeDeferredFinalizeWeights", err)
	}
	badWeights := [][]float32{
		{float32(math.NaN()), 0.5, 0.3, 0.7},
		{0.5, float32(math.Inf(1)), 0.3, 0.7},
		{0.5, 0.5, float32(math.Inf(-1)), 0.7},
	}
	for idx, bw := range badWeights {
		if _, err := MoeDeferredFinalize(rows, bw, ids, numExperts, numTokens, topk, hidden); !errors.Is(err, ErrMoeDeferredFinalizeWeights) {
			t.Errorf("non-finite weight case %d: err=%v, want ErrMoeDeferredFinalizeWeights", idx, err)
		}
	}
}
