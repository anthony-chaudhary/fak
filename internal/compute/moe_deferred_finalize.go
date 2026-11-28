package compute

// Low-batch MoE deferred finalize: expert down-projection accumulates
// unweighted/unit expert rows before the finalize/all-reduce pass applies
// per-token top-k router weights.
//
// ADAPT of sgl-project/sglang@2e7105d6fc8ac3b6e7f52215cd01709392d1f7ea
// (python/sglang/kernels/jit/csrc/moe/mxfp4_decode/moe_mxfp4.cuh, Apache-2.0):
// in the low-batch decode path (kMaxM=256), expert down-projection skips the
// inline router-weight multiply and emits unweighted rows; the top-k weights
// are folded into the subsequent finalize/all-reduce pass that already touches
// every element.
//
// Clean-room Go reference and oracle only ([SW-VERIFIED]): no hardware claim.

import (
	"errors"
	"fmt"
	"math"
)

// MoeDeferredFinalizeMaxM bounds the low-batch decode token count admitted by
// the deferred-finalize path, matching upstream kMaxM=256.
const MoeDeferredFinalizeMaxM = 256

var (
	// ErrMoeDeferredFinalizeGeometry is the closed refusal for invalid token,
	// expert, topk, hidden, inter, or rank-slice shapes.
	ErrMoeDeferredFinalizeGeometry = errors.New("compute: moe deferred finalize invalid geometry")
	// ErrMoeDeferredFinalizeWeights is the closed refusal for malformed or
	// non-finite router weights.
	ErrMoeDeferredFinalizeWeights = errors.New("compute: moe deferred finalize invalid weights")
)

func validateMoeDeferredCommon(numExperts, numTokens, topk, hidden int) (int, error) {
	if numExperts <= 0 || numExperts > MoeAlignMaxExperts {
		return 0, fmt.Errorf("%w: numExperts=%d", ErrMoeDeferredFinalizeGeometry, numExperts)
	}
	if numTokens <= 0 || numTokens > MoeDeferredFinalizeMaxM {
		return 0, fmt.Errorf("%w: numTokens=%d (want 1..%d)", ErrMoeDeferredFinalizeGeometry, numTokens, MoeDeferredFinalizeMaxM)
	}
	if topk <= 0 || hidden <= 0 {
		return 0, fmt.Errorf("%w: topk=%d hidden=%d", ErrMoeDeferredFinalizeGeometry, topk, hidden)
	}
	tokensTopk := numTokens * topk
	if tokensTopk <= 0 || tokensTopk > math.MaxInt/hidden {
		return 0, fmt.Errorf("%w: dimensions overflow", ErrMoeDeferredFinalizeGeometry)
	}
	return tokensTopk, nil
}

func validateMoeDeferredWeightsAndIDs(weights []float32, topkIDs []int32, tokensTopk int) error {
	if len(weights) != tokensTopk {
		return fmt.Errorf("%w: weights len=%d want %d", ErrMoeDeferredFinalizeWeights, len(weights), tokensTopk)
	}
	if len(topkIDs) != tokensTopk {
		return fmt.Errorf("%w: topkIDs len=%d want %d", ErrMoeDeferredFinalizeGeometry, len(topkIDs), tokensTopk)
	}
	for i, w := range weights {
		if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
			return fmt.Errorf("%w: weight[%d]=%v is non-finite", ErrMoeDeferredFinalizeWeights, i, w)
		}
	}
	return nil
}

// MoeDeferredExpertDown computes the unweighted (unit-scale) expert
// down-projection rows [numTokens, topk, hidden] from activated intermediates
// interOut [numTokens, topk, inter] and fc2Weights [numExperts, hidden, inter].
// Router weights are deliberately absent here: unroutable/padding topkIDs
// outside [0, numExperts) produce zero rows.
func MoeDeferredExpertDown(
	interOut []float32,
	fc2Weights []float32,
	topkIDs []int32,
	numExperts, numTokens, topk, hidden, inter int,
) ([]float32, error) {
	tokensTopk, err := validateMoeDeferredCommon(numExperts, numTokens, topk, hidden)
	if err != nil {
		return nil, err
	}
	if inter <= 0 {
		return nil, fmt.Errorf("%w: inter=%d", ErrMoeDeferredFinalizeGeometry, inter)
	}
	if len(topkIDs) != tokensTopk {
		return nil, fmt.Errorf("%w: topkIDs len=%d want %d", ErrMoeDeferredFinalizeGeometry, len(topkIDs), tokensTopk)
	}
	if tokensTopk > math.MaxInt/inter || len(interOut) != tokensTopk*inter {
		return nil, fmt.Errorf("%w: interOut len=%d want %d", ErrMoeDeferredFinalizeGeometry, len(interOut), tokensTopk*inter)
	}
	if numExperts > math.MaxInt/(hidden*inter) || len(fc2Weights) != numExperts*hidden*inter {
		return nil, fmt.Errorf("%w: fc2Weights len=%d want %d", ErrMoeDeferredFinalizeGeometry, len(fc2Weights), numExperts*hidden*inter)
	}

	ne := int32(numExperts)
	out := make([]float32, tokensTopk*hidden)
	for row := 0; row < tokensTopk; row++ {
		id := topkIDs[row]
		if id < 0 || id >= ne {
			continue
		}
		h := interOut[row*inter : (row+1)*inter]
		fc2 := fc2Weights[int(id)*hidden*inter : (int(id)+1)*hidden*inter]
		dst := out[row*hidden : (row+1)*hidden]
		for j := 0; j < hidden; j++ {
			wRow := fc2[j*inter : (j+1)*inter]
			var acc float32
			for i := 0; i < inter; i++ {
				acc += wRow[i] * h[i]
			}
			dst[j] = acc
		}
	}
	return out, nil
}

// MoeDeferredFinalize applies top-k router weights to unweighted deferredRows
// [numTokens, topk, hidden] and reduces across k in ascending slot order into
// [numTokens, hidden]. Entries whose topkIDs fall outside [0, numExperts) are
// skipped cleanly.
func MoeDeferredFinalize(
	deferredRows []float32,
	weights []float32,
	topkIDs []int32,
	numExperts, numTokens, topk, hidden int,
) ([]float32, error) {
	tokensTopk, err := validateMoeDeferredCommon(numExperts, numTokens, topk, hidden)
	if err != nil {
		return nil, err
	}
	if err := validateMoeDeferredWeightsAndIDs(weights, topkIDs, tokensTopk); err != nil {
		return nil, err
	}
	if len(deferredRows) != tokensTopk*hidden {
		return nil, fmt.Errorf("%w: deferredRows len=%d want %d", ErrMoeDeferredFinalizeGeometry, len(deferredRows), tokensTopk*hidden)
	}

	ne := int32(numExperts)
	out := make([]float32, numTokens*hidden)
	for t := 0; t < numTokens; t++ {
		dst := out[t*hidden : (t+1)*hidden]
		for j := 0; j < hidden; j++ {
			var acc float32
			for k := 0; k < topk; k++ {
				idx := t*topk + k
				id := topkIDs[idx]
				if id < 0 || id >= ne {
					continue
				}
				acc += weights[idx] * deferredRows[idx*hidden+j]
			}
			dst[j] = acc
		}
	}
	return out, nil
}

// MoeDeferredFinalizeAllReduce fuses tensor-parallel partial all-reduce across
// ranks with the deferred top-k router-weight application (and optional
// sharedRows [numTokens, hidden]) in a single finalize pass.
func MoeDeferredFinalizeAllReduce(
	rankDeferredRows [][]float32,
	weights []float32,
	topkIDs []int32,
	sharedRows []float32,
	numExperts, numTokens, topk, hidden int,
) ([]float32, error) {
	tokensTopk, err := validateMoeDeferredCommon(numExperts, numTokens, topk, hidden)
	if err != nil {
		return nil, err
	}
	if err := validateMoeDeferredWeightsAndIDs(weights, topkIDs, tokensTopk); err != nil {
		return nil, err
	}
	if len(rankDeferredRows) == 0 {
		return nil, fmt.Errorf("%w: empty rankDeferredRows", ErrMoeDeferredFinalizeGeometry)
	}
	wantRows := tokensTopk * hidden
	for r, rows := range rankDeferredRows {
		if len(rows) != wantRows {
			return nil, fmt.Errorf("%w: rank %d deferredRows len=%d want %d", ErrMoeDeferredFinalizeGeometry, r, len(rows), wantRows)
		}
	}
	if sharedRows != nil && len(sharedRows) != numTokens*hidden {
		return nil, fmt.Errorf("%w: sharedRows len=%d want %d", ErrMoeDeferredFinalizeGeometry, len(sharedRows), numTokens*hidden)
	}

	ne := int32(numExperts)
	out := make([]float32, numTokens*hidden)
	for t := 0; t < numTokens; t++ {
		for j := 0; j < hidden; j++ {
			var acc float32
			for k := 0; k < topk; k++ {
				idx := t*topk + k
				id := topkIDs[idx]
				if id < 0 || id >= ne {
					continue
				}
				var rankSum float32
				for r := range rankDeferredRows {
					rankSum += rankDeferredRows[r][idx*hidden+j]
				}
				acc += weights[idx] * rankSum
			}
			if sharedRows != nil {
				acc += sharedRows[t*hidden+j]
			}
			out[t*hidden+j] = acc
		}
	}
	return out, nil
}
