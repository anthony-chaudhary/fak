// dspark_sharded_greedy.go — DSpark sharded-vocabulary greedy build step at world_size 1.
//
// ADAPTED from sgl-project/sglang,
// python/sglang/kernels/ops/speculative/dspark/sharded_greedy.py:109 at
// 7d5696b3a1638a7c980e46ed77eb876faad158a2 (Apache-2.0). Re-expressed in
// idiomatic Go; no upstream code copied. Ref oss-port-sglang-dspark-sharded-greedy.
//
// PROBLEM / CONTEXT:
// DSpark speculative decoding selects the next token as the argmax of
// TP-local base logits plus a rounded Markov bias over a tensor-parallel
// sharded vocabulary. The upstream Triton op fuses three steps — build the
// local logits, gather the sharded vocab across ranks, argmax — so that one
// (value, index) pair per 4096-wide block crosses the wire instead of the
// full logit vector. At world_size 1 a single rank owns the whole shard, so
// the gather stage is a no-op and the fused step reduces to a partial stage
// (one pair per 4096-block of the padded partition width) followed by a
// finish combine, without ever materializing base+bias.
//
// SEMANTICS (pinned by dspark_sharded_greedy_test.go):
//   - value_i = baseLocal[i] + bias[i] in float32 for i in [0, len(bias));
//     baseLocal entries beyond len(bias) are masked out, including a
//     completely empty final block.
//   - Winner is the smallest global index (i + vocabStart) among values
//     equal to the maximum.
//   - NaN surfaces deterministically: if any value is NaN the result is the
//     smallest global index among the NaN values, never a finite value.
//
// The partial count is derived from the padded base-partition width, not the
// live bias width, so every rank of a future multi-rank transport produces
// the same number of slots; fully masked blocks carry the invalid sentinel
// and are dropped by the finish stage. Upstream moves indices as raw int32
// bits through the float transport and strips a sign bit the NVLink push
// path may flip; this port keeps indices typed, so no bit tricks are needed.

package compute

import (
	"errors"
	"math"
)

// DSparkShardedGreedyBlock is the fixed 4096-wide block width of the partial
// stage, matching the upstream kernel's BLOCK constexpr.
const DSparkShardedGreedyBlock = 4096

// DSparkShardedGreedyInvalidIndex marks a fully masked (empty) block slot in
// a partial. It is the int32 sentinel the upstream kernels exchange as bits.
const DSparkShardedGreedyInvalidIndex int32 = math.MaxInt32

// Sentinel errors closing the DSparkShardedGreedy* contract. Callers match
// these with errors.Is, never on prose.
var (
	// ErrDSparkShardedGreedyEmpty rejects a zero-width bias or base shard.
	ErrDSparkShardedGreedyEmpty = errors.New("compute: dspark sharded greedy: empty score row")

	// ErrDSparkShardedGreedyShapeMismatch rejects a bias wider than the
	// base shard (upstream asserts bias.shape[1] <= base_local.shape[1]).
	ErrDSparkShardedGreedyShapeMismatch = errors.New("compute: dspark sharded greedy: bias wider than base shard")

	// ErrDSparkShardedGreedyVocabStart rejects a negative global offset.
	ErrDSparkShardedGreedyVocabStart = errors.New("compute: dspark sharded greedy: negative vocab start")

	// ErrDSparkShardedGreedyIndexRange rejects rows whose global indices
	// would overflow the int32 index space the transport carries, and finish
	// partials with negative token indices.
	ErrDSparkShardedGreedyIndexRange = errors.New("compute: dspark sharded greedy: global index exceeds int32 range")

	// ErrDSparkShardedGreedyNoWinner rejects a finish stage whose partials
	// are all invalid (no live entry anywhere).
	ErrDSparkShardedGreedyNoWinner = errors.New("compute: dspark sharded greedy: no valid partial")
)

var (
	dsparkNaN    = float32(math.NaN())
	dsparkNegInf = float32(math.Inf(-1))
)

// DSparkShardedGreedyPartial is one (value, global-index) pair produced by
// the partial stage for a 4096-wide block. Value is the block maximum, or
// NaN when the block contains a NaN; Index is the winning global index, or
// DSparkShardedGreedyInvalidIndex for a fully masked block.
type DSparkShardedGreedyPartial struct {
	Value float32
	Index int32
}

// validateDSparkShardedGreedyRow checks the closed input contract shared by
// the partial and fused stages.
func validateDSparkShardedGreedyRow(bias, baseLocal []float32, vocabStart int) error {
	if len(bias) == 0 || len(baseLocal) == 0 {
		return ErrDSparkShardedGreedyEmpty
	}
	if len(bias) > len(baseLocal) {
		return ErrDSparkShardedGreedyShapeMismatch
	}
	if vocabStart < 0 {
		return ErrDSparkShardedGreedyVocabStart
	}
	if vocabStart > math.MaxInt32-len(bias) {
		return ErrDSparkShardedGreedyIndexRange
	}
	return nil
}

// DSparkShardedGreedyPartials runs the partial stage: one
// (block-maximum, winning-global-index) pair per 4096-block of the padded
// base partition, without materializing base+bias. Blocks are partitioned
// over len(baseLocal); entries at or beyond len(bias) are masked.
func DSparkShardedGreedyPartials(bias, baseLocal []float32, vocabStart int) ([]DSparkShardedGreedyPartial, error) {
	if err := validateDSparkShardedGreedyRow(bias, baseLocal, vocabStart); err != nil {
		return nil, err
	}
	nParts := (len(baseLocal) + DSparkShardedGreedyBlock - 1) / DSparkShardedGreedyBlock
	partials := make([]DSparkShardedGreedyPartial, nParts)
	for p := 0; p < nParts; p++ {
		lo := p * DSparkShardedGreedyBlock
		hi := lo + DSparkShardedGreedyBlock
		if hi > len(baseLocal) {
			hi = len(baseLocal)
		}
		if hi > len(bias) {
			hi = len(bias)
		}
		var (
			hasNaN bool
			nanIdx int
			best   float32
		)
		argIdx := -1 // -1 means "no live entry yet"; 0 is a real index
		for i := lo; i < hi; i++ {
			v := baseLocal[i] + bias[i]
			if v != v { // NaN: first (smallest) index wins, recorded once
				if !hasNaN {
					hasNaN = true
					nanIdx = i
				}
				continue
			}
			if argIdx < 0 || v > best { // ties keep the earlier index
				best = v
				argIdx = i
			}
		}
		switch {
		case hasNaN:
			partials[p] = DSparkShardedGreedyPartial{Value: dsparkNaN, Index: int32(nanIdx + vocabStart)}
		case argIdx >= 0:
			partials[p] = DSparkShardedGreedyPartial{Value: best, Index: int32(argIdx + vocabStart)}
		default: // fully masked block (padded partition wider than live data)
			partials[p] = DSparkShardedGreedyPartial{Value: dsparkNegInf, Index: DSparkShardedGreedyInvalidIndex}
		}
	}
	return partials, nil
}

// DSparkShardedGreedyFinish runs the finish stage: combine the per-block
// partials into the single winning global token id. Invalid slots are
// dropped; a NaN partial dominates and resolves to the smallest NaN index.
// Negative indices are malformed, not masked slots, and are refused before
// score comparison (including NaNs).
func DSparkShardedGreedyFinish(partials []DSparkShardedGreedyPartial) (int, error) {
	var (
		hasNaN bool
		nanIdx int32
		best   float32
	)
	bestIdx := int32(-1) // -1 means "no valid partial yet"; 0 is a real index
	for _, p := range partials {
		if p.Index < 0 {
			return 0, ErrDSparkShardedGreedyIndexRange
		}
		if p.Index == DSparkShardedGreedyInvalidIndex {
			continue
		}
		if p.Value != p.Value { // NaN partial
			if !hasNaN || p.Index < nanIdx {
				hasNaN = true
				nanIdx = p.Index
			}
			continue
		}
		if bestIdx < 0 || p.Value > best || (p.Value == best && p.Index < bestIdx) {
			best = p.Value
			bestIdx = p.Index
		}
	}
	switch {
	case hasNaN:
		return int(nanIdx), nil
	case bestIdx >= 0:
		return int(bestIdx), nil
	default:
		return 0, ErrDSparkShardedGreedyNoWinner
	}
}

// DSparkShardedGreedyStep runs the fused build-step + argmax at world_size 1:
// the exact greedy selection from TP-local base logits and rounded Markov
// bias, equivalent to the argmax of the materialized sum over the sharded
// vocab, but carrying only one pair per 4096-block instead of the full logit
// vector. The gather stage is a no-op at world_size 1.
func DSparkShardedGreedyStep(bias, baseLocal []float32, vocabStart int) (int, error) {
	partials, err := DSparkShardedGreedyPartials(bias, baseLocal, vocabStart)
	if err != nil {
		return 0, err
	}
	return DSparkShardedGreedyFinish(partials)
}
