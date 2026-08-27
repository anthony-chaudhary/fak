package gateway

// decode_footprint_load.go — the anticipatory decode-footprint load term for the
// fleet scorer (issue #5274, epic #50, borrow: NVIDIA Dynamo, INSPIRE / clean-room
// Go over an Apache-2.0 source).
//
// CacheAwarePolicy.effectiveLoad (residency_router.go) costs a worker as its
// resident occupancy plus a live in-flight COUNT. A count treats a request that
// will emit 20 tokens the same as one that will emit 20k — but a long-output
// request's growing decode KV is exactly what will inflate that worker's
// footprint AFTER it is placed. Dynamo pre-books the not-yet-grown decode blocks
// into the load term, scaled by a decay fraction toward the request's expected
// output length hint, so a worker about to grow a large decode footprint looks
// expensive at selection time, before its in-flight count reflects it.
//
// This file supplies ONLY the pure term: given a request's expected output
// length, the block size, and a speculative scale, it projects the anticipated
// decode-block footprint the request will grow into. It composes ADDITIVELY onto
// the existing count-based load without double-counting (the base already carries
// the request's live membership; this term adds only the blocks not yet grown).
// Deterministic and wall-clock-free — no network, no GPU, no time source.

// DecodeFootprintInputs carries the per-request signals the anticipatory term
// reads. All fields are optional; degenerate values fail closed to a zero
// contribution (see AnticipatedDecodeBlocks).
//
//   - ExpectedOutputTokens: the expected output length hint (fak's agentic
//     per-request hint carries this). The projection scales with it.
//   - GeneratedTokens: how much of the turn has already been emitted. The
//     anticipated remainder shrinks as the true output length becomes known, so
//     the projection decays toward the actual length as the turn grows.
//   - BlockTokens: tokens per decode block (the block granularity KV grows in).
//   - BlockBytes: bytes per decode block, for the companion byte projection
//     (AnticipatedDecodeBytes); unused by the block-count load term.
//   - Scale: a speculative discount in [0,1]. The hint is only a hint until
//     generation reveals the true length, so the projection is discounted. A
//     zero (or negative) scale disables the term (contributes nothing).
type DecodeFootprintInputs struct {
	ExpectedOutputTokens int
	GeneratedTokens      int
	BlockTokens          int
	BlockBytes           int
	Scale                float64
}

// anticipatedRemainderBlocks is the shared kernel: the number of whole decode
// blocks the request is still expected to grow, after discounting. It returns 0
// on any degenerate input so a malformed hint can never inflate a worker's load.
func anticipatedRemainderBlocks(in DecodeFootprintInputs) int {
	if in.ExpectedOutputTokens <= 0 || in.GeneratedTokens < 0 || in.BlockTokens <= 0 {
		return 0
	}
	scale := in.Scale
	if scale <= 0 {
		return 0
	}
	if scale > 1 {
		scale = 1
	}
	remaining := in.ExpectedOutputTokens - in.GeneratedTokens
	if remaining <= 0 {
		return 0
	}
	// ceil(remaining / blockTokens), written without an addition that can
	// overflow when the caller supplies a very large token count.
	rawBlocks := remaining / in.BlockTokens
	if remaining%in.BlockTokens != 0 {
		rawBlocks++
	}
	if scale >= 1 {
		return rawBlocks
	}
	scaled := int(float64(rawBlocks) * scale)
	if scaled < 0 {
		return 0
	}
	return scaled
}

// AnticipatedDecodeBlocks projects the decode-block footprint a request is
// expected to grow into its host worker, in the same whole-block unit the load
// term sums. This is the anticipatory half of the fleet scorer.
//
// The projection is ceil(remaining / BlockTokens) discounted by Scale, where
// remaining = max(0, ExpectedOutputTokens - GeneratedTokens). Properties the
// witness pins:
//   - Monotone: a larger expected output length yields a term that is at least as
//     large (never smaller), so a request expected to grow big raises the load.
//   - Zero base: a zero (or fully-generated) expected output contributes nothing,
//     so a short request is costed exactly as the count-only body costs it.
//   - Fail closed: non-positive block size or expected length, or a non-positive
//     scale, contribute nothing rather than corrupting the load.
func AnticipatedDecodeBlocks(in DecodeFootprintInputs) int {
	return anticipatedRemainderBlocks(in)
}

// AnticipatedDecodeBytes is the companion byte projection: the anticipated
// decode-block footprint expressed in bytes (blocks × BlockBytes), for callers
// that budget the fleet in bytes rather than in the whole-block load unit. It
// shares the same kernel, so it fails closed on the same degenerate inputs and
// is monotone in the expected output length. Returns 0 when BlockBytes is
// non-positive.
func AnticipatedDecodeBytes(in DecodeFootprintInputs) int {
	if in.BlockBytes <= 0 {
		return 0
	}
	blocks := anticipatedRemainderBlocks(in)
	if blocks > maxIntValue()/in.BlockBytes {
		return maxIntValue()
	}
	return blocks * in.BlockBytes
}

// EffectiveLoadWithDecodeFootprint composes the anticipatory decode-footprint
// term additively onto a worker's base load (its resident occupancy plus live
// in-flight count, as effectiveLoad already computes) WITHOUT double-counting:
// the base already reflects the request's live membership, and this adds only the
// not-yet-grown decode blocks on top. It is the additive form the issue names for
// effectiveLoad. A negative base is treated as 0 (fail closed).
func EffectiveLoadWithDecodeFootprint(base int, in DecodeFootprintInputs) int {
	if base < 0 {
		base = 0
	}
	return saturatingNonnegativeAdd(base, AnticipatedDecodeBlocks(in))
}

func saturatingNonnegativeAdd(a, b int) int {
	if a < 0 {
		a = 0
	}
	if b < 0 {
		b = 0
	}
	if a > maxIntValue()-b {
		return maxIntValue()
	}
	return a + b
}

func maxIntValue() int { return int(^uint(0) >> 1) }
