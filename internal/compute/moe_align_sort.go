package compute

// On-device MoE token align-and-sort per expert + weighted-sum finalize: the CPU
// reference of moe_align_sort.cu (ADAPT of vllm-project/vllm
// csrc/moe/moe_align_sum_kernels.cu at pin ae065918640a198a56f6f17eb45e251c3229fe41,
// Apache-2.0). This file is pure Go on purpose: it compiles and is tested in every
// build, while the device kernels ship behind the cuda build tag once the follow-on
// ticket wires moe_align_sort.cu into the single libfakcuda translation unit. The
// cgo launch wrapper lands with that wiring; declaring the C ABI here would break
// every -tags cuda link against a libfakcuda.a that does not carry the symbols yet.

import (
	"errors"
	"fmt"
)

// MoeAlignMaxExperts bounds the routed-expert count the align/sort ABI accepts:
// the device kernel runs one thread per expert lane in a single 1024-thread block,
// so the host oracle enforces the same bound and refuses anything wider.
const MoeAlignMaxExperts = 1024

var (
	// ErrMoeAlignGeometry is the closed refusal for MoeAlignBlockSizes inputs that
	// violate the expert/block bounds or overflow the int32 device ABI.
	ErrMoeAlignGeometry = errors.New("compute: moe align-block-size geometry refused")
	// ErrMoeSumGeometry is the closed refusal for MoeSumTopK shape mismatches.
	ErrMoeSumGeometry = errors.New("compute: moe top-k weighted-sum geometry refused")
)

// MoeAlignResult is the CPU image of what fcuda_moe_align_block_size produces on
// device for a flat topk expert-id panel [num_tokens][top_k] of length Numel.
//
//   - Every cumsum entry is a multiple of the block size: expert e owns the padded
//     region [cumsum[e], cumsum[e+1]) of SortedTokenIDs, whose length is
//     ceil(count_e/blockSize)*blockSize.
//   - SortedTokenIDs holds the sentinel Numel in every pad slot. For each expert e
//     the multiset of non-sentinel values in its region equals the set of input
//     indices i whose id routes to e; this oracle emits them in ascending input
//     order (the canonical stable form). The device scatter satisfies the same
//     multiset guarantee but leaves the intra-region order unspecified, so consumers
//     must treat a region as a set — the grouped GEMM that consumes these lists is
//     permutation-invariant inside a region.
//   - ExpertIDs[b] names the expert owning block b (sorted positions
//     [b*blockSize, (b+1)*blockSize)); -1 marks blocks past NumTokensPostPad.
type MoeAlignResult struct {
	SortedTokenIDs     []int32 // length MaxNumTokensPadded
	ExpertIDs          []int32 // length MaxNumMBlocks
	Cumsum             []int32 // length numExperts + 1
	NumTokensPostPad   int32   // == Cumsum[numExperts]
	Numel              int32   // input length; also the pad sentinel in SortedTokenIDs
	MaxNumTokensPadded int32   // numel + numExperts*(blockSize-1), the tight capacity bound
	MaxNumMBlocks      int32   // ceil(MaxNumTokensPadded / blockSize)
}

// MoeAlignBlockSizes computes the block-aligned per-expert token lists, expert
// offsets and num_tokens_post_pad for a flat topk expert-id panel. Entry i is valid
// iff 0 <= topkIDs[i] < numExperts; invalid ids (-1 padding, out-of-range) are
// skipped exactly like the device kernel's get_local_expert_id.
func MoeAlignBlockSizes(topkIDs []int32, numExperts, blockSize int) (*MoeAlignResult, error) {
	if numExperts <= 0 || numExperts > MoeAlignMaxExperts {
		return nil, fmt.Errorf("%w: numExperts=%d", ErrMoeAlignGeometry, numExperts)
	}
	if blockSize <= 0 {
		return nil, fmt.Errorf("%w: blockSize=%d", ErrMoeAlignGeometry, blockSize)
	}
	const maxInt32 = int32(1<<31 - 1)
	numel := len(topkIDs)
	if numel > int(maxInt32) {
		return nil, fmt.Errorf("%w: numel=%d overflows the int32 ABI", ErrMoeAlignGeometry, numel)
	}
	// Tight capacity bound: each expert pads by at most blockSize-1 entries.
	maxPadded := numel + numExperts*(blockSize-1)
	if maxPadded < 0 || maxPadded > int(maxInt32) {
		return nil, fmt.Errorf("%w: maxNumTokensPadded=%d overflows the int32 ABI", ErrMoeAlignGeometry, maxPadded)
	}
	maxMBlocks := (maxPadded + blockSize - 1) / blockSize

	ne := int32(numExperts) // validated <= MoeAlignMaxExperts above
	counts := make([]int, numExperts)
	perExpert := make([][]int32, numExperts)
	for i, id := range topkIDs {
		if id < 0 || id >= ne {
			continue
		}
		counts[id]++
		perExpert[id] = append(perExpert[id], int32(i))
	}

	cumsum := make([]int32, numExperts+1)
	postPad := 0
	for e := 0; e < numExperts; e++ {
		padded := (counts[e] + blockSize - 1) / blockSize * blockSize
		cumsum[e] = int32(postPad)
		postPad += padded
	}
	cumsum[numExperts] = int32(postPad)

	expertIDs := make([]int32, maxMBlocks)
	for b := range expertIDs {
		expertIDs[b] = -1
	}
	for e := 0; e < numExperts; e++ {
		first := int(cumsum[e]) / blockSize
		last := int(cumsum[e+1]) / blockSize
		for b := first; b < last; b++ {
			expertIDs[b] = int32(e)
		}
	}

	sentinel := int32(numel)
	sorted := make([]int32, maxPadded)
	for p := range sorted {
		sorted[p] = sentinel
	}
	for e := 0; e < numExperts; e++ {
		base := int(cumsum[e])
		entries := perExpert[e]
		for k, idx := range entries {
			sorted[base+k] = idx
		}
	}

	return &MoeAlignResult{
		SortedTokenIDs:     sorted,
		ExpertIDs:          expertIDs,
		Cumsum:             cumsum,
		NumTokensPostPad:   int32(postPad),
		Numel:              int32(numel),
		MaxNumTokensPadded: int32(maxPadded),
		MaxNumMBlocks:      int32(maxMBlocks),
	}, nil
}

// MoeSumTopK finalizes a token's topk expert outputs into one dense row:
//
//	out[t, j] = sum over k of weights[t, k] * expertOut[t, k, j]
//
// where the sum skips every k whose topkIDs[t, k] falls outside [0, numExperts)
// (padding or an unroutable expert contributes nothing), accumulated in float32 in
// ascending k order. expertOut is row-major [numTokens][topk][hidden], weights and
// topkIDs are row-major [numTokens][topk]. IEEE semantics: no NaN filtering, and
// each multiply and add is individually rounded, matching the device kernel's
// mul.rn.f32/add.rn.f32 discipline bit for bit.
func MoeSumTopK(expertOut []float32, weights []float32, topkIDs []int32, numExperts, numTokens, topk, hidden int) ([]float32, error) {
	if numExperts <= 0 || numExperts > MoeAlignMaxExperts {
		return nil, fmt.Errorf("%w: numExperts=%d", ErrMoeSumGeometry, numExperts)
	}
	if numTokens <= 0 || topk <= 0 || hidden <= 0 {
		return nil, fmt.Errorf("%w: tokens/topk/hidden=%d/%d/%d", ErrMoeSumGeometry, numTokens, topk, hidden)
	}
	tokensTopk := numTokens * topk
	if tokensTopk <= 0 || tokensTopk > 1<<31-1 {
		return nil, fmt.Errorf("%w: numTokens*topk overflows", ErrMoeSumGeometry)
	}
	if len(weights) != tokensTopk || len(topkIDs) != tokensTopk {
		return nil, fmt.Errorf("%w: weights/topkIDs length %d/%d want %d", ErrMoeSumGeometry, len(weights), len(topkIDs), tokensTopk)
	}
	need := int64(tokensTopk) * int64(hidden)
	if need <= 0 || need > 1<<62 {
		return nil, fmt.Errorf("%w: element count overflows", ErrMoeSumGeometry)
	}
	if len(expertOut) != int(need) {
		return nil, fmt.Errorf("%w: expertOut length %d want %d", ErrMoeSumGeometry, len(expertOut), need)
	}

	ne := int32(numExperts) // validated <= MoeAlignMaxExperts above
	out := make([]float32, numTokens*hidden)
	for t := 0; t < numTokens; t++ {
		for j := 0; j < hidden; j++ {
			var acc float32
			for k := 0; k < topk; k++ {
				id := topkIDs[t*topk+k]
				if id < 0 || id >= ne {
					continue
				}
				acc += weights[t*topk+k] * expertOut[(t*topk+k)*hidden+j]
			}
			out[t*hidden+j] = acc
		}
	}
	return out, nil
}
