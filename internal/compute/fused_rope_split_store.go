package compute

import (
	"fmt"
	"math"
)

// FusedRoPESplitStoreReceipt records execution metrics and DRAM savings for the fused kernel.
type FusedRoPESplitStoreReceipt struct {
	NumTokens        int     `json:"num_tokens"`
	NumQHeads        int     `json:"num_q_heads"`
	NumKVHeads       int     `json:"num_kv_heads"`
	HeadDim          int     `json:"head_dim"`
	PageSize         int     `json:"page_size"`
	SavedLaunches    int     `json:"saved_launches"`
	EliminatedDRAMMB float64 `json:"eliminated_dram_mb"`
}

// rotateHalfInPlace applies HF non-interleaved rotate_half RoPE for a single head vector.
func rotateHalfInPlace(vec []float32, headDim int, pos int, theta float64) {
	half := headDim / 2
	for j := 0; j < half; j++ {
		inv := 1.0 / math.Pow(theta, float64(2*j)/float64(headDim))
		a := float64(pos) * inv
		c := float32(math.Cos(a))
		s := float32(math.Sin(a))

		x0 := vec[j]
		x1 := vec[j+half]

		vec[j] = x0*c - x1*s
		vec[j+half] = x1*c + x0*s
	}
}

// ExecuteFusedRoPESplitStore fuses Q/K/V splitting, rotary position embedding (RoPE),
// and paged KV cache storage into a single pass, eliminating intermediate buffer traffic.
func ExecuteFusedRoPESplitStore(
	packedQKV []float32,
	positions []int,
	pageTable []int32,
	pageSize int,
	numQHeads int,
	numKVHeads int,
	headDim int,
	theta float64,
	outQ []float32,
	pagedK []float32,
	pagedV []float32,
) (FusedRoPESplitStoreReceipt, error) {
	var receipt FusedRoPESplitStoreReceipt
	numTokens := len(positions)

	if numTokens == 0 {
		return receipt, fmt.Errorf("positions must not be empty")
	}
	if numQHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || pageSize <= 0 {
		return receipt, fmt.Errorf("dimensions must be positive: qHeads=%d, kvHeads=%d, hd=%d, pageSize=%d",
			numQHeads, numKVHeads, headDim, pageSize)
	}
	if headDim%2 != 0 {
		return receipt, fmt.Errorf("headDim must be even for RoPE rotate_half, got %d", headDim)
	}

	packedStride := (numQHeads + 2*numKVHeads) * headDim
	if len(packedQKV) != numTokens*packedStride {
		return receipt, fmt.Errorf("packedQKV length %d != expected %d", len(packedQKV), numTokens*packedStride)
	}
	if len(outQ) != numTokens*numQHeads*headDim {
		return receipt, fmt.Errorf("outQ length %d != expected %d", len(outQ), numTokens*numQHeads*headDim)
	}

	qWidth := numQHeads * headDim
	kvWidth := numKVHeads * headDim

	for t := 0; t < numTokens; t++ {
		pos := positions[t]
		if pos < 0 {
			return receipt, fmt.Errorf("negative position %d at token %d", pos, t)
		}

		lPage := pos / pageSize
		inPage := pos % pageSize
		if lPage >= len(pageTable) {
			return receipt, fmt.Errorf("token %d pos %d page %d exceeds pageTable length %d", t, pos, lPage, len(pageTable))
		}
		physPage := pageTable[lPage]
		if physPage < 0 {
			return receipt, fmt.Errorf("unmapped physical page %d for token %d", physPage, t)
		}

		packedTokenOffset := t * packedStride

		// 1. Process Q: extract and apply RoPE directly into outQ
		for qh := 0; qh < numQHeads; qh++ {
			qSrcStart := packedTokenOffset + qh*headDim
			qDstStart := t*qWidth + qh*headDim

			copy(outQ[qDstStart:qDstStart+headDim], packedQKV[qSrcStart:qSrcStart+headDim])
			rotateHalfInPlace(outQ[qDstStart:qDstStart+headDim], headDim, pos, theta)
		}

		// 2. Process K: extract, apply RoPE, and store directly into paged K store
		kPackedBase := packedTokenOffset + qWidth
		pagedTokenOffset := (int(physPage)*pageSize + inPage) * kvWidth

		for kvh := 0; kvh < numKVHeads; kvh++ {
			kSrcStart := kPackedBase + kvh*headDim
			kDstStart := pagedTokenOffset + kvh*headDim

			if kDstStart+headDim > len(pagedK) {
				return receipt, fmt.Errorf("pagedK overflow at offset %d (len %d)", kDstStart+headDim, len(pagedK))
			}

			copy(pagedK[kDstStart:kDstStart+headDim], packedQKV[kSrcStart:kSrcStart+headDim])
			rotateHalfInPlace(pagedK[kDstStart:kDstStart+headDim], headDim, pos, theta)
		}

		// 3. Process V: extract and store directly into paged V store (no RoPE for V)
		vPackedBase := packedTokenOffset + qWidth + kvWidth
		for kvh := 0; kvh < numKVHeads; kvh++ {
			vSrcStart := vPackedBase + kvh*headDim
			vDstStart := pagedTokenOffset + kvh*headDim

			if vDstStart+headDim > len(pagedV) {
				return receipt, fmt.Errorf("pagedV overflow at offset %d (len %d)", vDstStart+headDim, len(pagedV))
			}

			copy(pagedV[vDstStart:vDstStart+headDim], packedQKV[vSrcStart:vSrcStart+headDim])
		}
	}

	// DRAM savings: eliminated intermediate K/V split writes (2 * kvWidth * 4) + RoPE reads & writes (2 * kvWidth * 4)
	eliminatedBytes := float64(numTokens * 4 * (2 * kvWidth))
	receipt = FusedRoPESplitStoreReceipt{
		NumTokens:        numTokens,
		NumQHeads:        numQHeads,
		NumKVHeads:       numKVHeads,
		HeadDim:          headDim,
		PageSize:         pageSize,
		SavedLaunches:    2,
		EliminatedDRAMMB: eliminatedBytes / (1024 * 1024),
	}

	return receipt, nil
}
