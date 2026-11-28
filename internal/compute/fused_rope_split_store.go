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

// Adapted from modular/modular 88ab8913af max/kernels/src/nn/head_rms_norm_rope.mojo:273
// (Apache-2.0 WITH LLVM-exception): the scale and gate contract of the
// per-head RMSNorm+RoPE+paged-store launch, on the host path.

// ExecuteFusedRoPESplitStoreScaled is ExecuteFusedRoPESplitStore with q_scale
// applied to Q and value_scale applied to V in the same pass. Q is scaled
// before RoPE and V before the store; both are linear so the result equals
// scaling after, and scale 1 reproduces the unscaled kernel exactly.
func ExecuteFusedRoPESplitStoreScaled(
	packedQKV []float32,
	positions []int,
	pageTable []int32,
	pageSize, numQHeads, numKVHeads, headDim int,
	theta float64,
	qScale, valueScale float32,
	outQ, pagedK, pagedV []float32,
) (FusedRoPESplitStoreReceipt, error) {
	stride := (numQHeads + 2*numKVHeads) * headDim
	if qScale == 1 && valueScale == 1 || stride <= 0 || len(packedQKV)%stride != 0 {
		return ExecuteFusedRoPESplitStore(packedQKV, positions, pageTable, pageSize, numQHeads, numKVHeads, headDim, theta, outQ, pagedK, pagedV)
	}
	scaled := append([]float32(nil), packedQKV...)
	qWidth, kvWidth := numQHeads*headDim, numKVHeads*headDim
	for t := 0; t < len(scaled)/stride; t++ {
		base := t * stride
		for i := 0; i < qWidth; i++ {
			scaled[base+i] *= qScale
		}
		for i := 0; i < kvWidth; i++ {
			scaled[base+qWidth+kvWidth+i] *= valueScale
		}
	}
	return ExecuteFusedRoPESplitStore(scaled, positions, pageTable, pageSize, numQHeads, numKVHeads, headDim, theta, outQ, pagedK, pagedV)
}

// HeadNormGate returns rms_norm(x) * sigmoid(gate): the per-head output gate
// the Qwen3.5 attention output gate maps to. x and gate must be equal length.
func HeadNormGate(x, gate []float32, eps float32) ([]float32, error) {
	if len(x) == 0 || len(x) != len(gate) {
		return nil, fmt.Errorf("HeadNormGate: x length %d and gate length %d must match and be nonzero", len(x), len(gate))
	}
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		sig := float32(1 / (1 + math.Exp(-float64(gate[i]))))
		out[i] = v * inv * sig
	}
	return out, nil
}
