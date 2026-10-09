package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// The FP8-to-BF16 lookup bytes below are adapted from ktransformers
// GemmKernel224FP8::bf16_{hi,lo}_{0,1}_val and fp8x64_to_bf16x64:
// https://github.com/kvcache-ai/ktransformers/blob/0c2912a5e25dc3459d728a59be2d8c7461ba64bd/kt-kernel/operators/amx/la/amx_raw_kernels.hpp
// Licensed under Apache-2.0; see the repository LICENSE. Modified for Fak:
// concatenate each pair of 64-byte tables, replace SIMD permutation with a Go
// scalar lookup, and preserve E4M3FN NaNs instead of decoding 0x7f as 480.
// Only the finite-byte decode is borrowed. Scale application and Q8 rounding
// retain Fak's existing expand-then-quantize arithmetic, not upstream's BF16
// activation or scale-after-accumulation GEMM semantics (partial fak#5240).
var fp8BF16Hi = [128]byte{
	0x00, 0x3b, 0x3b, 0x3b, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c,
	0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d,
	0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e,
	0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
	0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
	0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
	0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43,
}

var fp8BF16Lo = [128]byte{
	0x00, 0x00, 0x80, 0xc0, 0x00, 0x20, 0x40, 0x60, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
	0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
	0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
	0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
	0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
	0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
	0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
	0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
}

func fp8E4M3Lookup(b byte) float32 {
	i := b & 0x7f
	if i == 0x7f {
		// Match mathx.DecodeE4M3's canonical NaN bits for either sign.
		return float32(math.NaN())
	}
	return math.Float32frombits(uint32(fp8BF16Hi[i]|(b&0x80))<<24 | uint32(fp8BF16Lo[i])<<16)
}

// fp8DirectQ8Name admits only the ordinary quant-weight branch of
// quantizeDecodedFloatTensorInto. Embeddings, fused projections and source MoE
// layouts still need their existing transforms, including malformed-shape errors.
func fp8DirectQ8Name(cfg Config, name string, shape []int) (string, bool) {
	canonical, keep := quantSourceTensorName(cfg, name)
	if !keep || len(shape) != 2 || !isQuantWeight(canonical) {
		return "", false
	}
	// This GPT-OSS source tensor matches isQuantWeight even without an expert
	// index; its rank-3 transpose/shape check must run before the ordinary branch.
	if strings.HasSuffix(canonical, ".mlp.experts.down_proj.weight") {
		return "", false
	}
	return canonical, true
}

// quantizeFP8BlockScaleQ8 converts the checkpoint's 128x128-scaled E4M3 bytes
// directly into the existing Q8 resident representation. Each worker widens one
// 32-element block at a time, avoiding the whole-tensor float32 decode and byte
// serialization temporaries. q8PrepareAccelWeight remains unchanged and may
// deliberately allocate an accelerator-specific whole-matrix cache.
func quantizeFP8BlockScaleQ8(name string, shape []int, weight, scaleF32 []byte) (*q8Tensor, error) {
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 {
		return nil, fmt.Errorf("fp8 block-scale %s: shape %v, want positive rank-2 [O,I]", name, shape)
	}
	out, in := shape[0], shape[1]
	elems, ok := checkedShapeProduct(out, in)
	if !ok || len(weight) != elems {
		return nil, fmt.Errorf("fp8 block-scale %s: weight has %d bytes for shape %v", name, len(weight), shape)
	}
	if in%qBlk != 0 {
		return nil, fmt.Errorf("fp8 block-scale %s: Q8_0 reduction dim %d is not a multiple of %d", name, in, qBlk)
	}
	scaleRows, scaleCols := (out-1)/fp8BlockDim+1, (in-1)/fp8BlockDim+1
	scaleElems, ok := checkedShapeProduct(scaleRows, scaleCols)
	if !ok || len(scaleF32)%4 != 0 || len(scaleF32)/4 != scaleElems {
		return nil, fmt.Errorf("fp8 block-scale %s: scale has %d bytes, want %d float32 values", name, len(scaleF32), scaleElems)
	}
	nblk := in / qBlk
	qt := newQ8Tensor(out, in, nblk)
	parFor(out, currentWorkerCount(), func(lo, hi int) {
		var block [qBlk]float32
		for o := lo; o < hi; o++ {
			scaleRow := (o / fp8BlockDim) * scaleCols
			for b := 0; b < nblk; b++ {
				col := b * qBlk
				base := o*in + col
				si := scaleRow + col/fp8BlockDim
				scale := math.Float32frombits(binary.LittleEndian.Uint32(scaleF32[si*4:]))
				for i := range block {
					block[i] = fp8E4M3Lookup(weight[base+i]) * scale
				}
				// qBlk divides fp8BlockDim, so a Q8 block never crosses a scale tile.
				quantizeRowQ8scalar(block[:], qt.q[base:base+qBlk], qt.d[o*nblk+b:o*nblk+b+1], 1)
			}
		}
	})
	q8PrepareAccelWeight(qt)
	return qt, nil
}
