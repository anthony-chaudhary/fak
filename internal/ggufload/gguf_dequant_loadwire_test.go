package ggufload

// gguf_dequant_loadwire_test.go — witnesses the #1130 seam: the public K-quant and
// IQ3_XXS load-path dequant entrypoints route through the SIMD arch unpack (composed
// with #1102's parFor via dequantBlocks) when the CPU provides it, and fall back to
// the scalar unpack otherwise. Either way the load output must stay bit-identical to
// the pure scalar unpack — the loader's correctness is the model's correctness
// (epic #1124 acceptance 1). This test is arch-agnostic: on amd64 with AVX2 it
// exercises the SIMD body, elsewhere (and under FAK_QKERNEL=scalar) it exercises the
// scalar fallback, and it is the same assertion both ways.

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"
)

func TestDequantLoadPathMatchesScalarWiredSIMD(t *testing.T) {
	cases := []struct {
		name       string
		blockBytes int
		entry      func([]float32, []byte) // wired load-path entrypoint (SIMD when present)
		scalar     func([]float32, []byte) // pure scalar reference
		scaleAt    []int                   // in-block byte offsets to stamp with finite f16 scales
	}{
		{"Q4_K", blockQ4KBytes, dequantQ4K, dequantQ4KScalar, []int{0, 2}},
		{"Q5_K", blockQ5KBytes, dequantQ5K, dequantQ5KScalar, []int{0, 2}},
		{"Q6_K", blockQ6KBytes, dequantQ6K, dequantQ6KScalar, []int{blockQ6KBytes - 2}},
		{"IQ3_XXS", blockIQ3XXSBytes, dequantIQ3XXS, dequantIQ3XXSScalar, []int{0}},
	}
	// 17 blocks stays under dequantParallelMinBlocks (single-worker body); the larger
	// count crosses it so the parFor chunking (#1102) is exercised alongside the SIMD
	// body, proving the two compose without drifting from the scalar reference.
	for _, blocks := range []int{17, dequantParallelMinBlocks + 3} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/%d-blocks", tc.name, blocks), func(t *testing.T) {
				raw := kquantRawFixtureForWire(blocks, tc.blockBytes, tc.scaleAt)
				want := make([]float32, blocks*qkK)
				got := make([]float32, blocks*qkK)
				tc.scalar(want, raw)
				tc.entry(got, raw)
				assertF32BitsEqual(t, tc.name, got, want)
			})
		}
	}
}

// kquantRawFixtureForWire builds a random raw payload of `blocks` super-blocks and
// stamps each block's f16 scale field(s) (at scaleOffsets) with finite half-floats so
// scalar and SIMD dequant produce comparable finite values. The quantized code bytes
// stay random — every byte is a valid code/grid index for the K-quant and IQ3_XXS
// layouts, so no clamping is needed.
func kquantRawFixtureForWire(blocks, blockBytes int, scaleOffsets []int) []byte {
	raw := make([]byte, blocks*blockBytes)
	rng := rand.New(rand.NewSource(int64(1130 + blockBytes)))
	if _, err := rng.Read(raw); err != nil {
		panic(err)
	}
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockBytes:]
		for k, off := range scaleOffsets {
			binary.LittleEndian.PutUint16(blk[off:], finiteScaleF16ForWire(b+k))
		}
	}
	return raw
}

// finiteScaleF16ForWire cycles a handful of finite half-float bit patterns (±1, ±0.5,
// 2, 0.25) so stamped block scales never introduce NaN/Inf into the bit comparison.
func finiteScaleF16ForWire(i int) uint16 {
	vals := [...]uint16{0x3c00, 0x3800, 0x4000, 0xbc00, 0x3400, 0xb800}
	return vals[i%len(vals)]
}
