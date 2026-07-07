//go:build amd64

package model

import (
	"encoding/binary"
	"testing"
)

// quant_iquant_bench_test.go — the witnessed scalar→AVX2 microbench for the resident IQ3_XXS
// decode dequant. Both benches dequantize one 4096-wide weight row's worth of super-blocks
// (16 blocks) — the per-row work kQuantMatRowsRange pays before the f32 dot — calling the two
// kernels directly. (qtier is init-resolved, so an env toggle can't switch paths at runtime;
// calling the functions directly is also how the Q6_K f32/int8 benches isolate their kernels.)
// The ratio Scalar_ns / AVX2_ns is the reported speedup.

const iq3xxsBenchBlocks = 4096 / qkK // 16 super-blocks = one 4096-wide row

var iq3xxsBenchSink float32 // defeats dead-code elimination of the dequant writes

func iq3xxsBenchRaw() []byte {
	raw := make([]byte, iq3xxsBenchBlocks*iq3xxsBlockBytes)
	lcgBytes(raw, 0x1337C0DEA5A5F00D)
	for b := 0; b < iq3xxsBenchBlocks; b++ {
		binary.LittleEndian.PutUint16(raw[b*iq3xxsBlockBytes:], f16One)
	}
	return raw
}

func BenchmarkIQ3XXSDecodeDequantScalar(b *testing.B) {
	raw := iq3xxsBenchRaw()
	dst := make([]float32, qkK)
	var s float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for blk := 0; blk < iq3xxsBenchBlocks; blk++ {
			iq3xxsDequantSuperBlock(dst, raw[blk*iq3xxsBlockBytes:(blk+1)*iq3xxsBlockBytes])
			s += dst[0]
		}
	}
	iq3xxsBenchSink = s
}

func BenchmarkIQ3XXSDecodeDequantAVX2(b *testing.B) {
	if qtier < tierAVX2 {
		b.Skip("AVX2 not available — iq3xxs asm inactive")
	}
	raw := iq3xxsBenchRaw()
	dst := make([]float32, qkK)
	var s float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for blk := 0; blk < iq3xxsBenchBlocks; blk++ {
			iq3xxsDequantSuperBlockArch(dst, raw[blk*iq3xxsBlockBytes:(blk+1)*iq3xxsBlockBytes])
			s += dst[0]
		}
	}
	iq3xxsBenchSink = s
}
