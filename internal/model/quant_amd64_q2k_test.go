//go:build amd64

package model

import (
	"math"
	"math/rand"
	"testing"
)

func TestQ2KDequantAsmMatchesScalar(t *testing.T) {
	t.Run("AVX2", func(t *testing.T) {
		if !detectAVX2() {
			t.Skip("AVX2 unavailable")
		}
		testQ2KDequantAsmMatchesScalar(t, q2kDequantSuperBlockAsmAVX2)
	})
	t.Run("AVX512", func(t *testing.T) {
		if !detectAVX512() {
			t.Skip("AVX-512 unavailable")
		}
		testQ2KDequantAsmMatchesScalar(t, q2kDequantSuperBlockAsmAVX512)
	})
	t.Run("dispatch", func(t *testing.T) {
		rng := rand.New(rand.NewSource(11947))
		for block := 0; block < 10_000; block++ {
			blk := make([]byte, q2kBlockBytes)
			_, _ = rng.Read(blk)
			var want, got [qkK]float32
			q2kDequantSuperBlockScalar(want[:], blk)
			q2kDequantSuperBlock(got[:], blk)
			assertQ2KBitsEqual(t, block, got[:], want[:])
		}
	})
}

func testQ2KDequantAsmMatchesScalar(t *testing.T, kernel func(*float32, *byte, *float32)) {
	t.Helper()
	rng := rand.New(rand.NewSource(11947))
	const blocks = 10_000
	for block := 0; block < blocks; block++ {
		blk := make([]byte, q2kBlockBytes)
		_, _ = rng.Read(blk)
		var want, got [qkK]float32
		q2kDequantSuperBlockScalar(want[:], blk)
		tables := q2kLookupTables(blk)
		kernel(&got[0], &blk[qkK/16], &tables[0][0])
		assertQ2KBitsEqual(t, block, got[:], want[:])
	}
}

func assertQ2KBitsEqual(t *testing.T, block int, got, want []float32) {
	t.Helper()
	for lane := range want {
		if math.Float32bits(got[lane]) != math.Float32bits(want[lane]) {
			t.Fatalf("block %d lane %d: got %#08x (%g), want %#08x (%g)",
				block, lane, math.Float32bits(got[lane]), got[lane],
				math.Float32bits(want[lane]), want[lane])
		}
	}
}

func q2kLookupTables(blk []byte) [qkK / 16][4]float32 {
	dm := qkK/16 + qkK/4
	d := math.Float32frombits(F16BitsToF32Bits(uint16(blk[dm]) | uint16(blk[dm+1])<<8))
	minVal := math.Float32frombits(F16BitsToF32Bits(uint16(blk[dm+2]) | uint16(blk[dm+3])<<8))
	var tables [qkK / 16][4]float32
	for i, sc := range blk[:qkK/16] {
		dl, ml := d*float32(sc&0x0f), minVal*float32(sc>>4)
		tables[i] = [4]float32{0 - ml, dl - ml, dl*2 - ml, dl*3 - ml}
	}
	return tables
}

func benchmarkQ2KDequant(b *testing.B, kernel func([]float32, []byte)) {
	b.Helper()
	rng := rand.New(rand.NewSource(11947))
	blk := make([]byte, q2kBlockBytes)
	_, _ = rng.Read(blk)
	dst := make([]float32, qkK)
	b.SetBytes(qkK * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kernel(dst, blk)
	}
}

func BenchmarkQ2KDequantScalar(b *testing.B) {
	benchmarkQ2KDequant(b, q2kDequantSuperBlockScalar)
}

func BenchmarkQ2KDequantAVX2(b *testing.B) {
	if !detectAVX2() {
		b.Skip("AVX2 unavailable")
	}
	benchmarkQ2KDequant(b, func(dst []float32, blk []byte) {
		tables := q2kLookupTables(blk)
		q2kDequantSuperBlockAsmAVX2(&dst[0], &blk[qkK/16], &tables[0][0])
	})
}

func BenchmarkQ2KDequantAVX2AsmOnly(b *testing.B) {
	if !detectAVX2() {
		b.Skip("AVX2 unavailable")
	}
	rng := rand.New(rand.NewSource(11947))
	blk := make([]byte, q2kBlockBytes)
	_, _ = rng.Read(blk)
	tables := q2kLookupTables(blk)
	dst := make([]float32, qkK)
	b.SetBytes(qkK * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q2kDequantSuperBlockAsmAVX2(&dst[0], &blk[qkK/16], &tables[0][0])
	}
}

func BenchmarkQ2KLookupTables(b *testing.B) {
	rng := rand.New(rand.NewSource(11947))
	blk := make([]byte, q2kBlockBytes)
	_, _ = rng.Read(blk)
	var tables [qkK / 16][4]float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tables = q2kLookupTables(blk)
	}
	_ = tables
}

func BenchmarkQ2KDequantAVX512(b *testing.B) {
	if !detectAVX512() {
		b.Skip("AVX-512 unavailable")
	}
	benchmarkQ2KDequant(b, func(dst []float32, blk []byte) {
		tables := q2kLookupTables(blk)
		q2kDequantSuperBlockAsmAVX512(&dst[0], &blk[qkK/16], &tables[0][0])
	})
}

func BenchmarkQ2KDequantDispatch(b *testing.B) {
	benchmarkQ2KDequant(b, q2kDequantSuperBlock)
}

func BenchmarkQ2KDequantSuperBlock(b *testing.B) {
	b.Run("scalar", BenchmarkQ2KDequantScalar)
	b.Run("avx2", BenchmarkQ2KDequantAVX2)
	b.Run("avx512", BenchmarkQ2KDequantAVX512)
	b.Run("dispatch", BenchmarkQ2KDequantDispatch)
}
