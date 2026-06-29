//go:build amd64

package ggufload

import (
	"encoding/binary"
	"math/rand"
	"testing"
)

func TestKQuantDequantAVX2MatchesScalar(t *testing.T) {
	if !ggufLoadDequantAVX2() {
		t.Skip("AVX2 GGUF load dequant path unavailable")
	}
	for _, tc := range []struct {
		name       string
		blockBytes int
		blocks     int
		fixScales  func([]byte)
		scalar     func([]float32, []byte)
		arch       func([]float32, []byte) bool
	}{
		{
			name:       "Q4_K",
			blockBytes: blockQ4KBytes,
			blocks:     17,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ4KBytes; b++ {
					blk := raw[b*blockQ4KBytes:]
					binary.LittleEndian.PutUint16(blk[0:], finiteF16ForTest(b))
					binary.LittleEndian.PutUint16(blk[2:], finiteF16ForTest(b+3))
				}
			},
			scalar: dequantQ4KScalar,
			arch:   dequantQ4KArch,
		},
		{
			name:       "Q5_K",
			blockBytes: blockQ5KBytes,
			blocks:     17,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ5KBytes; b++ {
					blk := raw[b*blockQ5KBytes:]
					binary.LittleEndian.PutUint16(blk[0:], finiteF16ForTest(b+5))
					binary.LittleEndian.PutUint16(blk[2:], finiteF16ForTest(b+7))
				}
			},
			scalar: dequantQ5KScalar,
			arch:   dequantQ5KArch,
		},
		{
			name:       "Q6_K",
			blockBytes: blockQ6KBytes,
			blocks:     17,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ6KBytes; b++ {
					blk := raw[b*blockQ6KBytes:]
					binary.LittleEndian.PutUint16(blk[blockQ6KBytes-2:], finiteF16ForTest(b+11))
				}
			},
			scalar: dequantQ6KScalar,
			arch:   dequantQ6KArch,
		},
		{
			name:       "IQ3_XXS",
			blockBytes: blockIQ3XXSBytes,
			blocks:     17,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockIQ3XXSBytes; b++ {
					blk := raw[b*blockIQ3XXSBytes:]
					binary.LittleEndian.PutUint16(blk[0:], finiteF16ForTest(b+13))
				}
			},
			scalar: dequantIQ3XXSScalar,
			arch:   dequantIQ3XXSArch,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := randomKQuantRawForTest(tc.blocks, tc.blockBytes)
			tc.fixScales(raw)
			want := make([]float32, tc.blocks*qkK)
			got := make([]float32, tc.blocks*qkK)
			tc.scalar(want, raw)
			if !tc.arch(got, raw) {
				t.Fatalf("%s AVX2 path declined despite AVX2 gate", tc.name)
			}
			assertF32BitsEqual(t, tc.name+" scalar-vs-avx2", got, want)
		})
	}
}

func BenchmarkKQuantDequantLoadBody(b *testing.B) {
	for _, tc := range []struct {
		name       string
		blockBytes int
		scalar     func([]float32, []byte)
		arch       func([]float32, []byte) bool
		fixScales  func([]byte)
	}{
		{
			name:       "Q4_K",
			blockBytes: blockQ4KBytes,
			scalar:     dequantQ4KScalar,
			arch:       dequantQ4KArch,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ4KBytes; b++ {
					blk := raw[b*blockQ4KBytes:]
					binary.LittleEndian.PutUint16(blk[0:], finiteF16ForTest(b))
					binary.LittleEndian.PutUint16(blk[2:], finiteF16ForTest(b+3))
				}
			},
		},
		{
			name:       "Q5_K",
			blockBytes: blockQ5KBytes,
			scalar:     dequantQ5KScalar,
			arch:       dequantQ5KArch,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ5KBytes; b++ {
					blk := raw[b*blockQ5KBytes:]
					binary.LittleEndian.PutUint16(blk[0:], finiteF16ForTest(b+5))
					binary.LittleEndian.PutUint16(blk[2:], finiteF16ForTest(b+7))
				}
			},
		},
		{
			name:       "Q6_K",
			blockBytes: blockQ6KBytes,
			scalar:     dequantQ6KScalar,
			arch:       dequantQ6KArch,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ6KBytes; b++ {
					blk := raw[b*blockQ6KBytes:]
					binary.LittleEndian.PutUint16(blk[blockQ6KBytes-2:], finiteF16ForTest(b+11))
				}
			},
		},
		{
			name:       "IQ3_XXS",
			blockBytes: blockIQ3XXSBytes,
			scalar:     dequantIQ3XXSScalar,
			arch:       dequantIQ3XXSArch,
			fixScales: func(raw []byte) {
				for b := 0; b < len(raw)/blockIQ3XXSBytes; b++ {
					blk := raw[b*blockIQ3XXSBytes:]
					binary.LittleEndian.PutUint16(blk[0:], finiteF16ForTest(b+13))
				}
			},
		},
	} {
		const blocks = 4096
		raw := randomKQuantRawForTest(blocks, tc.blockBytes)
		tc.fixScales(raw)
		out := make([]float32, blocks*qkK)
		b.Run(tc.name+"/scalar", func(b *testing.B) {
			b.SetBytes(int64(len(raw)))
			for i := 0; i < b.N; i++ {
				tc.scalar(out, raw)
			}
		})
		b.Run(tc.name+"/avx2", func(b *testing.B) {
			if !ggufLoadDequantAVX2() {
				b.Skip("AVX2 GGUF load dequant path unavailable")
			}
			b.SetBytes(int64(len(raw)))
			for i := 0; i < b.N; i++ {
				if !tc.arch(out, raw) {
					b.Fatal("AVX2 path declined")
				}
			}
		})
	}
}

func randomKQuantRawForTest(blocks, blockBytes int) []byte {
	raw := make([]byte, blocks*blockBytes)
	rng := rand.New(rand.NewSource(1130))
	if _, err := rng.Read(raw); err != nil {
		panic(err)
	}
	return raw
}

func finiteF16ForTest(i int) uint16 {
	vals := [...]uint16{
		0x0000, // 0
		0x3400, // 0.25
		0x3800, // 0.5
		0x3c00, // 1
		0x4000, // 2
		0xbc00, // -1
		0xb800, // -0.5
	}
	return vals[i%len(vals)]
}
