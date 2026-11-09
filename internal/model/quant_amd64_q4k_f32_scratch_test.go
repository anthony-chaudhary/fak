//go:build amd64

package model

import (
	"bytes"
	"math"
	"math/rand"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestQ4KExactDecodeBoundedScaleScratch(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skipf("exact AVX2 Q4_K decode is unavailable at tier %d", qtier)
	}
	for _, width := range []int{256, 4096, 5120, 13824, 32768, 33024} {
		t.Run(strconv.Itoa(width), func(t *testing.T) {
			const rows, lo, hi = 5, 1, 4
			rng := rand.New(rand.NewSource(int64(width)))
			nblk := width / qkK
			raw := make([]byte, rows*nblk*q4kBlockBytes)
			alternate := make([]byte, len(raw))
			for _, data := range [][]byte{raw, alternate} {
				for offset := 0; offset < len(data); offset += q4kBlockBytes {
					randQ4KBlock(rng, data[offset:offset+q4kBlockBytes])
				}
			}
			qt := quantizeQ4KFromRaw(raw, rows, width)
			x := make([]float32, width)
			for i := range x {
				x[i] = float32(rng.NormFloat64()) * float32(1+i%7)
			}
			xBefore := append([]float32(nil), x...)
			rawBefore := append([]byte(nil), qt.raw...)
			alternateBefore := append([]byte(nil), alternate...)
			for _, override := range [][]byte{nil, alternate} {
				want, got := make([]float32, rows+2), make([]float32, rows+2)
				for i := range want {
					want[i], got[i] = -321.25, -321.25
				}
				q4kMatRowsRangeRaw(override, qt, x, want, lo, hi)
				for repeat := 0; repeat < 3; repeat++ {
					if !q4kMatRowsRangeArchRaw(override, qt, x, got, lo, hi) {
						t.Fatal("exact AVX2 route declined")
					}
					for i := range want {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("override=%v repeat=%d row=%d: bits %08x, want %08x", override != nil, repeat, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
						}
					}
				}
				allocs := testing.AllocsPerRun(10, func() {
					q4kMatRowsRangeArchRaw(override, qt, x, got, lo, hi)
				})
				t.Logf("width=%d override=%v allocations/range=%g", width, override != nil, allocs)
				if width <= 32768 && allocs != 0 {
					t.Fatalf("bounded exact decode allocated %g times, want zero scale/min heap scratch", allocs)
				}
			}
			if !bytes.Equal(qt.raw, rawBefore) || !bytes.Equal(alternate, alternateBefore) {
				t.Fatal("decode modified resident or alternate weight storage")
			}
			for i := range x {
				if math.Float32bits(x[i]) != math.Float32bits(xBefore[i]) {
					t.Fatalf("decode modified activation %d", i)
				}
			}
		})
	}
}

func BenchmarkQ4KExactDecodeScaleScratch(b *testing.B) {
	if qtier < tierAVX2 {
		b.Skipf("exact AVX2 Q4_K decode is unavailable at tier %d", qtier)
	}
	for _, width := range []int{5120, 13824, 32768, 33024} {
		b.Run(strconv.Itoa(width), func(b *testing.B) {
			qt, x := benchQ4KFixture(b, 32, width)
			y := make([]float32, qt.out)
			q4kMatRowsRangeArch(qt, x, y, 0, qt.out)
			b.ReportAllocs()
			b.SetBytes(int64(len(qt.raw)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q4kMatRowsRangeArch(qt, x, y, 0, qt.out)
			}
		})
	}
}
