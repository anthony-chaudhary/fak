//go:build amd64

package model

// awq_simd_amd64_test.go — the correctness + bench witness for the AWQ AVX2/AVX-512
// kernels (#1124 C4 / #1128). DEQUANT is asserted BIT-IDENTICAL (==) to the scalar
// reference; DOT is asserted COSINE-PARITY (lane-reduced sum) with a tight relative
// bound. The bench ladder (scalar → AVX2 → AVX-512) quotes the witnessed multiple.

import (
	"math"
	"math/rand"
	"testing"
)

// awqRandRow returns nb packed bytes (= 2*nb weights) and 2*nb random activations.
func awqRandRow(rng *rand.Rand, nb int) (packed []byte, x []float32) {
	packed = make([]byte, nb)
	x = make([]float32, nb*2)
	for i := range packed {
		packed[i] = byte(rng.Intn(256))
	}
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	return packed, x
}

// awqEdgeRow returns a row whose every byte is fill — exercises the zero-point edge
// codes: 0x00 → both nibbles -8, 0xff → both +7, 0x88 → both 0.
func awqEdgeRow(fill byte, nb int) []byte {
	p := make([]byte, nb)
	for i := range p {
		p[i] = fill
	}
	return p
}

// awqTestSizes spans pure-block, block+tail, and large rows for both ISA block widths
// (AVX2 = 4 bytes, AVX-512 = 8 bytes). n is the weight count (even).
var awqTestSizes = []int{8, 16, 32, 64, 128, 256, 4096, 20, 100, 4090, 514}

func TestAWQDequantRowAVX2MatchesScalar(t *testing.T) {
	if !detectAVX2() {
		t.Skip("no AVX2 on this host")
	}
	rng := rand.New(rand.NewSource(1))
	for _, n := range awqTestSizes {
		nb := n / 2
		packed, _ := awqRandRow(rng, nb)
		assertDequantBitIdentical(t, "AVX2/random", n, packed, 0.137, awqDequantRowAVX2)
	}
	for _, fill := range []byte{0x00, 0xff, 0x88, 0x0f, 0xf0} {
		packed := awqEdgeRow(fill, 64)
		assertDequantBitIdentical(t, "AVX2/edge", 128, packed, 0.5, awqDequantRowAVX2)
	}
}

func TestAWQDequantRowAVX512MatchesScalar(t *testing.T) {
	if !detectAVX512() {
		t.Skip("no AVX-512 on this host")
	}
	rng := rand.New(rand.NewSource(2))
	for _, n := range awqTestSizes {
		nb := n / 2
		packed, _ := awqRandRow(rng, nb)
		assertDequantBitIdentical(t, "AVX512/random", n, packed, 0.137, awqDequantRowAVX512)
	}
	for _, fill := range []byte{0x00, 0xff, 0x88, 0x0f, 0xf0} {
		packed := awqEdgeRow(fill, 64)
		assertDequantBitIdentical(t, "AVX512/edge", 128, packed, 0.5, awqDequantRowAVX512)
	}
}

func assertDequantBitIdentical(t *testing.T, tag string, n int, packed []byte, scale float32,
	fn func(dst []float32, scale float32, src *byte, n int)) {
	t.Helper()
	want := make([]float32, n)
	got := make([]float32, n)
	awqDequantRowScalar(want, scale, &packed[0], n)
	fn(got, scale, &packed[0], n)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s n=%d: dequant[%d] = %v (bits %#x), want %v (bits %#x) — not bit-identical",
				tag, n, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestAWQDotProductAVX2MatchesScalar(t *testing.T) {
	if !detectAVX2() {
		t.Skip("no AVX2 on this host")
	}
	assertDotCosineParity(t, "AVX2", rand.New(rand.NewSource(3)), awqDotProductAVX2)
}

func TestAWQDotProductAVX512MatchesScalar(t *testing.T) {
	if !detectAVX512() {
		t.Skip("no AVX-512 on this host")
	}
	assertDotCosineParity(t, "AVX512", rand.New(rand.NewSource(4)), awqDotProductAVX512)
}

func assertDotCosineParity(t *testing.T, tag string, rng *rand.Rand,
	fn func(src *byte, scale float32, x *float32, n int) float32) {
	t.Helper()
	// Per-row relative bound + an aggregate cosine across all rows' dot values.
	var scalarVec, asmVec []float32
	for _, n := range awqTestSizes {
		nb := n / 2
		packed, x := awqRandRow(rng, nb)
		scale := rng.Float32()*0.2 + 0.01
		want := awqDotProductScalar(&packed[0], scale, &x[0], n)
		got := fn(&packed[0], scale, &x[0], n)
		rel := relErr(got, want)
		if rel > 1e-4 {
			t.Fatalf("%s n=%d: dot = %v, scalar = %v, rel err %.3g > 1e-4", tag, n, got, want, rel)
		}
		scalarVec = append(scalarVec, want)
		asmVec = append(asmVec, got)
	}
	if sim := cosineSimilarity(asmVec, scalarVec); sim < 0.99999 {
		t.Fatalf("%s: cosine(asm, scalar) over %d rows = %f, want ≥0.99999", tag, len(asmVec), sim)
	}
}

func relErr(got, want float32) float32 {
	d := got - want
	if d < 0 {
		d = -d
	}
	a := want
	if a < 0 {
		a = -a
	}
	if a < 1 {
		a = 1
	}
	return d / a
}

// ---- bench ladder: scalar → AVX2 → AVX-512 ----------------------------------

const awqBenchN = 4096

func benchAWQDot(b *testing.B, fn func(src *byte, scale float32, x *float32, n int) float32) {
	rng := rand.New(rand.NewSource(7))
	packed, x := awqRandRow(rng, awqBenchN/2)
	b.SetBytes(int64(awqBenchN / 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fn(&packed[0], 0.1, &x[0], awqBenchN)
	}
}

func benchAWQDequant(b *testing.B, fn func(dst []float32, scale float32, src *byte, n int)) {
	rng := rand.New(rand.NewSource(8))
	packed, _ := awqRandRow(rng, awqBenchN/2)
	dst := make([]float32, awqBenchN)
	b.SetBytes(int64(awqBenchN / 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, 0.1, &packed[0], awqBenchN)
	}
}

// Distinct names from awq_test.go's BenchmarkAWQDotProductScalar; this is the
// full scalar→AVX2→AVX-512 ladder over one fixed random row.
func BenchmarkAWQDotLadderScalar(b *testing.B) { benchAWQDot(b, awqDotProductScalar) }
func BenchmarkAWQDotLadderAVX2(b *testing.B) {
	if !detectAVX2() {
		b.Skip("no AVX2")
	}
	benchAWQDot(b, awqDotProductAVX2)
}
func BenchmarkAWQDotLadderAVX512(b *testing.B) {
	if !detectAVX512() {
		b.Skip("no AVX-512")
	}
	benchAWQDot(b, awqDotProductAVX512)
}

func BenchmarkAWQDequantLadderScalar(b *testing.B) { benchAWQDequant(b, awqDequantRowScalar) }
func BenchmarkAWQDequantLadderAVX2(b *testing.B) {
	if !detectAVX2() {
		b.Skip("no AVX2")
	}
	benchAWQDequant(b, awqDequantRowAVX2)
}
func BenchmarkAWQDequantLadderAVX512(b *testing.B) {
	if !detectAVX512() {
		b.Skip("no AVX-512")
	}
	benchAWQDequant(b, awqDequantRowAVX512)
}
