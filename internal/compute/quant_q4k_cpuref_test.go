package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// quant_q4k_cpuref_test.go — independent witness for the cpu-ref Q4_K MatMul (quant_q4k.go).
// It checks the dequant-and-dot against a SEPARATE computation: dequant the Q4_K weight to a full
// f32 matrix and run the F32 MatMul (an 8-accumulator fdot — a different reduction than the Q4_K
// 4-accumulator dot, so this is not a tautology). The two must agree to a tight cosine and pick the
// same argmax. Also pins the resident-byte win (144 B / 256 weights). The model package holds the
// bit-exact-vs-model-reference witness; this keeps the backend self-tested.

// randQ4KBlockC fills a 144-byte super-block with random codes/scales, constraining the f16 d/dmin
// (bytes 0..3) to a small finite exponent so no NaN/Inf scale poisons the dot (mirrors the model
// test's randQ4KBlock).
func randQ4KBlockC(rng *rand.Rand, blk []byte) {
	for i := range blk {
		blk[i] = byte(rng.Intn(256))
	}
	// d, dmin as small positive f16: exponent 0x0c..0x0e (≈ 2^-3..2^-1), random 10-bit fraction.
	put := func(off int) {
		exp := uint16(0x0c + rng.Intn(3))
		frac := uint16(rng.Intn(1024))
		binary.LittleEndian.PutUint16(blk[off:], (exp<<10)|frac)
	}
	put(0)
	put(2)
}

func cosineC(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestCPURefQ4KMatMulVsF32Dequant(t *testing.T) {
	const out, in = 12, 768 // 3 super-blocks per row
	nblk := in / q4kSuper
	raw := make([]byte, out*nblk*q4kSuperBlock)
	rng := rand.New(rand.NewSource(5))
	for b := 0; b < out*nblk; b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}

	// Full f32 dequant of the whole weight, row-major [out,in].
	wf := make([]float32, out*in)
	buf := make([]float32, q4kSuper)
	rowBytes := nblk * q4kSuperBlock
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			q4kDequantBlock(buf, raw[o*rowBytes+b*q4kSuperBlock:o*rowBytes+(b+1)*q4kSuperBlock])
			copy(wf[o*in+b*q4kSuper:], buf)
		}
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	be := Default()
	yQ4K := be.Read(be.MatMul(NewQ4K(be, []int{out, in}, raw), be.Upload(NewF32(be, []int{in}, x), F32)))
	yF32 := be.Read(be.MatMul(NewF32(be, []int{out, in}, wf), be.Upload(NewF32(be, []int{in}, x), F32)))

	if a, b := argmaxF32(yQ4K), argmaxF32(yF32); a != b {
		t.Fatalf("Q4_K argmax %d != f32-dequant argmax %d", a, b)
	}
	if c := cosineC(yQ4K, yF32); c < 0.99999 {
		t.Fatalf("Q4_K vs f32-dequant cosine %.8f < 0.99999 (dequant/dot mismatch)", c)
	}

	// Resident-byte win: a Q4_K weight is 144 B per 256-weight super-block, far below f32's 1024.
	if got, want := len(raw), out*nblk*144; got != want {
		t.Fatalf("Q4_K raw bytes %d, want %d (144/super-block)", got, want)
	}
}
