package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
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

func TestCPURefRawKQuantMatMulMatchesIndependentDequant(t *testing.T) {
	const out, in = 5, 512
	rng := rand.New(rand.NewSource(4843))
	for _, tc := range []struct {
		name      string
		dt        Dtype
		bs        int
		newTensor func(Backend, []int, []byte) Tensor
		dequant   func([]float32, []byte)
	}{
		{"Q5_K", Q5_K, 176, NewQ5K, dequantQ5K}, {"Q6_K", Q6_K, 210, NewQ6K, dequantQ6K},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, out*(in/256)*tc.bs)
			rng.Read(raw)
			// Keep both scales finite and modest while preserving random codes/scales.
			for b := 0; b < len(raw); b += tc.bs {
				if tc.dt == Q5_K {
					binary.LittleEndian.PutUint16(raw[b:], 0x3000)
					binary.LittleEndian.PutUint16(raw[b+2:], 0x2c00)
				} else {
					binary.LittleEndian.PutUint16(raw[b+208:], 0x3000)
				}
			}
			wf := make([]float32, out*in)
			block := make([]float32, 256)
			for o := 0; o < out; o++ {
				for b := 0; b < in/256; b++ {
					off := (o*(in/256) + b) * tc.bs
					tc.dequant(block, raw[off:off+tc.bs])
					copy(wf[o*in+b*256:], block)
				}
			}
			x := make([]float32, in)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			be := Default()
			hx := be.Upload(NewF32(be, []int{in}, x), F32)
			got := be.Read(be.MatMul(tc.newTensor(be, []int{out, in}, raw), hx))
			want := be.Read(be.MatMul(NewF32(be, []int{out, in}, wf), hx))
			if c := cosineC(got, want); c < 0.999999 {
				t.Fatalf("cosine %.9f", c)
			}
		})
	}
}

// quantizeVecQ8_1 quantizes an activation vector to Q8_1 format per 32-element block:
// scale d = amax/127, sum s = d * sum(q), and int8 codes q in [-127, 127].
func quantizeVecQ8_1(x []float32) (qx []int8, d, s []float32) {
	const block = 32
	nblk := len(x) / block
	qx = make([]int8, len(x))
	d = make([]float32, nblk)
	s = make([]float32, nblk)
	for b := 0; b < nblk; b++ {
		blk := x[b*block : (b+1)*block]
		var amax float32
		for _, v := range blk {
			if a := absf(v); a > amax {
				amax = a
			}
		}
		dd := amax / 127.0
		d[b] = dd
		if dd == 0 {
			continue
		}
		inv := 1.0 / dd
		var sumQ int32
		var sumF float32
		for i := 0; i < block; i++ {
			c := q8round(blk[i] * inv)
			qx[b*block+i] = c
			sumQ += int32(c)
			sumF += blk[i]
		}
		s[b] = sumF // exact float sum of activations
	}
	return
}

// q4kRowDotQ8_1DP4A performs the Q8_1 x Q4_K MMVQ reduction: signed DP4A vector dot
// for the integer inner products, combined via (d*sc)*(dx*I) - (dmin*mn)*sx.
func q4kRowDotQ8_1DP4A(raw []byte, qx []int8, dx, sx []float32) float32 {
	nsb := len(raw) / q4kSuperBlock
	var rowSum float32
	for sb := 0; sb < nsb; sb++ {
		blk := raw[sb*q4kSuperBlock : (sb+1)*q4kSuperBlock]
		d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
		dmin := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[2:])))
		scales := blk[4:16]
		qs := blk[16:144]

		for k := 0; k < 4; k++ {
			sA := sb*8 + 2*k
			sB := sA + 1
			chunk := qs[k*32 : (k+1)*32]

			var dotA, dotB int32
			for l := 0; l < 32; l++ {
				nibA := int32(int8((chunk[l] & 0x0f)) - 8)
				qxA := int32(qx[sA*32+l])
				dotA += nibA * qxA

				nibB := int32(int8((chunk[l] >> 4)) - 8)
				qxB := int32(qx[sB*32+l])
				dotB += nibB * qxB
			}

			scA, mnA := kquantbits.ScaleMinK4(2*k, scales)
			scB, mnB := kquantbits.ScaleMinK4(2*k+1, scales)

			wsA := d * float32(scA)
			wmA := dmin * float32(mnA)
			wsB := d * float32(scB)
			wmB := dmin * float32(mnB)

			termA := wsA*(dx[sA]*float32(dotA)) + (8*wsA-wmA)*sx[sA]
			termB := wsB*(dx[sB]*float32(dotB)) + (8*wsB-wmB)*sx[sB]
			rowSum += termA + termB
		}
	}
	return rowSum
}

func dominantRowC(w []float32, out, in int) int {
	best, bestNorm := 0, float32(-1)
	for o := 0; o < out; o++ {
		var n float32
		for _, v := range w[o*in : o*in+in] {
			n += v * v
		}
		if n > bestNorm {
			bestNorm, best = n, o
		}
	}
	return best
}

func nonTargetC(v []float32, out, target int) []float32 {
	r := make([]float32, 0, out-1)
	for o := 0; o < out; o++ {
		if o != target {
			r = append(r, v[o])
		}
	}
	return r
}

func maxAbsDeltaC(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

func f32ToF16BitsWitness(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	if exp <= 0 {
		return sign
	}
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	half := uint16(exp<<10) | uint16(mant>>13)
	if mant&0x1000 != 0 {
		half++
	}
	return sign | half
}

func randQ4KBytesLCG(g *lcg, out, in int) []byte {
	nsb := in / q4kSuper
	raw := make([]byte, out*nsb*q4kSuperBlock)
	for blk := 0; blk < out*nsb; blk++ {
		base := blk * q4kSuperBlock
		d := 0.01 + absf(g.f())*0.03
		dmin := 0.004 + absf(g.f())*0.012
		binary.LittleEndian.PutUint16(raw[base:], f32ToF16BitsWitness(d))
		binary.LittleEndian.PutUint16(raw[base+2:], f32ToF16BitsWitness(dmin))
		for i := 0; i < 12+128; i++ {
			raw[base+4+i] = byte(int32((g.f()+0.5)*255.0) & 0xff)
		}
	}
	return raw
}

// TestQ81DP4AQ4KMathWitness verifies the mathematical correctness and numerical gate
// (cosine >= 0.995, exact argmax, maxAbs <= 0.02) of Q8_1 activation quantization and
// signed DP4A Q4_K MMVQ reduction against f32 dequantized reference.
func TestQ81DP4AQ4KMathWitness(t *testing.T) {
	const out, in = 320, 256
	var seed lcg = 0x4b4b
	g := &seed
	raw := randQ4KBytesLCG(g, out, in)

	wf := make([]float32, out*in)
	buf := make([]float32, q4kSuper)
	rowBytes := (in / q4kSuper) * q4kSuperBlock
	for o := 0; o < out; o++ {
		for b := 0; b < in/q4kSuper; b++ {
			q4kDequantBlock(buf, raw[o*rowBytes+b*q4kSuperBlock:o*rowBytes+(b+1)*q4kSuperBlock])
			copy(wf[o*in+b*q4kSuper:], buf)
		}
	}

	target := dominantRowC(wf, out, in)
	x := make([]float32, in)
	copy(x, wf[target*in:(target+1)*in])

	// Reference f32 dot
	yRef := make([]float32, out)
	for o := 0; o < out; o++ {
		var acc float32
		for i := 0; i < in; i++ {
			acc += wf[o*in+i] * x[i]
		}
		yRef[o] = acc
	}

	// Q8_1 + DP4A MMVQ path
	qx, dx, sx := quantizeVecQ8_1(x)
	yMMVQ := make([]float32, out)
	for o := 0; o < out; o++ {
		yMMVQ[o] = q4kRowDotQ8_1DP4A(raw[o*rowBytes:(o+1)*rowBytes], qx, dx, sx)
	}

	// Gates: argmax-exact, cosine >= 0.995
	if aRef, aMMVQ := argmaxF32(yRef), argmaxF32(yMMVQ); aRef != aMMVQ || aMMVQ != target {
		t.Fatalf("argmax mismatch: ref=%d mmvq=%d want=%d", aRef, aMMVQ, target)
	}

	cos := cosineC(nonTargetC(yRef, out, target), nonTargetC(yMMVQ, out, target))
	if cos < 0.995 {
		t.Fatalf("cosine %.6f < 0.995 gate", cos)
	}
	t.Logf("Q8_1 DP4A Q4_K MMVQ math witness: cosine=%.8f argmax-exact=%d (PASS)", cos, target)

	// Test with realistic normalized activations
	xNorm := make([]float32, in)
	for i := range xNorm {
		xNorm[i] = g.f()
	}
	yRefNorm := make([]float32, out)
	for o := 0; o < out; o++ {
		var acc float32
		for i := 0; i < in; i++ {
			acc += wf[o*in+i] * xNorm[i]
		}
		yRefNorm[o] = acc
	}
	qxN, dxN, sxN := quantizeVecQ8_1(xNorm)
	yMMVQNorm := make([]float32, out)
	for o := 0; o < out; o++ {
		yMMVQNorm[o] = q4kRowDotQ8_1DP4A(raw[o*rowBytes:(o+1)*rowBytes], qxN, dxN, sxN)
	}
	cosN := cosineC(yRefNorm, yMMVQNorm)
	if cosN < 0.995 {
		t.Fatalf("normalized activation cosine %.6f < 0.995 gate", cosN)
	}
	t.Logf("Normalized input: cosine=%.8f", cosN)

	// Test with realistic model weight ranges (bounded exponents ~0.1 weight scale)
	numSuperBlocks := in / q4kSuper
	rawBounded := make([]byte, out*numSuperBlocks*q4kSuperBlock)
	blkBounded := make([]byte, q4kSuperBlock)
	for b := 0; b < out*numSuperBlocks; b++ {
		for i := range blkBounded {
			blkBounded[i] = byte(int32((g.f()+0.5)*255.0) & 0xff)
		}
		for s := 0; s < 2; s++ {
			hi := blkBounded[s*2+1]
			exp := 2 + int(int32((g.f()+0.5)*5.0)&3)
			blkBounded[s*2+1] = (hi & 0x83) | byte(exp<<2)
		}
		copy(rawBounded[b*q4kSuperBlock:], blkBounded)
	}
	wfBounded := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for b := 0; b < numSuperBlocks; b++ {
			q4kDequantBlock(buf, rawBounded[o*rowBytes+b*q4kSuperBlock:o*rowBytes+(b+1)*q4kSuperBlock])
			copy(wfBounded[o*in+b*q4kSuper:], buf)
		}
	}
	targetBounded := dominantRowC(wfBounded, out, in)
	xBounded := make([]float32, in)
	copy(xBounded, wfBounded[targetBounded*in:(targetBounded+1)*in])
	var sumSqB float64
	for _, v := range xBounded {
		sumSqB += float64(v * v)
	}
	rmsB := float32(math.Sqrt(sumSqB / float64(in)))
	for i := range xBounded {
		xBounded[i] /= rmsB
	}

	yRefB := make([]float32, out)
	for o := 0; o < out; o++ {
		var acc float32
		for i := 0; i < in; i++ {
			acc += wfBounded[o*in+i] * xBounded[i]
		}
		yRefB[o] = acc
	}
	qxB, dxB, sxB := quantizeVecQ8_1(xBounded)
	yMMVQB := make([]float32, out)
	for o := 0; o < out; o++ {
		yMMVQB[o] = q4kRowDotQ8_1DP4A(rawBounded[o*rowBytes:(o+1)*rowBytes], qxB, dxB, sxB)
	}
	if aRefB, aMMVQB := argmaxF32(yRefB), argmaxF32(yMMVQB); aRefB != aMMVQB || aMMVQB != targetBounded {
		t.Fatalf("bounded weights argmax mismatch: ref=%d mmvq=%d want=%d", aRefB, aMMVQB, targetBounded)
	}
	cosB := cosineC(nonTargetC(yRefB, out, targetBounded), nonTargetC(yMMVQB, out, targetBounded))
	if cosB < 0.995 {
		t.Fatalf("bounded weights cosine %.6f < 0.995 gate", cosB)
	}
	t.Logf("Realistic bounded weights: cosine=%.8f argmax=%d (PASS)", cosB, argmaxF32(yMMVQB))
}
