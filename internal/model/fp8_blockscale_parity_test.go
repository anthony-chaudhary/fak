package model

import (
	"math"
	"math/rand"
	"testing"
)

// FP8 (E4M3) block-scale decode PARITY witness (#4209, proposal step 3). The golden
// tests beside this file (fp8_blockscale_test.go) pin the byte decode and the tiling
// bitwise, but every value there is exactly representable in e4m3, so they never
// exercise the QUANTIZATION LOSS the issue's tolerance ask is about. This test closes
// that gap: it quantizes a small mixture-of-experts' expert weights to float8_e4m3fn
// with 128x128 block scales (exactly the DeepSeek-V4 / GLM-FP8 / Hy3-class layout the
// loader now serves via decodeFP8BlockScale, safetensors.go:479), decodes them back
// through the shipped path, runs BOTH the f32-reference weights and the FP8-decoded
// weights through the same f32 GEMM, and asserts the FP8 decode agrees with the f32
// reference within a documented FP8 bound — the "FP8 weights compute through the
// existing f32 GEMM within tolerance" claim, witnessed on CPU with no GPU.
//
// This is the model-side analogue of dsparity's ToleranceFP8Bounded row (dsparity.go:66):
// a genuinely mixed-precision path where the comparison is a bounded abs/rel tolerance,
// not bitwise. The bound is derived from e4m3 precision, then confirmed by measurement
// (t.Logf below), never reverse-fit to force a pass.

// e4m3Max is the largest finite float8_e4m3fn magnitude (S.1111.110 = 448); block
// scaling maps a tile's absmax onto this so the tile uses e4m3's full dynamic range.
const e4m3Max = 448.0

// encodeE4M3 rounds one f32 to the NEAREST representable float8_e4m3fn byte by scanning
// the 254 finite codes and decoding each with the production decoder (fp8E4M3ToF32), so
// the encoder can never disagree with the decoder about what a code means. The two NaN
// codes (0x7F / 0xFF) are excluded; inputs are pre-scaled into [-448,448] by the block
// quantizer so no saturation branch is needed here. O(256) per element is irrelevant at
// test fixture sizes and buys exactness over hand-rolled exponent arithmetic.
func encodeE4M3(x float32) byte {
	best := byte(0)
	bestErr := math.Inf(1)
	for c := 0; c < 256; c++ {
		b := byte(c)
		if b&0x7f == 0x7f {
			continue // 0x7f / 0xff are the sole NaN pattern in e4m3fn
		}
		v := float64(fp8E4M3ToF32(b))
		e := math.Abs(v - float64(x))
		if e < bestErr {
			bestErr = e
			best = b
		}
	}
	return best
}

// quantizeBlockScaleE4M3 quantizes a row-major [O,I] f32 weight to the checkpoint layout
// decodeFP8BlockScale consumes: per 128x128 tile it takes the absmax, sets that tile's
// scaleInv = absmax/448 (mapping the tile's largest magnitude onto e4m3's max), and
// encodes each element as encodeE4M3(w/scaleInv). A tile of all zeros gets scaleInv 0 and
// all-zero codes (0*0 == 0 decodes exactly). The returned (weightBytes, scaleInv) is
// exactly what a real weight / weight_scale_inv companion pair holds.
func quantizeBlockScaleE4M3(O, I int, w []float32) ([]byte, []float32) {
	sO := (O + fp8BlockDim - 1) / fp8BlockDim
	sI := (I + fp8BlockDim - 1) / fp8BlockDim
	scaleInv := make([]float32, sO*sI)
	for bo := 0; bo < sO; bo++ {
		for bi := 0; bi < sI; bi++ {
			var amax float64
			for o := bo * fp8BlockDim; o < (bo+1)*fp8BlockDim && o < O; o++ {
				for i := bi * fp8BlockDim; i < (bi+1)*fp8BlockDim && i < I; i++ {
					if a := math.Abs(float64(w[o*I+i])); a > amax {
						amax = a
					}
				}
			}
			scaleInv[bo*sI+bi] = float32(amax / e4m3Max)
		}
	}
	out := make([]byte, O*I)
	for o := 0; o < O; o++ {
		for i := 0; i < I; i++ {
			s := scaleInv[(o/fp8BlockDim)*sI+i/fp8BlockDim]
			if s == 0 {
				out[o*I+i] = 0
				continue
			}
			out[o*I+i] = encodeE4M3(w[o*I+i] / s)
		}
	}
	return out, scaleInv
}

// gemvF32 computes y[o] = sum_i W[o,I+i] * x[i] for a row-major [O,I] weight — the linear
// projection at the heart of every expert (gate/up/down). f32 accumulate is the reference.
func gemvF32(O, I int, w, x []float32) []float32 {
	y := make([]float32, O)
	for o := 0; o < O; o++ {
		var acc float64
		for i := 0; i < I; i++ {
			acc += float64(w[o*I+i]) * float64(x[i])
		}
		y[o] = float32(acc)
	}
	return y
}

// TestFP8BlockScaleMoEDecodeParity is the #4209 tolerance witness: a two-expert MoE whose
// expert weights are served through the FP8 block-scale decode agrees with the f32
// reference within a documented FP8 bound.
func TestFP8BlockScaleMoEDecodeParity(t *testing.T) {
	// Small MoE: 2 experts, each a gate projection [O,I] spanning a 2x3 grid of 128x128
	// scale tiles (so block indexing and ragged edges are exercised), one input token,
	// router gates mixing the two experts. Deterministic PRNG => a fixed, reproducible
	// fixture; N(0,1) weights/inputs give a realistic dynamic range where the tile absmax
	// is a ~4-sigma outlier and typical elements sit well below it (the case block scaling
	// is meant to handle, and the case a bitwise golden never covers).
	const O, I = 200, 300
	const experts = 2
	rng := rand.New(rand.NewSource(0x4209))

	x := make([]float32, I)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	gates := []float32{0.7, 0.3} // fixed top-2 router mix

	refMix := make([]float32, O)
	fp8Mix := make([]float32, O)
	for e := 0; e < experts; e++ {
		w := make([]float32, O*I)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		// f32 reference: the full-precision expert.
		yRef := gemvF32(O, I, w, x)
		// FP8 path: quantize -> decode through the SHIPPED loader math -> same GEMM.
		qBytes, scaleInv := quantizeBlockScaleE4M3(O, I, w)
		wDeq, err := decodeFP8BlockScale("expert", O, I, qBytes, scaleInv)
		if err != nil {
			t.Fatalf("decodeFP8BlockScale expert %d: %v", e, err)
		}
		yFP8 := gemvF32(O, I, wDeq, x)
		for o := 0; o < O; o++ {
			refMix[o] += gates[e] * yRef[o]
			fp8Mix[o] += gates[e] * yFP8[o]
		}
	}

	// Compare the mixed MoE output vectors: cosine similarity (direction agreement, the
	// same observable the cuda accuracy gates floor at 0.995) and relative L2 error.
	var dot, nRef, nFP8, dNum, dDen float64
	for o := 0; o < O; o++ {
		r, f := float64(refMix[o]), float64(fp8Mix[o])
		dot += r * f
		nRef += r * r
		nFP8 += f * f
		d := r - f
		dNum += d * d
		dDen += r * r
	}
	cosine := dot / (math.Sqrt(nRef) * math.Sqrt(nFP8))
	relL2 := math.Sqrt(dNum / dDen)
	t.Logf("FP8 MoE decode parity: cosine=%.6f relL2=%.6f", cosine, relL2)

	// Documented FP8 bound. e4m3 carries 3 mantissa bits, so round-to-nearest bounds a
	// single element's relative error at ~2^-4 = 6.25%; block absmax scaling costs the
	// sub-absmax elements additional relative precision, but the GEMM sums O(I) such terms
	// so the per-element errors partially cancel and the OUTPUT relative L2 lands well
	// under a single element's worst case. cosine must stay above the 0.995 floor the
	// device Q4_K/AWQ accuracy gates already hold; relL2 must stay within an 8% FP8 band.
	// Thresholds are set with margin above the measured values (logged above), not fit to
	// them. If a decode regression widens the error, both assertions fail loudly.
	const minCosine = 0.995
	const maxRelL2 = 0.08
	if cosine < minCosine {
		t.Errorf("cosine %.6f < %.3f — FP8 decode drifted from the f32 reference direction", cosine, minCosine)
	}
	if relL2 > maxRelL2 {
		t.Errorf("relL2 %.6f > %.3f — FP8 decode exceeded the documented FP8 tolerance band", relL2, maxRelL2)
	}
}
