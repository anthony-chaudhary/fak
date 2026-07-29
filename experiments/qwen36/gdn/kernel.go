package gdn

import (
	"math"
	"runtime"
	"sync"
)

// Silu is the SiLU/swish activation x*sigmoid(x).
func Silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

// Sigmoidf is the logistic sigmoid in f32.
func Sigmoidf(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

// Softplus is log(1+exp(x)) in f32.
func Softplus(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) }

// L2NormInto reproduces qwen35.go:l2normInto — SUM (not mean) of squares, eps inside
// the sqrt. The GDN prefill path uses eps=1e-6.
func L2NormInto(dst, src []float32, eps float32) {
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss+eps)))
	for i := range src {
		dst[i] = src[i] * inv
	}
}

// RMSNormGain1p reproduces the (1+w) RMSNorm used by every non-GDN-readout norm in
// qwen35 — MEAN of squares, gain applied as (1+w).
func RMSNormGain1p(dst, src, w []float32, eps float32) {
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss/float32(len(src)))+float64(eps)))
	for i := range src {
		dst[i] = src[i] * inv * (1 + w[i])
	}
}

// RMSNormGatedInPlace reproduces qwen35.go:rmsNormGatedInPlace — the GDN readout's
// gated norm with PLAIN (not 1+w) weight and a Silu(gate) multiply.
func RMSNormGatedInPlace(x, w, gate []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss/float32(len(x)))+float64(eps)))
	for i := range x {
		x[i] = w[i] * (x[i] * inv) * Silu(gate[i])
	}
}

// QuantF16 round-trips a float32 through IEEE-754 half precision (round-to-nearest-even,
// normal range; tiny values flush toward zero). It models a kernel that stores the GDN
// state in f16 — the realistic worst case for the recurrent accumulator.
//
// The tie test reads the DROPPED bits of mant, not of (mant+half). Testing the sum instead
// is the same expression with a carry folded in, and it fires on mant&mask==0 — i.e. on
// every value already exactly representable in f16 — clearing a set bit `drop` and pushing
// each odd-mantissa half back by one ULP. That variant round-trips only half of all f16
// values to themselves (measured: 30720 of 61440 normal patterns move) and rounds true ties
// half-UP, so it injects up to a full ULP of error where a real f16 store injects none,
// inflating the modelled f16-state divergence. Keep the test on mant.
func QuantF16(x float32) float32 {
	b := math.Float32bits(x)
	sign := b & 0x80000000
	exp := int32((b>>23)&0xFF) - 127 // unbiased
	mant := b & 0x7FFFFF
	if exp == 128 { // Inf/NaN
		return x
	}
	if exp > 15 { // overflow -> saturate to max f16 (~65504), keep sign
		return math.Float32frombits(sign | math.Float32bits(65504))
	}
	if exp < -14 { // subnormal/zero in f16 -> flush to zero (state lives well above this)
		return math.Float32frombits(sign)
	}
	// Round the 23-bit mantissa to 10 bits, round-to-nearest-even.
	const drop = 23 - 10
	const dropped = uint32(1)<<drop - 1
	half := uint32(1) << (drop - 1)
	rounded := mant + half
	if mant&dropped == half { // exactly halfway -> round to even
		rounded &^= 1 << drop
	}
	// Carry into exponent if mantissa overflowed.
	newExp := exp
	if rounded&(1<<23) != 0 {
		rounded = 0
		newExp++
		if newExp > 15 {
			return math.Float32frombits(sign | math.Float32bits(65504))
		}
	}
	rounded &^= dropped
	out := sign | (uint32(newExp+127) << 23) | (rounded & 0x7FFFFF)
	return math.Float32frombits(out)
}

// DepthwiseCausalSilu is the causal depthwise conv1d + SiLU of
// metal_prefill_hybrid_core.go:145-169, run fresh (no carried conv history): taps that
// reach before position 0 contribute nothing.
func DepthwiseCausalSilu(dst, src, weights []float32, steps, channels, kernel int) {
	for t := 0; t < steps; t++ {
		outRow := dst[t*channels : (t+1)*channels]
		for c := 0; c < channels; c++ {
			var acc float32
			base := c * kernel
			for j := 0; j < kernel; j++ {
				source := t + j - (kernel - 1)
				if source >= 0 {
					acc += weights[base+j] * src[source*channels+c]
				}
			}
			outRow[c] = Silu(acc)
		}
	}
}

// ParMatmul computes Y[P,outDim] = X[P,inDim] * W[outDim,inDim]^T in f32. Output rows
// are partitioned across GOMAXPROCS workers, but each output element's dot product runs
// in the same ascending-j serial order regardless of the partition, so the result is
// bit-identical for any GOMAXPROCS.
func ParMatmul(Y, X, W []float32, P, outDim, inDim int) {
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	chunk := (outDim + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if hi > outDim {
			hi = outDim
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for t := 0; t < P; t++ {
				xr := X[t*inDim : (t+1)*inDim]
				yr := Y[t*outDim : (t+1)*outDim]
				for i := lo; i < hi; i++ {
					wr := W[i*inDim : (i+1)*inDim]
					var acc float32
					for j := 0; j < inDim; j++ {
						acc += wr[j] * xr[j]
					}
					yr[i] = acc
				}
			}
		}(lo, hi)
	}
	wg.Wait()
}

// RelDiv is the relative L2 divergence ||a-b|| / ||a||, accumulated in float64. A
// zero-norm reference returns 0 rather than NaN.
func RelDiv(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num) / math.Sqrt(den)
}
