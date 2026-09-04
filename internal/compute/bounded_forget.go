package compute

import (
	"math"
)

// MinBoundedDecay is the theoretical minimum decay factor exp(-5.0) ~ 0.006737947.
var MinBoundedDecay = float32(math.Exp(-5.0))

// MaxBoundedDecay is the theoretical maximum decay factor exp(0.0) = 1.0.
const MaxBoundedDecay float32 = 1.0

// BoundedAsymmetricForget computes the bounded asymmetric exponential forget gate
// for recurrent attention kernels (borrowed from ds4 / wkljohn/ds4-strix-halo-tp-odinlink
// rocm/ds4_rocm_glm5_kda.cuh:145-160).
//
// Formula:
//
//	scaled = exp(a_log) * (f_proj + dt_bias)
//	forget = -5.0f * sigmoid(scaled)
//	decay  = exp(forget)
//
// By scaling the sigmoid through -5.0, the forget term is strictly bounded in [-5.0, 0.0].
// This guarantees that the multiplicative decay factor exp(forget) lies strictly in
// [exp(-5) ≈ 0.006737947, 1.0], preventing catastrophic state explosion (decay > 1.0)
// and numerical underflow over long sequence lengths while retaining a wide dynamic range.
//
// Computation is numerically stabilized:
//   - When (f_proj + dt_bias) == 0, scaled is 0 without intermediate Inf*0 NaN.
//   - Extreme input ranges (±10^4) are evaluated safely in the log-domain or saturated
//     without intermediate float64 overflow or NaN generation.
//   - Decay is strictly clamped to [MinBoundedDecay, MaxBoundedDecay].
//
// Shape and Broadcasting:
//   - fProj of length N dictates output length.
//   - aLog and dtBias may have length N (1:1 elementwise), length 1 (scalar broadcast),
//     or length H where N is divisible by H (head-wise broadcast, e.g. tokens x heads).
//   - dtBias may be nil or empty, in which case zero bias is used.
//   - If aLog is empty/nil or shapes are incompatible, nil is returned.
func BoundedAsymmetricForget(aLog []float32, fProj, dtBias []float32) []float32 {
	if fProj == nil {
		return nil
	}
	n := len(fProj)
	if n == 0 {
		return []float32{}
	}
	nALog := len(aLog)
	if nALog == 0 {
		return nil
	}
	if nALog != 1 && nALog != n && n%nALog != 0 {
		return nil
	}

	nDT := len(dtBias)
	if nDT > 0 && nDT != 1 && nDT != n && nDT != nALog && n%nDT != 0 {
		return nil
	}

	out := make([]float32, n)
	for i := 0; i < n; i++ {
		a := aLog[i%nALog]
		var dt float32
		if nDT > 0 {
			dt = dtBias[i%nDT]
		}
		out[i] = computeBoundedForgetElement(a, fProj[i], dt)
	}
	return out
}

func computeBoundedForgetElement(aLog, fProj, dtBias float32) float32 {
	// Guard against NaN inputs fail-closed.
	if math.IsNaN(float64(aLog)) || math.IsNaN(float64(fProj)) || math.IsNaN(float64(dtBias)) {
		return MinBoundedDecay
	}

	u := float64(fProj) + float64(dtBias)
	if math.IsNaN(u) {
		return MinBoundedDecay
	}

	if u == 0 {
		// At zero input, scaled evaluates to 0, sigmoid to 0.5, and decay to exp(-2.5).
		decay := float32(math.Exp(-2.5))
		if decay < MinBoundedDecay {
			return MinBoundedDecay
		}
		if decay > MaxBoundedDecay {
			return MaxBoundedDecay
		}
		return decay
	}

	a := float64(aLog)
	var scaled float64
	if a >= -50.0 && a <= 50.0 {
		scaled = math.Exp(a) * u
	} else {
		// For |a| > 50, use log-domain to avoid intermediate float64 overflow/underflow.
		absU := math.Abs(u)
		logAbsU := math.Log(absU)
		s := a + logAbsU
		if s >= 40.0 {
			if u > 0 {
				scaled = 1e9
			} else {
				scaled = -1e9
			}
		} else if s <= -40.0 {
			scaled = 0.0
		} else {
			val := math.Exp(s)
			if u < 0 {
				scaled = -val
			} else {
				scaled = val
			}
		}
	}

	var sig float64
	if scaled >= 40.0 {
		sig = 1.0
	} else if scaled <= -40.0 {
		sig = 0.0
	} else {
		sig = 1.0 / (1.0 + math.Exp(-scaled))
	}

	forget := -5.0 * sig
	decay := float32(math.Exp(forget))

	if math.IsNaN(float64(decay)) || decay < MinBoundedDecay {
		decay = MinBoundedDecay
	} else if decay > MaxBoundedDecay {
		decay = MaxBoundedDecay
	}

	return decay
}
