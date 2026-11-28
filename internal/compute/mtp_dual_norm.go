package compute

// mtp_dual_norm.go: HOST-ONLY [SW-VERIFIED] oracle for the single-launch dual
// RMSNorm over the two MTP inputs (embedding row and hidden row). It pins the
// arithmetic contract a fused device kernel must meet; it launches nothing and
// makes no device or speed claim. The shipped CUDA path still runs the row norm
// twice (fcuda_qwen35_mtp_f32 -> two norm launches); this file is the reference
// that a one-kernel dual-sum implementation is held to.
//
// Adapted from modular/modular max/kernels/src/nn/mtp_eh_norm.mojo (Apache-2.0
// WITH LLVM-exception): one pass per token reduces BOTH sums of squares through
// one dual-sum block reduce and stores [normed_embed | normed_hidden]. Only the
// arithmetic contract is ported here; no source is copied.

import (
	"errors"
	"math"
)

// mtpDualNormLanes is the per-thread piece width of the fused kernel (two 8-channel
// pieces per row). The reduction order below mirrors it: lane partials first, then
// one sequential combine, identically for each row.
const mtpDualNormLanes = 8

var errMTPDualNorm = errors.New("compute: mtp dual norm: invalid shape or epsilon")

// mtpSumSquares reduces x*x with the fixed order the fused kernel uses: per-lane
// partials over a strided walk, then a sequential combine of the lane partials.
func mtpSumSquares(x []float32) float32 {
	var lane [mtpDualNormLanes]float32
	for i, v := range x {
		lane[i%mtpDualNormLanes] += v * v
	}
	var s float32
	for _, p := range lane {
		s += p
	}
	return s
}

// MTPRMSNormRow normalizes one row: x * rsqrt(mean(x^2) + eps) * w.
func MTPRMSNormRow(x, w []float32, eps float32) ([]float32, error) {
	if len(x) == 0 || len(x) != len(w) || !(eps >= 0) || math.IsInf(float64(eps), 0) {
		return nil, errMTPDualNorm
	}
	scale := float32(1.0 / math.Sqrt(float64(mtpSumSquares(x)/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * scale * w[i]
	}
	return out, nil
}

// MTPTwoNormComposition is the shipped shape: two independent row norms, outputs
// concatenated as [normed_embed | normed_hidden].
func MTPTwoNormComposition(embed, hidden, wEmbed, wHidden []float32, eps float32) ([]float32, error) {
	e, err := MTPRMSNormRow(embed, wEmbed, eps)
	if err != nil {
		return nil, err
	}
	h, err := MTPRMSNormRow(hidden, wHidden, eps)
	if err != nil {
		return nil, err
	}
	return append(e, h...), nil
}

// MTPDualSumNorm is the single-pass contract: both sums of squares come out of one
// walk over the two rows (each element read once for the reduction), then one write
// pass stores the joined output. Rows may differ in width (ragged shapes).
func MTPDualSumNorm(embed, hidden, wEmbed, wHidden []float32, eps float32) ([]float32, error) {
	if len(embed) == 0 || len(hidden) == 0 || len(embed) != len(wEmbed) || len(hidden) != len(wHidden) ||
		!(eps >= 0) || math.IsInf(float64(eps), 0) {
		return nil, errMTPDualNorm
	}
	var le, lh [mtpDualNormLanes]float32
	n := len(embed)
	if len(hidden) > n {
		n = len(hidden)
	}
	for i := 0; i < n; i++ {
		if i < len(embed) {
			le[i%mtpDualNormLanes] += embed[i] * embed[i]
		}
		if i < len(hidden) {
			lh[i%mtpDualNormLanes] += hidden[i] * hidden[i]
		}
	}
	var se, sh float32
	for k := 0; k < mtpDualNormLanes; k++ {
		se += le[k]
		sh += lh[k]
	}
	scaleE := float32(1.0 / math.Sqrt(float64(se/float32(len(embed))+eps)))
	scaleH := float32(1.0 / math.Sqrt(float64(sh/float32(len(hidden))+eps)))
	out := make([]float32, 0, len(embed)+len(hidden))
	for i, v := range embed {
		out = append(out, v*scaleE*wEmbed[i])
	}
	for i, v := range hidden {
		out = append(out, v*scaleH*wHidden[i])
	}
	return out, nil
}
