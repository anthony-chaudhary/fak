package compute

import (
	"errors"
	"math"
)

// ADAPT of FlagOpen/FlagGems (Apache-2.0), src/flag_gems/ops/rms_norm.py:1 @3f1f896fd90f:
// fused single-pass RMSNorm kernel contract over [rows, cols] float32 panels.
// Component and host oracle only: no hardware or speed claim.

// FusedRMSNormF32 normalizes each row of x[rows][cols] by 1/sqrt(mean(x^2)+eps)
// and multiplies by weight[cols] in a single fused pass.
func FusedRMSNormF32(x, weight []float32, rows, cols int, eps float32) ([]float32, error) {
	if rows <= 0 || cols <= 0 || len(x) != rows*cols || len(weight) != cols ||
		!(eps > 0) || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return nil, errors.New("compute: invalid fused RMSNorm inputs")
	}
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		base := r * cols
		var sumSq float64
		for c := 0; c < cols; c++ {
			v := float64(x[base+c])
			sumSq += v * v
		}
		invRMS := 1.0 / math.Sqrt(sumSq/float64(cols)+float64(eps))
		for c := 0; c < cols; c++ {
			out[base+c] = float32(float64(x[base+c]) * invRMS * float64(weight[c]))
		}
	}
	return out, nil
}
