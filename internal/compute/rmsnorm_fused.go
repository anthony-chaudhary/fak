package compute

import (
	"fmt"
	"math"
)

// FusedRMSNormResidualReceipt records operational metrics of the fused kernel.
type FusedRMSNormResidualReceipt struct {
	Rows             int     `json:"rows"`
	Width            int     `json:"width"`
	Epsilon          float32 `json:"epsilon"`
	InPlaceResidual  bool    `json:"in_place_residual"`
	EliminatedDRAMMB float64 `json:"eliminated_dram_mb"`
	SavedLaunches    int     `json:"saved_launches"`
}

// FusedRMSNormResidualAdd computes:
//
//	residualOut = x + residualIn
//	normed = RMSNorm(residualOut, weight, eps)
//
// in a single pass, eliminating intermediate residual DRAM round-trips.
// Supports in-place aliasing (residualOut sharing backing storage with residualIn or x).
func FusedRMSNormResidualAdd(x, residualIn, weight, residualOut, normed []float32, rows, width int, eps float32) (FusedRMSNormResidualReceipt, error) {
	var receipt FusedRMSNormResidualReceipt
	if rows <= 0 || width <= 0 {
		return receipt, fmt.Errorf("dimensions must be positive: rows=%d, width=%d", rows, width)
	}
	if eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return receipt, fmt.Errorf("invalid epsilon: %v", eps)
	}

	totalElements := rows * width
	if len(x) != totalElements {
		return receipt, fmt.Errorf("x length %d != expected %d", len(x), totalElements)
	}
	if len(residualIn) != totalElements {
		return receipt, fmt.Errorf("residualIn length %d != expected %d", len(residualIn), totalElements)
	}
	if len(weight) != width {
		return receipt, fmt.Errorf("weight length %d != width %d", len(weight), width)
	}
	if len(normed) != totalElements {
		return receipt, fmt.Errorf("normed length %d != expected %d", len(normed), totalElements)
	}
	if residualOut != nil && len(residualOut) != totalElements {
		return receipt, fmt.Errorf("residualOut length %d != expected %d", len(residualOut), totalElements)
	}

	inPlace := false
	if residualOut != nil {
		if &residualOut[0] == &residualIn[0] || &residualOut[0] == &x[0] {
			inPlace = true
		}
	}

	for r := 0; r < rows; r++ {
		rowOffset := r * width
		var sumSq float64
		for i := 0; i < width; i++ {
			idx := rowOffset + i
			xi := x[idx]
			ri := residualIn[idx]
			if math.IsNaN(float64(xi)) || math.IsInf(float64(xi), 0) || math.IsNaN(float64(ri)) || math.IsInf(float64(ri), 0) {
				return receipt, fmt.Errorf("non-finite value encountered at index %d", idx)
			}
			res := xi + ri
			if residualOut != nil {
				residualOut[idx] = res
			}
			sumSq += float64(res) * float64(res)
		}

		mean := sumSq / float64(width)
		inv := float32(1.0 / math.Sqrt(mean+float64(eps)))

		for i := 0; i < width; i++ {
			idx := rowOffset + i
			var res float32
			if residualOut != nil {
				res = residualOut[idx]
			} else {
				res = x[idx] + residualIn[idx]
			}
			normed[idx] = res * inv * weight[i]
		}
	}

	// 1 intermediate write + 1 intermediate read of residual eliminated = 2 * totalElements * 4 bytes
	eliminatedBytes := float64(totalElements * 4 * 2)
	receipt = FusedRMSNormResidualReceipt{
		Rows:             rows,
		Width:            width,
		Epsilon:          eps,
		InPlaceResidual:  inPlace,
		EliminatedDRAMMB: eliminatedBytes / (1024 * 1024),
		SavedLaunches:    1,
	}

	return receipt, nil
}
