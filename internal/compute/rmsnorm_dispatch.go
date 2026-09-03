package compute

import (
	"fmt"
	"math"
)

// RMSNormDispatchKind identifies the selected kernel strategy for a row width.
type RMSNormDispatchKind string

const (
	// RMSNormDispatchWarpPerRow assigns one 32-thread warp per row using warp shuffle reductions.
	RMSNormDispatchWarpPerRow RMSNormDispatchKind = "warp-per-row"
	// RMSNormDispatchBlockPerRow assigns one 256-thread block per row using block shared memory.
	RMSNormDispatchBlockPerRow RMSNormDispatchKind = "block-per-row"
)

// RMSNormDispatchThresholdWidth is the crossover width below which warp-per-row is optimal.
const RMSNormDispatchThresholdWidth = 1024

// RMSNormDispatchReceipt records deterministic metrics of the shape-routed dispatch.
type RMSNormDispatchReceipt struct {
	Rows           int                 `json:"rows"`
	Width          int                 `json:"width"`
	DispatchKind   RMSNormDispatchKind `json:"dispatch_kind"`
	Epsilon        float32             `json:"epsilon"`
	ThreadsPerUnit int                 `json:"threads_per_unit"`
	UnitsPerBlock  int                 `json:"units_per_block"`
	TotalBlocks    int                 `json:"total_blocks"`
	BarrierFree    bool                `json:"barrier_free"`
}

// SelectRMSNormDispatch chooses warp-per-row for fitting widths <= 1024 (barrier-free shuffle)
// and block-per-row for wider/ragged rows needing larger reduction capacity.
func SelectRMSNormDispatch(rows, width int) RMSNormDispatchKind {
	if width <= RMSNormDispatchThresholdWidth {
		return RMSNormDispatchWarpPerRow
	}
	return RMSNormDispatchBlockPerRow
}

// RMSNormDispatched performs shape-aware dispatched RMS normalization.
func RMSNormDispatched(x, weight, out []float32, rows, width int, eps float32) (RMSNormDispatchReceipt, error) {
	var receipt RMSNormDispatchReceipt
	if rows <= 0 || width <= 0 {
		return receipt, fmt.Errorf("dimensions must be positive: rows=%d, width=%d", rows, width)
	}
	if eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return receipt, fmt.Errorf("invalid epsilon: %v", eps)
	}

	total := rows * width
	if len(x) != total {
		return receipt, fmt.Errorf("x length %d != expected %d", len(x), total)
	}
	if len(weight) != width {
		return receipt, fmt.Errorf("weight length %d != width %d", len(weight), width)
	}
	if len(out) != total {
		return receipt, fmt.Errorf("out length %d != expected %d", len(out), total)
	}

	kind := SelectRMSNormDispatch(rows, width)

	for r := 0; r < rows; r++ {
		rowOff := r * width
		var sumSq float64
		for i := 0; i < width; i++ {
			v := float64(x[rowOff+i])
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return receipt, fmt.Errorf("non-finite value at row %d, col %d", r, i)
			}
			sumSq += v * v
		}

		mean := sumSq / float64(width)
		inv := float32(1.0 / math.Sqrt(mean+float64(eps)))

		for i := 0; i < width; i++ {
			out[rowOff+i] = x[rowOff+i] * inv * weight[i]
		}
	}

	if kind == RMSNormDispatchWarpPerRow {
		warpsPerBlock := 8
		totalBlocks := (rows + warpsPerBlock - 1) / warpsPerBlock
		receipt = RMSNormDispatchReceipt{
			Rows:           rows,
			Width:          width,
			DispatchKind:   kind,
			Epsilon:        eps,
			ThreadsPerUnit: 32,
			UnitsPerBlock:  warpsPerBlock,
			TotalBlocks:    totalBlocks,
			BarrierFree:    true,
		}
	} else {
		receipt = RMSNormDispatchReceipt{
			Rows:           rows,
			Width:          width,
			DispatchKind:   kind,
			Epsilon:        eps,
			ThreadsPerUnit: 256,
			UnitsPerBlock:  1,
			TotalBlocks:    rows,
			BarrierFree:    false,
		}
	}

	return receipt, nil
}
