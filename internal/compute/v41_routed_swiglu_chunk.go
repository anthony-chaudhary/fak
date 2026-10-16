package compute

import (
	"fmt"
	"math"
)

// V41RoutedSwiGLUPanelChunkBackend consumes one existing down-schedule chunk
// from borrowed gate/up panels. Selection precedes effects; selected errors
// never decline or replay. A missing optional module preserves the row path.
type V41RoutedSwiGLUPanelChunkBackend interface {
	SupportsV41RoutedSwiGLUPanelChunk(gate, up Tensor, first, rows int) bool
	// Output is fresh flat F32 [rows*width], directly consumable by token-panel
	// down. Status owns rows consecutive 16-byte records, Count=width, Site=0,
	// Kind=V41DeferredRoutedSwiGLU. Each fault index is row-local. A successful
	// call issues every row record. A returned error is a grouped activation
	// submission boundary, not evidence of any old per-row issue prefix.
	// Inputs stay borrowed; every partial output/status owner belongs to caller.
	V41RoutedSwiGLUPanelChunkDeferred(gate, up Tensor, first int, limit float32, weights []float32) (Tensor, V41DeferredStatus, error)
}

// V41RoutedSwiGLUPanelChunkPlan bounds complete input storage and addressed
// output. 2/4/8 exactly match the existing down schedule; singletons stay scalar.
type V41RoutedSwiGLUPanelChunkPlan struct {
	PanelRows, Width, First, Rows                        int
	PanelBytes, OutputBytes, StatusBytes, OffsetElements int
}

func PlanV41RoutedSwiGLUPanelChunk(panelRows, width, first, rows int) (V41RoutedSwiGLUPanelChunkPlan, error) {
	if panelRows < 1 || panelRows > 64 || width < 1 || (rows != 2 && rows != 4 && rows != 8) || first < 0 || first > panelRows-rows || uint64(width) > uint64(math.MaxInt32)/4/uint64(panelRows) {
		return V41RoutedSwiGLUPanelChunkPlan{}, fmt.Errorf("compute: V4.1 activation chunk requires bounded full panels and an addressed2/4/8-row extent")
	}
	return V41RoutedSwiGLUPanelChunkPlan{panelRows, width, first, rows, panelRows * width * 4, rows * width * 4, rows * 16, first * width}, nil
}
func validateV41RoutedSwiGLUPanelChunk(gate, up Tensor, first, rows int, limit float32, weights []float32) (V41RoutedSwiGLUPanelChunkPlan, error) {
	for _, t := range []Tensor{gate, up} {
		if t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || len(t.Shape) != 2 {
			return V41RoutedSwiGLUPanelChunkPlan{}, fmt.Errorf("compute: V4.1 activation chunk requires unquantized row-major F32 panels")
		}
	}
	if gate.Shape[0] != up.Shape[0] || gate.Shape[1] != up.Shape[1] || !finite32(limit) || len(weights) != rows {
		return V41RoutedSwiGLUPanelChunkPlan{}, fmt.Errorf("compute: V4.1 activation chunk shape, limit or weights differ")
	}
	for _, w := range weights {
		if !finite32(w) {
			return V41RoutedSwiGLUPanelChunkPlan{}, fmt.Errorf("compute: V4.1 activation chunk weight is non-finite")
		}
	}
	return PlanV41RoutedSwiGLUPanelChunk(gate.Shape[0], gate.Shape[1], first, rows)
}
