package compute

import (
	"fmt"
	"math"
)

// V41RoutedSwiGLUPanelRowBackend optionally consumes one row directly from
// borrowed gate/up F32 [rows,width] panels. It avoids materializing two row
// copies without batching activations or changing their issue order.
// Capability refusal must precede selection; a selected error is fatal and
// must never trigger scalar replay. Implementations must check live ownership,
// storage extents, supported addressing and disjoint output/status buffers.
type V41RoutedSwiGLUPanelRowBackend interface {
	SupportsV41RoutedSwiGLUPanelRow(gate, up Tensor, row int) bool
	// The result is fresh caller-owned F32 [width]; the independent status has
	// Kind V41DeferredRoutedSwiGLU, Count width, Site 0 and one record. Its fault
	// indices are row-local, exactly as V41RoutedSwiGLUDeferred. Both panels stay
	// borrowed until checked completion. Partial output/status owners returned
	// with an error still belong to the caller. No input may be freed or mutated.
	V41RoutedSwiGLUPanelRowDeferred(gate, up Tensor, row int, limit, weight float32) (Tensor, V41DeferredStatus, error)
}

// V41RoutedSwiGLUPanelRowPlan bounds addressing independently of a native
// implementation. Native capability must additionally qualify its own address
// alignment/module ABI; this plan alone never enables a backend or caller.
type V41RoutedSwiGLUPanelRowPlan struct {
	Rows, Width, Row                  int
	PanelBytes, RowBytes, OffsetBytes int
}

func PlanV41RoutedSwiGLUPanelRow(rows, width, row int) (V41RoutedSwiGLUPanelRowPlan, error) {
	if rows < 1 || rows > 64 || width < 1 || row < 0 || row >= rows || uint64(width) > uint64(math.MaxInt32)/4/uint64(rows) {
		return V41RoutedSwiGLUPanelRowPlan{}, fmt.Errorf("compute: V4.1 routed panel row requires 1..64 rows, a valid row and bounded F32 storage")
	}
	return V41RoutedSwiGLUPanelRowPlan{Rows: rows, Width: width, Row: row, PanelBytes: rows * width * 4, RowBytes: width * 4, OffsetBytes: row * width * 4}, nil
}

// Geometry/scalar validation reads no tensor data and transfers no ownership.
func validateV41RoutedSwiGLUPanelRow(gate, up Tensor, row int, limit, weight float32) (V41RoutedSwiGLUPanelRowPlan, error) {
	for _, t := range []Tensor{gate, up} {
		if t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || len(t.Shape) != 2 {
			return V41RoutedSwiGLUPanelRowPlan{}, fmt.Errorf("compute: V4.1 routed panel operands must be unquantized row-major F32 matrices")
		}
	}
	if gate.Shape[0] != up.Shape[0] || gate.Shape[1] != up.Shape[1] || !finite32(limit) || !finite32(weight) {
		return V41RoutedSwiGLUPanelRowPlan{}, fmt.Errorf("compute: V4.1 routed panel shapes must match and scalars must be finite")
	}
	return PlanV41RoutedSwiGLUPanelRow(gate.Shape[0], gate.Shape[1], row)
}
