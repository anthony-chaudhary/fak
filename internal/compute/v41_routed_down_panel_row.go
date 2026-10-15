package compute

import (
	"fmt"
	"math"
)

// V41RoutedDownPanelRowBackend optionally publishes one row of a borrowed
// F32 down panel at the fixed ordinary RoutedDown site. Capability refusal
// precedes selection; selected errors are fatal and must not be replayed.
// Output and status are fresh caller-owned allocations, including partial
// results returned with an error. Input stays borrowed until completion.
type V41RoutedDownPanelRowBackend interface {
	SupportsV41RoutedDownPanelRow(panel Tensor, row int) bool
	// The output is F32 [width]. Status has Kind V41DeferredBF16Publish,
	// Count width, Site V41BF16PublishRoutedDown and one row-local record.
	// Finite/BF16-overflow checks and integer rounding match ordinary publication.
	V41RoutedDownPanelRowDeferred(panel Tensor, row int) (Tensor, V41DeferredStatus, error)
}

// This validates geometry only; native admission additionally checks module
// ABI, live ownership, storage extents and disjoint output/status bindings.
func validateV41RoutedDownPanelRow(panel Tensor, row int) (rows, width int, err error) {
	if panel.Dtype != F32 || panel.Layout != RowMajor || panel.Quant != nil || len(panel.Shape) != 2 {
		return 0, 0, fmt.Errorf("compute: routed down panel requires an unquantized row-major F32 matrix")
	}
	rows, width = panel.Shape[0], panel.Shape[1]
	if rows < 1 || rows > 64 || width < 1 || row < 0 || row >= rows || uint64(width) > uint64(math.MaxInt32)/4/uint64(rows) {
		return 0, 0, fmt.Errorf("compute: routed down panel row exceeds bounded F32 storage")
	}
	return rows, width, nil
}
