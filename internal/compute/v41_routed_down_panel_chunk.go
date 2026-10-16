package compute

import (
	"fmt"
	"math"
)

// V41RoutedDownPanelChunkBackend publishes one existing down-schedule chunk.
// Inputs remain borrowed through completion. Output and status are fresh and
// caller-owned, including partial error results. Selected failures never replay.
// Successful output is flat F32 [rows*width]; the fixed site is RoutedDown21.
// One physical status contains rows row-local records, Count=width and
// Kind=V41DeferredBF16Publish. A grouped submission error asserts no row prefix.
type V41RoutedDownPanelChunkBackend interface {
	SupportsV41RoutedDownPanelChunk(panel Tensor, first, rows int) bool
	V41RoutedDownPanelChunkDeferred(panel Tensor, first, rows int) (Tensor, V41DeferredStatus, error)
}

// This plan bounds full input storage, addressed output and row-local status.
type V41RoutedDownPanelChunkPlan struct {
	PanelRows, Width, First, Rows                        int
	PanelBytes, OutputBytes, StatusBytes, OffsetElements int
}

func PlanV41RoutedDownPanelChunk(panelRows, width, first, rows int) (V41RoutedDownPanelChunkPlan, error) {
	if panelRows < 1 || panelRows > 64 || width < 1 || (rows != 2 && rows != 4 && rows != 8) || first < 0 || first > panelRows-rows || uint64(width) > uint64(math.MaxInt32)/4/uint64(panelRows) {
		return V41RoutedDownPanelChunkPlan{}, fmt.Errorf("compute: V4.1 down publication requires bounded panels and an addressed 2/4/8-row extent")
	}
	return V41RoutedDownPanelChunkPlan{panelRows, width, first, rows, panelRows * width * 4, rows * width * 4, rows * 16, first * width}, nil
}

func validateV41RoutedDownPanelChunk(panel Tensor, first, rows int) (V41RoutedDownPanelChunkPlan, error) {
	if panel.Dtype != F32 || panel.Layout != RowMajor || panel.Quant != nil || len(panel.Shape) != 2 {
		return V41RoutedDownPanelChunkPlan{}, fmt.Errorf("compute: V4.1 down publication requires an unquantized row-major F32 panel")
	}
	return PlanV41RoutedDownPanelChunk(panel.Shape[0], panel.Shape[1], first, rows)
}
