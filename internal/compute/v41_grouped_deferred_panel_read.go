package compute

import "math"

// V41GroupedDeferredPanelReader transfers one grouped-output panel into
// independent token rows. Only the exact canonical provider may bind it.
// Resolve transport, row count and numeric verdicts before taking the result.
// The callback is call-local and consumes statuses once, including on failure.
type V41GroupedDeferredPanelReader func(statuses []V41DeferredStatus, panel Tensor, rows, width int) ([]error, V41DeferredReadResult, error)

type v41GroupedDeferredPanelReaderBinder interface {
	v41BindGroupedDeferredPanelReader(original V41DeferredStatusBackend) V41GroupedDeferredPanelReader
}

// BindV41GroupedDeferredPanelReader is effect-free. It must be paired with
// V41GroupedDeferredPanelEligible before publication. Wrapper promotion alone
// cannot authorize the canonical provider's independent storage guarantee.
func BindV41GroupedDeferredPanelReader(backend V41DeferredStatusBackend) V41GroupedDeferredPanelReader {
	if b, ok := backend.(v41GroupedDeferredPanelReaderBinder); ok {
		return b.v41BindGroupedDeferredPanelReader(backend)
	}
	return nil
}

// V41GroupedDeferredPanelEligible bounds one grouped output independently of
// the routed reader's 32-row cohort and the ordinary typed-prefix allowance.
// Each group contributes two scalar statuses, followed by one final status.
func V41GroupedDeferredPanelEligible(rows, groups, width int) bool {
	if rows < 1 || rows > 64 || groups < 1 || groups > 31 || width < 1 || width > math.MaxInt32/4/rows {
		return false
	}
	prefix := (2*groups + 1) * 16
	return width <= (math.MaxInt-prefix)/4/rows
}

type v41DeferredReadMode uint8

const (
	v41DeferredReadOrdinary v41DeferredReadMode = iota
	v41DeferredReadRoutedPanel
	v41DeferredReadGroupedPanel
)

func v41GroupedDeferredPanelLayout(statuses []V41DeferredStatus, panels []Tensor, layout []int, width int) (int, bool) {
	if len(panels) != 1 || len(layout) != 1 || len(statuses) < 3 || len(statuses) > 63 || len(statuses)%2 != 1 {
		return 0, false
	}
	rows := layout[0]
	if !V41GroupedDeferredPanelEligible(rows, (len(statuses)-1)/2, width) {
		return 0, false
	}
	panel := panels[0]
	if panel.Dtype != F32 || panel.Layout != RowMajor || panel.Quant != nil || len(panel.Shape) != 1 || panel.Shape[0] != rows*width {
		return 0, false
	}
	for _, status := range statuses {
		if status.Kind != V41DeferredBF16Publish || status.Site != V41BF16PublishGroupedOutputProjection || status.Records > 1 || status.Count <= 0 || status.Count%rows != 0 {
			return 0, false
		}
	}
	if statuses[len(statuses)-1].Count != rows*width {
		return 0, false
	}
	return rows, true
}
