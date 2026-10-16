package compute

import (
	"encoding/binary"
	"math"
)

// V41DeferredPanelReader is a call-local canonical reader for a mixture of
// published 1/2/4/8-row tensors. It consumes each physical status once, performs
// one ordinary deferred transfer and returns independently allocated logical
// rows, never views into a packed slab. All numeric verdicts must be resolved
// before TakeRows. Transport errors return no rows or ownership authority.
// The callback must not be retained beyond its current backend execution scope.
type V41DeferredPanelReader func(statuses []V41DeferredStatus, panels []Tensor, rows []int, width int) ([]error, V41DeferredReadResult, error)

type v41DeferredPanelReaderBinder interface {
	v41BindDeferredPanelReader(original V41DeferredStatusBackend) V41DeferredPanelReader
}

// BindV41DeferredPanelReader does not infer ownership from a backend name or
// function identity. The canonical provider checks the original exact receiver;
// a promoted private method on a wrapped backend cannot qualify that wrapper.
// Binding is effect-free and must precede selecting any packed publication.
func BindV41DeferredPanelReader(backend V41DeferredStatusBackend) V41DeferredPanelReader {
	if b, ok := backend.(v41DeferredPanelReaderBinder); ok {
		return b.v41BindDeferredPanelReader(backend)
	}
	return nil
}

// Allocation-free geometry check. The sum is capped by the existing cold
// expert cohort cap; it never enables larger cohorts or a new memory budget.
// A recording failure passes no payloads but still reads its issued statuses.
func v41DeferredPanelRows(panels []Tensor, rows []int, width int) (int, bool) {
	if width < 1 || width > math.MaxInt32/4 || len(panels) != len(rows) || len(rows) > 32 {
		return 0, false
	}
	total := 0
	for i, n := range rows {
		if (n != 1 && n != 2 && n != 4 && n != 8) || n > 32-total || width > math.MaxInt32/4/n {
			return 0, false
		}
		p := panels[i]
		if p.Dtype != F32 || p.Layout != RowMajor || p.Quant != nil || len(p.Shape) != 1 || p.Shape[0] != n*width {
			return 0, false
		}
		total += n
	}
	return total, true
}

// The logical layout has already been checked before transfer. Every logical
// row gets its own full-cap allocation, even when several rows share one native
// transfer region. The transfer image and sibling rows cannot be retained.
func (image v41HostReadbackImage) materializePanelRows(sizes []int, first int, rows []int, width, total int) [][]float32 {
	values := make([][]float32, total)
	next := 0
	for i, n := range rows {
		raw := image.region(first+i, sizes[first+i])
		for r := 0; r < n; r++ {
			row := make([]float32, width)
			start := r * width * 4
			for j := range row {
				row[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[start+4*j:]))
			}
			values[next] = row
			next++
		}
	}
	return values
}
