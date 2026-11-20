package compute

import (
	"fmt"
)

// BlockTableSpanInt32Bound is the explicit int32 bound on block-table span counts and
// column counts: every count must fit in a non-negative int32 (<= 2**31 - 1). A span or
// row width beyond this bound is malformed and rejected fail-closed.
const BlockTableSpanInt32Bound = int32(1<<31 - 1)

// BlockTableSpan is the (released, pages) span of one block-table row. The row's closed
// contract is [0*released, values*pages, -1*padding] where padding = totalCols -
// released - pages: a released prefix of holes (0), a live run of page IDs (values in
// [1, 2**31)), and a padding tail (-1). The span plus the live values fully determine the
// row, so only the live window needs a host-to-device upload; the device rebuilds the full
// row from the span and the live values. The span is a value type, so a table of spans is
// a zero-copy [rows, 2] int32 view.
type BlockTableSpan struct {
	// Released is the count of leading hole entries (0) in the row.
	Released int32
	// Pages is the count of live page-ID entries (values) in the row.
	Pages int32
}

// Validate reports whether the span is well-formed for a row of totalCols columns. It
// fails closed: a negative count, a count beyond the int32 bound, or released+pages
// exceeding the row width is rejected. The overflow check is arranged so it cannot
// overflow int32.
func (s BlockTableSpan) Validate(totalCols int32) error {
	if totalCols < 0 {
		return fmt.Errorf("compute: block-table span: row width %d is negative", totalCols)
	}
	if s.Released < 0 || s.Pages < 0 {
		return fmt.Errorf("compute: block-table span: negative span (released=%d, pages=%d)", s.Released, s.Pages)
	}
	if s.Released > BlockTableSpanInt32Bound || s.Pages > BlockTableSpanInt32Bound || totalCols > BlockTableSpanInt32Bound {
		return fmt.Errorf("compute: block-table span: count exceeds the int32 bound (released=%d, pages=%d, cols=%d)", s.Released, s.Pages, totalCols)
	}
	if s.Released > totalCols {
		return fmt.Errorf("compute: block-table span: released %d exceeds row width %d", s.Released, totalCols)
	}
	if s.Pages > totalCols-s.Released {
		return fmt.Errorf("compute: block-table span: released+pages (%d+%d) exceeds row width %d", s.Released, s.Pages, totalCols)
	}
	return nil
}

// ExportBlockTableSpan extracts the (released, pages) span and the live values from a
// full block-table row following the closed 0 / values / -1 contract. The live values are
// a zero-copy view into row. It fails closed on a row that violates the contract: a 0 or
// a negative other than -1 inside the live run, or a non-(-1) entry in the padding tail.
func ExportBlockTableSpan(row []int32) (BlockTableSpan, []int32, error) {
	totalCols := int32(len(row))
	released := 0
	for released < len(row) && row[released] == 0 {
		released++
	}
	pages := 0
	for i := released; i < len(row); i++ {
		if row[i] == -1 {
			break
		}
		if row[i] <= 0 {
			return BlockTableSpan{}, nil, fmt.Errorf("compute: block-table span: hole or invalid value %d at column %d inside the live run", row[i], i)
		}
		pages++
	}
	for i := released + pages; i < len(row); i++ {
		if row[i] != -1 {
			return BlockTableSpan{}, nil, fmt.Errorf("compute: block-table span: padding entry %d at column %d is not -1", row[i], i)
		}
	}
	span := BlockTableSpan{Released: int32(released), Pages: int32(pages)}
	if err := span.Validate(totalCols); err != nil {
		return BlockTableSpan{}, nil, err
	}
	return span, row[released : released+pages], nil
}

// RebuildBlockTableRow reconstructs the full block-table row from its span and live
// values, following the closed 0 / values / -1 contract. It fails closed on a malformed
// span (see Validate), a live-values length that disagrees with the span, or a live value
// outside [1, 2**31).
func RebuildBlockTableRow(span BlockTableSpan, live []int32, totalCols int32) ([]int32, error) {
	if err := span.Validate(totalCols); err != nil {
		return nil, err
	}
	if int32(len(live)) != span.Pages {
		return nil, fmt.Errorf("compute: block-table span: live-values length %d disagrees with span pages %d", len(live), span.Pages)
	}
	for i, v := range live {
		if v <= 0 || v > BlockTableSpanInt32Bound {
			return nil, fmt.Errorf("compute: block-table span: live value %d at index %d is outside [1, 2**31)", v, i)
		}
	}
	row := make([]int32, totalCols) // zero-init: the released prefix is 0
	copy(row[span.Released:span.Released+span.Pages], live)
	for i := span.Released + span.Pages; i < totalCols; i++ {
		row[i] = -1
	}
	return row, nil
}
