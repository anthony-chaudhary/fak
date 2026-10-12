package compute

import (
	"errors"
	"fmt"
	"math"
)

// V41RowLogBackend keeps DeepSeek-V4.1 decode KV rows resident on the device.
// Every tensor is an unquantized row-major F32 [rows, width] buffer owned by
// this backend. The operations are byte copies: no arithmetic, rounding or
// reordering within a row. Capability absence permits the caller's per-step
// host upload; a selected failure must propagate without CPU replay.
//
//   - V41RowLogAlloc returns a fresh caller-owned [rows, width] buffer whose
//     contents are unspecified until written.
//   - V41RowLogCopy overwrites dst rows [dstRow, dstRow+rows) with src rows
//     [srcRow, srcRow+rows) in place. Overlapping ranges of one buffer refuse.
//   - V41RowLogGather returns a fresh caller-owned panel holding every span's
//     rows in span order, repetitions preserved.
type V41RowLogBackend interface {
	V41RowLogAvailable() bool
	V41RowLogAlloc(rows, width int) (Tensor, error)
	V41RowLogCopy(dst Tensor, dstRow int, src Tensor, srcRow, rows int) error
	V41RowLogGather(spans []V41RowSpan, width int) (Tensor, error)
}

// V41RowSpan names rows [Row, Row+Rows) of Src.
type V41RowSpan struct {
	Src       Tensor
	Row, Rows int
}

// ErrV41RowLogGeometry marks a row-log request refused before any device work.
var ErrV41RowLogGeometry = errors.New("compute: invalid V4.1 row-log geometry")

func v41RowLogShape(t Tensor) (rows, width int, ok bool) {
	if t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 {
		return 0, 0, false
	}
	rows, width = t.Shape[0], t.Shape[1]
	if width > math.MaxInt32/4 || rows > math.MaxInt/(width*4) {
		return 0, 0, false
	}
	return rows, width, true
}

// ValidateV41RowLogAlloc checks an allocation geometry.
func ValidateV41RowLogAlloc(rows, width int) error {
	if rows <= 0 || width <= 0 || width > math.MaxInt32/4 || rows > math.MaxInt/(width*4) {
		return fmt.Errorf("%w: alloc rows=%d width=%d", ErrV41RowLogGeometry, rows, width)
	}
	return nil
}

// ValidateV41RowLogCopy checks an in-place copy and returns the row byte stride.
func ValidateV41RowLogCopy(dst Tensor, dstRow int, src Tensor, srcRow, rows int) (rowBytes int, err error) {
	dRows, dWidth, dOK := v41RowLogShape(dst)
	sRows, sWidth, sOK := v41RowLogShape(src)
	if !dOK || !sOK || dWidth != sWidth {
		return 0, fmt.Errorf("%w: copy operands must be F32 [rows,width] with equal widths", ErrV41RowLogGeometry)
	}
	if rows <= 0 || dstRow < 0 || srcRow < 0 || dstRow > dRows-rows || srcRow > sRows-rows {
		return 0, fmt.Errorf("%w: copy rows=%d dst=[%d,+%d) of %d src=[%d,+%d) of %d", ErrV41RowLogGeometry, rows, dstRow, rows, dRows, srcRow, rows, sRows)
	}
	if dst.Buf() != nil && dst.Buf() == src.Buf() && dstRow < srcRow+rows && srcRow < dstRow+rows {
		return 0, fmt.Errorf("%w: overlapping in-place copy", ErrV41RowLogGeometry)
	}
	return dWidth * 4, nil
}

// ValidateV41RowLogGather checks every span and returns the panel row count.
func ValidateV41RowLogGather(spans []V41RowSpan, width int) (total int, err error) {
	if len(spans) == 0 || width <= 0 {
		return 0, fmt.Errorf("%w: gather needs spans and a positive width", ErrV41RowLogGeometry)
	}
	for i, span := range spans {
		rows, w, ok := v41RowLogShape(span.Src)
		if !ok || w != width || span.Rows <= 0 || span.Row < 0 || span.Row > rows-span.Rows {
			return 0, fmt.Errorf("%w: gather span %d rows [%d,+%d) width %d", ErrV41RowLogGeometry, i, span.Row, span.Rows, width)
		}
		if total > math.MaxInt32-span.Rows {
			return 0, fmt.Errorf("%w: gather row count overflows", ErrV41RowLogGeometry)
		}
		total += span.Rows
	}
	return total, ValidateV41RowLogAlloc(total, width)
}
