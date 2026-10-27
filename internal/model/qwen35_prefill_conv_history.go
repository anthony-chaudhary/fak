package model

import (
	"slices"
	"unsafe"
)

// pushConvPrefillTail publishes a completed convolution panel's bounded history.
// Convolution workers have joined and no history reader runs during publication.
// The original loop copies every prompt row although only the final keep survive.
// Retain its initial fill/allocation order and final backing-row permutation, but
// omit middle copies that are overwritten before any consumer can observe them.
func (st *linearAttnLayerState) pushConvPrefillTail(mixed []float32, rows, width, keep int) bool {
	if st == nil || rows <= 0 || width <= 0 || keep <= 0 || rows <= keep ||
		len(mixed)%width != 0 || len(mixed)/width != rows || len(st.conv) > keep {
		return false
	}
	fill := keep - len(st.conv)
	if rows-keep <= fill {
		return false // no middle row to omit after the original initial fill
	}
	// Admit every extent and alias before any mutation. Existing rows have the
	// exact eventual length, so skipped rotations cannot hide reallocations or
	// bounds failures. Generic aliased panels and malformed history use the old
	// loop, including its original prefix writes and panic behavior.
	if uintptr(len(mixed)) > ^uintptr(0)/4 {
		return false
	}
	for i, row := range st.conv {
		if len(row) != width || convPrefillRowsOverlap(row, mixed) {
			return false
		}
		for _, prior := range st.conv[:i] {
			if convPrefillRowsOverlap(row, prior) {
				return false
			}
		}
	}
	for t := 0; t < fill; t++ {
		st.pushConvRow(mixed[t*width:(t+1)*width], keep)
	}
	// All keep buffers now exist with sufficient capacity. Skipping middle
	// writes must still advance the original ring's physical buffer order.
	shift := (rows - fill - keep) % keep
	if shift != 0 {
		slices.Reverse(st.conv[:shift])
		slices.Reverse(st.conv[shift:])
		slices.Reverse(st.conv)
	}
	for t := rows - keep; t < rows; t++ {
		st.pushConvRow(mixed[t*width:(t+1)*width], keep)
	}
	return true
}

// Address-only overlap admission; no pointer is reconstructed or dereferenced.
// The caller bounds both lengths by the admitted mixed panel before this call.
func convPrefillRowsOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	ap := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bp := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if ap <= bp {
		return bp-ap < uintptr(len(a))*4
	}
	return ap-bp < uintptr(len(b))*4
}
