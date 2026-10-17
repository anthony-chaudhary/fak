package compute

import (
	"encoding/binary"
	"math"
	"unsafe"
)

// The current deferred readers return zero or one tensor. Only that one-row
// case can share its host image: an escaped row must never retain peer rows.
// At most two status records per routed pick precede the row. The additional
// requested backing is bounded both absolutely and by the payload; allocator
// size-class rounding is not included in this requested-byte bound.
const v41ReadbackRetainedPrefixMax = 2 * V41RoutedIDMaxPicks * 16

type v41HostReadbackImage struct {
	host       []byte
	floats     []float32
	offsets    []int
	typedRow   int
	typedBytes int
}

func v41NativeLittleEndian() bool {
	word := uint32(1)
	return *(*byte)(unsafe.Pointer(&word)) == 1
}

// This selector performs no allocation or pointer construction on decline.
// The admitted row is last, so it retains at most the bounded status prefix.
// Unknown/multirow layouts keep the independent legacy row allocations.
func v41TypedReadbackLayout(sizes []int, rows int, littleEndian bool) (prefix, total int, ok bool) {
	if !littleEndian || rows != 1 || len(sizes) == 0 || len(sizes) > v41ReadbackRetainedPrefixMax/16+1 {
		return 0, 0, false
	}
	// Index selection has exactly one 4-byte status word. Its row remains
	// float32-aligned; the shared layout pads this prefix to 16 bytes.
	indexStatus := len(sizes) == 2 && sizes[0] == 4
	for _, n := range sizes[:len(sizes)-1] {
		if n <= 0 || (n%16 != 0 && !indexStatus) || n > v41ReadbackRetainedPrefixMax-prefix {
			return 0, 0, false
		}
		if indexStatus {
			prefix += 16 // v41DeferredLayout aligns the following row to 16 bytes.
		} else {
			prefix += n
		}
	}
	payload := sizes[len(sizes)-1]
	if payload <= 0 || payload%4 != 0 || prefix > payload || payload > math.MaxInt-prefix {
		return 0, 0, false
	}
	total = prefix + payload
	if total <= 0 || total%4 != 0 {
		return 0, 0, false
	}
	return prefix, total, true
}

// A typed image is allocated as []float32, never obtained by assuming an
// arbitrary []byte slice is aligned. Its byte view covers that exact owned,
// pointer-free Go allocation. Layout checks prove a positive multiple-of-four
// extent before constructing the view. The native reader fills it synchronously
// and never retains the host pointer. Both views keep the Go allocation live.
func newV41HostReadbackImage(sizes []int, rows int, littleEndian bool) v41HostReadbackImage {
	if _, total, ok := v41TypedReadbackLayout(sizes, rows, littleEndian); ok {
		offsets, _ := v41DeferredLayout(sizes)
		floats := make([]float32, total/4)
		host := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(floats))), total)
		return v41HostReadbackImage{host: host, floats: floats, offsets: offsets, typedRow: len(sizes) - 1, typedBytes: sizes[len(sizes)-1]}
	}
	// Retain the existing layout, allocation, and failure behavior on decline.
	offsets, total := v41DeferredLayout(sizes)
	return v41HostReadbackImage{host: make([]byte, total), offsets: offsets}
}

func (image v41HostReadbackImage) region(index, size int) []byte {
	start := image.offsets[index]
	return image.host[start : start+size : start+size]
}

// v41DecodeF32Row materializes one independent full-cap float32 row from a raw
// little-endian byte region. On a little-endian host the float32 bit pattern is
// the raw bytes, so one bulk copy replaces the per-element decode loop; a
// big-endian host keeps the explicit decode. The row is always a fresh
// allocation: callers may retain any single row, so rows must never alias the
// shared image or a peer row.
func v41DecodeF32Row(raw []byte, n int) []float32 {
	row := make([]float32, n)
	if n == 0 {
		return row
	}
	if v41NativeLittleEndian() {
		copy(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(row))), 4*n), raw[:4*n])
		return row
	}
	for j := range row {
		row[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
	}
	return row
}

// materializeRows is called only after the unchanged status-verdict loop.
// The typed arm exposes exactly one full-cap row; all other layouts decode
// into independent allocations, preserving their original retention behavior.
func (image v41HostReadbackImage) materializeRows(sizes []int, first int) [][]float32 {
	values := make([][]float32, len(sizes)-first)
	for i := range values {
		index, n := first+i, sizes[first+i]/4
		if image.floats != nil && len(values) == 1 && index == image.typedRow && sizes[index] == image.typedBytes {
			start := image.offsets[index] / 4
			values[i] = image.floats[start : start+n : start+n]
			continue
		}
		raw := image.region(index, sizes[index])
		values[i] = v41DecodeF32Row(raw, n)
	}
	return values
}
