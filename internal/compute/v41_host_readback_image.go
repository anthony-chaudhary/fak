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
	for _, n := range sizes[:len(sizes)-1] {
		if n <= 0 || n%16 != 0 || n > v41ReadbackRetainedPrefixMax-prefix {
			return 0, 0, false
		}
		prefix += n
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
		row := make([]float32, n)
		for j := range row {
			row[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
		}
		values[i] = row
	}
	return values
}
