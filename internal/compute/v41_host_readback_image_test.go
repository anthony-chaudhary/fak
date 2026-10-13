package compute

import (
	"bytes"
	"encoding/binary"
	"math"
	"reflect"
	"runtime"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41TypedReadbackAdmissionBounds(t *testing.T) {
	for _, c := range []struct {
		name   string
		sizes  []int
		rows   int
		little bool
		ok     bool
	}{
		{"one row", []int{16, 64}, 1, true, true},
		{"rows only", []int{4}, 1, true, true},
		{"max prefix", []int{128, 128, 256}, 1, true, true},
		{"status only", []int{16}, 0, true, false},
		{"two rows", []int{16, 64, 64}, 2, true, false},
		{"unknown endian", []int{16, 64}, 1, false, false},
		{"empty", nil, 1, true, false},
		{"zero row", []int{16, 0}, 1, true, false},
		{"negative row", []int{16, -4}, 1, true, false},
		{"zero status", []int{0, 64}, 1, true, false},
		{"negative status", []int{-16, 64}, 1, true, false},
		{"unaligned status", []int{12, 64}, 1, true, false},
		{"odd payload", []int{16, 65}, 1, true, false},
		{"small payload", []int{16, 4}, 1, true, false},
		{"large prefix", []int{256, 16, 1024}, 1, true, false},
		{"huge status", []int{math.MaxInt, 64}, 1, true, false},
		{"addition overflow", []int{16, math.MaxInt &^ 3}, 1, true, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			prefix, total, ok := v41TypedReadbackLayout(c.sizes, c.rows, c.little)
			if ok != c.ok {
				t.Fatalf("admission=%t, want %t", ok, c.ok)
			}
			if ok && (total <= 0 || total%4 != 0 || prefix > v41ReadbackRetainedPrefixMax || prefix > total-prefix) {
				t.Fatal("unsafe requested-byte bound")
			}
		})
	}
	many := make([]int, v41ReadbackRetainedPrefixMax/16+2)
	for i := range many {
		many[i] = 16
	}
	if _, _, ok := v41TypedReadbackLayout(many, 1, true); ok {
		t.Fatal("unbounded status count admitted")
	}
	// Exercise the fallback constructor's zero extent without taking a pointer
	// to element zero. Invalid native read geometry remains the wrapper's job.
	if image := newV41HostReadbackImage(nil, 1, true); len(image.host) != 0 || image.floats != nil {
		t.Fatal("zero image unexpectedly typed")
	}
}

func v41ReadbackTestFill(image v41HostReadbackImage, sizes []int, status []byte, words []uint32) {
	if len(status) != 0 {
		copy(image.region(0, sizes[0]), status)
	}
	raw := image.region(len(sizes)-1, sizes[len(sizes)-1])
	for i := range words {
		binary.LittleEndian.PutUint32(raw[4*i:], words[i])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41TypedReadbackBitsAndStatusParity(t *testing.T) {
	words := []uint32{0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234, 0xff801234}
	sizes := []int{16, 4 * len(words)}
	for _, c := range []struct {
		status V41DeferredStatus
		record []byte
	}{
		{V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: len(words), Site: V41BF16PublishRoutedDown}, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedDown, math.MaxUint32, uint32(len(words)))},
		{V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: len(words), Site: V41BF16PublishRoutedDown}, v41DeferredRecord(v41BF16CheckedStatusMagic, V41BF16PublishRoutedDown, 3, uint32(len(words)))},
		{V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: len(words), Site: V41BF16PublishRoutedDown}, v41DeferredRecord(0, V41BF16PublishRoutedDown, 3, uint32(len(words)))},
		{V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: len(words)}, v41DeferredRecord(v41RoutedSwiGLUStatusMagic, uint32(len(words)), 3, math.MaxUint32)},
		{V41DeferredStatus{Kind: 255, Count: len(words)}, v41DeferredRecord(0, 1, 2, 3)},
	} {
		legacy := newV41HostReadbackImage(sizes, 1, false)
		next := newV41HostReadbackImage(sizes, 1, v41NativeLittleEndian())
		v41ReadbackTestFill(legacy, sizes, c.record, words)
		v41ReadbackTestFill(next, sizes, c.record, words)
		oldErr := v41DeferredVerdict("vulkan", c.status, legacy.region(0, 16))
		newErr := v41DeferredVerdict("vulkan", c.status, next.region(0, 16))
		if !reflect.DeepEqual(oldErr, newErr) {
			t.Fatalf("status changed: old=%v new=%v", oldErr, newErr)
		}
		a, b := legacy.materializeRows(sizes, 1)[0], next.materializeRows(sizes, 1)[0]
		for i, want := range words {
			if math.Float32bits(a[i]) != want || math.Float32bits(b[i]) != want {
				t.Fatalf("word %d changed", i)
			}
		}
		if !bytes.Equal(next.region(0, 16), c.record) {
			t.Fatal("status bits changed while materializing floats")
		}
		if cap(b) != len(b) {
			t.Fatal("row exposes status/padding capacity")
		}
	}
}

// Causal RED keeps the same typed image and native wrapper, but uses the
// original copy/LE-decode loop in materializeRows. Numeric parity still holds;
// only this production materialization identity witness fails.
// fak-test:runtime fast est=10ms lane=default
func TestV41TypedReadbackAvoidsSecondPayloadAllocation(t *testing.T) {
	if !v41NativeLittleEndian() {
		t.Skip("native typed arena requires little endian")
	}
	sizes := []int{16, 64}
	image := newV41HostReadbackImage(sizes, 1, true)
	row := image.materializeRows(sizes, 1)[0]
	if image.floats == nil || &row[0] != &image.floats[image.offsets[1]/4] {
		t.Fatal("row was copied out of its owned readback arena")
	}
	if uintptr(unsafe.Pointer(&row[0]))%unsafe.Alignof(float32(0)) != 0 {
		t.Fatal("float row is not naturally aligned")
	}
	if len(image.host)-4*len(row) != 16 || len(image.host) != 4*len(image.floats) {
		t.Fatal("requested backing accounting changed")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestV41TypedReadbackLifetimeAndFallbackIsolation(t *testing.T) {
	makeRow := func() ([]float32, []byte, int) {
		sizes := []int{16, 64}
		image := newV41HostReadbackImage(sizes, 1, v41NativeLittleEndian())
		row := image.materializeRows(sizes, 1)[0]
		row[0] = math.Float32frombits(0x3f818000)
		return row, image.region(0, 16), len(image.host)
	}
	row, status, requested := makeRow()
	escaped := func() []float32 { onlyRow, _, _ := makeRow(); return onlyRow }()
	runtime.GC()
	if math.Float32bits(escaped[0]) != 0x3f818000 {
		t.Fatal("row without byte-view owner did not survive GC")
	}
	other, _, _ := makeRow()
	other[0] = 77
	if math.Float32bits(row[0]) != 0x3f818000 || requested != 80 || cap(row) != len(row) {
		t.Fatal("escaped row lifetime or bound changed")
	}
	row[1] = 88
	if !bytes.Equal(status, make([]byte, 16)) {
		t.Fatal("row overwrote status")
	}
	extended := append(row, 99)
	if &extended[0] == &row[0] {
		t.Fatal("append reused full-cap arena")
	}

	// Unknown multirow lifetimes retain separate allocations, even though all
	// source rows arrived in one byte image. A caller can retain any one row.
	sizes := []int{16, 20, 24}
	image := newV41HostReadbackImage(sizes, 2, true)
	if image.floats != nil {
		t.Fatal("peer payload retention admitted")
	}
	rows := image.materializeRows(sizes, 1)
	for i := range image.host {
		image.host[i] = 0xff
	}
	rows[0][0] = 1
	if rows[1][0] != 0 {
		t.Fatal("fallback peers alias")
	}
	grown := append(rows[0], 2, 3, 4, 5, 6, 7, 8)
	grown[len(grown)-1] = 9
	if rows[1][0] != 0 {
		t.Fatal("fallback append corrupted peer")
	}
	if rows[0][1] != 0 {
		t.Fatal("fallback row borrows readback image")
	}
}
