package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41PackedRowExtent(t *testing.T) {
	t.Parallel()
	e, err := NewV41PackedRowExtent(V41MainKVPer16E4M3NoGlobal, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	if e.Format() != V41MainKVPer16E4M3NoGlobal || e.LogicalWidth() != 512 || e.RecordBytes() != 288 || e.Rows() != 2 || e.Capacity() != 4 {
		t.Fatalf("incorrect main-KV extent: %+v", e)
	}
	if n, err := e.CapacityBytes(); err != nil || n != 1152 {
		t.Fatalf("capacity bytes = %d, %v", n, err)
	}
	if nope, rope, err := e.ScaleOffsets(); err != nil || nope != 256 || rope != 284 {
		t.Fatalf("scale offsets = %d, %d, %v", nope, rope, err)
	}
	if offset, n, err := e.ByteRange(1, 1); err != nil || offset != 288 || n != 288 {
		t.Fatalf("second row = %d, %d, %v", offset, n, err)
	}
	if offset, n, err := e.ByteRange(2, 0); err != nil || offset != 576 || n != 0 {
		t.Fatalf("empty end span = %d, %d, %v", offset, n, err)
	}
	for _, span := range [][2]int{{2, 1}, {-1, 1}, {0, -1}, {math.MaxInt, 1}, {1, math.MaxInt}} {
		if _, _, err := e.ByteRange(span[0], span[1]); !errors.Is(err, ErrV41PackedRowGeometry) {
			t.Fatalf("accepted invalid span %v: %v", span, err)
		}
	}
	for _, invalid := range []V41PackedRowExtent{{}, {format: 255}, {format: V41MainKVPer16E4M3NoGlobal, rows: -1}, {format: V41MainKVPer16E4M3NoGlobal, rows: 2, capacity: 1}, {format: V41MainKVPer16E4M3NoGlobal, capacity: math.MaxInt/288 + 1}} {
		if _, err := NewV41PackedRowExtent(invalid.format, invalid.rows, invalid.capacity); !errors.Is(err, ErrV41PackedRowGeometry) {
			t.Fatalf("accepted invalid extent %+v: %v", invalid, err)
		}
		if _, _, err := invalid.ByteRange(0, 0); !errors.Is(err, ErrV41PackedRowGeometry) {
			t.Fatalf("invalid extent admitted an empty read: %v", err)
		}
	}
	limit, err := NewV41PackedRowExtent(V41MainKVPer16E4M3NoGlobal, math.MaxInt/288, math.MaxInt/288)
	if err != nil {
		t.Fatal(err)
	}
	if _, n, err := limit.ByteRange(0, limit.Rows()); err != nil || n != (math.MaxInt/288)*288 {
		t.Fatalf("maximum representable extent = %d, %v", n, err)
	}
	empty, err := NewV41PackedRowExtent(V41MainKVPer16E4M3NoGlobal, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if offset, n, err := empty.ByteRange(0, 0); err != nil || offset != 0 || n != 0 {
		t.Fatalf("empty extent = %d, %d, %v", offset, n, err)
	}
}
