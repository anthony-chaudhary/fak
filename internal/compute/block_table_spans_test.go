package compute

import (
	"reflect"
	"testing"
)

// TestBlockTableSpanRebuild proves the (released, pages) span contract: a full
// block-table row following the closed 0 / values / -1 contract exports to a span plus a
// zero-copy live window, and rebuilding the row from the span and the live window matches
// the original row byte-for-byte. It covers empty, full, all-hole, all-padding, and tile
// edges, plus malformed-span rejection.
func TestBlockTableSpanRebuild(t *testing.T) {
	valid := []struct {
		name string
		row  []int32
	}{
		{"empty", []int32{}},
		{"full-no-holes", []int32{5, 7, 9, 11}},
		{"all-hole", []int32{0, 0, 0, 0}},
		{"all-padding", []int32{-1, -1, -1, -1}},
		{"hole-and-padding", []int32{0, 0, 3, 4, -1, -1}},
		{"single-live", []int32{0, 0, 0, 42, -1, -1, -1}},
		{"tile-edge-released", []int32{0, 0, 0, 0, 1, 2, 3, 4}},
		{"tile-edge-pages", []int32{1, 2, 3, 4, -1, -1, -1, -1}},
		{"wide", []int32{0, 0, 100, 101, 102, -1, -1, -1, -1, -1}},
	}
	for _, tc := range valid {
		t.Run(tc.name, func(t *testing.T) {
			span, live, err := ExportBlockTableSpan(tc.row)
			if err != nil {
				t.Fatalf("ExportBlockTableSpan(%v) unexpected error: %v", tc.row, err)
			}
			if len(live) != int(span.Pages) {
				t.Fatalf("live length = %d, want %d", len(live), span.Pages)
			}
			rebuilt, err := RebuildBlockTableRow(span, live, int32(len(tc.row)))
			if err != nil {
				t.Fatalf("RebuildBlockTableRow(%+v, %v, %d) unexpected error: %v", span, live, len(tc.row), err)
			}
			if !reflect.DeepEqual(rebuilt, tc.row) {
				t.Fatalf("rebuild mismatch:\n got  %v\n want %v", rebuilt, tc.row)
			}
		})
	}

	t.Run("malformed", func(t *testing.T) {
		if err := (BlockTableSpan{Released: -1, Pages: 0}).Validate(4); err == nil {
			t.Error("expected error for negative released")
		}
		if err := (BlockTableSpan{Released: 0, Pages: -1}).Validate(4); err == nil {
			t.Error("expected error for negative pages")
		}
		if err := (BlockTableSpan{Released: 5, Pages: 0}).Validate(4); err == nil {
			t.Error("expected error for released exceeding width")
		}
		if err := (BlockTableSpan{Released: 3, Pages: 3}).Validate(4); err == nil {
			t.Error("expected error for released+pages exceeding width")
		}
		if _, err := RebuildBlockTableRow(BlockTableSpan{Released: 0, Pages: 2}, []int32{1}, 4); err == nil {
			t.Error("expected error for live length disagreeing with span")
		}
		if _, err := RebuildBlockTableRow(BlockTableSpan{Released: 0, Pages: 1}, []int32{0}, 4); err == nil {
			t.Error("expected error for live value 0")
		}
		if _, err := RebuildBlockTableRow(BlockTableSpan{Released: 0, Pages: 1}, []int32{-5}, 4); err == nil {
			t.Error("expected error for negative live value")
		}
		if _, _, err := ExportBlockTableSpan([]int32{0, 0, 5, 0, 6}); err == nil {
			t.Error("expected error for a 0 inside the live run")
		}
		if _, _, err := ExportBlockTableSpan([]int32{5, -1, 6}); err == nil {
			t.Error("expected error for a non-(-1) entry in the padding tail")
		}
	})
}
