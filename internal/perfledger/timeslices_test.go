package perfledger

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestTimeslices(t *testing.T) {
	s, err := NewColumnStore(ColumnStoreLimits{})
	if err != nil {
		t.Fatal(err)
	}
	// Record 0: [0, 80], Record 1: [50, 150], Record 2: [200, 250]
	if err := s.Ingest(0, 0, 80, map[string]any{"ttft": 10.0}); err != nil {
		t.Fatal(err)
	}
	if err := s.Ingest(1, 50, 150, map[string]any{"ttft": 20.0}); err != nil {
		t.Fatal(err)
	}
	if err := s.Ingest(2, 200, 250, map[string]any{"ttft": 30.0}); err != nil {
		t.Fatal(err)
	}

	// Clip at 250 with slice width 100 -> 3 slices: [0,100] complete, [100,200] complete, [200,250] incomplete.
	grid, err := BuildTimeslices(s, 100, 250)
	if err != nil {
		t.Fatalf("BuildTimeslices: %v", err)
	}
	if len(grid.Slices) != 3 {
		t.Fatalf("len(Slices) = %d, want 3", len(grid.Slices))
	}
	if !grid.Slices[0].IsComplete || !grid.Slices[1].IsComplete || grid.Slices[2].IsComplete {
		t.Fatalf("unexpected IsComplete flags: %+v", grid.Slices)
	}
	if grid.Slices[2].EndNS != 250 {
		t.Fatalf("clipped EndNS = %v, want 250", grid.Slices[2].EndNS)
	}
	if got := grid.SliceRows(0); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("slice 0 rows = %v, want [0 1]", got)
	}
	if got := grid.SliceRows(1); !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("slice 1 rows = %v, want [1 2]", got)
	}
	if got := grid.SliceRows(2); !reflect.DeepEqual(got, []int{2}) {
		t.Fatalf("slice 2 rows = %v, want [2]", got)
	}

	// NaN clipEndNS treats all slices as complete.
	unclipped, err := BuildTimeslices(s, 100, math.NaN())
	if err != nil {
		t.Fatalf("unclipped BuildTimeslices: %v", err)
	}
	if !unclipped.Slices[2].IsComplete {
		t.Fatalf("unclipped trailing slice must have IsComplete=true")
	}
}
