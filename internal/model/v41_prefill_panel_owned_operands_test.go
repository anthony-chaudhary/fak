package model

import "testing"

var v41PanelOperandSink []float32

// fak-test:runtime fast est=20ms lane=default
func TestV41PrefillPanelOwnedOperandRows(t *testing.T) {
	const width = 17
	packed := make([]float32, 3*width)
	row := v41PrefillPanelWeightedRow(packed, 1, width, nil)
	if len(row) != width || cap(row) != width || &row[0] != &packed[width] {
		t.Fatal("host row must be capped private destination view")
	}
	row[0] = 7
	if packed[width] != 7 || packed[0] != 0 || packed[2*width] != 0 {
		t.Fatal("wrong destination or sibling mutation")
	}
	grown := append(row, 9)
	if grown[width] != 9 || packed[2*width] != 0 {
		t.Fatal("append escaped row capacity")
	}
	selected := v41BF16BoundaryFunc(func(_ int, _ string, x []float32) ([]float32, error) { return x, nil })
	independent := v41PrefillPanelWeightedRow(packed, 1, width, selected)
	if len(independent) != width || cap(independent) != width || &independent[0] == &packed[width] {
		t.Fatal("selected publication lost independent ownership")
	}
	independent[0] = 11
	if packed[width] != 7 {
		t.Fatal("selected publication input aliases packed destination")
	}
	hostAllocs := testing.AllocsPerRun(100, func() { v41PanelOperandSink = v41PrefillPanelWeightedRow(packed, 1, width, nil) })
	selectedAllocs := testing.AllocsPerRun(100, func() { v41PanelOperandSink = v41PrefillPanelWeightedRow(packed, 1, width, selected) })
	v41PanelOperandSink = nil
	if hostAllocs != 0 || selectedAllocs != 1 {
		t.Fatalf("row allocation counts host=%g selected=%g; want0/1", hostAllocs, selectedAllocs)
	}
}
