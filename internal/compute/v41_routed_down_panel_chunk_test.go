package compute

import (
	"math"
	"testing"
)

// fak-test: unit
func TestV41RoutedDownChunkGeometry(t *testing.T) {
	for panel := 1; panel <= 64; panel++ {
		for _, rows := range []int{2, 4, 8} {
			for first := 0; first <= panel; first++ {
				p, e := PlanV41RoutedDownPanelChunk(panel, 65, first, rows)
				if first+rows > panel {
					if e == nil {
						t.Fatal("extent accepted")
					}
					continue
				}
				if e != nil || p.PanelBytes != panel*260 || p.OutputBytes != rows*260 || p.StatusBytes != rows*16 || p.OffsetElements != first*65 {
					t.Fatalf("plan %+v %v", p, e)
				}
				if p.OutputBytes+p.StatusBytes != rows*(65*4+16) {
					t.Fatal("aggregate requested output/status changed")
				}
			}
		}
	}
	for _, v := range [][4]int{{0, 1, 0, 2}, {65, 1, 0, 2}, {8, 0, 0, 2}, {8, 1, -1, 2}, {8, 1, 0, 1}, {8, 1, 0, 3}, {8, 1, 1, 8}, {8, math.MaxInt32, 0, 8}} {
		if _, e := PlanV41RoutedDownPanelChunk(v[0], v[1], v[2], v[3]); e == nil {
			t.Fatalf("accepted %v", v)
		}
	}
}

// fak-test: unit
func TestV41RoutedDownChunkValidation(t *testing.T) {
	p := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{8, 65}}
	if _, e := validateV41RoutedDownPanelChunk(p, 0, 8); e != nil {
		t.Fatal(e)
	}
	for _, shape := range [][]int{{8 * 65}, {8, 65, 1}, {0, 65}} {
		x := p
		x.Shape = shape
		if _, e := validateV41RoutedDownPanelChunk(x, 0, 8); e == nil {
			t.Fatal("shape")
		}
	}
	x := p
	x.Dtype = F16
	if _, e := validateV41RoutedDownPanelChunk(x, 0, 8); e == nil {
		t.Fatal("dtype")
	}
	x = p
	x.Layout = Layout(255)
	if _, e := validateV41RoutedDownPanelChunk(x, 0, 8); e == nil {
		t.Fatal("layout")
	}
}
