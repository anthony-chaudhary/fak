package compute

import (
	"math"
	"testing"
)

// fak-test: unit
func TestV41RoutedSwiGLUChunkGeometry(t *testing.T) {
	for panel := 1; panel <= 64; panel++ {
		for _, rows := range []int{2, 4, 8} {
			for first := 0; first <= panel; first++ {
				p, err := PlanV41RoutedSwiGLUPanelChunk(panel, 65, first, rows)
				if first+rows > panel {
					if err == nil {
						t.Fatal("invalid extent accepted")
					}
					continue
				}
				if err != nil || p.PanelBytes != panel*65*4 || p.OutputBytes != rows*65*4 || p.StatusBytes != 16*rows || p.OffsetElements != first*65 {
					t.Fatalf("wrong plan: %+v %v", p, err)
				}
				for r := 0; r < rows; r++ {
					for i := 0; i < 65; i++ {
						if p.OffsetElements+r*65+i >= panel*65 {
							t.Fatal("input overrun")
						}
					}
				}
				// Old schedule retained each row plus a packed copy; chunk owns only packed output.
				if p.OutputBytes+p.StatusBytes > 2*rows*65*4+rows*16 {
					t.Fatal("payload peak increased")
				}
			}
		}
	}
	for _, v := range [][4]int{{0, 1, 0, 2}, {65, 1, 0, 2}, {8, 0, 0, 2}, {8, 1, -1, 2}, {8, 1, 0, 1}, {8, 1, 0, 3}, {8, 1, 1, 8}, {8, math.MaxInt32, 0, 8}} {
		if _, err := PlanV41RoutedSwiGLUPanelChunk(v[0], v[1], v[2], v[3]); err == nil {
			t.Fatalf("accepted %v", v)
		}
	}
}

// fak-test: unit
func TestV41RoutedSwiGLUChunkValidation(t *testing.T) {
	g := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{8, 65}}
	weights := []float32{0, math.Float32frombits(0x80000000), 1, -1}
	if _, err := validateV41RoutedSwiGLUPanelChunk(g, g, 4, 4, 0, weights); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := validateV41RoutedSwiGLUPanelChunk(g, g, 0, 4, bad, weights); err == nil {
			t.Fatal("nonfinite limit")
		}
		w := append([]float32(nil), weights...)
		w[3] = bad
		if _, err := validateV41RoutedSwiGLUPanelChunk(g, g, 0, 4, 0, w); err == nil {
			t.Fatal("nonfinite route")
		}
	}
	if _, err := validateV41RoutedSwiGLUPanelChunk(g, g, 0, 4, 0, weights[:3]); err == nil {
		t.Fatal("short route")
	}
	u := g
	u.Shape = []int{7, 65}
	if _, err := validateV41RoutedSwiGLUPanelChunk(g, u, 0, 4, 0, weights); err == nil {
		t.Fatal("shape mismatch")
	}
}
