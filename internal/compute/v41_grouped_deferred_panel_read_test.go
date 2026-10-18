package compute

import (
	"encoding/binary"
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedDeferredPanelRead(t *testing.T) {
	fixture := func(rows, width int) ([]V41DeferredStatus, []Tensor, []int) {
		statuses := make([]V41DeferredStatus, 17)
		for i := range statuses {
			statuses[i] = V41DeferredStatus{Kind: V41DeferredBF16Publish, Site: V41BF16PublishGroupedOutputProjection, Count: rows * width}
		}
		return statuses, []Tensor{{Dtype: F32, Layout: RowMajor, Shape: []int{rows * width}}}, []int{rows}
	}
	t.Run("separate-geometry", func(t *testing.T) {
		for _, rows := range []int{1, 2, 4, 8, 11, 63, 64} {
			statuses, panels, layout := fixture(rows, 5120)
			if n, ok := v41GroupedDeferredPanelLayout(statuses, panels, layout, 5120); !ok || n != rows {
				t.Fatalf("grouped geometry %d declined", rows)
			}
			if rows > 8 {
				if _, ok := v41DeferredPanelRows(panels, layout, 5120); ok {
					t.Fatal("routed reader inherited grouped geometry")
				}
			}
		}
		for _, rows := range []int{0, 65, math.MaxInt} {
			if V41GroupedDeferredPanelEligible(rows, 8, 5120) {
				t.Fatalf("row bound accepted %d", rows)
			}
		}
		for _, groups := range []int{0, 32, math.MaxInt} {
			if V41GroupedDeferredPanelEligible(64, groups, 5120) {
				t.Fatalf("group bound accepted %d", groups)
			}
		}
		if V41GroupedDeferredPanelEligible(64, 8, math.MaxInt32/4/64+1) {
			t.Fatal("byte extent overflow accepted")
		}
		// Five eight-row panels still exceed the routed cohort's old32 bound.
		panels := make([]Tensor, 5)
		for i := range panels {
			panels[i] = Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{8}}
		}
		if _, ok := v41DeferredPanelRows(panels, []int{8, 8, 8, 8, 8}, 1); ok {
			t.Fatal("routed total widened")
		}
	})
	t.Run("malformed-metadata", func(t *testing.T) {
		for _, mutation := range []string{"short", "even", "many", "multi-record", "kind", "site", "count", "final-count", "rank", "dtype", "layout", "quant", "two-panels"} {
			statuses, panels, layout := fixture(64, 5120)
			switch mutation {
			case "short":
				statuses = statuses[:1]
			case "even":
				statuses = statuses[:16]
			case "many":
				statuses = append(statuses, make([]V41DeferredStatus, 48)...)
			case "multi-record":
				statuses[0].Records = 2
			case "kind":
				statuses[0].Kind = V41DeferredRoutedSwiGLU
			case "site":
				statuses[0].Site = 0
			case "count":
				statuses[0].Count = 1
			case "final-count":
				statuses[16].Count = 64
			case "rank":
				panels[0].Shape = []int{64, 5120}
			case "dtype":
				panels[0].Dtype = Q2_K
			case "layout":
				panels[0].Layout = Layout(255)
			case "quant":
				panels[0].Quant = &QuantSpec{}
			case "two-panels":
				panels = append(panels, panels[0])
				layout = append(layout, 64)
			}
			if _, ok := v41GroupedDeferredPanelLayout(statuses, panels, layout, 5120); ok {
				t.Fatalf("accepted %s", mutation)
			}
		}
	})
	t.Run("independent-materialization", func(t *testing.T) {
		for _, rows := range []int{11, 64} {
			const width = 4
			sizes := make([]int, 18)
			for i := range sizes[:17] {
				sizes[i] = 16
			}
			sizes[17] = rows * width * 4
			if _, _, ok := v41TypedReadbackLayout(sizes, 1, true); ok {
				t.Fatal("ordinary256-byte prefix cap changed")
			}
			image := newV41HostReadbackImage(sizes, rows, v41NativeLittleEndian())
			if image.floats != nil {
				t.Fatal("packed reader retained typed panel")
			}
			raw := image.region(17, sizes[17])
			for i := 0; i < rows*width; i++ {
				binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(float32(i)))
			}
			values := image.materializePanelRows(sizes, 17, []int{rows}, width, rows)
			for i, row := range values {
				if len(row) != width || cap(row) != width {
					t.Fatal("row geometry/capacity changed")
				}
				for j, v := range row {
					if v != float32(i*width+j) {
						t.Fatal("row bits changed")
					}
				}
			}
			values[0][0] = 99
			if values[1][0] != 4 {
				t.Fatal("peer rows alias")
			}
			for i := range image.host {
				image.host[i] = 0xff
			}
			if values[1][0] != 4 {
				t.Fatal("returned row retained image")
			}
			if got := testing.AllocsPerRun(20, func() {
				v := image.materializePanelRows(sizes, 17, []int{rows}, width, rows)
				if len(v) != rows {
					panic("row count")
				}
			}); got != float64(rows+1) {
				t.Fatalf("allocations %v want%d; no intermediate panel permitted", got, rows+1)
			}
		}
	})
}
