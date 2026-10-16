package compute

import (
	"encoding/binary"
	"math"
	"testing"
)

// fak-test: unit
func TestV41DeferredPanelGeometry(t *testing.T) {
	rows := []int{8, 4, 2, 1}
	panels := make([]Tensor, len(rows))
	for i, n := range rows {
		panels[i] = Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{n * 5}}
	}
	if n, ok := v41DeferredPanelRows(panels, rows, 5); !ok || n != 15 {
		t.Fatal(n, ok)
	}
	if n, ok := v41DeferredPanelRows(nil, nil, 5); !ok || n != 0 {
		t.Fatal("issued-status-only read")
	}
	for _, width := range []int{0, -1, math.MaxInt32} {
		if _, ok := v41DeferredPanelRows(panels, rows, width); ok {
			t.Fatal("bad width")
		}
	}
	if _, ok := v41DeferredPanelRows(panels, rows[:3], 5); ok {
		t.Fatal("count")
	}
	if _, ok := v41DeferredPanelRows(panels, []int{8, 4, 2, 3}, 5); ok {
		t.Fatal("invalid chunk")
	}
	panels[0].Shape = []int{8, 5}
	if _, ok := v41DeferredPanelRows(panels, rows, 5); ok {
		t.Fatal("matrix is not flat output")
	}
	many := make([]Tensor, 5)
	for i := range many {
		many[i] = Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{40}}
	}
	if _, ok := v41DeferredPanelRows(many, []int{8, 8, 8, 8, 8}, 5); ok {
		t.Fatal("cohort overflow")
	}
}

// fak-test: unit
func TestV41DeferredPanelMaterializationIndependent(t *testing.T) {
	for _, layout := range [][]int{{2}, {4}, {8}, {8, 4, 2, 1}, {8, 8, 8, 8}} {
		width, total := 5, 0
		sizes := []int{32, 16}
		for _, n := range layout {
			sizes = append(sizes, n*width*4)
			total += n
		}
		for _, little := range []bool{false, true} {
			image := newV41HostReadbackImage(sizes, total, little)
			if image.floats != nil {
				t.Fatal("packed read retained typed arena")
			}
			bits := []uint32{0, 0x80000000, 0x3f808000, 0x7fc01234, 0xff800000}
			k := 0
			for i := range layout {
				raw := image.region(i+2, sizes[i+2])
				for j := 0; j < len(raw); j += 4 {
					binary.LittleEndian.PutUint32(raw[j:], bits[k%len(bits)])
					k++
				}
			}
			got := image.materializePanelRows(sizes, 2, layout, width, total)
			if len(got) != total {
				t.Fatal("logical count")
			}
			for _, row := range got {
				if len(row) != width || cap(row) != width {
					t.Fatal("extent")
				}
				for j, v := range row {
					if math.Float32bits(v) != bits[j] {
						t.Fatal("bits")
					}
				}
			}
			for i := range image.host {
				image.host[i] = 0xff
			}
			got[0][0] = 9
			for i, row := range got {
				if i > 0 && math.Float32bits(row[0]) != 0 {
					t.Fatal("peer alias")
				}
				if math.Float32bits(row[1]) != 0x80000000 {
					t.Fatal("image alias")
				}
			}
		}
	}
}
