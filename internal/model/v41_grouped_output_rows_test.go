package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputRowsMatchFlat(t *testing.T) {
	for _, rows := range []int{2, 11, 65} {
		t.Run(itoa(rows), func(t *testing.T) {
			m, flatSession, _ := v41GroupedPanelFixture(t)
			_, rowSession, _ := v41GroupedPanelFixture(t)
			defer flatSession.Close()
			defer rowSession.Close()
			c := m.Cfg
			width := c.NumHeads * c.HeadDim
			input := v41GroupedPanelInput(rows, width)
			before := append([]float32(nil), input...)
			views := make([][]float32, rows)
			for i := range views {
				views[i] = input[i*width : (i+1)*width : (i+1)*width]
			}
			want, wo, we := flatSession.v41GroupedOutputPanel(0, input, rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			got, go_, ge := rowSession.v41GroupedOutputPanelRows(0, views, rows, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			if we != nil || ge != nil || wo != v41ProjectionHandled || go_ != wo || !reflect.DeepEqual(got, want) {
				t.Fatalf("row view differs: flat=%v/%v rows=%v/%v", wo, we, go_, ge)
			}
			for i := range input {
				if math.Float32bits(input[i]) != math.Float32bits(before[i]) {
					t.Fatalf("input changed at%d", i)
				}
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputRowsValidateBeforeBackend(t *testing.T) {
	for _, invalid := range []string{"late_nonfinite", "late_width", "backend_decline"} {
		t.Run(invalid, func(t *testing.T) {
			m, s, b := v41GroupedPanelFixture(t)
			defer s.Close()
			c := m.Cfg
			width := c.NumHeads * c.HeadDim
			input := v41GroupedPanelInput(3, width)
			views := [][]float32{input[:width:width], input[width : 2*width : 2*width], input[2*width : 3*width : 3*width]}
			switch invalid {
			case "late_nonfinite":
				views[2][width-1] = float32(math.NaN())
			case "late_width":
				views[2] = views[2][:width-1]
			case "backend_decline":
				b.deny = true
			}
			got, outcome, err := s.v41GroupedOutputPanelRows(0, views, 3, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
			if got != nil || err != nil || outcome != v41ProjectionDeclined || b.uploads != 0 || b.matmuls != 0 || b.deferredPublishes != 0 {
				t.Fatalf("invalid input reached backend: %v/%v/%v effects%d/%d/%d", got, outcome, err, b.uploads, b.matmuls, b.deferredPublishes)
			}
		})
	}
}
