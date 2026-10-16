package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41CohortHostRowTransferAndLegacyIsolation(t *testing.T) {
	for _, owned := range []bool{false, true} {
		row := []float32{math.Float32frombits(0x80000000), math.Float32frombits(0x00010000), 1, -2}
		got := v41CohortHostRow(row, owned)
		for i, v := range row {
			if math.Float32bits(got[i]) != math.Float32bits(v) {
				t.Fatal("bits changed")
			}
		}
		if (&got[0] == &row[0]) != owned {
			t.Fatal("ownership selection")
		}
		row[0] = 7
		if !owned && math.Float32bits(got[0]) != 0x80000000 {
			t.Fatal("legacy result aliases scratch")
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41CohortHostRowPayloadAllocations(t *testing.T) {
	rows := make([][]float32, 32)
	for i := range rows {
		rows[i] = make([]float32, 5120)
	}
	output := make([][]float32, len(rows))
	measure := func(owned bool) float64 {
		return testing.AllocsPerRun(10, func() {
			for i, row := range rows {
				output[i] = v41CohortHostRow(row, owned)
			}
		})
	}
	legacy, owned := measure(false), measure(true)
	if legacy != 32 || owned != 0 {
		t.Fatalf("payload allocations %g -> %g", legacy, owned)
	}
	t.Logf("payload only: %g -> %g allocations; typed grant overhead measured separately", legacy, owned)
}
