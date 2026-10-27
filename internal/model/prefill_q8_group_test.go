package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=200ms
func TestQ8PrefillGroupedLaunchPreservesSeparateProjectionResults(t *testing.T) {
	oldMode, oldGroup, oldMaxP := qgemmMode, qgemmGroup, qgemmGroupMaxP
	t.Cleanup(func() { qgemmMode, qgemmGroup, qgemmGroupMaxP = oldMode, oldGroup, oldMaxP })
	qgemmGroupMaxP = 1024
	cfg := llamaArchConfig()
	cfg.NumLayers = 2
	m := NewSynthetic(cfg)
	m.Quantize()
	if q8PrefillNeedsTokenLoop(cfg) {
		t.Fatal("fixture must reach ordinary batched Q8 prefill")
	}
	exact := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length%d want%d", label, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s cell%d=%g want%g", label, i, got[i], want[i])
			}
		}
	}
	for _, mode := range []string{qgemmModeTile, qgemmModeLegacy} {
		qgemmMode = mode
		separate, grouped := m.NewSession(), m.NewSession()
		defer separate.Close()
		defer grouped.Close()
		separate.Quant, grouped.Quant = true, true
		for _, prompt := range [][]int{{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37}, {2, 4, 6}} {
			qgemmGroup = false
			want := separate.prefillBatchedQ(prompt)
			qgemmGroup = true
			got := grouped.prefillBatchedQ(prompt)
			exact(mode+" hidden", got, want)
			if grouped.Cache.Len() != separate.Cache.Len() {
				t.Fatal("grouping changed cache position")
			}
			for layer := 0; layer < cfg.NumLayers; layer++ {
				exact(mode+" K", grouped.Cache.K[layer], separate.Cache.K[layer])
				exact(mode+" Kraw", grouped.Cache.Kraw[layer], separate.Cache.Kraw[layer])
				exact(mode+" V", grouped.Cache.V[layer], separate.Cache.V[layer])
			}
		}
	}
}
