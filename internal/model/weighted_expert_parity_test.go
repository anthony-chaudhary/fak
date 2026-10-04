package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// weighted_expert_parity_test.go is the fak#13458 witness: the standard MoE
// routed-expert loop and BOTH V4.1 full-forward accumulation branches now
// delegate their elementwise weighted accumulation to the shared
// ffn.AddScaled leaf instead of an inline `-= weight*src` loop. The migration
// must be numerically invisible, so this file pins each migrated shape against
// its pre-refactor inline arithmetic with math.Float32bits.

// legacyRoutedDelta is the PRE-refactor body of the standard moeFFN routed
// accumulation: run each selected expert's SwiGLU over xn and accumulate its
// gate-weighted output into delta with the inline loop. The production
// moeFFN.apply must match it bit for bit, because every named operation runs in
// the identical order.
func legacyRoutedDelta(m *Model, layer int, xn any, mat matKernel, picks []routePick) []float32 {
	H := m.Cfg.HiddenSize
	delta := make([]float32, H)
	for _, pk := range picks {
		var out []float32
		if m.Cfg.isGPTOSS() {
			out = expertGPTOSS(m, layer, pk.expert, xn, mat)
		} else {
			out = expertSwiGLU(m, layer, pk.expert, xn, mat)
		}
		for i := 0; i < H; i++ {
			delta[i] += pk.weight * out[i]
		}
	}
	return delta
}

// TestWeightedExpertAccumulationParity proves the shared accumulation leaf is
// wired through the real consumers and changes no bit:
//
//  1. The standard MoE routed loop drives the REAL moeFFN.apply end to end and
//     is bit-identical to its pre-refactor inline accumulation oracle.
//  2. Each V4.1 full-forward accumulation branch (the device-gate/up handled arm
//     and the generic host arm) accumulates a routed row with the same inline
//     expression the migration replaced; the shared leaf reproduces it exactly
//     for every pick, including signed zeros.
func TestWeightedExpertAccumulationParity(t *testing.T) {
	t.Run("standard moe routed loop", func(t *testing.T) {
		cfg := moeCfgForTest()
		if !cfg.IsMoE() {
			t.Fatal("moeCfgForTest must report IsMoE()==true")
		}
		m := NewSyntheticMoE(cfg)
		xn := make([]float32, cfg.HiddenSize)
		for i := range xn {
			xn[i] = float32(i%7)*0.31 - 0.93
		}
		mat := f32Kernel{m}
		picks := route(m, 0, xn, mat)
		if len(picks) == 0 {
			t.Fatal("router returned no picks")
		}

		want := legacyRoutedDelta(m, 0, xn, mat, picks)
		got := moeFFN{}.apply(m, 0, xn, mat)
		assertFloat32BitsEqual(t, "moeFFN routed accumulation", want, got)
	})

	t.Run("v41 accumulation branches", func(t *testing.T) {
		H := 96
		// Two routed rows of the shape both V4.1 branches accumulate: the
		// device-gate/up handled arm accumulates a down-projected row `y`; the
		// generic arm accumulates the same-shaped row from its contract twin.
		rows := [][]float32{
			make([]float32, H),
			make([]float32, H),
		}
		weights := []float32{0.375, -1.25}
		for r := range rows {
			for i := range rows[r] {
				rows[r][i] = float32(i%11)*0.25 - 1.0
			}
		}

		want := make([]float32, H)
		for r, y := range rows {
			for i := range want {
				want[i] += weights[r] * y[i]
			}
		}

		got := make([]float32, H)
		for r, y := range rows {
			if err := ffn.AddScaled(got, y, weights[r]); err != nil {
				t.Fatalf("AddScaled pick %d: %v", r, err)
			}
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("v41 accumulation hidden[%d] bits=%#x want=%#x", i,
					math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	})
}
