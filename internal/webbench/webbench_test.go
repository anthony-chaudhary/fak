// Package webbench tests.
//
// TestGeometryComputeArms verifies the pure prefill-cost arithmetic in
// Geometry.ComputeArms. The three arms are closed-form functions of
// (Prefix, Turns, Action, DOMState, workers):
//
//	growth = Action + DOMState
//	A (naive):      workers * (Turns*Prefix + growth*Turns*(Turns-1)/2)
//	B (per-agent):  workers * (Prefix + growth*(Turns-1))
//	C (fak fused):  Prefix + workers*growth*(Turns-1)
//
// Expected values are computed by hand below so the test fails on any
// regression of the formulas.
package webbench

import "testing"

func TestGeometryComputeArms(t *testing.T) {
	tests := []struct {
		name    string
		geom    Geometry
		workers int
		want    ArmCost
	}{
		{
			// growth = 150+2000 = 2150
			// A: 4 * (5*3400 + 2150*(5*4/2)) = 4*(17000+21500) = 154000
			// B: 4 * (3400 + 2150*4) = 4*12000 = 48000
			// C: 3400 + 4*2150*4 = 3400+34400 = 37800
			name:    "multi-turn multi-worker",
			geom:    Geometry{Prefix: 3400, Turns: 5, Action: 150, DOMState: 2000},
			workers: 4,
			want:    ArmCost{Workers: 4, ANaive: 154000, BAgent: 48000, CFak: 37800},
		},
		{
			// Single turn: growth*(Turns-1) terms vanish; the quadratic
			// term Turns*(Turns-1)/2 = 0 as well.
			// A: 3 * (1*1000 + 0) = 3000
			// B: 3 * (1000 + 0)   = 3000
			// C: 1000 + 0         = 1000
			name:    "single turn",
			geom:    Geometry{Prefix: 1000, Turns: 1, Action: 100, DOMState: 500},
			workers: 3,
			want:    ArmCost{Workers: 3, ANaive: 3000, BAgent: 3000, CFak: 1000},
		},
		{
			// Single worker, two turns. growth = 10+20 = 30.
			// A: 1 * (2*100 + 30*(2*1/2)) = 200 + 30 = 230
			// B: 1 * (100 + 30*1) = 130
			// C: 100 + 1*30*1 = 130
			name:    "single worker two turns",
			geom:    Geometry{Prefix: 100, Turns: 2, Action: 10, DOMState: 20},
			workers: 1,
			want:    ArmCost{Workers: 1, ANaive: 230, BAgent: 130, CFak: 130},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.geom.ComputeArms(tt.workers)
			if got != tt.want {
				t.Errorf("ComputeArms(%d) = %+v, want %+v", tt.workers, got, tt.want)
			}
		})
	}
}

// TestGeometryComputeArmsOrdering checks the invariant the arms exist to
// demonstrate: for a realistic multi-turn, multi-worker geometry, the fak
// fused arm (C) does strictly less prefill work than per-agent KV (B),
// which in turn does less than naive re-prefill (A).
func TestGeometryComputeArmsOrdering(t *testing.T) {
	g := Geometry{Prefix: 3400, Turns: 12, Action: 150, DOMState: 2000}
	c := g.ComputeArms(8)
	if !(c.CFak < c.BAgent && c.BAgent < c.ANaive) {
		t.Errorf("expected CFak < BAgent < ANaive, got A=%d B=%d C=%d", c.ANaive, c.BAgent, c.CFak)
	}
}
