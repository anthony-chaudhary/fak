package model

import "testing"

func TestQuantizeBatchPanelAccelF32PanelContract(t *testing.T) {
	const P, in = 3, 64
	X := mkVec(P*in, 123)
	qp := quantizeBatchPanel(X, P, in)
	if !qgemmAccelDefault() {
		if len(qp.f32) != 0 {
			t.Fatalf("default build retained qp.f32 len = %d, want 0", len(qp.f32))
		}
		return
	}
	if len(qp.f32) != len(X) {
		t.Fatalf("qp.f32 len = %d, want %d", len(qp.f32), len(X))
	}
	if &qp.f32[0] != &X[0] {
		t.Fatal("qp.f32 does not reference the original activation panel")
	}
	X[5] = 99
	if qp.f32[5] != 99 {
		t.Fatal("qp.f32 is not the live activation panel")
	}
}
