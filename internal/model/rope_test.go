package model

import (
	"math"
	"testing"
)

func TestRopeRowsShareInvFreqBitExact(t *testing.T) {
	cfg := Config{HeadDim: 8, RopeTheta: 10000}
	for pos := 0; pos < 16; pos++ {
		wantCos, wantSin := ropeRow(cfg, pos)
		gotCos, gotSin := ropeRowFromInv(invFreq(cfg, 0), pos)
		assertFloat32BitsEqual(t, "cos", wantCos, gotCos)
		assertFloat32BitsEqual(t, "sin", wantSin, gotSin)
	}
}

func TestForwardRopeAppliesSharedRotation(t *testing.T) {
	cfg := Config{HeadDim: 8, RopeTheta: 10000}
	rp := newRope(cfg, 8)
	for pos := 0; pos < 8; pos++ {
		got := []float32{0.25, -0.5, 0.75, -1, 1.25, -1.5, 1.75, -2}
		want := append([]float32(nil), got...)
		cos, sin := ropeRow(cfg, pos)
		rp.apply(got, pos)
		applyRopeRow(want, cos, sin)
		assertFloat32BitsEqual(t, "rope", want, got)
	}
}

func TestLayerSpecificRopeTheta(t *testing.T) {
	cfg := Config{
		HeadDim:           8,
		RopeTheta:         10000,
		RopeThetaPerLayer: []float64{10000, 1000000},
	}
	c0, s0 := ropeRowForLayer(cfg, 0, 7)
	c1, s1 := ropeRowForLayer(cfg, 1, 7)
	baseC, baseS := ropeRow(cfg, 7)
	assertFloat32BitsEqual(t, "layer0 cos == base", baseC, c0)
	assertFloat32BitsEqual(t, "layer0 sin == base", baseS, s0)
	if float32BitsEqual(c0, c1) && float32BitsEqual(s0, s1) {
		t.Fatal("layer-specific RoPE theta produced identical rows")
	}
}

func TestPartialRotaryFactorRotatesOnlyPrefix(t *testing.T) {
	cfg := Config{HeadDim: 8, RopeTheta: 10000, PartialRotaryFactor: 0.25}
	inv := invFreq(cfg, 0)
	if len(inv) != 1 {
		t.Fatalf("partial rotary inv len = %d, want 1", len(inv))
	}
	cos, sin := ropeRow(cfg, 3)
	if len(cos) != 1 || len(sin) != 1 {
		t.Fatalf("partial rotary row len = %d/%d, want 1/1", len(cos), len(sin))
	}
	got := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	orig := append([]float32(nil), got...)
	applyRopeRow(got, cos, sin)
	if got[2] != orig[2] || got[3] != orig[3] || got[4] != orig[4] || got[5] != orig[5] || got[6] != orig[6] || got[7] != orig[7] {
		t.Fatalf("partial rotary changed tail: got %v orig %v", got, orig)
	}
	if got[0] == orig[0] && got[1] == orig[1] {
		t.Fatal("partial rotary did not rotate the prefix")
	}
}

func assertFloat32BitsEqual(t *testing.T, name string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s len = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s[%d] = %08x, want %08x (%g vs %g)",
				name, i, math.Float32bits(got[i]), math.Float32bits(want[i]), got[i], want[i])
		}
	}
}
