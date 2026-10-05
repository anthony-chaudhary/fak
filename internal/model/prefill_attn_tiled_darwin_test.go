//go:build darwin && arm64 && cgo

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// tiledPrefillAttnMaxAbs bounds device-vs-host attention output drift. Both routes are f32; they
// differ only in summation order (the device uses 8x8 simdgroup MMA tiles and an online softmax,
// the host a per-(row, head) dot and a two-pass softmax). Outputs are convex combinations of
// unit-normal v rows, so reordering error is ~sqrt(kvLen)*2^-24 per accumulation, ~1e-6 at kvLen 4096;
// 5e-5 is a wide margin that a wrong mask, head mapping or normalization (O(1e-2..1)) cannot meet.
const tiledPrefillAttnMaxAbs = 5e-5

type tiledAttnPanel struct {
	P, base, nH, nKV, hd, W int
	Q, K, V                 []float32
	scale                   float32
}

func newTiledAttnPanel(P, base, nH, nKV, hd, W int, seed int64) tiledAttnPanel {
	kvLen := base + P
	return tiledAttnPanel{
		P: P, base: base, nH: nH, nKV: nKV, hd: hd, W: W,
		Q:     randomVecF(P*nH*hd, seed),
		K:     randomVecF(kvLen*nKV*hd, seed+1),
		V:     randomVecF(kvLen*nKV*hd, seed+2),
		scale: float32(1 / math.Sqrt(float64(hd))),
	}
}

func (p tiledAttnPanel) host(attnCap float32, obs AttnObserver) []float32 {
	out := make([]float32, p.P*p.nH*p.hd)
	attnPrefillInto(out, p.Q, p.K, p.V, p.P, p.base, p.nH, p.hd, p.nKV*p.hd, p.nH/p.nKV, p.W, 0, p.scale, attnCap, fdot, obs)
	return out
}

func (p tiledAttnPanel) dispatch(s *Session, device bool, attnCap float32, obs AttnObserver) []float32 {
	out := make([]float32, p.P*p.nH*p.hd)
	s.attnPrefillDispatch(device, out, p.Q, p.K, p.V, p.P, p.base, p.nH, p.hd, p.nKV*p.hd, p.nH/p.nKV, p.W, 0, p.scale, attnCap, fdot, obs)
	return out
}

func tiledAttnBitIdentical(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func tiledAttnMaxAbs(t *testing.T, want, got []float32) float64 {
	t.Helper()
	m := 0.0
	for i := range want {
		g := float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("non-finite device output[%d]=%g", i, g)
		}
		m = math.Max(m, math.Abs(g-float64(want[i])))
	}
	return m
}

// TestTiledPrefillAttnParity is the fak#13695 parity witness the issue's definition of done names:
// the device route of attnPrefillDispatch against the attnPrefillInto host oracle on a 4k-token
// panel at the Qwen2.5-7B attention geometry (28 query heads, 4 KV heads, head_dim 128), plus a
// later chunk over a cached prefix and a sliding-window layer. It also pins that a successful
// device route records no fallback, and that every gate that must keep the host route (obs,
// soft-cap, P < 8, device=false, unsupported head_dim) is bit-identical to attnPrefillInto.
func TestTiledPrefillAttnParity(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	const nH, nKV, hd = 28, 4, 128
	if !metalgemm.PrefillAttentionSupported(hd, nH, nKV) {
		t.Fatal("tiled prefill attention unsupported for the 7B geometry on a Metal host")
	}

	for _, tc := range []struct {
		name       string
		P, base, W int
	}{
		{"fresh_4096", 4096, 0, -1},
		{"chunk_512_after_3584", 512, 3584, -1},
		{"window_1024_P777", 777, 300, 1024},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newTiledAttnPanel(tc.P, tc.base, nH, nKV, hd, tc.W, 13695+int64(tc.P))
			s := &Session{PhaseProfiler: NewPhaseProfiler()}
			got := p.dispatch(s, true, 0, nil)
			want := p.host(0, nil)
			maxAbs := tiledAttnMaxAbs(t, want, got)
			t.Logf("%s: device vs attnPrefillInto maxAbs=%.3e", tc.name, maxAbs)
			if maxAbs > tiledPrefillAttnMaxAbs {
				t.Fatalf("%s: device vs attnPrefillInto maxAbs=%.3e > %g", tc.name, maxAbs, tiledPrefillAttnMaxAbs)
			}
			// The admitted route really ran on the device: a host recomputation would be bit-identical.
			if tiledAttnBitIdentical(got, want) {
				t.Fatalf("%s: admitted device route is bit-identical to the host loop; the device kernel did not run", tc.name)
			}
			receipt, err := s.PhaseProfiler.MetalFallbackReceipt()
			if err != nil {
				t.Fatal(err)
			}
			if len(receipt.Events) != 0 {
				t.Fatalf("%s: successful device route recorded fallbacks %+v", tc.name, receipt.Events)
			}
		})
	}

	// Gating: every case below must take the host loop, so its output is bit-identical to
	// attnPrefillInto (the admitted control above proves the device route is not).
	panel := newTiledAttnPanel(64, 16, nH, nKV, hd, -1, 136951)
	s := &Session{PhaseProfiler: NewPhaseProfiler()}
	admitted := panel.dispatch(s, true, 0, nil)
	if tiledAttnBitIdentical(admitted, panel.host(0, nil)) {
		t.Fatal("gating control: admitted P=64 device route is bit-identical to the host loop")
	}
	small := newTiledAttnPanel(7, 16, nH, nKV, hd, -1, 136952)
	hd96 := newTiledAttnPanel(64, 16, 8, 2, 96, -1, 136953)
	if metalgemm.PrefillAttentionSupported(96, 8, 2) {
		t.Fatal("head_dim 96 reported supported")
	}
	const softCap = 50
	for _, gc := range []struct {
		name    string
		p       tiledAttnPanel
		device  bool
		attnCap float32
		obs     AttnObserver
	}{
		{"device_false", panel, false, 0, nil},
		{"observer_attached", panel, true, 0, AttnObserver(func(int, int, int, []int, []float32) {})},
		{"softcap_set", panel, true, softCap, nil},
		{"P_below_8", small, true, 0, nil},
		{"unsupported_hd96", hd96, true, 0, nil},
	} {
		got := gc.p.dispatch(s, gc.device, gc.attnCap, gc.obs)
		want := gc.p.host(gc.attnCap, gc.obs)
		if !tiledAttnBitIdentical(got, want) {
			t.Errorf("%s: dispatch is not bit-identical to attnPrefillInto; the device route was taken", gc.name)
		}
	}
	receipt, err := s.PhaseProfiler.MetalFallbackReceipt()
	if err != nil {
		t.Fatal(err)
	}
	if len(receipt.Events) != 0 {
		t.Fatalf("gated host routes recorded fallbacks %+v; a gate is not a device failure", receipt.Events)
	}
}
