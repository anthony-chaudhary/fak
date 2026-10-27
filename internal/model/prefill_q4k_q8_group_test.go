//go:build !(darwin && arm64 && cgo)

package model

import (
	"math"
	"slices"
	"testing"
)

func assertHybridQ8GroupBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: cell %d = %g want %g", label, i, got[i], want[i])
		}
	}
}

// fak-test:runtime fast est=100ms
func TestCPUHybridQ8GroupPreservesMembersAndRefusals(t *testing.T) {
	oldMode, oldGroup, oldMaxP, oldProfile := qgemmMode, qgemmGroup, qgemmGroupMaxP, qprofOn
	t.Cleanup(func() { qgemmMode, qgemmGroup, qgemmGroupMaxP, qprofOn = oldMode, oldGroup, oldMaxP, oldProfile })
	qgemmMode, qgemmGroup, qgemmGroupMaxP, qprofOn = qgemmModeTile, true, 1024, false
	const in, P = 256, 19
	m := &Model{q8w: make(map[string]*q8Tensor), q2w: make(map[string]*q2Tensor)}
	for j, name := range []string{"q", "k", "shadow"} {
		out := 7 + j*4
		w := make([]float32, out*in)
		for i := range w {
			w[i] = float32(i%37-18) / 31
		}
		m.q8w[name] = quantizeQ8(w, out, in)
	}
	// Existing resident-format precedence wins over a Q8 shadow.
	m.q2w["shadow"] = &q2Tensor{}
	x := make([]float32, P*in)
	for i := range x {
		x[i] = float32(i%29-14) / 23
	}
	clear(x[3*in : 4*in])
	qp := quantizeBatchPanel(x, P, in)
	beforeQ, beforeD := slices.Clone(qp.q), slices.Clone(qp.d)
	names := []string{"q", "shadow", "missing", "k"}
	s := &Session{M: m}
	wantQ := s.q8GemmDispatch("q", m.q8w["q"], qp)
	wantK := s.q8GemmDispatch("k", m.q8w["k"], qp)
	got := s.q8GemmGroupDispatchDirect(names, qp, P)
	if len(got) != len(names) || got[1] != nil || got[2] != nil {
		t.Fatal("mixed membership changed")
	}
	assertHybridQ8GroupBits(t, "q", got[0], wantQ)
	assertHybridQ8GroupBits(t, "k", got[3], wantK)
	oldK := got[3][0]
	got[0][0]++
	if math.Float32bits(got[3][0]) != math.Float32bits(oldK) {
		t.Fatal("target outputs alias")
	}
	if !slices.Equal(qp.q, beforeQ) || !slices.Equal(qp.d, beforeD) {
		t.Fatal("input panel mutated")
	}
	decline := func(label string) {
		t.Helper()
		if s.q8GemmGroupDispatchDirect(names, qp, P) != nil {
			t.Fatalf("accepted %s", label)
		}
	}
	qprofOn = true
	decline("profiling MAC attribution")
	qprofOn = false
	qgemmGroup = false
	decline("group opt-out")
	qgemmGroup = true
	qgemmMode = qgemmModeLegacy
	decline("legacy mode")
	qgemmMode = qgemmModeTile
	qgemmGroupMaxP = P - 1
	decline("over group cap")
	qgemmGroupMaxP = 1024
	m.prism = &prismHadamardState{}
	decline("per-weight rotation")
	m.prism = nil
	if s.q8GemmGroupDispatchDirect(names[:1], qp, P) != nil || s.q8GemmGroupDispatchDirect(names, qp, 1) != nil {
		t.Fatal("singleton accepted")
	}
	oldQ := qp.q
	qp.q = qp.q[:len(qp.q)-1]
	decline("short panel")
	qp.q = oldQ
	m.q8w["k"], m.q8w["missing"] = m.q8w["missing"], m.q8w["k"]
	m.q8w["missing"].in = in / 2
	decline("mismatched target width")
}

// fak-test:runtime fast est=200ms
func TestCPUHybridQ8GroupedOrdinaryPrefillMatchesSeparate(t *testing.T) {
	oldMode, oldGroup, oldMaxP, oldProfile, oldForce := qgemmMode, qgemmGroup, qgemmGroupMaxP, qprofOn, q4kSDOTForce
	t.Cleanup(func() {
		qgemmMode, qgemmGroup, qgemmGroupMaxP, qprofOn, q4kSDOTForce = oldMode, oldGroup, oldMaxP, oldProfile, oldForce
	})
	qgemmGroupMaxP, qprofOn, q4kSDOTForce = 1024, false, -1
	cfg := Config{HiddenSize: 256, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, AttnSoftcap: 50, RopeTheta: 10000}
	m := NewSynthetic(cfg)
	m.Quantize()
	// Q/K/V and gate/up remain Q8; output/down keep the ordinary resident Q4_K route.
	var resident [][2]any
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		resident = append(resident, [2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize}, [2]any{p + "mlp.down_proj.weight", cfg.HiddenSize})
	}
	fillQ4KW(t, m, resident, 15015)
	for _, mode := range []string{qgemmModeTile, qgemmModeLegacy} {
		qgemmMode = mode
		separate, grouped := m.NewSession(), m.NewSession()
		defer separate.Close()
		defer grouped.Close()
		separate.Q4K, grouped.Q4K = true, true
		for _, prompt := range [][]int{{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37}, {2, 4, 6}} {
			qgemmGroup = false
			want := separate.prefillBatchedQ4K(prompt)
			qgemmGroup = true
			got := grouped.prefillBatchedQ4K(prompt)
			assertHybridQ8GroupBits(t, mode+" hidden", got, want)
			if grouped.Cache.Len() != separate.Cache.Len() {
				t.Fatal("cache position changed")
			}
			for l := 0; l < cfg.NumLayers; l++ {
				assertHybridQ8GroupBits(t, mode+" K", grouped.Cache.K[l], separate.Cache.K[l])
				assertHybridQ8GroupBits(t, mode+" Kraw", grouped.Cache.Kraw[l], separate.Cache.Kraw[l])
				assertHybridQ8GroupBits(t, mode+" V", grouped.Cache.V[l], separate.Cache.V[l])
			}
		}
	}
}
