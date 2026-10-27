//go:build !(darwin && arm64 && cgo)

package model

import (
	"bytes"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// fak-test:runtime fast est=100ms
func TestQ4KPrefillGroupSharesActivationWithoutChangingTargets(t *testing.T) {
	if !q4kExtractOnceGemmEnabled() {
		t.Skip("extract-once prefill unavailable on this CPU tier")
	}
	oldForce, oldGroup, oldMode, oldHook, oldProfile := q4kSDOTForce, qgemmGroup, qgemmMode, q8PanelBuildHook, qprofOn
	q4kSDOTForce, qgemmGroup, qprofOn = 1, true, false
	q8PanelBuildHook = nil
	t.Cleanup(func() {
		q4kSDOTForce, qgemmGroup, qgemmMode, q8PanelBuildHook, qprofOn = oldForce, oldGroup, oldMode, oldHook, oldProfile
	})
	const in, P = 768, 19
	rng := rand.New(rand.NewSource(14048))
	m := &Model{q4kw: make(map[string]*q4kTensor)}
	for i, name := range []string{"gate", "up"} {
		rows := 9 + i*8
		raw := make([]byte, rows*(in/qkK)*q4kBlockBytes)
		for b := 0; b < len(raw); b += q4kBlockBytes {
			randQ4KBlockBounded(rng, raw[b:b+q4kBlockBytes], 8, 12)
		}
		m.q4kw[name] = quantizeQ4KFromRaw(raw, rows, in)
	}
	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32(rng.NormFloat64())
	}
	clear(X[5*in : 6*in])
	beforeX := slices.Clone(X)
	beforeGate, beforeUp := bytes.Clone(m.q4kw["gate"].raw), bytes.Clone(m.q4kw["up"].raw)
	names := []string{"gate", "q8-minority", "up"}
	s := &Session{M: m}
	for _, mode := range []string{qgemmModeTile, qgemmModeLegacy} {
		qgemmMode = mode
		q8PanelBuildHook = nil
		wantGate := s.q4kGemmDispatch("gate", m.q4kw["gate"], X, P)
		wantUp := s.q4kGemmDispatch("up", m.q4kw["up"], X, P)
		builds := 0
		q8PanelBuildHook = func(rows, width int) {
			builds++
			if rows != P || width != in {
				t.Fatalf("unexpected panel %dx%d", rows, width)
			}
		}
		s.ResetQ4KPrefillGEMMObservation()
		got := s.q4kGemmGroupDispatch(names, X, P)
		if len(got) != len(names) || got[1] != nil || builds != 1 {
			t.Fatalf("%s: members=%d builds=%d, want three slots/one panel", mode, len(got), builds)
		}
		for j, want := range [][]float32{wantGate, nil, wantUp} {
			if len(got[j]) != len(want) {
				t.Fatalf("%s: member%d length=%d want%d", mode, j, len(got[j]), len(want))
			}
			for i := range want {
				if math.Float32bits(got[j][i]) != math.Float32bits(want[i]) {
					t.Fatalf("%s: member%d value%d=%g want%g", mode, j, i, got[j][i], want[i])
				}
			}
		}
		if s.Q4KPrefillGEMMObservation() != Q4KPrefillGEMMCPU {
			t.Fatal("group did not retain the CPU route observation")
		}
		// Replacing one result cell cannot alias another target or caller input.
		upFirst := got[2][0]
		got[0][0]++
		if math.Float32bits(got[2][0]) != math.Float32bits(upFirst) {
			t.Fatal("group results share mutable cells")
		}
	}
	if !slices.Equal(X, beforeX) || !bytes.Equal(m.q4kw["gate"].raw, beforeGate) || !bytes.Equal(m.q4kw["up"].raw, beforeUp) {
		t.Fatal("group mutated activation or resident weight payload")
	}
	q8PanelBuildHook = func(int, int) { t.Fatal("declined group quantized an activation") }
	qprofOn = true
	if s.q4kGemmGroupDispatch(names, X, P) != nil {
		t.Fatal("profiling must retain per-target MAC accounting")
	}
	qprofOn = false
	if s.q4kGemmGroupDispatch(names[:1], X, P) != nil || s.q4kGemmGroupDispatch(names, X[:in], 1) != nil {
		t.Fatal("singleton group/token must retain the per-weight route")
	}
	m.prism = &prismHadamardState{}
	if s.q4kGemmGroupDispatch(names, X, P) != nil {
		t.Fatal("Prism group must retain per-target rotation")
	}
	m.prism = nil
	q4kSDOTForce = -1
	if s.q4kGemmGroupDispatch(names, X, P) != nil {
		t.Fatal("f32 gate must retain the ordinary reduction")
	}
	q4kSDOTForce, qgemmGroup = 1, false
	if s.q4kGemmGroupDispatch(names, X, P) != nil {
		t.Fatal("group opt-out ignored")
	}
}
