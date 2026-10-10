package model

import (
	"math"
	"reflect"
	"testing"
)

func v41ReaderWidthFixture(t *testing.T) *Model {
	t.Helper()
	m := v41CompressorTestFixture(t)
	m.Cfg.DeepSeekV41.CompressRatios = []int{2, 2}
	m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{0}
	m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
	m.Cfg.IndexTopK = 0
	if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full {
		t.Fatalf("fixture geometry=(%v,%v), want full", full, err)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("fixture admission: %v", err)
	}
	plan, err := m.v41AttentionPlan(1)
	if err != nil || plan.Role != V41AttentionRoleReader ||
		plan.Ratio != 2 || plan.KVSourceLayer != 0 || plan.TopKWidth != 0 {
		t.Fatalf("no-index ratio-2 reader plan=%+v err=%v", plan, err)
	}
	return m
}

// The source's span remains authoritative when the reader's declared ratio
// differs. This is a pure plan invariant, not a runnable ratio-one assembly or
// a claim that a mixed-ratio source/reader schedule is a published checkpoint.
// fak-test:runtime fast est=5ms lane=default
func TestV41ReaderSourceWidthPlan(t *testing.T) {
	cfg := Config{NumLayers: 2, DeepSeekV41: &DeepSeekV41Config{
		CompressRatios: []int{2, 1}, KVSourceLayerIDs: []int{0}, CandidateSourceLayerID: -1,
	}}
	plan, err := v41AttentionPlanFor(cfg, 1, v41AttentionRoles(cfg))
	if err != nil || plan.Role != V41AttentionRoleReader || plan.Ratio != 1 || plan.KVSourceLayer != 0 || plan.kvGroupSize(cfg) != 2 {
		t.Fatalf("reader plan=%+v err=%v, want source span 2 despite reader ratio 1", plan, err)
	}
}

func v41ReaderWidthFinite(t *testing.T, values []float32) {
	t.Helper()
	if len(values) == 0 {
		t.Fatal("empty comparison")
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatal("non-finite comparison")
		}
	}
}

func v41ReaderWidthClose(t *testing.T, got, want []float32, label string) {
	t.Helper()
	v41ReaderWidthFinite(t, got)
	v41ReaderWidthFinite(t, want)
	if len(got) != len(want) {
		t.Fatalf("%s widths=%d/%d", label, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i])-float64(want[i])) > 1e-6 {
			t.Fatalf("%s[%d]=%g, want %g", label, i, got[i], want[i])
		}
	}
}

// Public Prefill seeds the actual source/reader state. An explicit recompute
// reads all position logits from the same production forward, and its last row
// must match the public result. This is continuation parity, not a scalar oracle.
func v41ReaderWidthPosition(t *testing.T, m *Model, ids []int) ([]float32, [][]float32) {
	t.Helper()
	s := m.NewSession()
	t.Cleanup(s.Close)
	public := s.Prefill(ids)
	act, err := m.forwardV41(nil, s.v41State())
	if err != nil || act == nil || len(act.Logits) != len(ids) {
		t.Fatalf("production full-history recompute: %v", err)
	}
	v41ReaderWidthClose(t, lastLogits(act), public, "public prefill/recompute")
	for _, row := range act.Logits {
		v41ReaderWidthFinite(t, row)
	}
	if s.v41Forward.attn == nil {
		t.Fatal("production Prefill lost the source registry")
	}
	rows, ok := s.v41Forward.attn.KVSourceRows(0)
	if !ok || len(rows) != len(ids)/2 {
		t.Fatalf("source rows present=%v count=%d", ok, len(rows))
	}
	for _, row := range rows {
		if len(row) != m.Cfg.HeadDim {
			t.Fatal("source row width differs")
		}
		v41ReaderWidthFinite(t, row)
	}
	return append([]float32(nil), act.Logits[1]...), rows
}

// No parallel execution: existing route observers are package-global.
// Estimated runtime is provisional until executable qualification.
// fak-test:runtime medium est=10s lane=default
func TestV41ReaderSourceWidthCausality(t *testing.T) {
	m := v41ReaderWidthFixture(t)
	a := []int{1, 2, 3, 4}
	b := []int{1, 2, 5, 6}
	positionA, rowsA := v41ReaderWidthPosition(t, m, a)
	positionB, rowsB := v41ReaderWidthPosition(t, m, b)
	prefixPosition, prefixRows := v41ReaderWidthPosition(t, m, a[:2])
	v41ReaderWidthClose(t, rowsA[0], rowsB[0], "unchanged source group")
	v41ReaderWidthClose(t, rowsA[0], prefixRows[0], "cold prefix source group")
	futureDelta, magnitude := float64(0), float64(0)
	for i := range rowsA[1] {
		futureDelta = math.Max(futureDelta, math.Abs(float64(rowsA[1][i])-float64(rowsB[1][i])))
	}
	for _, rows := range [][][]float32{rowsA, rowsB} {
		for _, row := range rows {
			for _, value := range row {
				magnitude = math.Max(magnitude, math.Abs(float64(value)))
			}
		}
	}
	if futureDelta <= 1e-4 || magnitude <= 1e-4 {
		t.Fatal("changed tail did not produce a distinct, nonzero future source payload")
	}
	v41ReaderWidthClose(t, positionA, positionB, "position1 ignores changed future group")
	v41ReaderWidthClose(t, positionA, prefixPosition, "position1 matches cold two-token prefix")

	s := m.NewSession()
	t.Cleanup(s.Close)
	v41ReaderWidthFinite(t, s.Prefill(a[:2]))
	if !s.v41IncrementalEligible() {
		t.Fatal("production seed is not incrementally eligible")
	}
	registry := s.v41Forward.attn
	read, restore := v41SessionIncrementalCalls(t)
	defer restore()
	history := append([]int(nil), a[:2]...)
	for i, token := range a[2:] {
		got := s.Step(token)
		history = append(history, token)
		if read() != i+1 || s.v41Forward.attn != registry ||
			!reflect.DeepEqual(s.v41Forward.history, history) {
			t.Fatal("Step lost incremental route, retained registry, or history")
		}
		cold := m.NewSession()
		t.Cleanup(cold.Close)
		want := cold.Prefill(history)
		v41ReaderWidthClose(t, got, want, "incremental/cold continuation")
		rows, ok := s.v41Forward.attn.KVSourceRows(0)
		coldRows, coldOK := cold.v41Forward.attn.KVSourceRows(0)
		if !ok || !coldOK || len(rows) != len(history)/2 || len(rows) != len(coldRows) {
			t.Fatal("incremental source group count differs from cold")
		}
		for g := range rows {
			v41ReaderWidthClose(t, rows[g], coldRows[g], "incremental/cold source payload")
		}
		partial := s.v41Forward.layerState(0)
		if len(partial.partialInputs) != len(history)%2 {
			t.Fatal("source partial group differs from consumed history")
		}
	}
}
