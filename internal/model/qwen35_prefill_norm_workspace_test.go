package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_norm_workspace*
func TestQwen35PrefillNormWorkspaceAdmission(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	s := &Session{M: &Model{Cfg: cfg}}
	w := s.newQwen35PrefillNormWorkspace(16)
	if (w != nil) != qwen35PrefillNormWorkspaceQualification {
		t.Fatal("default/qualification selection changed")
	}
	if w != nil && (w.rows != 16 || w.width != cfg.HiddenSize || w.normalized != nil) {
		t.Fatal("workspace allocated early or changed the invocation extent")
	}
	for _, edit := range []func(*Session){
		func(s *Session) { s.Metal = true },
		func(s *Session) { s.MetalQ4K = true },
		func(s *Session) { s.M.Cfg.LayerTypes = nil; s.M.Cfg.FullAttentionInterval = 0 },
		func(s *Session) { s.M.Cfg.HiddenSize = 0 },
	} {
		x := &Session{M: &Model{Cfg: cfg}}
		edit(x)
		if x.newQwen35PrefillNormWorkspace(16) != nil {
			t.Fatal("unsupported producer/configuration admitted")
		}
	}
	for _, n := range []int{0, -1, int(^uint(0) >> 1)} {
		if s.newQwen35PrefillNormWorkspace(n) != nil {
			t.Fatal("invalid panel extent admitted")
		}
	}
}

// fak-test:runtime fast est=20ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_norm_workspace*
func TestQwen35PrefillNormWorkspaceOverwrite(t *testing.T) {
	const rows, width = 5, 32
	x, weight := make([]float32, rows*width), make([]float32, width)
	for i := range x {
		x[i] = float32(i%19-9) / 7
	}
	for i := range weight {
		weight[i] = float32(i%7-3) / 9
	}
	w := &qwen35PrefillNormWorkspace{rows: rows, width: width}
	first := w.panel(rows, width)
	for _, cfg := range []Config{{}, {NormGain1p: true}, {LayerNorm: true}, {LayerNorm: true, NormGain1p: true}} {
		got, want := w.panel(rows, width), make([]float32, rows*width)
		if &got[0] != &first[0] || len(got) != rows*width || cap(got) != rows*width {
			t.Fatal("workspace changed backing allocation or exact capacity")
		}
		for i := range got {
			got[i] = float32(math.NaN())
		}
		parFor(rows, 2, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				a, b := row*width, (row+1)*width
				if cfg.NormGain1p || cfg.LayerNorm {
					qwen35PrefillNormIntoDisjoint(got[a:b], x[a:b], weight, 1e-5, cfg)
					qwen35PrefillNormIntoDisjoint(want[a:b], x[a:b], weight, 1e-5, cfg)
				} else {
					rmsnormInto(got[a:b], x[a:b], weight, 1e-5)
					rmsnormInto(want[a:b], x[a:b], weight, 1e-5)
				}
			}
		})
		assertFloat32BitsEqual(t, "fully overwritten norm panel", want, got)
		for _, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("prior contents survived the joined overwrite")
			}
		}
	}
	other := (&qwen35PrefillNormWorkspace{rows: rows, width: width}).panel(rows, width)
	if &other[0] == &first[0] {
		t.Fatal("two invocations share storage")
	}
	before := append([]float32(nil), first...)
	mismatch := w.panel(rows-1, width)
	if &mismatch[0] == &first[0] {
		t.Fatal("different extent reused the owned panel")
	}
	assertFloat32BitsEqual(t, "mismatched shape did not alter owner", before, first)
}

var qwen35NormWorkspaceTestSink []float32

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_norm_workspace*
func TestQwen35PrefillNormWorkspaceAllocations(t *testing.T) {
	t.Cleanup(func() { qwen35NormWorkspaceTestSink = nil })
	const rows, width, uses = 16, 64, 8
	var legacy *qwen35PrefillNormWorkspace
	baseline := testing.AllocsPerRun(4, func() {
		for i := 0; i < uses; i++ {
			qwen35NormWorkspaceTestSink = legacy.panel(rows, width)
		}
	})
	candidate := testing.AllocsPerRun(4, func() {
		w := &qwen35PrefillNormWorkspace{rows: rows, width: width}
		for i := 0; i < uses; i++ {
			qwen35NormWorkspaceTestSink = w.panel(rows, width)
		}
	})
	if baseline < uses || candidate >= baseline {
		t.Fatalf("invocation panel allocations baseline=%g candidate=%g", baseline, candidate)
	}
	w := &qwen35PrefillNormWorkspace{rows: rows, width: width}
	w.panel(rows, width)
	if got := testing.AllocsPerRun(4, func() {
		qwen35NormWorkspaceTestSink = w.panel(rows, width)
	}); got != 0 {
		t.Fatalf("reused panel allocated: %g", got)
	}
	t.Logf("component allocations for %d sequential borrows: baseline=%g candidate=%g", uses, baseline, candidate)
}

// fak-test:runtime fast est=500ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_norm_workspace*
func TestQwen35PrefillNormWorkspaceWholeHidden(t *testing.T) {
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67}
	for _, q4k := range []bool{false, true} {
		cfg := qwen35HybridTestCfg()
		if q4k {
			cfg = qwen35HybridQ4KTestCfg()
		}
		m := NewSynthetic(cfg)
		m.Quantize()
		if q4k {
			fillQ4KMajority(t, m, cfg)
		}
		fresh, reused := m.NewSession(), m.NewSession()
		fresh.Quant, reused.Quant = !q4k, !q4k
		fresh.Q4K, reused.Q4K = q4k, q4k
		var prior, priorSnapshot []float32
		for _, ids := range [][]int{prompt, prompt[:3]} {
			w := &qwen35PrefillNormWorkspace{rows: len(ids), width: cfg.HiddenSize}
			var want, got []float32
			if q4k {
				want = fresh.prefillQwen35HybridQ4KHiddenWithNormWorkspace(ids, nil)
				got = reused.prefillQwen35HybridQ4KHiddenWithNormWorkspace(ids, w)
			} else {
				want = fresh.prefillQwen35HybridQHiddenWithNormWorkspace(ids, nil)
				got = reused.prefillQwen35HybridQHiddenWithNormWorkspace(ids, w)
			}
			assertFloat32BitsEqual(t, "whole hidden panel parity", want, got)
			assertKVCacheUnchanged(t, "whole prefill state parity", fresh.Cache, reused.Cache)
			if len(w.normalized) != len(ids)*cfg.HiddenSize || cap(w.normalized) != len(w.normalized) {
				t.Fatal("actual whole producer did not use one exact-sized workspace")
			}
			for _, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("nonfinite whole-prefill witness")
				}
			}
			// The returned hidden row is independent of the dead scratch owner.
			copyOfGot := append([]float32(nil), got...)
			for i := range w.normalized {
				w.normalized[i] = float32(math.NaN())
			}
			assertFloat32BitsEqual(t, "returned row owns its values", copyOfGot, got)
			assertKVCacheUnchanged(t, "state does not borrow norm scratch", fresh.Cache, reused.Cache)
			if prior != nil {
				assertFloat32BitsEqual(t, "prior invocation result", priorSnapshot, prior)
			}
			prior, priorSnapshot = got, copyOfGot
		}
	}
}

// fak-test:runtime fast est=100ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_norm_workspace*
func TestQwen35PrefillNormWorkspaceCallbacksStayFresh(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	s := m.NewSession()
	const rows = 3
	x := make([]float32, rows*cfg.HiddenSize)
	for i := range x {
		x[i] = float32(i%23-11) / 13
	}
	var retained, snapshots [][]float32
	qz := func(x []float32, rows, width int, _ ...string) *q8Panel {
		return quantizeBatchPanel(x, rows, width)
	}
	proj := func(name string, _ []float32, q *q8Panel) []float32 {
		return qGemm8(m.q8(name), q)
	}
	group := func(names []string, x []float32, q *q8Panel) [][]float32 {
		retained = append(retained, x)
		snapshots = append(snapshots, append([]float32(nil), x...))
		out := make([][]float32, len(names))
		for i, name := range names {
			out[i] = proj(name, x, q)
		}
		return out
	}
	// This callback seam may retain Xn and Xn2; it must never inherit the
	// ordinary builtin producer's private reuse authority, even in a tagged build.
	s.prefillQwen35HybridQ4KLayerWithRopeRows(3, x, rows, 0, float32(cfg.RMSNormEps), proj, group, qz, false, nil)
	if len(retained) != 2 || &retained[0][0] == &retained[1][0] {
		t.Fatal("callback normalization inputs shared storage")
	}
	for i := range retained {
		assertFloat32BitsEqual(t, "retained callback input", snapshots[i], retained[i])
	}
}
