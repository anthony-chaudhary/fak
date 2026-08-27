//go:build darwin && arm64 && cgo

package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestQwen35HybridQ4KStepBatchActiveMatchesSerial(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	cfg.ModelType = "qwen3_5_text"
	cfg.QKNorm = true
	cfg.QKNormEps = 3e-5
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	t.Cleanup(func() {
		m.releaseMetalQ8Residency()
		releaseMetalQ4KResidency(m)
	})

	const lanes = 5 // four active rows after ragged compaction
	serial := make([]*Session, lanes)
	batched := make([]*Session, lanes)
	prompts := make([][]int, lanes)
	for i := 0; i < lanes; i++ {
		prompts[i] = make([]int, i+2)
		for j := range prompts[i] {
			prompts[i][j] = (17*i + 11*j + 3) % cfg.VocabSize
		}
		serial[i] = qwen35BatchPreparedSession(t, m, prompts[i])
		batched[i] = qwen35BatchPreparedSession(t, m, prompts[i])
		defer serial[i].Close()
		defer batched[i].Close()
	}

	active := []bool{true, false, true, true, true}
	ids := []int{41, 43, 47, 53, 59}
	beforeInactive := qwen35BatchStateSnapshot(t, batched[1])
	want := make([][]float32, lanes)
	for i := range serial {
		if active[i] {
			want[i] = serial[i].Step(ids[i])
		}
	}
	bs := &BatchSession{M: m, Seqs: batched}
	got := bs.StepBatchActive(ids, active)
	if bs.LastStepMACs() == 0 || bs.LastStepSharedPanels() == 0 {
		t.Fatalf("macs=%d shared=%d", bs.LastStepMACs(), bs.LastStepSharedPanels())
	}
	for i := range got {
		if !active[i] {
			if got[i] != nil {
				t.Fatalf("inactive lane %d produced logits", i)
			}
			continue
		}
		assertCosineAtLeast(t, "batch logits", want[i], got[i], Qwen35GDNParityCosineMin)
		assertMaxAbsAtMost(t, "batch logits", want[i], got[i], 2e-3)
		if argmax(want[i]) != argmax(got[i]) {
			t.Fatalf("lane %d argmax=%d want %d", i, argmax(got[i]), argmax(want[i]))
		}
		qwen35BatchCompareSession(t, i, serial[i], batched[i])
	}
	if after := qwen35BatchStateSnapshot(t, batched[1]); !reflect.DeepEqual(beforeInactive, after) {
		t.Fatal("inactive lane state mutated")
	}
}

func qwen35BatchPreparedSession(t *testing.T, m *Model, prompt []int) *Session {
	t.Helper()
	s := m.NewSession()
	s.Q4K, s.MetalQ4K = true, true
	// Build independent host state first, then seed an equally independent device owner.
	_ = s.Prefill(prompt)
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err == nil {
		t.Fatal("non-fresh session unexpectedly admitted")
	}
	backend := newQwen35MetalGDNSequenceBackend()
	accepted, err := s.initQwen35GDNPreprojectedSequence(backend)
	if err != nil || !accepted {
		t.Fatalf("init owners accepted=%v err=%v", accepted, err)
	}
	cfg := m.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	snaps := make([]qwen35GDNLayerSnapshot, 0)
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		st := s.Cache.linear.layer(cfg, l)
		conv := make([]float32, 0, (cfg.LinearConvKernelDim-1)*convDim)
		for _, row := range st.conv {
			conv = append(conv, row...)
		}
		rec := make([]float32, 0, nV*kHd*vHd)
		for _, row := range st.recurrent {
			rec = append(rec, row...)
		}
		snaps = append(snaps, qwen35GDNLayerSnapshot{layer: l, conv: conv, recurrent: rec})
	}
	if ok, err := s.promoteQwen35MetalGDNDecode(snaps); err != nil || !ok {
		t.Fatalf("promote owners ok=%v err=%v", ok, err)
	}
	return s
}

type qwen35BatchSnapshot struct {
	pos             int
	k, kr, v        [][]float32
	conv, recurrent [][]float32
}

func qwen35BatchStateSnapshot(t *testing.T, s *Session) qwen35BatchSnapshot {
	t.Helper()
	x := qwen35BatchSnapshot{pos: s.Cache.Len(), k: cloneQwen35Matrix(s.Cache.K), kr: cloneQwen35Matrix(s.Cache.Kraw), v: cloneQwen35Matrix(s.Cache.V)}
	if s.qwen35HAL == nil {
		return x
	}
	if snap, ok := s.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter); ok {
		for _, h := range s.qwen35HAL.sequenceLayers {
			if !h.valid() {
				continue
			}
			c, r, e := snap.SnapshotQwen35GDNAuxState(h)
			if e != nil {
				t.Fatal(e)
			}
			x.conv = append(x.conv, append([]float32(nil), c...))
			x.recurrent = append(x.recurrent, append([]float32(nil), r...))
		}
	}
	return x
}
func cloneQwen35Matrix(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i := range x {
		out[i] = append([]float32(nil), x[i]...)
	}
	return out
}
func qwen35BatchCompareSession(t *testing.T, lane int, want, got *Session) {
	t.Helper()
	if want.Cache.Len() != got.Cache.Len() {
		t.Fatalf("lane %d pos=%d want %d", lane, got.Cache.Len(), want.Cache.Len())
	}
	for l := 0; l < want.M.Cfg.NumLayers; l++ {
		if want.M.Cfg.isLinearAttnLayer(l) {
			continue
		}
		assertCosineAtLeast(t, "K", want.Cache.K[l], got.Cache.K[l], .99999)
		assertMaxAbsAtMost(t, "K", want.Cache.K[l], got.Cache.K[l], 2e-4)
		assertCosineAtLeast(t, "Kraw", want.Cache.Kraw[l], got.Cache.Kraw[l], .99999)
		assertMaxAbsAtMost(t, "Kraw", want.Cache.Kraw[l], got.Cache.Kraw[l], 2e-4)
		assertCosineAtLeast(t, "V", want.Cache.V[l], got.Cache.V[l], .99999)
		assertMaxAbsAtMost(t, "V", want.Cache.V[l], got.Cache.V[l], 2e-4)
	}
	cfg := want.M.Cfg
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		host := want.Cache.linear.layer(cfg, l)
		snap := got.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
		c, r, e := snap.SnapshotQwen35GDNAuxState(got.qwen35HAL.sequenceLayers[l])
		if e != nil {
			t.Fatal(e)
		}
		var wc, wr []float32
		for _, x := range host.conv {
			wc = append(wc, x...)
		}
		for _, x := range host.recurrent {
			wr = append(wr, x...)
		}
		assertCosineAtLeast(t, "conv", wc, c, Qwen35GDNParityCosineMin)
		assertMaxAbsAtMost(t, "conv", wc, c, 2e-4)
		assertCosineAtLeast(t, "recurrent", wr, r, Qwen35GDNParityCosineMin)
		assertMaxAbsAtMost(t, "recurrent", wr, r, 2e-4)
	}
}
