//go:build cuda && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// hal_prompt_panel_cuda_test.go is the device witness for the CUDA prompt panel: on a Q8_0
// synthetic model with q/k/v biases, panel prefill leaves logits and every layer's K, F32
// pre-RoPE Kraw and V bitwise equal to the per-token HAL loop for any prompt length and any
// chunking, and panel scratch never reaches the decode allocator pool.

func promptPanelCUDA(t *testing.T) compute.Backend {
	t.Helper()
	be := compute.Pick("cuda")
	if be.Name() != "cuda" {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	return be
}

func promptPanelCUDAModel() *Model {
	cfg := Config{
		HiddenSize: 128, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 32,
		IntermediateSize: 256, VocabSize: 96, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, AttentionBias: true,
	}
	extra := map[string][]int{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"self_attn.q_proj.bias"] = []int{cfg.NumHeads * cfg.HeadDim}
		extra[p+"self_attn.k_proj.bias"] = []int{cfg.NumKVHeads * cfg.HeadDim}
		extra[p+"self_attn.v_proj.bias"] = []int{cfg.NumKVHeads * cfg.HeadDim}
	}
	m := newSyntheticExtra(cfg, extra)
	m.Quantize()
	return m
}

func promptPanelCUDASession(t *testing.T, m *Model, be compute.Backend) *Session {
	t.Helper()
	s := m.NewBackendSession(be)
	s.Quant = true
	return s
}

func promptPanelIDs(n, vocab int) []int {
	ids := make([]int, n)
	x := uint32(12345)
	for i := range ids {
		x = x*1664525 + 1013904223
		ids[i] = int(x % uint32(vocab))
	}
	return ids
}

// promptPanelTokenLoop prefills ids one token at a time, the pre-panel HAL loop: no-logit
// steps, then the last token with logits. One-token prefills never open a panel, so the
// reference is token-loop only (asserted by expectPromptPanel).
func promptPanelTokenLoop(s *Session, ids []int) []float32 {
	for _, id := range ids[:len(ids)-1] {
		s.PrefillNoLogits([]int{id})
	}
	return append([]float32(nil), s.Prefill(ids[len(ids)-1:])...)
}

type promptPanelCounterBackend interface {
	CUDADebugPromptPanelCounters() compute.CUDAPromptPanelCounters
}

// expectPromptPanel runs f and requires it to open exactly scopes panel scopes issuing
// gemms per-row panel GEMMs, with no panel buffer left held when the last scope closed.
func expectPromptPanel(t *testing.T, be compute.Backend, label string, scopes, gemms uint64, f func()) {
	t.Helper()
	counters, ok := be.(promptPanelCounterBackend)
	if !ok {
		t.Fatal("cuda backend does not expose CUDADebugPromptPanelCounters")
	}
	before := counters.CUDADebugPromptPanelCounters()
	f()
	after := counters.CUDADebugPromptPanelCounters()
	if got := after.Begins - before.Begins; got != scopes {
		t.Fatalf("%s: %d panel scopes, want %d", label, got, scopes)
	}
	if got := after.MatMuls - before.MatMuls; got != gemms {
		t.Fatalf("%s: %d panel GEMMs, want %d", label, got, gemms)
	}
	if scopes > 0 && after.LastReleaseLive != 0 {
		t.Fatalf("%s: %d panel buffers still held at the panel end", label, after.LastReleaseLive)
	}
}

// promptPanelGEMMs is the panel GEMM count for a P-token prompt in chunks of chunk rows.
func promptPanelGEMMs(m *Model, P, chunk int) uint64 {
	return uint64(len(promptPanelProjections) * m.Cfg.NumLayers * ((P + chunk - 1) / chunk))
}

// TestCUDAPromptPanelBackendAdmitsQ8 pins the seam the panel tests rely on: the registered CUDA
// backend is a promptPanelBackend whose only per-row exact weight dtype is Q8_0.
func TestCUDAPromptPanelBackendAdmitsQ8(t *testing.T) {
	be := promptPanelCUDA(t)
	panel, ok := be.(promptPanelBackend)
	if !ok {
		t.Fatal("cuda backend does not implement promptPanelBackend")
	}
	if !panel.PromptPanelRowExact(compute.Q8_0) {
		t.Fatal("cuda PromptPanelRowExact(Q8_0) = false")
	}
	for _, dt := range []compute.Dtype{compute.F32, compute.F16, compute.Q4_K, compute.Q2_K} {
		if panel.PromptPanelRowExact(dt) {
			t.Fatalf("cuda PromptPanelRowExact(%s) = true, want only Q8_0", dt)
		}
	}
}

// TestCUDAPromptPanelBitExactVsTokenLoop: P=1 is token-loop only (no panel scope); every
// P>=2 must run as panels and match the token loop bitwise.
func TestCUDAPromptPanelBitExactVsTokenLoop(t *testing.T) {
	be := promptPanelCUDA(t)
	// The widest chunk: P=600 runs as 512 + 88 rows, every other length as one panel.
	setPromptPanelChunk(t, promptPanelMaxTokens)
	m := promptPanelCUDAModel()
	for _, P := range []int{1, 127, 128, 130, 256, 600} {
		ids := promptPanelIDs(P, m.Cfg.VocabSize)
		ref := promptPanelCUDASession(t, m, be)
		var want, got []float32
		expectPromptPanel(t, be, "token loop", 0, 0, func() { want = promptPanelTokenLoop(ref, ids) })
		s := promptPanelCUDASession(t, m, be)
		scopes, gemms := uint64(1), promptPanelGEMMs(m, P, promptPanelMaxTokens)
		if P == 1 {
			scopes, gemms = 0, 0
		}
		expectPromptPanel(t, be, "panel prefill", scopes, gemms, func() { got = append([]float32(nil), s.Prefill(ids)...) })
		assertSameBits(t, "panel logits", got, want)
		assertSameHALKV(t, "panel", s, ref)
		assertSameBits(t, "decode after panel", append([]float32(nil), s.Step(5)...), append([]float32(nil), ref.Step(5)...))
		s.Close()
		ref.Close()
	}
}

func TestCUDAPromptPanelChunkingKeepsBits(t *testing.T) {
	be := promptPanelCUDA(t)
	m := promptPanelCUDAModel()
	ids := promptPanelIDs(256, m.Cfg.VocabSize)
	ref := promptPanelCUDASession(t, m, be)
	defer ref.Close()
	var want []float32
	expectPromptPanel(t, be, "token loop", 0, 0, func() { want = promptPanelTokenLoop(ref, ids) })
	for _, chunk := range []int{256, 128, 37, 1, 0} { // 0: the size derived from the transient plan
		setPromptPanelChunk(t, chunk)
		s := promptPanelCUDASession(t, m, be)
		rows := s.promptPanelChunkTokens()
		var got []float32
		expectPromptPanel(t, be, "chunked panel", 1, promptPanelGEMMs(m, len(ids), rows), func() {
			got = append([]float32(nil), s.Prefill(ids)...)
		})
		assertSameBits(t, "chunked panel logits", got, want)
		assertSameHALKV(t, "chunked panel", s, ref)
		s.Close()
	}
	// A prefix-cache hit followed by a suffix prefill is just another chunk boundary.
	setPromptPanelChunk(t, 512)
	s := promptPanelCUDASession(t, m, be)
	defer s.Close()
	var got []float32
	expectPromptPanel(t, be, "prefix plus suffix", 2, promptPanelGEMMs(m, 128, 512)*2, func() {
		s.PrefillNoLogits(ids[:128])
		got = append([]float32(nil), s.Prefill(ids[128:])...)
	})
	assertSameBits(t, "prefix plus suffix logits", got, want)
	assertSameHALKV(t, "prefix plus suffix", s, ref)
}

// TestCUDAPromptPanelTailChunkStaysInChunkBudget: a prompt of two full chunks plus a shorter
// tail must peak at the panel bytes of one full chunk; the tail may not allocate its own
// buffers beside the cached full-chunk ones.
func TestCUDAPromptPanelTailChunkStaysInChunkBudget(t *testing.T) {
	be := promptPanelCUDA(t)
	stats, ok := be.(interface{ CUDADebugPoolStats() compute.CUDAPoolStats })
	if !ok {
		t.Fatal("cuda backend does not expose CUDADebugPoolStats")
	}
	const chunk = 64
	setPromptPanelChunk(t, chunk)
	m := promptPanelCUDAModel()
	peak := func(P int) int64 {
		s := promptPanelCUDASession(t, m, be)
		defer s.Close()
		s.Prefill(promptPanelIDs(P, m.Cfg.VocabSize))
		return stats.CUDADebugPoolStats().PanelPeakBytes
	}
	full := peak(chunk)
	if full <= 0 {
		t.Fatalf("one-chunk prefill reported panel peak %d", full)
	}
	if got := peak(2*chunk + 20); got != full {
		t.Fatalf("two chunks plus a 20-row tail peaked at %d panel bytes, want the one-chunk peak %d", got, full)
	}
}

// TestCUDAPromptPanelGraphPoolBounded runs panel prefills of different lengths, each followed by
// three decode steps under CUDA graph capture (whose prewarm deepens every pooled bucket). The
// decode pool must come out identical whatever the prompt length: no panel-sized bucket enters
// it, and the panel free list is empty once each prefill ends.
func TestCUDAPromptPanelGraphPoolBounded(t *testing.T) {
	be := promptPanelCUDA(t)
	graph, ok := be.(interface{ CUDADebugSetGraphEnabled(bool) bool })
	if !ok {
		t.Fatal("cuda backend does not expose CUDADebugSetGraphEnabled")
	}
	// The pool growth this guards comes from graph prewarm, so capture is forced on.
	restore := graph.CUDADebugSetGraphEnabled(true)
	t.Cleanup(func() { graph.CUDADebugSetGraphEnabled(restore) })
	stats, ok := be.(interface{ CUDADebugPoolStats() compute.CUDAPoolStats })
	if !ok {
		t.Fatal("cuda backend does not expose CUDADebugPoolStats")
	}
	// Panel sizes must vary with P for this to bite; the derived chunk on this tiny model is
	// one row, the size decode already pools.
	setPromptPanelChunk(t, promptPanelMaxTokens)
	m := promptPanelCUDAModel()
	run := func(P int) compute.CUDAPoolStats {
		s := promptPanelCUDASession(t, m, be)
		s.Prefill(promptPanelIDs(P, m.Cfg.VocabSize))
		if st := stats.CUDADebugPoolStats(); st.PanelCachedBytes != 0 || st.PanelLive != 0 {
			t.Fatalf("P=%d: panel free list holds %d bytes and %d live buffers after prefill", P, st.PanelCachedBytes, st.PanelLive)
		}
		for _, id := range []int{5, 9, 13} {
			s.Step(id)
		}
		s.Close()
		st := stats.CUDADebugPoolStats()
		st.PanelPeakBytes = 0 // scales with P by design; the decode pool is what must not
		return st
	}
	run(130) // warm every decode and graph size class; the second run settles prewarm spares
	run(130)
	base := run(130)
	for _, P := range []int{300, 257, 600} {
		got := run(P)
		if got != base {
			t.Fatalf("P=%d decode pool %+v, want the P=130 pool %+v (panel sizes leaked into it)", P, got, base)
		}
	}
}
