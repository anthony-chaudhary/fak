package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// promptPanelRecordingBackend wraps cpu-ref with the prompt-panel seam, implemented row by
// row through cpu-ref's own single-position ops, and counts which projection form ran.
type promptPanelRecordingBackend struct {
	compute.Backend
	rowExact                                            bool
	matmuls, panelMatMuls, panelAttention, begins, ends int
	chunks                                              []int
}

func newPromptPanelRecordingBackend() *promptPanelRecordingBackend {
	return &promptPanelRecordingBackend{Backend: compute.Pick("cpu-ref"), rowExact: true}
}

func (b *promptPanelRecordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}

// AddBias broadcasts the bias over panel rows, the device contract the panel relies on.
func (b *promptPanelRecordingBackend) AddBias(dst, bias compute.Tensor) {
	d, _ := b.Host(dst)
	bv, _ := b.Host(bias)
	for i := range d {
		d[i] += bv[i%len(bv)]
	}
}

// PromptPanelRowExact: cpu-ref's F32 BatchedMatMul dots each row exactly as MatMul does.
func (b *promptPanelRecordingBackend) PromptPanelRowExact(dt compute.Dtype) bool {
	return b.rowExact && dt == compute.F32
}

func (b *promptPanelRecordingBackend) PromptPanelBegin()         { b.begins++ }
func (b *promptPanelRecordingBackend) PromptPanelChunk(rows int) { b.chunks = append(b.chunks, rows) }
func (b *promptPanelRecordingBackend) PromptPanelEnd()           { b.ends++ }

func (b *promptPanelRecordingBackend) PromptPanelMatMul(w, x compute.Tensor, tokens int) compute.Tensor {
	b.panelMatMuls++
	return b.Backend.BatchedMatMul(w, x, tokens)
}

func (b *promptPanelRecordingBackend) row(x compute.Tensor, t, width int) compute.Tensor {
	xf, _ := b.Host(x)
	return compute.NewF32(compute.Default(), []int{width}, append([]float32(nil), xf[t*width:(t+1)*width]...))
}

func (b *promptPanelRecordingBackend) PromptPanelRoPE(x compute.Tensor, tokens, start, nHeads, headDim int, theta float64) compute.Tensor {
	width := nHeads * headDim
	out := make([]float32, tokens*width)
	for t := 0; t < tokens; t++ {
		r, _ := b.Host(b.Backend.RoPE(b.row(x, t, width), start+t, nHeads, headDim, theta))
		copy(out[t*width:], r)
	}
	return compute.NewF32(compute.Default(), []int{tokens, width}, out)
}

func (b *promptPanelRecordingBackend) PromptPanelAppendKV(kv compute.KVStore, layer int, kRaw, kRoPE, v compute.Tensor, start, tokens int) {
	width := kRaw.Numel() / tokens
	for t := 0; t < tokens; t++ {
		kv.AppendKV(layer, b.row(kRaw, t, width), b.row(kRoPE, t, width), b.row(v, t, width), start+t)
	}
}

// promptPanelPrefixKV exposes only the first n cached positions, so cpu-ref Attention
// computes exactly the causal window of one panel row.
type promptPanelPrefixKV struct {
	compute.KVStore
	n int
}

func (p promptPanelPrefixKV) KVConfig() compute.KVConfig {
	return p.KVStore.(compute.KVGeometry).KVConfig()
}

func (p promptPanelPrefixKV) prefix(full compute.Tensor) compute.Tensor {
	cfg := p.KVConfig()
	width := cfg.NumKVHeads * cfg.HeadDim
	f, _ := compute.Default().Host(full)
	return compute.NewF32(compute.Default(), []int{p.n, width}, f[:p.n*width])
}

func (p promptPanelPrefixKV) KeysView(layer int) compute.Tensor {
	return p.prefix(p.KVStore.KeysView(layer))
}

func (p promptPanelPrefixKV) ValuesView(layer int) compute.Tensor {
	return p.prefix(p.KVStore.ValuesView(layer))
}

func (b *promptPanelRecordingBackend) PromptPanelAttention(q compute.Tensor, kv compute.KVStore, layer, tokens, start, grp int, scale float32) compute.Tensor {
	b.panelAttention++
	width := q.Numel() / tokens
	out := make([]float32, tokens*width)
	for t := 0; t < tokens; t++ {
		ctx := b.Backend.Attention(b.row(q, t, width), promptPanelPrefixKV{KVStore: kv, n: start + t + 1}, layer, true, grp, scale)
		r, _ := b.Host(ctx)
		copy(out[t*width:], r)
	}
	return compute.NewF32(compute.Default(), []int{tokens, width}, out)
}

func (b *promptPanelRecordingBackend) PromptPanelRow(x compute.Tensor, row, width int) compute.Tensor {
	return b.row(x, row, width)
}

// promptPanelBiasModel is a dense synthetic model with q/k/v projection biases, so the panel's
// AttentionBias branch runs.
func promptPanelBiasModel() *Model {
	cfg := Config{
		HiddenSize: 16, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 4,
		IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: 63, AttentionBias: true,
	}
	extra := map[string][]int{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"self_attn.q_proj.bias"] = []int{cfg.NumHeads * cfg.HeadDim}
		extra[p+"self_attn.k_proj.bias"] = []int{cfg.NumKVHeads * cfg.HeadDim}
		extra[p+"self_attn.v_proj.bias"] = []int{cfg.NumKVHeads * cfg.HeadDim}
	}
	return newSyntheticExtra(cfg, extra)
}

func setPromptPanelChunk(t *testing.T, n int) {
	t.Helper()
	old := promptPanelChunkOverride
	promptPanelChunkOverride = n
	t.Cleanup(func() { promptPanelChunkOverride = old })
}

func assertSameBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %v (0x%08x), want %v (0x%08x)", label, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// assertSameHALKV compares every layer's rotated keys, F32 pre-RoPE keys, values and positions.
func assertSameHALKV(t *testing.T, label string, got, want *Session) {
	t.Helper()
	gs, err := got.halKV.(compute.KVHostSnapshotter).SnapshotToHost()
	if err != nil {
		t.Fatalf("%s: snapshot: %v", label, err)
	}
	ws, err := want.halKV.(compute.KVHostSnapshotter).SnapshotToHost()
	if err != nil {
		t.Fatalf("%s: reference snapshot: %v", label, err)
	}
	if len(gs.Pos) != len(ws.Pos) {
		t.Fatalf("%s: %d positions, want %d", label, len(gs.Pos), len(ws.Pos))
	}
	for i := range gs.Pos {
		if gs.Pos[i] != ws.Pos[i] {
			t.Fatalf("%s: pos[%d] = %d, want %d", label, i, gs.Pos[i], ws.Pos[i])
		}
	}
	for l := range ws.K {
		assertSameBits(t, label+" K layer", gs.K[l], ws.K[l])
		assertSameBits(t, label+" Kraw layer", gs.KRaw[l], ws.KRaw[l])
		assertSameBits(t, label+" V layer", gs.V[l], ws.V[l])
	}
}

// TestHALPromptPrefillRunsOneGEMMPerProjection is the regression for CUDA prefill running at
// decode speed: a backend that offers the prompt-panel seam must ingest a dense prompt with one
// panel GEMM per projection per layer, not one MatMul per token, and must leave the same
// logits and KV state, bit for bit, as the per-token HAL loop.
func TestHALPromptPrefillRunsOneGEMMPerProjection(t *testing.T) {
	setPromptPanelChunk(t, 64)
	m := promptPanelBiasModel()
	cfg := m.Cfg
	prompt := []int{3, 7, 11, 19, 23}

	ref := m.NewBackendSession(compute.Pick("cpu-ref"))
	defer ref.Close()
	wantLogits := ref.Prefill(prompt)

	be := newPromptPanelRecordingBackend()
	s := m.NewBackendSession(be)
	defer s.Close()
	gotLogits := s.Prefill(prompt)
	assertSameBits(t, "panel prefill logits", gotLogits, wantLogits)

	const projectionsPerLayer = 7 // q, k, v, o, gate, up, down
	if want := projectionsPerLayer * cfg.NumLayers; be.panelMatMuls != want {
		t.Fatalf("prompt prefill panel GEMMs = %d, want %d (one per projection per layer)", be.panelMatMuls, want)
	}
	if be.matmuls > 1 {
		t.Fatalf("prompt prefill MatMul calls = %d, want at most 1 (the last-token head)", be.matmuls)
	}
	if be.panelAttention != cfg.NumLayers || be.begins != 1 || be.ends != 1 {
		t.Fatalf("panel attention=%d begin=%d end=%d, want %d/1/1", be.panelAttention, be.begins, be.ends, cfg.NumLayers)
	}
	assertSameHALKV(t, "panel prefill", s, ref)
	assertSameBits(t, "decode after panel prefill", s.Step(29), ref.Step(29))

	// A continuation prefill starts at the cached length and attends over the prefix.
	more := []int{5, 41, 2}
	assertSameBits(t, "continuation panel prefill logits", s.Prefill(more), ref.Prefill(more))
	assertSameHALKV(t, "continuation panel prefill", s, ref)
}

// TestHALPromptPrefillChunkingKeepsBits runs one prompt whole, in uneven chunks and one row
// per chunk, and as a prefix plus suffix prefill; every split leaves the token loop's bits.
func TestHALPromptPrefillChunkingKeepsBits(t *testing.T) {
	m := promptPanelBiasModel()
	prompt := []int{3, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47}
	ref := m.NewBackendSession(compute.Pick("cpu-ref"))
	defer ref.Close()
	wantLogits := ref.Prefill(prompt)
	for _, chunk := range []int{len(prompt), 4, 3, 1} {
		setPromptPanelChunk(t, chunk)
		be := newPromptPanelRecordingBackend()
		s := m.NewBackendSession(be)
		assertSameBits(t, "chunked panel logits", s.Prefill(prompt), wantLogits)
		assertSameHALKV(t, "chunked panel", s, ref)
		if want := (len(prompt) + chunk - 1) / chunk * m.Cfg.NumLayers; be.panelAttention != want {
			t.Fatalf("chunk %d: panel attention calls = %d, want %d", chunk, be.panelAttention, want)
		}
		rows := 0
		for i, n := range be.chunks {
			if n < 1 || n > chunk || (n != chunk && i != len(be.chunks)-1) {
				t.Fatalf("chunk %d: chunk rows %v, want full chunks and one shorter tail", chunk, be.chunks)
			}
			rows += n
		}
		if rows != len(prompt) {
			t.Fatalf("chunk %d: chunk rows %v sum to %d, want %d", chunk, be.chunks, rows, len(prompt))
		}
		s.Close()
	}
	setPromptPanelChunk(t, 64)
	s := m.NewBackendSession(newPromptPanelRecordingBackend())
	defer s.Close()
	s.PrefillNoLogits(prompt[:6])
	assertSameBits(t, "prefix plus suffix panel logits", s.Prefill(prompt[6:]), wantLogits)
	assertSameHALKV(t, "prefix plus suffix panel", s, ref)
}

// TestHALPromptPrefillEvictThenStep evicts a span after a panel prefill; the survivors are
// re-rotated from the panel's F32 pre-RoPE keys, so the next step matches the token loop's.
func TestHALPromptPrefillEvictThenStep(t *testing.T) {
	setPromptPanelChunk(t, 64)
	m := promptPanelBiasModel()
	prompt := []int{3, 7, 11, 19, 23, 29, 31, 37}
	ref := m.NewBackendSession(compute.Pick("cpu-ref"))
	defer ref.Close()
	ref.Prefill(prompt)
	s := m.NewBackendSession(newPromptPanelRecordingBackend())
	defer s.Close()
	s.Prefill(prompt)
	if got, want := s.evictKV(2, 3), ref.evictKV(2, 3); got != want || got != 3 {
		t.Fatalf("evicted %d positions, reference %d, want 3", got, want)
	}
	assertSameHALKV(t, "evicted panel", s, ref)
	assertSameBits(t, "step after evict", s.Step(9), ref.Step(9))
}

// TestHALPromptPrefillKeepsTokenStepWhenNotRowExact keeps one-token prompts and weights without
// a per-row exact panel GEMM on the decode step.
func TestHALPromptPrefillKeepsTokenStepWhenNotRowExact(t *testing.T) {
	m := promptPanelBiasModel()
	be := newPromptPanelRecordingBackend()
	s := m.NewBackendSession(be)
	s.Prefill([]int{9})
	s.Close()
	inexact := newPromptPanelRecordingBackend()
	inexact.rowExact = false
	s = m.NewBackendSession(inexact)
	s.Prefill([]int{3, 7, 11})
	s.Close()
	for _, b := range []*promptPanelRecordingBackend{be, inexact} {
		if b.panelMatMuls != 0 || b.panelAttention != 0 || b.begins != 0 {
			t.Fatalf("prompt took the panel: gemms=%d attention=%d begins=%d", b.panelMatMuls, b.panelAttention, b.begins)
		}
	}
}
