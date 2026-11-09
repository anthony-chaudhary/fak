package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
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

// PromptPanelSetRow writes src back into row of the panel, the inverse of PromptPanelRow.
func (b *promptPanelRecordingBackend) PromptPanelSetRow(x compute.Tensor, row, width int, src compute.Tensor) {
	d, _ := b.Host(x)
	s, _ := b.Host(src)
	copy(d[row*width:(row+1)*width], s)
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
// decode speed: a backend that offers the prompt-panel seam must ingest a prompt with one panel
// GEMM per batched projection per layer, not one MatMul per token, and must leave the same
// logits and KV state, bit for bit, as the per-token HAL loop. The dense-bias case covers the
// original seven-projection layer; the deepseek-v4-moe case covers the MoE arch the panel
// admits: its attention sublayer batches over the rows while its routed FFN stays the proven
// per-row live-expert primitive, so a layer contributes four panel GEMMs plus one single-row
// router MatMul per row. [SW-VERIFIED]: cpu-ref witness, no GPU host.
func TestHALPromptPrefillRunsOneGEMMPerProjection(t *testing.T) {
	setPromptPanelChunk(t, 64)
	prompt := []int{3, 7, 11, 19, 23}
	cases := []struct {
		name               string
		model              func() *Model
		wire               func(t *testing.T, m *Model, s *Session) // per-session extras (fake expert runtime)
		panelProjsPerLayer int                                      // panel GEMMs per layer (dense: 7, deepseek_v4: 4 attention projections)
		routerRowsPerLayer int                                      // single-row router MatMuls per layer (0 for dense)
	}{
		{name: "dense-bias", model: promptPanelBiasModel, panelProjsPerLayer: 7},
		{name: "deepseek-v4-moe", model: promptPanelV4Model, wire: wirePanelV4FakeRuntime, panelProjsPerLayer: 4, routerRowsPerLayer: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model()
			cfg := m.Cfg

			ref := m.NewBackendSession(compute.Pick("cpu-ref"))
			defer ref.Close()
			var refFakes []*panelV4FakeRouted
			if tc.wire != nil {
				tc.wire(t, m, ref)
				refFakes = append(refFakes, ref.v4Expert.(*panelV4FakeRouted))
			}
			wantLogits := ref.Prefill(prompt)

			be := newPromptPanelRecordingBackend()
			s := m.NewBackendSession(be)
			defer s.Close()
			if tc.wire != nil {
				tc.wire(t, m, s)
			}
			gotLogits := s.Prefill(prompt)
			assertSameBits(t, "panel prefill logits", gotLogits, wantLogits)

			if want := tc.panelProjsPerLayer * cfg.NumLayers; be.panelMatMuls != want {
				t.Fatalf("prompt prefill panel GEMMs = %d, want %d (one per batched projection per layer)", be.panelMatMuls, want)
			}
			wantMatMuls := len(prompt)*tc.routerRowsPerLayer*cfg.NumLayers + 1 // per-row routers + the last-token head
			if be.matmuls != wantMatMuls {
				t.Fatalf("prompt prefill MatMul calls = %d, want %d (per-row routers + the last-token head)", be.matmuls, wantMatMuls)
			}
			if be.panelAttention != cfg.NumLayers || be.begins != 1 || be.ends != 1 {
				t.Fatalf("panel attention=%d begin=%d end=%d, want %d/1/1", be.panelAttention, be.begins, be.ends, cfg.NumLayers)
			}
			if tc.wire != nil {
				assertSameV4FFNCalls(t, "prefill", s.v4Expert.(*panelV4FakeRouted), refFakes[0], cfg.NumLayers, prompt)
			}
			assertSameHALKV(t, "panel prefill", s, ref)
			assertSameBits(t, "decode after panel prefill", s.Step(29), ref.Step(29))

			// A continuation prefill starts at the cached length and attends over the prefix.
			more := []int{5, 41, 2}
			assertSameBits(t, "continuation panel prefill logits", s.Prefill(more), ref.Prefill(more))
			assertSameHALKV(t, "continuation panel prefill", s, ref)
		})
	}
}

// assertSameV4FFNCalls proves both paths ran the per-row routed FFN once for every
// (layer, prompt token) pair: the panel path visits them layer-major, the token loop
// token-major, so the multisets must agree exactly.
func assertSameV4FFNCalls(t *testing.T, label string, got, want *panelV4FakeRouted, layers int, prompt []int) {
	t.Helper()
	gc, wc := got.callsSnapshot(), want.callsSnapshot()
	if len(gc) != layers*len(prompt) || len(wc) != layers*len(prompt) {
		t.Fatalf("%s: FFN forward calls = %d/%d, want %d on both paths", label, len(gc), len(wc), layers*len(prompt))
	}
	key := func(c panelV4Call) string { return fmt.Sprintf("%d:%d", c.layer, c.tokenID) }
	gk := make(map[string]int, len(gc))
	for _, c := range gc {
		gk[key(c)]++
	}
	for _, c := range wc {
		if gk[key(c)] == 0 {
			t.Fatalf("%s: token-loop FFN call (%d,%d) has no panel-path twin", label, c.layer, c.tokenID)
		}
		gk[key(c)]--
	}
	for k, n := range gk {
		if n != 0 {
			t.Fatalf("%s: panel-path FFN call %s ran %d extra time(s)", label, k, n)
		}
	}
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

// promptPanelV4Model builds a small deepseek_v4 MoE model: attention projections, layernorms and
// the per-layer router gate, but no dense MLP triple (V4 has none) and no expert tensors — the
// routed FFN is served by an injected fake v4LiveExpertRuntime, so no expert bytes are ever read.
// The layout mirrors NewSyntheticMoE's tensor set with the V4 identity, which is what
// promptPanelFor's named MoE subset keys on.
func promptPanelV4Model() *Model {
	cfg := Config{
		ModelType: "deepseek_v4", HiddenSize: 16, NumLayers: 2, NumHeads: 4, NumKVHeads: 2,
		HeadDim: 4, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: 4, NumExpertsPerTok: 2, NSharedExperts: 0,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	H, E := cfg.HiddenSize, cfg.NumExperts
	type ts = synthTensor
	var tensors []ts
	tensors = append(tensors, ts{"model.embed_tokens.weight", []int{cfg.VocabSize, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "self_attn.q_proj.weight", []int{cfg.NumHeads * cfg.HeadDim, H}},
			ts{p + "self_attn.k_proj.weight", []int{cfg.NumKVHeads * cfg.HeadDim, H}},
			ts{p + "self_attn.v_proj.weight", []int{cfg.NumKVHeads * cfg.HeadDim, H}},
			ts{p + "self_attn.o_proj.weight", []int{H, cfg.NumHeads * cfg.HeadDim}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{routerName(l), []int{E, H}},
		)
	}
	tensors = append(tensors, ts{"model.norm.weight", []int{H}})
	man, raw := synthBuildRawResolved(tensors, func(name string) synthElemFill {
		if strings.HasSuffix(name, "layernorm.weight") || name == "model.norm.weight" {
			return synthConstFill(1.0)
		}
		return synthMatmulFillFor(name)
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// panelV4Call is one recorded live-expert forward: the layer it ran in and the prompt token id
// whose row it contracted.
type panelV4Call struct{ layer, tokenID int }

// panelV4FakeRouted is a deterministic v4LiveExpertRuntime for the panel MoE witness: its output
// is a pure function of the router logits and the normalized row (top-k by logit value, ties to
// the lower expert index, each pick weighted by its own logit), so two sessions that feed it
// identical rows get identical bits back. It records every forward call so a witness can compare
// the (layer, tokenID) coverage of the panel and token-loop paths.
type panelV4FakeRouted struct {
	mu     sync.Mutex
	hidden int
	topK   int
	calls  []panelV4Call
}

func (f *panelV4FakeRouted) forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error) {
	f.mu.Lock()
	f.calls = append(f.calls, panelV4Call{layer, tokenID})
	f.mu.Unlock()
	out := make([]float32, f.hidden)
	order := make([]int, len(logits))
	for e := range order {
		order[e] = e
	}
	sort.SliceStable(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
	k := min(f.topK, len(order))
	for rank := 0; rank < k; rank++ {
		e := order[rank]
		w := logits[e] * float32(rank+1)
		for i := 0; i < f.hidden; i++ {
			out[i] += w * x[(i+e)%f.hidden]
		}
	}
	return out, nil
}

func (f *panelV4FakeRouted) Close() error { return nil }

func (f *panelV4FakeRouted) Stats() v4ExpertRuntimeStats { return v4ExpertRuntimeStats{} }

func (f *panelV4FakeRouted) callsSnapshot() []panelV4Call {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]panelV4Call(nil), f.calls...)
}

// wirePanelV4FakeRuntime installs the deterministic fake routed-expert runtime on one session
// over the V4 fixture: ensureV4LiveExpert requires an indexed source directory, and the fake
// means no weight is ever read from it.
func wirePanelV4FakeRuntime(t *testing.T, m *Model, s *Session) {
	t.Helper()
	if m.sourceDir == "" {
		m.sourceDir = t.TempDir()
	}
	s.v4Expert = &panelV4FakeRouted{hidden: m.Cfg.HiddenSize, topK: m.Cfg.NumExpertsPerTok}
}

// TestHALPromptPrefillMoEGreedyParity is the greedy-sha parity witness between the two prefill
// paths for the MoE arch the panel admits: the batched panel path and the per-token HAL loop
// must produce bit-identical prefill logits, KV state and greedy generations. [SW-VERIFIED]:
// cpu-ref reference backend, no GPU host, no hardware timing claim.
func TestHALPromptPrefillMoEGreedyParity(t *testing.T) {
	setPromptPanelChunk(t, 64)
	const steps = 8
	prompt := []int{3, 7, 11, 19, 23, 29}

	ref := promptPanelV4Model()
	refBe := compute.Pick("cpu-ref")
	refS := ref.NewBackendSession(refBe)
	defer refS.Close()
	wirePanelV4FakeRuntime(t, ref, refS)
	wantLogits := refS.Prefill(prompt)

	panel := promptPanelV4Model()
	be := newPromptPanelRecordingBackend()
	s := panel.NewBackendSession(be)
	defer s.Close()
	wirePanelV4FakeRuntime(t, panel, s)
	gotLogits := s.Prefill(prompt)
	assertSameBits(t, "greedy-parity prefill logits", gotLogits, wantLogits)
	assertSameHALKV(t, "greedy-parity prefill", s, refS)
	if be.panelMatMuls != 4*panel.Cfg.NumLayers {
		t.Fatalf("panel GEMMs = %d, want %d (attention projections only)", be.panelMatMuls, 4*panel.Cfg.NumLayers)
	}
	wantGreedy := greedySequence(t, refS, argmaxF32(wantLogits), steps)
	gotGreedy := greedySequence(t, s, argmaxF32(gotLogits), steps)

	if len(gotGreedy) != len(wantGreedy) {
		t.Fatalf("greedy lengths = %d/%d, want equal", len(gotGreedy), len(wantGreedy))
	}
	for i := range gotGreedy {
		if gotGreedy[i] != wantGreedy[i] {
			t.Fatalf("greedy[%d] = %d, want %d (panel vs token loop diverged)", i, gotGreedy[i], wantGreedy[i])
		}
	}
	t.Logf("[SW-VERIFIED] greedy-sha parity: panel sha256=%s token-loop sha256=%s tokens=%v",
		greedySHA256(gotGreedy), greedySHA256(wantGreedy), gotGreedy)
}

// greedySequence decodes n tokens greedily from seed on s and returns their ids.
func greedySequence(t *testing.T, s *Session, seed int, n int) []int {
	t.Helper()
	out := make([]int, 0, n)
	cur := seed
	for i := 0; i < n; i++ {
		logits := s.Step(cur)
		cur = argmaxF32(logits)
		out = append(out, cur)
	}
	return out
}

// greedySHA256 digests a generated token sequence as little-endian uint32 rows.
func greedySHA256(seq []int) string {
	buf := make([]byte, 4*len(seq))
	for i, id := range seq {
		binary.LittleEndian.PutUint32(buf[4*i:], uint32(id))
	}
	return fmt.Sprintf("%x", sha256.Sum256(buf))
}

// promptPanelWitnessCount reads one reason's count out of a witness snapshot. The switch
// pins the field/reason pairing, so renaming either side fails compilation here.
func promptPanelWitnessCount(c PromptPanelSelectionCounts, r promptPanelWitnessReason) int64 {
	switch r {
	case promptPanelSelected:
		return c.Selected
	case promptPanelRefusalShortPrompt:
		return c.ShortPrompt
	case promptPanelRefusalNoBackend:
		return c.NoBackend
	case promptPanelRefusalNoHALKV:
		return c.NoHALKV
	case promptPanelRefusalBackendLacksSeam:
		return c.BackendLacksSeam
	case promptPanelRefusalArchQ8TokenLoop:
		return c.ArchQ8TokenLoop
	case promptPanelRefusalArchDeepSeekV4NonPanel:
		return c.ArchDeepSeekV4NonPanel
	case promptPanelRefusalArchQKNorm:
		return c.ArchQKNorm
	case promptPanelRefusalArchMLAMoELayout:
		return c.ArchMLAMoELayout
	case promptPanelRefusalArchMiniMaxSparseAttn:
		return c.ArchMiniMaxSparseAttn
	case promptPanelRefusalArchGemma4:
		return c.ArchGemma4
	case promptPanelRefusalCaptureTarget:
		return c.CaptureTarget
	case promptPanelRefusalQ2KEmbedding:
		return c.Q2KEmbedding
	case promptPanelRefusalSplitGPULayers:
		return c.SplitGPULayers
	case promptPanelRefusalDtypeNotRowExact:
		return c.DtypeNotRowExact
	default:
		panic(fmt.Sprintf("model: unknown prompt-panel witness reason %d", r))
	}
}

// fak-test:runtime fast est=20ms lane=default
// TestPromptPanelSelectionMatrix is the coverage-matrix tripwire at the prompt-panel
// selection seam: every refusal axis of promptPanelFor must emit its typed, counted reason,
// so a batched-capable arch served the token loop can never do so silently. Each row drives
// one session to a known decision and asserts (a) the decision, (b) the expected reason's
// witness delta, and (c) no silent refusal - exactly one outcome is counted per call.
// Refusals attribute to the first matching axis in left-to-right order, mirroring the
// original OR: a deepseek_v4 MoE with QKNorm therefore lands on the q8-token-loop axis
// (IsMoE), while the bare QKNorm axis needs a non-MoE config to be reachable.
// [SW-VERIFIED]: cpu-ref witness, no GPU host.
func TestPromptPanelSelectionMatrix(t *testing.T) {
	dense := func() Config { return promptPanelBiasModel().Cfg }
	minimal := func(cfg Config, be compute.Backend) *Session {
		return &Session{M: &Model{Cfg: cfg}, Backend: be, halKV: be.NewKV(halKVConfig(cfg))}
	}
	rows := []struct {
		name      string
		session   func(t *testing.T) *Session
		ids       []int
		wantPanel bool
		reason    promptPanelWitnessReason
	}{
		{
			name: "dense-f32-panel",
			session: func(t *testing.T) *Session {
				return promptPanelBiasModel().NewBackendSession(newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: true, reason: promptPanelSelected,
		},
		{
			name: "deepseek-v4-moe-panel",
			session: func(t *testing.T) *Session {
				return promptPanelV4Model().NewBackendSession(newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: true, reason: promptPanelSelected,
		},
		{
			name: "short-prompt",
			session: func(t *testing.T) *Session {
				return promptPanelBiasModel().NewBackendSession(newPromptPanelRecordingBackend())
			},
			ids: []int{3}, wantPanel: false, reason: promptPanelRefusalShortPrompt,
		},
		{
			name: "no-backend",
			session: func(t *testing.T) *Session {
				return &Session{M: promptPanelBiasModel()}
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalNoBackend,
		},
		{
			name: "no-hal-kv",
			session: func(t *testing.T) *Session {
				return &Session{M: promptPanelBiasModel(), Backend: newPromptPanelRecordingBackend()}
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalNoHALKV,
		},
		{
			name: "backend-lacks-seam",
			session: func(t *testing.T) *Session {
				m := promptPanelBiasModel()
				be := compute.Pick("cpu-ref")
				return &Session{M: m, Backend: be, halKV: be.NewKV(halKVConfig(m.Cfg))}
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalBackendLacksSeam,
		},
		{
			name: "dense-attn-output-gate-q8-token-loop",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.AttnOutputGate = true
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchQ8TokenLoop,
		},
		{
			name: "moe-generic-q8-token-loop",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.NumExperts = 4
				cfg.NumExpertsPerTok = 2
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchQ8TokenLoop,
		},
		{
			name: "deepseek-v4-moe-qknorm-attributes-to-q8-axis",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.ModelType = "deepseek_v4"
				cfg.NumExperts = 4
				cfg.NumExpertsPerTok = 2
				cfg.QKNorm = true
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchQ8TokenLoop,
		},
		{
			name: "deepseek-v4-non-panel",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.ModelType = "deepseek_v4"
				cfg.QKNorm = true // breaks panelMoE; NumExperts stays 0 so the q8 axis stays off
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchDeepSeekV4NonPanel,
		},
		{
			name: "qknorm",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.QKNorm = true
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchQKNorm,
		},
		{
			name: "mla-moe-layout",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.ModelType = "deepseek2"
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchMLAMoELayout,
		},
		{
			name: "minimax-sparse-attn",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.ModelType = "minimax_m3"
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchMiniMaxSparseAttn,
		},
		{
			name: "gemma4",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.ModelType = "gemma4"
				return minimal(cfg, newPromptPanelRecordingBackend())
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalArchGemma4,
		},
		{
			name: "capture-target",
			session: func(t *testing.T) *Session {
				s := promptPanelBiasModel().NewBackendSession(newPromptPanelRecordingBackend())
				s.captureTargetHidden = true
				return s
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalCaptureTarget,
		},
		{
			name: "q2k-embedding",
			session: func(t *testing.T) *Session {
				cfg := dense()
				cfg.HiddenSize = 256 // Q2_K rows pack in 256-wide superblocks
				q2k, err := NewQ2KEmbedding(makeTestQ2KPayload(cfg.VocabSize, cfg.HiddenSize), cfg.VocabSize, cfg.HiddenSize)
				if err != nil {
					t.Fatal(err)
				}
				be := newPromptPanelRecordingBackend()
				return &Session{M: &Model{Cfg: cfg, Q2KEmbedding: q2k}, Backend: be, halKV: be.NewKV(halKVConfig(cfg))}
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalQ2KEmbedding,
		},
		{
			name: "split-gpu-layers",
			session: func(t *testing.T) *Session {
				s := promptPanelBiasModel().NewBackendSession(newPromptPanelRecordingBackend())
				s.DenseGPULayers = 1 // of 2 layers: partial placement
				return s
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalSplitGPULayers,
		},
		{
			name: "dtype-not-row-exact",
			session: func(t *testing.T) *Session {
				be := newPromptPanelRecordingBackend()
				be.rowExact = false
				return promptPanelBiasModel().NewBackendSession(be)
			},
			ids: []int{3, 7}, wantPanel: false, reason: promptPanelRefusalDtypeNotRowExact,
		},
	}
	for _, tc := range rows {
		t.Run(tc.name, func(t *testing.T) {
			s := tc.session(t)
			defer s.Close()
			before := PromptPanelSelectionWitness()
			_, ok := s.promptPanelFor(tc.ids)
			after := PromptPanelSelectionWitness()
			if ok != tc.wantPanel {
				t.Fatalf("panel = %t, want %t", ok, tc.wantPanel)
			}
			if got := promptPanelWitnessCount(after, tc.reason) - promptPanelWitnessCount(before, tc.reason); got != 1 {
				t.Fatalf("reason %v: witness delta = %d, want 1", tc.reason, got)
			}
			if got := after.Total() - before.Total(); got != 1 {
				t.Fatalf("total witness delta = %d, want 1 (every outcome must count exactly one typed reason)", got)
			}
		})
	}
}

// TestHALPromptPrefillMoEFallbackKeepsTokenLoop pins the named-subset boundary: a generic MoE
// arch (not deepseek_v4) keeps the per-token loop even when the backend offers the panel seam,
// because its routed FFN has no per-row HAL primitive yet; a deepseek_v4 config carrying an
// axis the panel does not model (QKNorm here) is refused the same way. [SW-VERIFIED]
func TestHALPromptPrefillMoEFallbackKeepsTokenLoop(t *testing.T) {
	generic := Config{HiddenSize: 16, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 4,
		IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: 4, NumExpertsPerTok: 2, TieWordEmbeddings: true}
	m := NewSyntheticMoE(generic)
	be := newPromptPanelRecordingBackend()
	s := m.NewBackendSession(be)
	defer s.Close()
	if _, ok := s.promptPanelFor([]int{3, 7}); ok {
		t.Fatal("generic MoE took the panel, want the token loop")
	}
	if be.panelMatMuls != 0 || be.begins != 0 {
		t.Fatalf("generic MoE ran panel ops: gemms=%d begins=%d", be.panelMatMuls, be.begins)
	}

	v4 := promptPanelV4Model()
	v4.Cfg.QKNorm = true
	be2 := newPromptPanelRecordingBackend()
	s2 := v4.NewBackendSession(be2)
	defer s2.Close()
	if _, ok := s2.promptPanelFor([]int{3, 7}); ok {
		t.Fatal("deepseek_v4 with QKNorm took the panel, want the token loop")
	}
}
