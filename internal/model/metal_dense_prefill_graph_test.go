//go:build darwin && arm64 && cgo

package model

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// denseGraphQ6KTensor is a Q6_K tensor with random payload and the f16 super-block scale pinned
// to 2^-13, so dequantized weights stay in the ~0.1 band the synthetic forward tolerates.
func denseGraphQ6KTensor(out, in int, seed uint64) *kQuantTensor {
	nblk := in / qkK
	bb := kindQ6K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, seed)
	for i := 0; i < out*nblk; i++ {
		binary.LittleEndian.PutUint16(raw[i*bb+bb-2:], 0x0800)
	}
	return quantizeKQuantFromRaw(raw, out, in, kindQ6K)
}

// denseGraphTestModel is the Qwen2.5 q4_k_m residency shape in miniature: q/k Q8 (q8w), v and
// down Q6_K on even layers / Q4_K on odd layers, o/gate/up Q4_K, plus q/k/v projection bias.
func denseGraphTestModel(t *testing.T, layers int) *Model {
	t.Helper()
	cfg := Config{HiddenSize: 256, NumLayers: layers, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 512, VocabSize: 64, RMSNormEps: 1e-6, RopeTheta: 10000, AttentionBias: true}
	extra := map[string][]int{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"self_attn.q_proj.bias"] = []int{cfg.NumHeads * cfg.HeadDim}
		extra[p+"self_attn.k_proj.bias"] = []int{cfg.NumKVHeads * cfg.HeadDim}
		extra[p+"self_attn.v_proj.bias"] = []int{cfg.NumKVHeads * cfg.HeadDim}
	}
	m := newSyntheticExtra(cfg, extra)
	m.Quantize()
	var projs [][2]any
	m.kqw = map[string]*kQuantTensor{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		kv := cfg.NumKVHeads * cfg.HeadDim
		projs = append(projs,
			[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize})
		if l%2 == 0 {
			m.kqw[p+"self_attn.v_proj.weight"] = denseGraphQ6KTensor(kv, cfg.HiddenSize, uint64(100+l))
			m.kqw[p+"mlp.down_proj.weight"] = denseGraphQ6KTensor(cfg.HiddenSize, cfg.IntermediateSize, uint64(200+l))
		} else {
			projs = append(projs,
				[2]any{p + "self_attn.v_proj.weight", kv},
				[2]any{p + "mlp.down_proj.weight", cfg.HiddenSize})
		}
	}
	fillQ4KW(t, m, projs, 13599)
	return m
}

type denseGraphRun struct {
	hidden            [][]float32
	cache             *KVCache
	graphCBs, dispCBs int
	graphDeclines     int
}

// runDenseGraphPrefill prefills chunks on a fresh MetalQ4K session with the dense layer graph
// on or off, returning each chunk's last hidden, the final cache, and the CB split.
func runDenseGraphPrefill(t *testing.T, m *Model, chunks [][]int, graph bool) denseGraphRun {
	t.Helper()
	SetDenseQ4KPrefillGraph(graph)
	t.Cleanup(func() { SetDenseQ4KPrefillGraph(true) })
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Q4K, s.MetalQ4K = true, true
	s.PhaseProfiler = NewPhaseProfiler() // a decline must record, not panic, with a profiler attached
	if got := s.denseQ4KPrefillGraphEligible(); got != graph {
		t.Fatalf("dense graph eligible=%v, want %v", got, graph)
	}
	s.ResetMetalCommandBuffers()
	var r denseGraphRun
	for _, ids := range chunks {
		r.hidden = append(r.hidden, s.prefillBatchedQ4K(ids))
	}
	r.cache = s.Cache
	r.graphCBs, r.dispCBs = s.MetalGraphCommandBuffers(), s.MetalDispatchCommandBuffers()
	receipt, err := s.PhaseProfiler.MetalFallbackReceipt()
	if err != nil {
		t.Fatalf("metal fallback receipt: %v", err)
	}
	for _, ev := range receipt.Events {
		if ev.Route == MetalFallbackDensePrefillGraphHost {
			r.graphDeclines++
		}
	}
	return r
}

func denseGraphChunks(sizes ...int) [][]int {
	var chunks [][]int
	next := 0
	for _, n := range sizes {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (next*7 + 3) % 64
			next++
		}
		chunks = append(chunks, ids)
	}
	return chunks
}

// requireDenseGraphParity holds the graph route to the per-projection host route it replaces:
// same GEMM kernels and weights, so the only drift is device RMSNorm/SwiGLU/residual float order
// and the device vs host Q8 activation quantize — a wiring bug (wrong weight, missed residual,
// dropped bias) diverges O(1) and blows past these bounds.
func requireDenseGraphParity(t *testing.T, m *Model, ref, got denseGraphRun) {
	t.Helper()
	for i := range ref.hidden {
		cos := cosine(ref.hidden[i], got.hidden[i])
		mx, _ := maxAbsDiff(ref.hidden[i], got.hidden[i])
		if cos < 0.9999 || math.IsNaN(float64(cos)) {
			t.Fatalf("chunk %d last-hidden cosine %.7f < 0.9999 (max-abs %.3e)", i, cos, mx)
		}
		t.Logf("chunk %d last-hidden cosine=%.7f max-abs=%.3e", i, cos, mx)
	}
	if ref.cache.Len() != got.cache.Len() {
		t.Fatalf("cache len %d != %d", got.cache.Len(), ref.cache.Len())
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for _, side := range []struct {
			name string
			a, b []float32
		}{{"K", ref.cache.K[l], got.cache.K[l]}, {"Kraw", ref.cache.Kraw[l], got.cache.Kraw[l]}, {"V", ref.cache.V[l], got.cache.V[l]}} {
			if len(side.a) != len(side.b) {
				t.Fatalf("layer %d %s len %d != %d", l, side.name, len(side.b), len(side.a))
			}
			// q/k are Q8 projections: a device-vs-host float-order difference in the norm can
			// flip one Q8 activation code, an error of one Q8 step (1/127 of the block absmax).
			// So the elementwise bound sits above that granularity while the aggregate relative
			// RMS stays tight; a wiring bug is O(1) on both.
			var sq, dsq, mx float64
			for i := range side.a {
				d := math.Abs(float64(side.a[i] - side.b[i]))
				mx = math.Max(mx, d)
				dsq += d * d
				sq += float64(side.a[i]) * float64(side.a[i])
			}
			if sq > 1e-24 {
				if rel, peak := math.Sqrt(dsq/sq), mx/math.Sqrt(sq/float64(len(side.a))); rel > 1e-3 || peak > 1e-2 {
					t.Fatalf("layer %d %s relative RMS %.3e (max 1e-3), max-abs/RMS %.3e (max 1e-2)", l, side.name, rel, peak)
				}
			}
		}
	}
	for i := range ref.cache.pos {
		if ref.cache.pos[i] != got.cache.pos[i] {
			t.Fatalf("pos[%d] %d != %d", i, got.cache.pos[i], ref.cache.pos[i])
		}
	}
}

// TestDenseQ4KPrefillGraphMatchesHostRoute is the #13599 prefill gate: the dense layer graph
// matches the per-projection host route over unequal chunks (nonzero RoPE/cache bases), pays
// exactly L+1 command buffers per prefill and no per-projection dispatches, while the host route
// pays at most 5 per layer (q+k group, v, o, gate+up group, down — Step A).
func TestDenseQ4KPrefillGraphMatchesHostRoute(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	m := denseGraphTestModel(t, 4)
	L := m.Cfg.NumLayers
	chunks := denseGraphChunks(37, 5, 64)

	ref := runDenseGraphPrefill(t, m, chunks, false)
	if ref.graphCBs != 0 || ref.dispCBs > 5*L*len(chunks) {
		t.Fatalf("host route CBs graph=%d dispatch=%d, want 0 and <= %d", ref.graphCBs, ref.dispCBs, 5*L*len(chunks))
	}
	got := runDenseGraphPrefill(t, m, chunks, true)
	if want := (L + 1) * len(chunks); got.graphCBs != want || got.dispCBs != 0 {
		t.Fatalf("graph route CBs graph=%d dispatch=%d, want %d and 0", got.graphCBs, got.dispCBs, want)
	}
	t.Logf("CBs per prefill: host route %d dispatch, graph route %d graph (L=%d)", ref.dispCBs/len(chunks), got.graphCBs/len(chunks), L)
	requireDenseGraphParity(t, m, ref, got)
}

// TestDenseQ4KPrefillGraphMidPrefillDeclineFallsBack injects a decline at the second segment:
// the first segment's device result is kept, the rest of the prefill runs on the host route
// from the unchanged residual, the outcome still matches, and the decline is recorded.
func TestDenseQ4KPrefillGraphMidPrefillDeclineFallsBack(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	m := denseGraphTestModel(t, 3)
	chunks := denseGraphChunks(33)
	ref := runDenseGraphPrefill(t, m, chunks, false)
	for _, seg := range []int{1, m.Cfg.NumLayers} {
		denseQ4KGraphFailSegmentForTest = seg
		got := runDenseGraphPrefill(t, m, chunks, true)
		denseQ4KGraphFailSegmentForTest = -1
		if got.graphDeclines != 1 {
			t.Fatalf("decline at segment %d recorded %d dense-graph fallbacks, want 1", seg, got.graphDeclines)
		}
		if got.graphCBs != seg || got.dispCBs == 0 {
			t.Fatalf("decline at segment %d: CBs graph=%d dispatch=%d, want %d graph and host dispatches after", seg, got.graphCBs, got.dispCBs, seg)
		}
		requireDenseGraphParity(t, m, ref, got)
	}
}

// TestDenseQ4KPrefillGraphLongPromptPanels walks a prompt longer than one graph panel and holds
// it to the single host pass.
// fak-test:runtime medium est=6s
func TestDenseQ4KPrefillGraphLongPromptPanels(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	m := denseGraphTestModel(t, 2)
	chunks := denseGraphChunks(denseQ4KPrefillGraphMaxRows + 77)
	ref := runDenseGraphPrefill(t, m, chunks, false)
	got := runDenseGraphPrefill(t, m, chunks, true)
	if want := 2 * (m.Cfg.NumLayers + 1); got.graphCBs != want {
		t.Fatalf("long prompt graph CBs %d, want %d (two panels)", got.graphCBs, want)
	}
	requireDenseGraphParity(t, m, ref, got)
}

// TestDenseQ4KPrefillGraphDeclinesUnsupportedShapes keeps the graph off where its device ops
// would change semantics: LayerNorm, GELU, and o_proj bias all stay on the host route.
func TestDenseQ4KPrefillGraphDeclinesUnsupportedShapes(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	for _, tc := range []struct {
		name   string
		mutate func(*Model)
	}{
		{"layernorm", func(m *Model) { m.Cfg.LayerNorm = true }},
		{"gelu", func(m *Model) { m.Cfg.ActGeluTanh = true }},
		{"gain1p", func(m *Model) { m.Cfg.NormGain1p = true }},
		{"o-bias", func(m *Model) {
			name := layerName(1, "self_attn.o_proj.bias")
			m.manifest[name] = tensorMeta{Shape: []int{m.Cfg.HiddenSize}}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := denseGraphTestModel(t, 2)
			tc.mutate(m)
			s := m.NewSession()
			defer s.Close()
			s.Q4K, s.MetalQ4K = true, true
			if s.denseQ4KPrefillGraphEligible() {
				t.Fatal("dense prefill graph admitted an unsupported shape")
			}
		})
	}
}
