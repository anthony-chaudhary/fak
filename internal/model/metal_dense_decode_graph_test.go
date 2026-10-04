//go:build darwin && arm64 && cgo

package model

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// denseDecodeGraphTestCfg is a dense Qwen2-shaped PreNorm SwiGLU config: GQA (4 query
// heads over 2 KV heads, hd=64), q/k/v projection bias, no softcap, so the route admits it.
func denseDecodeGraphTestCfg() Config {
	return Config{
		HiddenSize: 256, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, RopeTheta: 10000.0,
		AttentionBias: true, EOSTokenID: -1,
	}
}

type denseDecodeGraphVariant struct {
	name string
	// q6 makes v_proj, down_proj and the LM head resident Q6_K instead of Q4_K, the
	// Qwen2.5 Q4_K_M mix.
	q6 bool
	// q8 makes k_proj and the LM head resident Q8_0 (the Q8 projection and Q8 head
	// branches), and allBias adds o_proj/gate/up/down biases to the q/k/v ones.
	q8, allBias bool
}

// newDenseDecodeGraphTestModel builds a deterministic dense model whose seven layer
// projections and untied LM head are resident Q4_K (or the Q6_K mix). Two calls build
// byte-identical models, which the parity tests need because a Metal session may free
// its model's CPU copy (single residency).
func newDenseDecodeGraphTestModel(t *testing.T, cfg Config, v denseDecodeGraphVariant) *Model {
	t.Helper()
	extra := map[string][]int{}
	for l := 0; l < cfg.NumLayers; l++ {
		extra[layerName(l, "self_attn.q_proj.bias")] = []int{cfg.NumHeads * cfg.HeadDim}
		extra[layerName(l, "self_attn.k_proj.bias")] = []int{cfg.NumKVHeads * cfg.HeadDim}
		extra[layerName(l, "self_attn.v_proj.bias")] = []int{cfg.NumKVHeads * cfg.HeadDim}
		if v.allBias {
			extra[layerName(l, "self_attn.o_proj.bias")] = []int{cfg.HiddenSize}
			extra[layerName(l, "mlp.gate_proj.bias")] = []int{cfg.IntermediateSize}
			extra[layerName(l, "mlp.up_proj.bias")] = []int{cfg.IntermediateSize}
			extra[layerName(l, "mlp.down_proj.bias")] = []int{cfg.HiddenSize}
		}
	}
	m := newSyntheticExtra(cfg, extra)
	if v.allBias {
		// Scale the o_proj/MLP biases from the synthetic ±0.1 to ±2 so a dropped or
		// misnamed bias moves the logits past the parity gate instead of hiding in it.
		for l := 0; l < cfg.NumLayers; l++ {
			for _, suffix := range []string{"self_attn.o_proj.bias", "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias"} {
				meta := m.manifest[layerName(l, suffix)]
				for off := meta.Offset; off < meta.Offset+meta.Nbytes; off += 4 {
					x := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[off:]))
					binary.LittleEndian.PutUint32(m.raw[off:], math.Float32bits(20*x))
				}
			}
		}
	}
	qw, kvw := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	var projs [][2]any
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		projs = append(projs,
			[2]any{p + "self_attn.q_proj.weight", qw},
			[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize},
		)
		if !v.q8 {
			projs = append(projs, [2]any{p + "self_attn.k_proj.weight", kvw})
		}
		if !v.q6 {
			projs = append(projs,
				[2]any{p + "self_attn.v_proj.weight", kvw},
				[2]any{p + "mlp.down_proj.weight", cfg.HiddenSize},
			)
		}
	}
	fillQ4KW(t, m, projs, 4242)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	if v.q6 {
		for l := 0; l < cfg.NumLayers; l++ {
			m.kqw[layerName(l, "self_attn.v_proj.weight")] = denseTestQ6K(kvw, cfg.HiddenSize, int64(5100+l))
			m.kqw[layerName(l, "mlp.down_proj.weight")] = denseTestQ6K(cfg.HiddenSize, cfg.IntermediateSize, int64(5200+l))
		}
		m.kqw["lm_head.weight"] = denseTestQ6K(cfg.VocabSize, cfg.HiddenSize, 5300)
	} else if !v.q8 {
		m.q4kw["lm_head.weight"] = randomQ4KTensor(cfg.VocabSize, cfg.HiddenSize, 5301)
	}
	if v.q8 {
		if m.q8w == nil {
			m.q8w = map[string]*q8Tensor{}
		}
		for l := 0; l < cfg.NumLayers; l++ {
			name := layerName(l, "self_attn.k_proj.weight")
			m.q8w[name] = quantizeQ8(m.tensor(name), kvw, cfg.HiddenSize)
		}
		// An untied Q8_0 head held only in q8w (the Qwen2.5-7B lean-load shape); its
		// values come from the embedding, which is [vocab, hidden] like the head.
		m.q8w["lm_head.weight"] = quantizeQ8(m.tensor("model.embed_tokens.weight"), cfg.VocabSize, cfg.HiddenSize)
	}
	return m
}

// denseTestQ6K is randomQ6KTensor with the f16 super-block scale pinned near 5e-4 so the
// dequantized weights stay O(1) like the bounded Q4_K projections.
func denseTestQ6K(out, in int, seed int64) *kQuantTensor {
	qt := randomQ6KTensor(out, in, seed)
	for b := 0; b < len(qt.raw)/q6kBlockBytes; b++ {
		hi := b*q6kBlockBytes + q6kBlockBytes - 1
		qt.raw[hi] = 0x10 | (qt.raw[hi] & 0x03)
	}
	return qt
}

func newDenseDecodeGraphSession(t *testing.T, cfg Config, v denseDecodeGraphVariant, metal bool) *Session {
	t.Helper()
	s := newDenseDecodeGraphTestModel(t, cfg, v).NewSession()
	s.Q4K = true
	s.MetalQ4K = metal
	t.Cleanup(s.Close)
	return s
}

func denseDecodeGraphPrompt(cfg Config, n int) []int {
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = (i*7 + 3) % cfg.VocabSize
	}
	return prompt
}

func requireDenseDecodeParity(t *testing.T, label string, cpu, gpu []float32) {
	t.Helper()
	if cos, maxRel := cosineAndMaxRel(cpu, gpu); cos < 0.999 {
		t.Fatalf("%s: logits cosine %.6f (maxRel %.3g) below 0.999", label, cos, maxRel)
	}
	if a, b := argmaxF(cpu), argmaxF(gpu); a != b {
		t.Fatalf("%s: argmax cpu=%d gpu=%d", label, a, b)
	}
}

func requireDenseDecodeKVClose(t *testing.T, label string, cpu, gpu *KVCache) {
	t.Helper()
	if cpu.Len() != gpu.Len() {
		t.Fatalf("%s: cache Len cpu=%d gpu=%d", label, cpu.Len(), gpu.Len())
	}
	for l := range cpu.K {
		for _, side := range []struct {
			name     string
			cpu, gpu []float32
		}{{"Kraw", cpu.Kraw[l], gpu.Kraw[l]}, {"K", cpu.K[l], gpu.K[l]}, {"V", cpu.V[l], gpu.V[l]}} {
			if len(side.cpu) != len(side.gpu) {
				t.Fatalf("%s: layer %d %s len cpu=%d gpu=%d", label, l, side.name, len(side.cpu), len(side.gpu))
			}
			if cos, maxRel := cosineAndMaxRel(side.cpu, side.gpu); cos < 0.9999 {
				t.Fatalf("%s: layer %d %s cosine %.6f maxRel %.3g", label, l, side.name, cos, maxRel)
			}
		}
	}
}

// TestMetalDenseQ4KDecodeGraphMatchesCPU is the dense decode graph's parity and
// single-command-buffer gate: a MetalQ4K session decoding through the graph must match
// the CPU Q4_K blockStep session token-for-token (argmax, logits cosine, KV rows) and
// commit at most two command buffers per decoded token.
//
// fak-test:runtime slow est=10s
func TestMetalDenseQ4KDecodeGraphMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := denseDecodeGraphTestCfg()
	for _, v := range []denseDecodeGraphVariant{{name: "q4k"}, {name: "q4k_m_q6k_mix", q6: true},
		{name: "q8_head_q8_kproj_all_bias", q8: true, allBias: true}} {
		t.Run(v.name, func(t *testing.T) {
			cpu := newDenseDecodeGraphSession(t, cfg, v, false)
			gpu := newDenseDecodeGraphSession(t, cfg, v, true)
			prompt := denseDecodeGraphPrompt(cfg, 16)
			cpuLg, gpuLg := cpu.Prefill(prompt), gpu.Prefill(prompt)
			requireDenseDecodeParity(t, "prefill", cpuLg, gpuLg)
			var cpuSeq, gpuSeq []int
			for step := 0; step < 10; step++ {
				cpuNext, gpuNext := argmaxF(cpuLg), argmaxF(gpuLg)
				cpuSeq, gpuSeq = append(cpuSeq, cpuNext), append(gpuSeq, gpuNext)
				gpu.ResetMetalCommandBuffers()
				cpuLg, gpuLg = cpu.Step(cpuNext), gpu.Step(gpuNext)
				if cbs := gpu.MetalCommandBuffers(); cbs > 2 {
					t.Fatalf("step %d committed %d Metal command buffers, want <= 2 (receipt %+v)", step, cbs, gpu.DenseQ4KDecodeGraphReceipt())
				}
				r := gpu.DenseQ4KDecodeGraphReceipt()
				if !r.Accepted || r.CommandBuffers != 1 || r.HostReadbackBytes == 0 {
					t.Fatalf("step %d: dense decode graph receipt %+v", step, r)
				}
				requireDenseDecodeParity(t, "step", cpuLg, gpuLg)
			}
			for i := range cpuSeq {
				if cpuSeq[i] != gpuSeq[i] {
					t.Fatalf("greedy token %d differs: cpu=%v gpu=%v", i, cpuSeq, gpuSeq)
				}
			}
			requireDenseDecodeKVClose(t, "after decode", cpu.Cache, gpu.Cache)
			r := gpu.DenseQ4KDecodeGraphReceipt()
			if r.AcceptedTokens < 10 || r.GraphFallbacks != 0 {
				t.Fatalf("receipt after decode %+v", r)
			}
			if r.DeviceKVSeedBytes != 0 {
				t.Fatalf("steady-state decode re-uploaded %d KV bytes", r.DeviceKVSeedBytes)
			}
			t.Logf("%s: greedy %v, receipt encoders=%d gpu_ms=%.3f up=%dB down=%dB reseeds=%d",
				v.name, gpuSeq, r.Encoders, r.GPUMilliseconds, r.HostUploadBytes, r.HostReadbackBytes, r.DeviceKVReseeds)
		})
	}
}

// TestMetalDenseQ4KDecodeGraphKillSwitch pins FAK_DENSE_Q4K_DECODE_GRAPH=0: the token
// takes the historical per-op path (more than two command buffers) and the receipt
// records the decline.
//
// fak-test:runtime integration est=1s lane=default
func TestMetalDenseQ4KDecodeGraphKillSwitch(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	cfg := denseDecodeGraphTestCfg()
	s := newDenseDecodeGraphSession(t, cfg, denseDecodeGraphVariant{name: "q4k"}, true)
	lg := s.Prefill(denseDecodeGraphPrompt(cfg, 8))
	t.Setenv("FAK_DENSE_Q4K_DECODE_GRAPH", "0")
	s.ResetMetalCommandBuffers()
	s.Step(argmaxF(lg))
	if cbs := s.MetalCommandBuffers(); cbs <= 2 {
		t.Fatalf("kill switch: %d command buffers, want the per-op path (> 2)", cbs)
	}
	if r := s.DenseQ4KDecodeGraphReceipt(); r.Accepted || r.DeclineReason != denseDeclineDisabled {
		t.Fatalf("kill switch receipt %+v", r)
	}
}

// TestMetalDenseQ4KDecodeGraphFailOpenAndCoherence covers the mirror lifecycle against
// the CPU reference: an injected post-submit failure replays through blockStep exactly
// once (no double append) with correct logits, Evict, RestoreSpan and a speculative
// rollback (Truncate) each invalidate and reseed the mirror, and a long decode crosses the mirror's capacity
// and regrows it, all while staying in parity.
//
// fak-test:runtime slow est=15s
func TestMetalDenseQ4KDecodeGraphFailOpenAndCoherence(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := denseDecodeGraphTestCfg()
	// A 64-token context window caps the first mirror at 64 rows, so the long decode
	// below crosses it (and regrows geometrically past the window) in ~50 steps.
	cfg.MaxPositionEmbeddings = 64
	v := denseDecodeGraphVariant{name: "q4k"}
	cpu := newDenseDecodeGraphSession(t, cfg, v, false)
	gpu := newDenseDecodeGraphSession(t, cfg, v, true)
	prompt := denseDecodeGraphPrompt(cfg, 16)
	cpuLg, gpuLg := cpu.Prefill(prompt), gpu.Prefill(prompt)
	step := func(label string, id int) {
		t.Helper()
		cpuLg, gpuLg = cpu.Step(id), gpu.Step(id)
		requireDenseDecodeParity(t, label, cpuLg, gpuLg)
	}
	step("warm", argmaxF(cpuLg))
	if r := gpu.DenseQ4KDecodeGraphReceipt(); !r.Accepted || r.DeviceKVReseeds != 1 {
		t.Fatalf("warm receipt %+v", r)
	}

	// Injected post-submit failure: the graph commits and completes but reports failure
	// before any host mutation, so blockStep replays the token and appends it once.
	gpu.denseDecodeState().injectPostSubmitFailure = true
	before := gpu.Cache.Len()
	step("injected", argmaxF(cpuLg))
	if got := gpu.Cache.Len(); got != before+1 {
		t.Fatalf("injected failure: cache Len %d, want %d (exactly one append)", got, before+1)
	}
	r := gpu.DenseQ4KDecodeGraphReceipt()
	if r.Accepted || r.DeclineReason != denseDeclineGraph || r.GraphFallbacks != 1 {
		t.Fatalf("injected failure receipt %+v", r)
	}
	requireDenseDecodeKVClose(t, "after injected failure", cpu.Cache, gpu.Cache)
	step("after injected", argmaxF(cpuLg))
	if r := gpu.DenseQ4KDecodeGraphReceipt(); !r.Accepted || r.DeviceKVReseeds != 2 {
		t.Fatalf("post-failure reseed receipt %+v", r)
	}

	// Evict compacts rows and re-RoPEs the survivors: the mirror must reseed.
	cpu.Cache.Evict(2, 3)
	gpu.Cache.Evict(2, 3)
	step("after evict", argmaxF(cpuLg))
	if r := gpu.DenseQ4KDecodeGraphReceipt(); !r.Accepted || r.DeviceKVReseeds != 3 {
		t.Fatalf("evict reseed receipt %+v", r)
	}
	requireDenseDecodeKVClose(t, "after evict", cpu.Cache, gpu.Cache)

	// RestoreSpan reinstates evicted rows; both caches evict then restore the same span
	// (the generation bumps on each), and the next token reseeds once. It runs before the
	// rollback below because Truncate leaves the token lineage SerializeSpan requires
	// longer than the resident rows.
	for _, c := range []*KVCache{cpu.Cache, gpu.Cache} {
		blob, err := c.SerializeSpan(1, 2)
		if err != nil {
			t.Fatal(err)
		}
		c.Evict(1, 2)
		if _, err := c.RestoreSpan(blob); err != nil {
			t.Fatal(err)
		}
	}
	step("after restore", argmaxF(cpuLg))
	if r := gpu.DenseQ4KDecodeGraphReceipt(); !r.Accepted || r.DeviceKVReseeds != 4 {
		t.Fatalf("restore reseed receipt %+v", r)
	}
	requireDenseDecodeKVClose(t, "after restore", cpu.Cache, gpu.Cache)

	// Speculative rollback truncates both caches; the mirror's mutation generation no
	// longer matches, so the next token reseeds from the truncated host cache.
	cpu.RollbackSpeculative(3)
	gpu.RollbackSpeculative(3)
	step("after rollback", (argmaxF(cpuLg)+1)%cfg.VocabSize)
	if r := gpu.DenseQ4KDecodeGraphReceipt(); !r.Accepted || r.DeviceKVReseeds != 5 || r.DeviceKVSeedBytes == 0 {
		t.Fatalf("rollback reseed receipt %+v", r)
	}
	requireDenseDecodeKVClose(t, "after rollback", cpu.Cache, gpu.Cache)

	// A long greedy decode crosses the initial capacity and regrows the mirror.
	startCap := gpu.DenseQ4KDecodeGraphReceipt().DeviceKVCapacity
	for gpu.Cache.Len() <= startCap {
		step("long", argmaxF(cpuLg))
		if r := gpu.DenseQ4KDecodeGraphReceipt(); !r.Accepted {
			t.Fatalf("long decode declined at Len=%d: %+v", gpu.Cache.Len(), r)
		}
	}
	r = gpu.DenseQ4KDecodeGraphReceipt()
	if startCap != cfg.MaxPositionEmbeddings || r.DeviceKVCapacity <= startCap || r.DeviceKVReseeds != 6 {
		t.Fatalf("capacity growth receipt %+v (start capacity %d)", r, startCap)
	}
	requireDenseDecodeKVClose(t, "after growth", cpu.Cache, gpu.Cache)

	// Close frees the mirror.
	gpu.Close()
	if st := gpu.denseDecode; st == nil || st.kv != nil {
		t.Fatal("Session.Close did not free the dense decode device KV mirror")
	}
}

// TestMetalDenseQ4KDecodeGraphKVBudgetDeclines pins the budget bound: a context that no
// longer fits the device KV budget declines to blockStep and stays correct.
//
// fak-test:runtime integration est=2s lane=default
func TestMetalDenseQ4KDecodeGraphKVBudgetDeclines(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	cfg := denseDecodeGraphTestCfg()
	perToken := int64(denseDecodeKVSides * cfg.NumLayers * cfg.NumKVHeads * cfg.HeadDim * 4)
	SetDenseQ4KDecodeKVBudgetBytes(8 * perToken)
	t.Cleanup(func() { SetDenseQ4KDecodeKVBudgetBytes(0) })
	s := newDenseDecodeGraphSession(t, cfg, denseDecodeGraphVariant{name: "q4k"}, true)
	lg := s.Prefill(denseDecodeGraphPrompt(cfg, 6))
	lg = s.Step(argmaxF(lg)) // need 7 <= 8 rows: admitted
	if r := s.DenseQ4KDecodeGraphReceipt(); !r.Accepted || r.DeviceKVCapacity != 8 {
		t.Fatalf("in-budget receipt %+v", r)
	}
	s.Step(argmaxF(lg)) // need 8: still fits
	s.Step(argmaxF(lg)) // need 9 > 8: declines
	if r := s.DenseQ4KDecodeGraphReceipt(); r.Accepted || r.DeclineReason != denseDeclineKVBudget {
		t.Fatalf("over-budget receipt %+v", r)
	}
	if s.Cache.Len() != 9 {
		t.Fatalf("cache Len %d after the declined token, want 9", s.Cache.Len())
	}
}

// TestMetalDenseQ4KDecodeGraphCachesNotReadyVerdict pins that a model the graph cannot
// serve is scanned once per session: the failed readiness verdict is cached, so later
// tokens decline with the same reason without re-resolving every projection.
//
// fak-test:runtime integration est=1s lane=default
func TestMetalDenseQ4KDecodeGraphCachesNotReadyVerdict(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	cfg := denseDecodeGraphTestCfg()
	s := newDenseDecodeGraphSession(t, cfg, denseDecodeGraphVariant{name: "q4k"}, true)
	// An f32 head (no resident Q4_K/Q6_K/Q8 store) is one the graph cannot encode.
	delete(s.M.q4kw, "lm_head.weight")
	lg := s.Prefill(denseDecodeGraphPrompt(cfg, 6))
	lg = s.Step(argmaxF(lg))
	st := s.denseDecode
	if r := s.DenseQ4KDecodeGraphReceipt(); r.Accepted || r.DeclineReason != denseDeclineHead {
		t.Fatalf("f32-head receipt %+v", r)
	}
	if st == nil || st.notReadyModel != s.M || st.notReadyReason != denseDeclineHead || st.readyModel != nil {
		t.Fatalf("not-ready verdict not cached: %+v", st)
	}
	// A cached verdict is served as-is: poison it and the next token must report it.
	st.notReadyReason = denseDeclineTensors
	s.Step(argmaxF(lg))
	if r := s.DenseQ4KDecodeGraphReceipt(); r.DeclineReason != denseDeclineTensors || r.DeclineCounts[denseDeclineHead] != 1 {
		t.Fatalf("second token re-scanned the model: receipt %+v", r)
	}
}
