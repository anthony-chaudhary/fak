package model

import "testing"

// prefillBatchedMoETestConfig is a Qwen3-MoE-shaped synthetic config (routed experts,
// QK-norm, GQA, PreNorm, shared RoPE theta): the arch prefillBatchedMoEOK admits.
func prefillBatchedMoETestConfig() Config {
	cfg := llamaArchConfig()
	cfg.NumExperts = 5
	cfg.NumExpertsPerTok = 2
	cfg.NormTopKProb = true
	cfg.IntermediateSize = 24
	cfg.QKNorm = true
	return cfg
}

// prefillBatchedQMoETestConfig is the same arch with a Q8_0-blockable expert width:
// Quantize needs every reduction dim (the expert down_proj's is IntermediateSize) to be a
// multiple of 32, and 64 keeps it distinct from HiddenSize.
func prefillBatchedQMoETestConfig() Config {
	cfg := prefillBatchedMoETestConfig()
	cfg.IntermediateSize = 64
	return cfg
}

func prefillBatchedMoEPrompt(cfg Config, n int) []int {
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = (i*1299709 + 17) % cfg.VocabSize
	}
	return prompt
}

func assertPrefillCachesBitEqual(t *testing.T, cfg Config, ref, bat *Session) {
	t.Helper()
	for l := 0; l < cfg.NumLayers; l++ {
		for name, pair := range map[string][2][]float32{
			"K":    {ref.Cache.K[l], bat.Cache.K[l]},
			"Kraw": {ref.Cache.Kraw[l], bat.Cache.Kraw[l]},
			"V":    {ref.Cache.V[l], bat.Cache.V[l]},
		} {
			if !float32BitsEqual(pair[0], pair[1]) {
				t.Fatalf("layer %d %s cache differs between token loop and batched lane", l, name)
			}
		}
	}
}

// TestPrefillBatchedMoEMatchesTokenLoop pins the admitted MoE subset to the per-token
// loop bit-for-bit: logits and every layer's K/Kraw/V, including a second prefill that
// appends at a non-zero base. Routing differs per row, so a lane that shared one row's
// picks across the panel would diverge here. [SW-VERIFIED]
// fak-test:runtime fast est=50ms lane=default
func TestPrefillBatchedMoEMatchesTokenLoop(t *testing.T) {
	cfg := prefillBatchedMoETestConfig()
	m := NewSyntheticMoE(cfg)
	if !prefillBatchedMoEOK(m) || prefillNeedsTokenLoop(m) {
		t.Fatal("premise broken: the synthetic MoE config is not admitted to prefillBatched")
	}
	if !q8PrefillNeedsTokenLoop(m.Cfg) {
		t.Fatal("q8PrefillNeedsTokenLoop must still refuse MoE for the Q4K/verify/HAL lanes")
	}

	first := prefillBatchedMoEPrompt(cfg, 11)
	second := []int{4, 60, 9, 33}

	ref := m.NewSession()
	ref.prefillTokenLoop(first)
	refLogits := ref.prefillTokenLoop(second)

	bat := m.NewSession()
	bat.head(bat.prefillBatched(first))
	batLogits := bat.head(bat.prefillBatched(second))

	if !float32BitsEqual(refLogits, batLogits) {
		t.Fatal("batched MoE prefill logits differ from the token loop")
	}
	assertPrefillCachesBitEqual(t, cfg, ref, bat)

	// Non-vacuity proxy: the router must split the prompt's rows across different expert
	// sets (probed on the embedding rows), or a lane that reused one row's route could
	// pass the parity check above.
	distinct := false
	for l := 0; l < cfg.NumLayers && !distinct; l++ {
		var firstPick []int
		for t0, id := range first {
			x := append([]float32(nil), m.embedRows()[id*cfg.HiddenSize:(id+1)*cfg.HiddenSize]...)
			picks := route(m, l, x, f32Kernel{m})
			ex := make([]int, len(picks))
			for i, p := range picks {
				ex[i] = p.expert
			}
			if t0 == 0 {
				firstPick = ex
				continue
			}
			for i := range ex {
				if ex[i] != firstPick[i] {
					distinct = true
				}
			}
		}
	}
	if !distinct {
		t.Fatal("every prompt row routes to the same experts: the parity check is vacuous")
	}
}

// TestPrefillRoutesAdmittedMoEToBatchedLane proves Prefill actually takes prefillBatched
// for the admitted subset: the token loop's blockStep records the full_attn_decode
// phase once per (token, layer), the batched lane never does. [SW-VERIFIED]
// fak-test:runtime fast est=50ms lane=default
func TestPrefillRoutesAdmittedMoEToBatchedLane(t *testing.T) {
	cfg := prefillBatchedMoETestConfig()
	m := NewSyntheticMoE(cfg)
	prompt := prefillBatchedMoEPrompt(cfg, 7)

	s := m.NewSession()
	s.PhaseProfiler = NewPhaseProfiler()
	got := s.Prefill(prompt)
	if st := s.PhaseProfiler.stat["full_attn_decode"]; st != nil {
		t.Fatalf("admitted MoE ran blockStep attention %d times, want the batched lane", st.Calls)
	}
	ref := m.NewSession()
	if !float32BitsEqual(got, ref.prefillTokenLoop(prompt)) {
		t.Fatal("Prefill logits on the batched MoE lane differ from the token loop")
	}
}

// TestPrefillMoEExcludedAxesKeepTokenLoop pins the admission boundary: an MoE config that
// also carries an axis prefillBatched does not model keeps the per-token loop, observed
// through Prefill itself (blockStep attention runs once per token and layer). [SW-VERIFIED]
// fak-test:runtime fast est=100ms lane=default
func TestPrefillMoEExcludedAxesKeepTokenLoop(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Config)
	}{
		{"attn-output-gate", func(c *Config) { c.AttnOutputGate = true }},
		{"sandwich-norm", func(c *Config) { c.BlockTopology = SandwichNorm }},
		{"gpt-oss", func(c *Config) { c.Architectures = []string{"GptOssForCausalLM"}; c.ModelType = "gpt_oss" }},
		{"layer-rope-theta", func(c *Config) { c.RopeThetaPerLayer = []float64{10000, 500000} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := prefillBatchedMoETestConfig()
			tc.mutate(&cfg)
			m := NewSyntheticMoE(cfg)
			if prefillBatchedMoEOK(m) || !prefillNeedsTokenLoop(m) {
				t.Fatal("excluded MoE axis was admitted to prefillBatched")
			}
			if tc.name == "attn-output-gate" || tc.name == "gpt-oss" {
				// The gated q_proj is 2x wide and GPT-OSS experts need their own tensors;
				// the synthetic MoE builder emits neither, so assert the predicate only.
				return
			}
			prompt := prefillBatchedMoEPrompt(cfg, 5)
			s := m.NewSession()
			s.PhaseProfiler = NewPhaseProfiler()
			s.Prefill(prompt)
			st := s.PhaseProfiler.stat["full_attn_decode"]
			if st == nil || st.Calls != len(prompt)*cfg.NumLayers {
				t.Fatalf("excluded MoE axis did not run the token loop (full_attn_decode=%v)", st)
			}
		})
	}

	// An enabled residual hook runs only inside blockStep, so an admitted MoE session with
	// one keeps the token loop (and the hook still fires once per token and layer).
	hookCfg := prefillBatchedMoETestConfig()
	hookCfg.EnableResidualHook = true
	hm := NewSyntheticMoE(hookCfg)
	calls := 0
	hm.SetResidualHook(func(int, []float32) { calls++ })
	hookPrompt := prefillBatchedMoEPrompt(hookCfg, 5)
	hm.NewSession().Prefill(hookPrompt)
	if calls != len(hookPrompt)*hookCfg.NumLayers {
		t.Fatalf("residual hook fired %d times, want %d (MoE session with a hook left the token loop)",
			calls, len(hookPrompt)*hookCfg.NumLayers)
	}

	// A layer carrying attention sinks under a non-GPT-OSS arch name is refused by tensor
	// presence, since prefillBatched's softmax applies no sink.
	cfg := prefillBatchedMoETestConfig()
	m := NewSyntheticMoE(cfg)
	addSyntheticTensor(m, layerName(1, "self_attn.sinks"), cfg.NumHeads)
	if prefillBatchedMoEOK(m) {
		t.Fatal("MoE layer with self_attn.sinks was admitted to prefillBatched")
	}
}

// addSyntheticTensor appends one zero-valued f32 tensor to a synthetic model manifest.
func addSyntheticTensor(m *Model, name string, n int) {
	off := len(m.raw)
	m.raw = append(m.raw, make([]byte, n*4)...)
	m.manifest[name] = tensorMeta{Dtype: "F32", Shape: []int{n}, Offset: off, Nbytes: n * 4}
}

// TestPrefillBatchedQMoEMatchesTokenLoop pins the Q8 lane (prefillBatchedQ) for the
// admitted MoE subset to its per-token loop (tokenHiddenQ, whose MoE blockStep runs on
// q8Kernel). Under FAK_QGEMM=legacy, the same mode TestQ8PrefillAppliesAttentionSoftcap
// pins the dense Q8 lane in, it is bit-for-bit: logits and every layer's K/Kraw/V across
// two appended prefills. In the default mode the tiled qGemm8 batched GEMM accumulates
// its Q8_0 block partials in a different float order than the qMatRows GEMV (~1 ulp per
// output, the reduction-order shift the Q4K lane's parity test also tolerates), so there
// the gate is argmax-equal logits with cosine >= 0.99999. [SW-VERIFIED]
// fak-test:runtime fast est=100ms lane=default
func TestPrefillBatchedQMoEMatchesTokenLoop(t *testing.T) {
	oldMode := qgemmMode
	qgemmMode = qgemmModeLegacy
	defer func() { qgemmMode = oldMode }()

	cfg := prefillBatchedQMoETestConfig()
	m := NewSyntheticMoE(cfg)
	m.Quantize()
	if m.q8layers != nil {
		t.Fatal("premise broken: an MoE model built the dense q8layers cache")
	}

	first := prefillBatchedMoEPrompt(cfg, 11)
	second := []int{4, 60, 9, 33}

	ref := m.NewSession()
	ref.Quant = true
	ref.tokenLoopHidden(ref.tokenHiddenQ, first)
	refLogits := ref.headQ(ref.tokenLoopHidden(ref.tokenHiddenQ, second))

	bat := m.NewSession()
	bat.Quant = true
	bat.prefillBatchedQ(first)
	batLogits := bat.headQ(bat.prefillBatchedQ(second))

	if !float32BitsEqual(refLogits, batLogits) {
		d, i := maxAbsDiff(refLogits, batLogits)
		t.Fatalf("batched Q8 MoE prefill logits differ from the token loop: max abs diff %.3e at %d", d, i)
	}
	assertPrefillCachesBitEqual(t, cfg, ref, bat)

	qgemmMode = oldMode
	tiled := m.NewSession()
	tiled.Quant = true
	tiled.prefillBatchedQ(first)
	tiledLogits := tiled.headQ(tiled.prefillBatchedQ(second))
	if c := cosine(refLogits, tiledLogits); c < 0.99999 || argmaxF32(refLogits) != argmaxF32(tiledLogits) {
		d, _ := maxAbsDiff(refLogits, tiledLogits)
		t.Fatalf("default-mode batched Q8 MoE prefill drifted from the token loop: cosine %.7f, max abs %.3e", c, d)
	}
}

// TestPrefillQRoutesAdmittedMoEToBatchedLane proves quantized Prefill and PrefillNoLogits
// take prefillBatchedQ for the admitted subset (no per-token blockStep attention) and that
// a target-hidden capture session keeps the token loop, since only tokenHiddenQ records
// the per-position hidden rows. [SW-VERIFIED]
// fak-test:runtime fast est=100ms lane=default
func TestPrefillQRoutesAdmittedMoEToBatchedLane(t *testing.T) {
	oldMode := qgemmMode
	qgemmMode = qgemmModeLegacy // bit-exact mode; see TestPrefillBatchedQMoEMatchesTokenLoop
	defer func() { qgemmMode = oldMode }()

	cfg := prefillBatchedQMoETestConfig()
	m := NewSyntheticMoE(cfg)
	m.Quantize()
	prompt := prefillBatchedMoEPrompt(cfg, 7)

	s := m.NewSession()
	s.Quant = true
	s.PhaseProfiler = NewPhaseProfiler()
	got := s.Prefill(prompt)
	if st := s.PhaseProfiler.stat["full_attn_decode"]; st != nil {
		t.Fatalf("admitted Q8 MoE ran blockStep attention %d times, want prefillBatchedQ", st.Calls)
	}
	ref := m.NewSession()
	ref.Quant = true
	if want := ref.headQ(ref.tokenLoopHidden(ref.tokenHiddenQ, prompt)); !float32BitsEqual(got, want) {
		t.Fatal("quantized Prefill logits on the batched MoE lane differ from the token loop")
	}

	nl := m.NewSession()
	nl.Quant = true
	nl.PhaseProfiler = NewPhaseProfiler()
	nl.PrefillNoLogits(prompt)
	if st := nl.PhaseProfiler.stat["full_attn_decode"]; st != nil {
		t.Fatalf("admitted Q8 MoE PrefillNoLogits ran blockStep attention %d times", st.Calls)
	}
	assertPrefillCachesBitEqual(t, cfg, ref, nl)

	for _, quant := range []bool{false, true} {
		c := m.NewSession()
		c.Quant = quant
		c.captureTargetHidden = true
		c.PhaseProfiler = NewPhaseProfiler()
		c.Prefill(prompt)
		if st := c.PhaseProfiler.stat["full_attn_decode"]; st == nil || st.Calls != len(prompt)*cfg.NumLayers {
			t.Fatalf("target-hidden capture session left the token loop (quant=%v, full_attn_decode=%v)", quant, st)
		}
	}
}
