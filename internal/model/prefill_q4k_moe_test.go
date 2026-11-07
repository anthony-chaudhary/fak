package model

import "testing"

// q4kMoETestModel is a Qwen3-MoE-shaped synthetic model with every reduction dim a
// multiple of Q4_K's 256 super-block. Attention and expert gate/up projections are
// resident Q4_K; router and expert down_proj stay off q4kw, so sessionQ4KKernel serves
// them through its Q8 minority fallback, as on a q4_k_m GGUF.
func q4kMoETestModel(t *testing.T, mutate func(*Config)) *Model {
	t.Helper()
	cfg := Config{
		HiddenSize: 256, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, RopeTheta: 10000.0,
		NumExperts: 4, NumExpertsPerTok: 2, NormTopKProb: true, QKNorm: true,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	m := NewSyntheticMoE(cfg)
	var projs [][2]any
	nHhd, nKVhd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		projs = append(projs,
			[2]any{p + "self_attn.q_proj.weight", nHhd},
			[2]any{p + "self_attn.k_proj.weight", nKVhd},
			[2]any{p + "self_attn.v_proj.weight", nKVhd},
			[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
		)
		for e := 0; e < cfg.NumExperts; e++ {
			projs = append(projs,
				[2]any{expertName(l, e, "gate_proj.weight"), cfg.IntermediateSize},
				[2]any{expertName(l, e, "up_proj.weight"), cfg.IntermediateSize},
			)
		}
	}
	fillQ4KW(t, m, projs, 7)
	return m
}

func q4kMoEPrompt(n, vocab int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = (i*37 + 5) % vocab
	}
	return p
}

// TestPrefillBatchedQ4KMoEMatchesTokenLoop pins the resident-Q4_K lane
// (prefillBatchedQ4K) for the admitted MoE subset to its per-token loop (tokenHiddenQ,
// whose blockStep runs on sessionQ4KKernel) across two appended prefills. Each row's
// routed FFN runs through prefillRowFFN on that same kernel, so the FFN arithmetic is the
// token loop's own. The batched attention projections are q4kGemm/qGemm8 panels and the
// scores use the panel reduction, which differ from the per-token GEMVs by reduction order
// only, so this holds the tolerance TestPrefillBatchedQ4KMatchesSerial holds for the dense
// lane (hidden cosine >= 0.999, K/Kraw/V max-abs/RMS <= 2e-3), tightened to cosine >= 0.9999.
// [SW-VERIFIED]
// fak-test:runtime fast est=300ms lane=default
func TestPrefillBatchedQ4KMoEMatchesTokenLoop(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	m := q4kMoETestModel(t, nil)
	cfg := m.Cfg
	if !prefillBatchedMoEOK(m) {
		t.Fatal("premise broken: the synthetic Q4_K MoE config is not admitted")
	}
	first := q4kMoEPrompt(12, cfg.VocabSize)
	second := []int{4, 60, 9, 33, 21}

	ref := m.NewSession()
	ref.Q4K = true
	if _, ok := ref.tokenHiddenQKernel().(sessionQ4KKernel); !ok {
		t.Fatal("premise broken: the Q4_K MoE token loop does not run sessionQ4KKernel")
	}
	ref.tokenLoopHidden(ref.tokenHiddenQ, first)
	refHidden := ref.tokenLoopHidden(ref.tokenHiddenQ, second)

	bat := m.NewSession()
	bat.Q4K = true
	bat.prefillBatchedQ4K(first)
	batHidden := bat.prefillBatchedQ4K(second)

	if c := cosine(refHidden, batHidden); c < 0.9999 {
		d, _ := maxAbsDiff(refHidden, batHidden)
		t.Fatalf("batched Q4_K MoE last hidden drifted from the token loop: cosine %.7f, max abs %.3e", c, d)
	}
	d, _ := maxAbsDiff(refHidden, batHidden)
	t.Logf("last-hidden cosine=%.9f max-abs=%.3e", cosine(refHidden, batHidden), d)
	if bat.Cache.Len() != ref.Cache.Len() {
		t.Fatalf("cache len %d != %d", bat.Cache.Len(), ref.Cache.Len())
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for name, pair := range map[string][2][]float32{
			"K": {ref.Cache.K[l], bat.Cache.K[l]}, "Kraw": {ref.Cache.Kraw[l], bat.Cache.Kraw[l]}, "V": {ref.Cache.V[l], bat.Cache.V[l]},
		} {
			if c := cosine(pair[0], pair[1]); c < 0.9999 {
				t.Fatalf("layer %d %s cache cosine %.7f < 0.9999", l, name, c)
			}
		}
	}
}

// TestPrefillQ4KRoutesAdmittedMoEToBatchedLane proves resident-Q4_K Prefill and
// PrefillNoLogits take prefillBatchedQ4K for the admitted subset (no per-token blockStep
// attention), and that an excluded MoE axis keeps the token loop. [SW-VERIFIED]
// fak-test:runtime fast est=300ms lane=default
func TestPrefillQ4KRoutesAdmittedMoEToBatchedLane(t *testing.T) {
	m := q4kMoETestModel(t, nil)
	prompt := q4kMoEPrompt(6, m.Cfg.VocabSize)
	for _, noLogits := range []bool{false, true} {
		s := m.NewSession()
		s.Q4K = true
		s.PhaseProfiler = NewPhaseProfiler()
		if noLogits {
			s.PrefillNoLogits(prompt)
		} else {
			s.Prefill(prompt)
		}
		if st := s.PhaseProfiler.stat["full_attn_decode"]; st != nil {
			t.Fatalf("admitted Q4_K MoE (noLogits=%v) ran blockStep attention %d times", noLogits, st.Calls)
		}
	}

	for _, tc := range []struct {
		name   string
		mutate func(*Config)
		setup  func(*Session)
	}{
		{"sandwich-norm", func(c *Config) { c.BlockTopology = SandwichNorm }, func(*Session) {}},
		{"target-hidden-capture", nil, func(s *Session) { s.captureTargetHidden = true }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			em := q4kMoETestModel(t, tc.mutate)
			s := em.NewSession()
			s.Q4K = true
			tc.setup(s)
			s.PhaseProfiler = NewPhaseProfiler()
			s.Prefill(prompt)
			if st := s.PhaseProfiler.stat["full_attn_decode"]; st == nil || st.Calls != len(prompt)*em.Cfg.NumLayers {
				t.Fatalf("excluded Q4_K MoE session left the token loop (full_attn_decode=%v)", st)
			}
		})
	}
}
