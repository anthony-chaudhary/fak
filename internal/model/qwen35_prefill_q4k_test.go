package model

import "testing"

// qwen35HybridQ4KTestCfg is the qwen35 hybrid test config with every projection reduction
// dim a multiple of qkK=256, the Q4_K precondition (quantizeQ4KFromRaw panics otherwise).
// H=256 (in for q/k/v/in_proj_*/gate/up), nH*hd=256 (in for self_attn.o_proj),
// valDim=nV*vHd=256 (in for linear_attn.out_proj), I=256 (in for down_proj). VocabSize stays a small multiple of 256 so a
// quantized tied head is also Q4_K-legal. Same topology as qwen35HybridTestCfg (3 linear +
// 1 full attention, AttnOutputGate, NormGain1p), so q8Qwen35HybridPrefillOK admits it.
func qwen35HybridQ4KTestCfg() Config {
	return Config{
		HiddenSize:            256,
		NumLayers:             4,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               64,
		IntermediateSize:      256,
		VocabSize:             512,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      64,
		LinearNumKeyHeads:     2,
		LinearValueHeadDim:    64,
		LinearNumValueHeads:   4,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
}

// fillQ4KMajority populates m.q4kw for exactly the projections ResidentQ4KEligible holds raw
// (the identity-normalized matmul weights: self_attn v_proj/o_proj on full-attn layers, and
// mlp gate/up/down on every layer). The q/k projections and the whole linear_attn.* family
// are reordered/unpermuted for qwen35 → left out of q4kw, so they resolve through m.q8 (Q8),
// exactly as sessionQ4KKernel and the batched proj dispatch them in the real 27B model. This
// makes the test exercise BOTH dispatch branches the way the real model does, not an all-q4k
// shortcut.
func fillQ4KMajority(t *testing.T, m *Model, cfg Config) {
	t.Helper()
	projs := [][2]any{}
	nKVhd := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		projs = append(projs,
			[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.down_proj.weight", cfg.HiddenSize},
		)
		if !cfg.isLinearAttnLayer(l) {
			projs = append(projs,
				[2]any{p + "self_attn.v_proj.weight", nKVhd},
				[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			)
		}
	}
	fillQ4KW(t, m, projs, 99)
}

// TestPrefillQwen35HybridQ4KMatchesTokenLoop is the load-bearing correctness gate for the
// batched resident-Q4_K hybrid prefill (QWEN36-NATIVE-PERF-PLAN P3). It proves Prefill's
// batched path (prefillQwen35HybridQ4K, reached via q4kQwen35HybridPrefillOK) produces the
// same logits + KV/linear cache as the proven per-token Q4K decode loop (tokenHiddenQ via
// sessionQ4KKernel) — the two share the identical f32 recurrence and the identical per-weight
// kernel dispatch, so the q4_k_m majority is bit-identical (q4kGemm == q4kMatRows per (o,t),
// TestQ4KGemmMatchesMatRows) and the only residual is the Q8 minority (self_attn q/k and the
// whole linear_attn.* family, which ResidentQ4KEligible keeps out of q4kw).
//
// That Q8-minority residual is the documented decode-vs-prefill reduction difference, NOT a
// wiring bug: the per-token decode GEMV (qdot8GEMV) and the batched prefill GEMM
// (qgemm8tile, a register-blocked deferred reduction) sum each dot in a different float order,
// which drifts ~1e-6 per dot exactly as parallel.go documents for fdot. Witnessed directly:
// under FAK_QGEMM=legacy (the prefill GEMM falls back to the same per-element qdot8 the decode
// GEMV uses) every K/Kraw/V layer is BIT-IDENTICAL (max|Δ|=0) — so the drift is entirely that
// kernel choice, with zero contribution from the Q4_K V/o/gate/up/down majority or the
// recurrence. The drift lands ~10× harder on V than on K/Kraw because K/Kraw are squeezed
// through the contractive per-head RMSNorm of applyLayerQKNorm while V is cached raw: measured
// at this synthetic seed, K/Kraw stay ≈4e-6 but V reaches ~1.2e-4 (mean 5e-6; ~5e-6 RELATIVE
// to its |V|≈22.6 outliers), so V gets the 2e-4 bound below and everything else stays at the
// strict 1e-5. A real wiring bug (wrong store, q4k/q8 mismatch, a recurrence mis-copy) would
// diverge O(1) per layer, survive the legacy fallback, and blow past these bounds by orders of
// magnitude.
func TestPrefillQwen35HybridQ4KMatchesTokenLoop(t *testing.T) {
	// Force the f32 decode GEMV for the per-token reference so it compares against the f32
	// batched q4kGemm — the q4_k majority is then bit-identical and the only drift is the Q8
	// minority's FMA rounding. The int8 SDOT decode path's activation-quant band (gated
	// separately by TestQ4KInt8DotMatchesF32) is a decode-vs-prefill phase difference, not a
	// dispatch bug; production decode stays int8. Same scoping knob TestPrefillBatchedQ4KMatchesSerial uses.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize() // build q8w for the q/k + linear_attn.* minority
	fillQ4KMajority(t, m, cfg)
	// 16 tokens: meets qwen35HybridQBatchMinPrompt so Prefill takes the batched path.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	ref := m.NewSession()
	ref.Q4K = true
	var refHidden []float32
	for _, id := range prompt {
		refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}
	want := ref.headResident(refHidden)

	got := m.NewSession()
	got.Q4K = true
	gotLogits := got.Prefill(prompt)

	assertQuantLogitsClose(t, "hybrid q4k batched prefill logits", want, gotLogits)
	// K/Kraw stay at the strict 1e-5 (their RMSNorm damps the propagated drift); V takes a
	// 2e-4 bound for the documented Q8-minority deferred-reduction drift carried raw into the
	// V cache (see this test's doc comment for the legacy-fallback bit-identity witness).
	assertKVCacheQuantCloseTol(t, "hybrid q4k batched prefill", ref.Cache, got.Cache, 1e-5, 2e-4)
	assertLinearAttnCacheQuantClose(t, "hybrid q4k batched prefill", ref.Cache.linear, got.Cache.linear)
}

// TestPrefillQwen35HybridQ4KNoLogitsMatchesState proves PrefillNoLogits advances the same
// KV/linear state as the logits-producing Prefill on the batched Q4K hybrid path (the
// teacher-forced context-growth entry must build a cache identical to the generating one).
func TestPrefillQwen35HybridQ4KNoLogitsMatchesState(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	want := m.NewSession()
	want.Q4K = true
	if logits := want.Prefill(prompt); len(logits) != cfg.VocabSize {
		t.Fatalf("Prefill logits len = %d, want %d", len(logits), cfg.VocabSize)
	}

	got := m.NewSession()
	got.Q4K = true
	got.PrefillNoLogits(prompt)

	assertKVCacheQuantClose(t, "hybrid q4k no-logits prefill", want.Cache, got.Cache)
	assertLinearAttnCacheQuantClose(t, "hybrid q4k no-logits prefill", want.Cache.linear, got.Cache.linear)
}

// TestPrefillQwen35HybridQ4KDeterministic confirms the batched Q4_K hybrid prefill is
// reproducible: identical prompt → bit-identical logits + recurrent state across runs. This
// catches a non-deterministic parallel-reduction bug (e.g. a per-worker accumulator escaping
// its head range in the GDN scan) that a single-run parity test would miss.
func TestPrefillQwen35HybridQ4KDeterministic(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	s1 := m.NewSession()
	s1.Q4K = true
	l1 := s1.Prefill(prompt)
	s2 := m.NewSession()
	s2.Q4K = true
	l2 := s2.Prefill(prompt)

	assertFloat32BitsEqual(t, "hybrid q4k prefill determinism", l1, l2)
	assertLinearAttnCacheQuantClose(t, "hybrid q4k prefill determinism", s1.Cache.linear, s2.Cache.linear)
}
