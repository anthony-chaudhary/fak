package model

import (
	"math"
	"testing"
	"time"
)

func qwen35HybridTestCfg() Config {
	return Config{
		HiddenSize:            32,
		NumLayers:             4,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		IntermediateSize:      64,
		VocabSize:             97,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      8,
		LinearNumKeyHeads:     2,
		LinearValueHeadDim:    8,
		LinearNumValueHeads:   4,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
}

func TestQwen35HybridSessionMatchesForwardAndPersistsState(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	prompt := []int{3, 7, 11, 5, 17, 19, 23}

	forward := m.Forward(prompt).Logits[len(prompt)-1]
	prefill := m.NewSession().Prefill(prompt)
	if d := maxAbsDelta(forward, prefill); d > 2e-5 {
		t.Fatalf("hybrid Prefill differs from cacheless Forward, max|delta|=%g", d)
	}

	full := m.NewSession().Prefill(prompt)
	split := m.NewSession()
	split.Prefill(prompt[:3])
	var splitLogits []float32
	for _, id := range prompt[3:] {
		splitLogits = split.Step(id)
	}
	assertFloat32BitsEqual(t, "hybrid split prefill/decode", full, splitLogits)

	base := m.NewSession()
	base.Prefill(prompt[:5])
	reuse := m.SessionFromPrefix(base.Cache)
	reuseLogits := reuse.Step(prompt[5])
	recompute := m.NewSession().Prefill(prompt[:6])
	assertFloat32BitsEqual(t, "hybrid cloned recurrent prefix", recompute, reuseLogits)
}

func TestQwen35HybridQuantTokenLoopPersistsState(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	prompt := []int{3, 7, 11, 5, 17, 19, 23}

	full := m.NewSession()
	full.Quant = true
	fullLogits := full.Prefill(prompt)

	split := m.NewSession()
	split.Quant = true
	split.Prefill(prompt[:4])
	var splitLogits []float32
	for _, id := range prompt[4:] {
		splitLogits = split.Step(id)
	}
	assertFloat32BitsEqual(t, "hybrid q8 split prefill/decode", fullLogits, splitLogits)
	for i, v := range splitLogits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("hybrid q8 logit[%d] = %v", i, v)
		}
	}
}

func TestQwen35HybridQuantBatchedPrefillMatchesTokenLoop(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	ref := m.NewSession()
	ref.Quant = true
	var refHidden []float32
	for _, id := range prompt {
		refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}
	want := ref.headQ(refHidden)

	gotSession := m.NewSession()
	gotSession.Quant = true
	got := gotSession.Prefill(prompt)

	assertQuantLogitsClose(t, "hybrid q8 batched prefill logits", want, got)
	assertKVCacheQuantClose(t, "hybrid q8 batched prefill", ref.Cache, gotSession.Cache)
	assertLinearAttnCacheQuantClose(t, "hybrid q8 batched prefill", ref.Cache.linear, gotSession.Cache.linear)
}

func TestQwen35HybridQuantBatchedPrefillNoLogitsMatchesPrefillState(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	want := m.NewSession()
	want.Quant = true
	if logits := want.Prefill(prompt); len(logits) != m.Cfg.VocabSize {
		t.Fatalf("Prefill logits len = %d, want %d", len(logits), m.Cfg.VocabSize)
	}

	got := m.NewSession()
	got.Quant = true
	got.PrefillNoLogits(prompt)

	assertKVCacheQuantClose(t, "hybrid q8 batched no-logits prefill", want.Cache, got.Cache)
	assertLinearAttnCacheQuantClose(t, "hybrid q8 batched no-logits prefill", want.Cache.linear, got.Cache.linear)
}

func TestQwen35HybridQPhaseProfilerRecordsPrefillAndDecode(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	s := m.NewSession()
	s.Quant = true
	pp := NewPhaseProfiler()
	s.PhaseProfiler = pp
	t0 := time.Now()
	logits := s.Prefill(prompt)
	prefill := pp.Snapshot("prefill", len(prompt), 0, time.Since(t0).Nanoseconds())
	if len(logits) != m.Cfg.VocabSize {
		t.Fatalf("Prefill logits len = %d, want %d", len(logits), m.Cfg.VocabSize)
	}
	for _, phase := range []string{
		"q8_panel_quantize",
		"qwen35_linear_recurrent",
		"qwen35_full_attn",
		"mlp_gate_up_proj",
		"lm_head_q8",
	} {
		if phaseCalls(prefill, phase) == 0 {
			t.Fatalf("prefill phase %q was not recorded: %#v", phase, prefill.Phases)
		}
	}

	pp = NewPhaseProfiler()
	s.PhaseProfiler = pp
	t0 = time.Now()
	logits = s.Step(67)
	decode := pp.Snapshot("decode", len(prompt), 1, time.Since(t0).Nanoseconds())
	if len(logits) != m.Cfg.VocabSize {
		t.Fatalf("Step logits len = %d, want %d", len(logits), m.Cfg.VocabSize)
	}
	for _, phase := range []string{
		"qwen35_linear_step_recurrent",
		"full_attn_decode",
		"mlp_decode",
		"lm_head_q8",
	} {
		if phaseCalls(decode, phase) == 0 {
			t.Fatalf("decode phase %q was not recorded: %#v", phase, decode.Phases)
		}
	}
}

func TestLinearAttnConvWindowReusesFullBuffer(t *testing.T) {
	var st linearAttnLayerState
	r1 := []float32{1, 2, 3}
	r2 := []float32{4, 5, 6}
	r3 := []float32{7, 8, 9}

	st.pushConvRow(r1, 2)
	st.pushConvRow(r2, 2)
	firstBuf := &st.conv[0][0]
	secondBuf := &st.conv[1][0]

	st.pushConvRow(r3, 2)
	if len(st.conv) != 2 {
		t.Fatalf("conv len=%d, want 2", len(st.conv))
	}
	if &st.conv[0][0] != secondBuf {
		t.Fatal("conv window did not retain the newer history row")
	}
	if &st.conv[1][0] != firstBuf {
		t.Fatal("conv window did not reuse the evicted row buffer")
	}
	want := [][]float32{r2, r3}
	for i := range want {
		for j := range want[i] {
			if st.conv[i][j] != want[i][j] {
				t.Fatalf("conv[%d][%d]=%v, want %v", i, j, st.conv[i][j], want[i][j])
			}
		}
	}
	r3[0] = 99
	if st.conv[1][0] != 7 {
		t.Fatal("conv row aliases caller buffer")
	}
}

func phaseCalls(p *PhaseProfile, phase string) int {
	for _, st := range p.Phases {
		if st.Phase == phase {
			return st.Calls
		}
	}
	return 0
}

func assertLinearAttnCacheQuantClose(t *testing.T, label string, want, got *linearAttnCache) {
	t.Helper()
	if want == nil || got == nil {
		if want != got {
			t.Fatalf("%s linear cache nil mismatch", label)
		}
		return
	}
	if len(want.layers) != len(got.layers) {
		t.Fatalf("%s linear cache layers = %d, want %d", label, len(got.layers), len(want.layers))
	}
	for l := range want.layers {
		wl, gl := want.layers[l], got.layers[l]
		if len(wl.recurrent) != len(gl.recurrent) {
			t.Fatalf("%s layer %d recurrent heads = %d, want %d", label, l, len(gl.recurrent), len(wl.recurrent))
		}
		for h := range wl.recurrent {
			assertMaxAbsAtMost(t, label+" layer "+itoa(l)+" recurrent head "+itoa(h), wl.recurrent[h], gl.recurrent[h], 1e-5)
		}
		if len(wl.conv) != len(gl.conv) {
			t.Fatalf("%s layer %d conv rows = %d, want %d", label, l, len(gl.conv), len(wl.conv))
		}
		for i := range wl.conv {
			assertMaxAbsAtMost(t, label+" layer "+itoa(l)+" conv row "+itoa(i), wl.conv[i], gl.conv[i], 1e-5)
		}
	}
}
