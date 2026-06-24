package model

import (
	"errors"
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

// TestQwen35HybridEvictReturnsTypedUnsupportedVerdictFailClosed pins the #444 boundary:
// a hybrid Gated-DeltaNet session's KVCache.Evict / TryEvict / CanEvict surface a TYPED
// unsupported verdict (RecurrentEvictUnsupportedError naming the recurrent layers) instead
// of the old opaque package-internal panic, and the verdict fails CLOSED — the cache is
// left byte-for-byte unchanged, so the partial softmax-KV deletion never happens silently.
// Non-vacuous: it proves the verdict path AND that no state was mutated, AND it confirms an
// ordinary (non-recurrent) cache still reports nil and evicts normally.
func TestQwen35HybridEvictReturnsTypedUnsupportedVerdictFailClosed(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11}
	poison := []int{5, 17}
	query := []int{19, 23}

	s := m.NewSession()
	s.Prefill(prefix)
	s.Prefill(poison)
	s.Prefill(query)

	wantLayers := s.Cache.linear.recurrentLayers()
	if len(wantLayers) == 0 {
		t.Fatal("synthetic hybrid cache has no recurrent layers; test would be vacuous")
	}

	// Snapshot the full cache state BEFORE the eviction attempt so we can prove fail-closed.
	snap := s.Cache.Clone()

	// CanEvict is the witnessable predicate: a typed verdict naming the recurrent layers.
	canErr := s.Cache.CanEvict()
	var verdict *RecurrentEvictUnsupportedError
	if !errors.As(canErr, &verdict) {
		t.Fatalf("CanEvict() = %v, want *RecurrentEvictUnsupportedError", canErr)
	}
	if !eq(verdict.Layers, wantLayers) {
		t.Fatalf("verdict.Layers = %v, want %v", verdict.Layers, wantLayers)
	}

	// TryEvict surfaces the SAME typed verdict, removes nothing, and mutates nothing.
	removed, err := s.Cache.TryEvict(len(prefix), len(poison))
	if removed != 0 {
		t.Fatalf("TryEvict removed = %d, want 0 (fail-closed)", removed)
	}
	var tryVerdict *RecurrentEvictUnsupportedError
	if !errors.As(err, &tryVerdict) {
		t.Fatalf("TryEvict err = %v, want *RecurrentEvictUnsupportedError", err)
	}
	assertKVCacheUnchanged(t, "TryEvict fail-closed", snap, s.Cache)

	// The convenience Evict wrapper panics the SAME typed value for an unchecked caller.
	func() {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("Evict on hybrid cache did not panic")
			}
			perr, ok := r.(*RecurrentEvictUnsupportedError)
			if !ok {
				t.Fatalf("Evict panic = %T (%v), want *RecurrentEvictUnsupportedError", r, r)
			}
			if !eq(perr.Layers, wantLayers) {
				t.Fatalf("panic verdict.Layers = %v, want %v", perr.Layers, wantLayers)
			}
		}()
		s.Cache.Evict(len(prefix), len(poison))
	}()
	assertKVCacheUnchanged(t, "Evict panic fail-closed", snap, s.Cache)

	// An ordinary softmax-KV cache (no recurrent state) reports nil and evicts normally —
	// the #444 boundary is specific to the hybrid path, softmax-KV eviction stays witnessed.
	denseCfg := Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
	}
	dense := NewSynthetic(denseCfg).NewSession()
	dense.Prefill([]int{1, 2, 3, 4, 5})
	if err := dense.Cache.CanEvict(); err != nil {
		t.Fatalf("dense CanEvict() = %v, want nil", err)
	}
	if removed, err := dense.Cache.TryEvict(1, 2); err != nil || removed != 2 {
		t.Fatalf("dense TryEvict = (%d, %v), want (2, nil)", removed, err)
	}
}

// assertKVCacheUnchanged proves a hybrid cache was left byte-for-byte identical to a
// pre-attempt clone — the fail-closed guarantee of the #444 typed eviction verdict.
func assertKVCacheUnchanged(t *testing.T, label string, want, got *KVCache) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("%s: Len = %d, want %d", label, got.Len(), want.Len())
	}
	for l := range want.K {
		assertFloat32BitsEqual(t, label+" K layer "+itoa(l), want.K[l], got.K[l])
		assertFloat32BitsEqual(t, label+" Kraw layer "+itoa(l), want.Kraw[l], got.Kraw[l])
		assertFloat32BitsEqual(t, label+" V layer "+itoa(l), want.V[l], got.V[l])
	}
	if (want.linear == nil) != (got.linear == nil) {
		t.Fatalf("%s: linear cache nil mismatch", label)
	}
	if want.linear != nil {
		for l := range want.linear.layers {
			wl, gl := want.linear.layers[l], got.linear.layers[l]
			for h := range wl.recurrent {
				assertFloat32BitsEqual(t, label+" recurrent l"+itoa(l)+" h"+itoa(h), wl.recurrent[h], gl.recurrent[h])
			}
			for i := range wl.conv {
				assertFloat32BitsEqual(t, label+" conv l"+itoa(l)+" r"+itoa(i), wl.conv[i], gl.conv[i])
			}
		}
	}
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
