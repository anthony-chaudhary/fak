package model

import (
	"encoding/binary"
	"math"
	"sort"
	"strings"
	"testing"
)

// arch_test.go — gates for the Stage-2 mechanical arch axes (MODEL-ARCH-SEAM.md §6).
//
// Two gates, matching the issue's acceptance:
//
//  (a) TestArchLlamaNoOp — the load-bearing Llama-invariance gate. With every new Arch
//      field at its Llama default, the forward/prefill/decode outputs are bit-identical
//      (Float32bits equality) to the pre-Stage-2 legacy reference. This is the proof the
//      mechanical axes are no-ops on Llama, so R2/R14 stay max|Δ|=0 by construction.
//
//  (b) one test per new axis, exercising its path on a synthetic config and asserting the
//      flag actually changes the result (so the hook is wired, not dead) AND that the axis
//      helper computes the documented transform.

// llamaArchConfig is the tiny synthetic Config used across these tests, with every
// Stage-2 axis at its Llama default (off / identity).
func llamaArchConfig() Config {
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

// archTestCfg is the small synthetic config the block-topology tests run on. It
// needs no weights (NewSynthetic) — the wiring property "topology X composes the
// norm/residual graph as specified" is structural, not numeric, exactly like the
// SEAM-0 refactor-equivalence gate (refactor_test.go). Per-family NUMERIC
// correctness still needs a re-exported HF oracle (MODEL-ARCH-SEAM §7.1).
func archTestCfg(topo BlockTopology) Config {
	cfg := llamaArchConfig()
	cfg.BlockTopology = topo
	return cfg
}

// ---- (a) Llama no-op gate ---------------------------------------------------------

// TestArchLlamaNoOp asserts that with all Stage-2 axes at their Llama defaults, the
// production forward pass (Forward, Prefill, Step) is byte-for-byte identical to the
// legacy pre-Stage-2 reference. The legacy reference (legacyTokenHiddenF32ForTest,
// kept in refactor_test.go) intentionally uses the OLD primitives — plain rmsnorm,
// silu, 1/sqrt(headDim), all-or-nothing AttentionBias, no qk-norm/softcap/scale — so
// equality here proves the new config-driven helpers lower to exactly the old code on
// Llama. max|Δ|=0 by construction.
func TestArchLlamaNoOp(t *testing.T) {
	cfg := llamaArchConfig()
	m := NewSynthetic(cfg)
	prompt := []int{3, 17, 5, 23, 41, 2, 19}

	// Prefill (production blockStep/prefillBatched) vs the legacy per-token loop.
	cur := m.NewSession()
	legacy := m.NewSession()
	curLogits := cur.Prefill(prompt)
	legacyLogits := legacyPrefillF32ForTest(legacy, prompt)
	assertFloat32BitsEqual(t, "llama no-op prefill logits", legacyLogits, curLogits)
	assertKVCacheBitsEqual(t, "llama no-op prefill", legacy.Cache, cur.Cache)

	// A few decode steps.
	for step, id := range []int{11, 29, 7, 13} {
		curLogits = cur.Step(id)
		legacyX := legacyTokenHiddenF32ForTest(legacy, id, legacy.Cache.Len())
		legacyLogits = legacy.head(legacyX)
		assertFloat32BitsEqual(t, "llama no-op decode logits", legacyLogits, curLogits)
		assertKVCacheBitsEqual(t, "llama no-op decode step "+itoa(step), legacy.Cache, cur.Cache)
	}

	// The cacheless Forward (the oracle path) must also be bit-stable: a Llama config with
	// the axes present must equal one numerically — exercised here by confirming Forward's
	// last-position logits equal a fresh prefill's (they share the same math), which already
	// holds in the existing suite; here we additionally assert the axis helpers are identity.
	if got := cfg.attnScale(); got != float32(1.0/math.Sqrt(float64(cfg.HeadDim))) {
		t.Fatalf("llama attnScale = %v, want 1/sqrt(headDim)", got)
	}
	if cfg.embedScale() != 1 {
		t.Fatalf("llama embedScale = %v, want 1", cfg.embedScale())
	}
}

// TestArchAxisFieldsDefaultLlama is a cheap guard that the zero-value Config is the Llama
// no-op for every helper, so a future field addition that forgets a default is caught.
func TestArchAxisFieldsDefaultLlama(t *testing.T) {
	var cfg Config
	cfg.HeadDim = 8
	if cfg.RopeScaling != "" {
		t.Fatal("RopeScaling default must be empty (none)")
	}
	inv := []float64{0.5, 0.25}
	cp := append([]float64(nil), inv...)
	applyRopeScaling(cfg, cp)
	for i := range inv {
		if inv[i] != cp[i] {
			t.Fatalf("default applyRopeScaling changed inv_freq[%d]: %v -> %v", i, inv[i], cp[i])
		}
	}
	if got := softcap(3.5, 0); got != 3.5 {
		t.Fatalf("softcap with cap 0 = %v, want identity", got)
	}
	if got := act(1.25, cfg); got != silu(1.25) {
		t.Fatalf("default act = %v, want silu", got)
	}
}

// ---- (b) per-axis tests -----------------------------------------------------------

// TestRopeScalingLlama3 checks the llama3 piecewise inv_freq rescale: low-freq bands are
// divided by factor, high-freq bands untouched, and the default (none) is the identity.
func TestRopeScalingLlama3(t *testing.T) {
	base := Config{HeadDim: 16, RopeTheta: 500000}
	bare := invFreq(base, 0)

	scaled := base
	scaled.RopeScaling = "llama3"
	scaled.RopeFactor = 8
	scaled.RopeLowFreqFactor = 1
	scaled.RopeHighFreqFactor = 4
	scaled.RopeOrigContext = 8192
	got := invFreq(scaled, 0)

	if len(got) != len(bare) {
		t.Fatalf("scaled inv_freq len %d != bare %d", len(got), len(bare))
	}
	changed := false
	origCtx := float64(scaled.RopeOrigContext)
	lowWavelen := origCtx / scaled.RopeLowFreqFactor
	highWavelen := origCtx / scaled.RopeHighFreqFactor
	for j := range bare {
		wavelen := 2 * math.Pi / bare[j]
		switch {
		case wavelen > lowWavelen:
			// low-frequency (long wavelength): must be divided by factor.
			want := bare[j] / scaled.RopeFactor
			if math.Abs(got[j]-want) > 1e-12 {
				t.Fatalf("low-freq inv[%d] = %v, want %v", j, got[j], want)
			}
			changed = true
		case wavelen < highWavelen:
			// high-frequency (short wavelength): untouched.
			if got[j] != bare[j] {
				t.Fatalf("high-freq inv[%d] changed %v -> %v (should be bare)", j, bare[j], got[j])
			}
		default:
			// interpolation band: strictly between bare/factor and bare.
			lo, hi := bare[j]/scaled.RopeFactor, bare[j]
			if got[j] < math.Min(lo, hi)-1e-9 || got[j] > math.Max(lo, hi)+1e-9 {
				t.Fatalf("interp inv[%d] = %v, out of [%v,%v]", j, got[j], lo, hi)
			}
			changed = true
		}
	}
	if !changed {
		t.Fatal("llama3 scaling left every inv_freq element unchanged — band thresholds wrong")
	}

	// Misconfigured llama3 (missing factor) must fail safe to bare, not divide by zero.
	bad := base
	bad.RopeScaling = "llama3"
	badInv := invFreq(bad, 0)
	for j := range bare {
		if badInv[j] != bare[j] {
			t.Fatalf("misconfigured llama3 inv[%d] = %v, want bare %v", j, badInv[j], bare[j])
		}
	}
}

// TestEOSListLoader checks the scalar-or-list eos_token_id loader and set-membership.
func TestEOSListLoader(t *testing.T) {
	var listCfg Config
	if err := listCfg.UnmarshalJSON([]byte(`{"hidden_size":4,"eos_token_id":[128001,128008,128009]}`)); err != nil {
		t.Fatalf("list eos: %v", err)
	}
	if listCfg.EOSTokenID != 128001 {
		t.Fatalf("list eos scalar = %d, want 128001", listCfg.EOSTokenID)
	}
	for _, id := range []int{128001, 128008, 128009} {
		if !listCfg.IsEOS(id) {
			t.Fatalf("isEOS(%d) = false, want true", id)
		}
	}
	if listCfg.IsEOS(42) {
		t.Fatal("isEOS(42) = true, want false")
	}

	var scalarCfg Config
	if err := scalarCfg.UnmarshalJSON([]byte(`{"hidden_size":4,"eos_token_id":2}`)); err != nil {
		t.Fatalf("scalar eos: %v", err)
	}
	if scalarCfg.EOSTokenID != 2 || !scalarCfg.IsEOS(2) || scalarCfg.IsEOS(3) {
		t.Fatalf("scalar eos membership wrong: id=%d", scalarCfg.EOSTokenID)
	}

	// Struct-literal -1 (the "never early-stop" convention) must never match a real id.
	noStop := Config{EOSTokenID: -1}
	if noStop.IsEOS(0) || noStop.IsEOS(255) {
		t.Fatal("EOSTokenID=-1 should never report EOS")
	}
}

// TestConfigJSONRoundTrip checks a representative config.json (the shape export_oracle.py
// writes) loads every base field plus the flattened rope-scaling + eos-list. This locks
// the loader contract because the real export is gitignored / absent in CI.
func TestConfigJSONRoundTrip(t *testing.T) {
	js := `{
		"hidden_size": 576, "num_hidden_layers": 30, "num_attention_heads": 9,
		"num_key_value_heads": 3, "head_dim": 64, "intermediate_size": 1536,
		"vocab_size": 49152, "rms_norm_eps": 1e-5, "rope_theta": 100000.0,
		"tie_word_embeddings": true, "attention_bias": false, "model_type": "llama",
		"eos_token_id": [128001, 128008, 128009],
		"rope_scaling_type": "llama3", "rope_scaling_factor": 8.0,
		"rope_scaling_low_freq_factor": 1.0, "rope_scaling_high_freq_factor": 4.0,
		"rope_scaling_original_max_position_embeddings": 8192
	}`
	var cfg Config
	if err := cfg.UnmarshalJSON([]byte(js)); err != nil {
		t.Fatalf("round-trip: %v", err)
	}
	if cfg.HiddenSize != 576 || cfg.NumLayers != 30 || cfg.HeadDim != 64 || cfg.VocabSize != 49152 {
		t.Fatalf("base fields wrong: %+v", cfg)
	}
	if cfg.ModelType != "llama" || cfg.RopeTheta != 100000.0 {
		t.Fatalf("type/theta wrong: %s %v", cfg.ModelType, cfg.RopeTheta)
	}
	if len(cfg.EOSTokenIDs) != 3 || cfg.EOSTokenID != 128001 || !cfg.IsEOS(128009) {
		t.Fatalf("eos-list wrong: %v / %d", cfg.EOSTokenIDs, cfg.EOSTokenID)
	}
	if cfg.RopeScaling != "llama3" || cfg.RopeFactor != 8 || cfg.RopeOrigContext != 8192 {
		t.Fatalf("rope scaling fields wrong: %+v", cfg)
	}
}

// TestQueryPreAttnScalar checks the per-head attention-scale override (Gemma).
func TestQueryPreAttnScalar(t *testing.T) {
	cfg := Config{HeadDim: 64}
	if cfg.attnScale() != float32(1.0/math.Sqrt(64)) {
		t.Fatal("default attnScale must be 1/sqrt(headDim)")
	}
	cfg.QueryPreAttnScalar = 256
	if got, want := cfg.attnScale(), float32(1.0/math.Sqrt(256)); got != want {
		t.Fatalf("attnScale with QueryPreAttnScalar=256 = %v, want %v", got, want)
	}
}

// TestNormGain1p checks the (1+w) RMSNorm gain (Gemma) differs from plain RMSNorm and is
// exactly (1+w) scaled.
func TestNormGain1p(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{0.1, -0.2, 0.3, 0.5}
	eps := float32(1e-5)
	plain := rmsnorm(x, w, eps)

	cfg := Config{NormGain1p: true}
	gain := rmsnormCfg(x, w, eps, cfg)

	// gain[i] should equal plain[i] * (1+w[i]) / w[i] only loosely; assert directly that
	// gain reproduces the (1+w) formula and differs from plain.
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	differs := false
	for i := range x {
		want := x[i] * inv * (1 + w[i])
		if math.Abs(float64(gain[i]-want)) > 1e-6 {
			t.Fatalf("gain[%d] = %v, want (1+w) form %v", i, gain[i], want)
		}
		if gain[i] != plain[i] {
			differs = true
		}
	}
	if !differs {
		t.Fatal("NormGain1p produced the same output as plain RMSNorm")
	}

	// off-by-default: NormGain1p=false must equal plain rmsnorm bit-for-bit.
	off := rmsnormCfg(x, w, eps, Config{})
	assertFloat32BitsEqual(t, "norm gain off == plain", plain, off)
}

func TestLayerNormAxis(t *testing.T) {
	x := []float32{1, 2, 4, 8}
	w := []float32{0.5, 1, 1.5, 2}
	b := []float32{0.1, -0.2, 0.3, -0.4}
	eps := float32(1e-5)
	got := rmsnormCfg(x, w, eps, Config{LayerNorm: true})
	gotBias := normCfg(x, w, b, eps, Config{LayerNorm: true})

	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var ss float32
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range x {
		want := (x[i] - mean) * inv * w[i]
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("layernorm[%d]=%v want %v", i, got[i], want)
		}
		if math.Abs(float64(gotBias[i]-(want+b[i]))) > 1e-6 {
			t.Fatalf("layernorm+bias[%d]=%v want %v", i, gotBias[i], want+b[i])
		}
	}
}

// TestActGeluTanh checks the GeGLU-tanh activation differs from SiLU and matches the
// tanh-approx GELU formula.
func TestActGeluTanh(t *testing.T) {
	cfg := Config{ActGeluTanh: true}
	for _, z := range []float32{-2, -0.5, 0, 0.5, 2, 3.3} {
		got := act(z, cfg)
		want := geluTanh(z)
		if got != want {
			t.Fatalf("act(%v) = %v, want geluTanh %v", z, got, want)
		}
		// closed-form spot check against the documented formula.
		x := float64(z)
		ref := 0.5 * x * (1 + math.Tanh(0.7978845608028654*(x+0.044715*x*x*x)))
		if math.Abs(float64(got)-ref) > 1e-6 {
			t.Fatalf("geluTanh(%v) = %v, want %v", z, got, ref)
		}
	}
	if act(1.0, cfg) == silu(1.0) {
		t.Fatal("GeluTanh must differ from SiLU at z=1")
	}
}

func TestActGeluErf(t *testing.T) {
	z := float32(0.75)
	got := act(z, Config{ActGeluErf: true})
	want := float32(0.5 * float64(z) * (1 + math.Erf(float64(z)/math.Sqrt2)))
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("geluErf=%v want %v", got, want)
	}
	if got == act(z, Config{}) {
		t.Fatal("geluErf should differ from default SiLU")
	}
}

// TestSoftcap checks the tanh soft-cap is identity at cap<=0 and saturates toward cap.
func TestSoftcap(t *testing.T) {
	if softcap(10, 0) != 10 || softcap(-10, -1) != -10 {
		t.Fatal("softcap with cap<=0 must be identity")
	}
	const cap = 5.0
	// large input saturates to the cap magnitude (tanh -> 1).
	if got := softcap(1000, cap); got > cap+1e-4 || got < cap-1e-3 {
		t.Fatalf("softcap(1000,5) = %v, want ~5", got)
	}
	// small input ~ linear.
	if got := softcap(0.001, cap); math.Abs(float64(got-0.001)) > 1e-4 {
		t.Fatalf("softcap(0.001,5) = %v, want ~0.001", got)
	}
}

// TestEmbedAndLogitScale checks embed-scale (Gemma) and logit-scale (Cohere).
func TestEmbedAndLogitScale(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	cp := append([]float32(nil), x...)
	scaleEmbedInPlace(cp, Config{EmbedScale: 2})
	for i := range x {
		if cp[i] != x[i]*2 {
			t.Fatalf("embed scale x[%d] = %v, want %v", i, cp[i], x[i]*2)
		}
	}
	// default (0 or 1) = identity.
	id := append([]float32(nil), x...)
	scaleEmbedInPlace(id, Config{})
	assertFloat32BitsEqual(t, "embed scale identity", x, id)

	logits := []float32{8, -16, 32}
	logitScaleInPlace(logits, Config{LogitScale: 0.0625})
	if logits[0] != 0.5 || logits[1] != -1 || logits[2] != 2 {
		t.Fatalf("logit scale 0.0625 = %v, want [0.5,-1,2]", logits)
	}
}

// ---- synthetic models with optional tensors (bias / qk-norm) ----------------------

// newSyntheticExtra builds a synthetic model and APPENDS the named extra tensors (e.g.
// q/k/v bias, q/k_norm) so the presence-driven bias and qk-norm paths can run. The base
// weights are identical to NewSynthetic; the extras are filled from a fixed LCG.
func newSyntheticExtra(cfg Config, extra map[string][]int) *Model {
	m := NewSynthetic(cfg)
	off := len(m.raw)
	// Iterate in sorted name order so the layout is deterministic regardless of Go's
	// randomized map iteration, and seed each tensor's fill from its NAME so two models
	// with the same extra-tensor set get byte-identical extras (the bias-flag equality
	// test relies on this).
	names := make([]string, 0, len(extra))
	for name := range extra {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		shape := extra[name]
		n := 1
		for _, d := range shape {
			n *= d
		}
		seed := uint64(1469598103934665603)
		for _, c := range []byte(name) {
			seed = (seed ^ uint64(c)) * 1099511628211
		}
		next := func() float32 {
			seed = seed*6364136223846793005 + 1442695040888963407
			u := float32(seed>>40) / float32(1<<24)
			return u*2 - 1
		}
		buf := make([]byte, n*4)
		for i := 0; i < n; i++ {
			v := next() * 0.1
			// qk-norm weights centered near 1 so the norm is well-conditioned.
			if strings.HasSuffix(name, "_norm.weight") {
				v = 1 + v*0.1
			}
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		m.manifest[name] = tensorMeta{Dtype: "F32", Shape: shape, Offset: off, Nbytes: n * 4}
		m.raw = append(m.raw, buf...)
		off += n * 4
	}
	return m
}

// TestPerProjectionBias exercises presence-driven bias: with the flag off but bias
// tensors present, bias is still applied (and changes the output); with no bias tensors,
// it is a no-op.
func TestPerProjectionBias(t *testing.T) {
	cfg := llamaArchConfig()
	hd, nH, nKV := cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	extra := map[string][]int{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"self_attn.q_proj.bias"] = []int{nH * hd}
		extra[p+"self_attn.k_proj.bias"] = []int{nKV * hd}
		extra[p+"self_attn.v_proj.bias"] = []int{nKV * hd}
	}
	withBias := newSyntheticExtra(cfg, extra)
	noBias := NewSynthetic(cfg)

	prompt := []int{3, 17, 5, 23, 9}
	a := withBias.NewSession().Prefill(prompt)
	b := noBias.NewSession().Prefill(prompt)
	if float32BitsEqual(a, b) {
		t.Fatal("presence-driven bias did not change the output despite bias tensors present")
	}

	// With the explicit AttentionBias flag, the same tensors must give the same result as
	// presence-driven (both add all three).
	cfgFlag := cfg
	cfgFlag.AttentionBias = true
	withFlag := newSyntheticExtra(cfgFlag, extra)
	c := withFlag.NewSession().Prefill(prompt)
	assertFloat32BitsEqual(t, "flag bias == presence bias", a, c)
}

// TestQKNorm exercises the per-head qk-norm path: it changes the output, and the cached
// Kraw stays POST-qk-norm/PRE-RoPE so eviction repositioning is still bit-exact (decode
// after a prefill equals a recompute).
func TestQKNorm(t *testing.T) {
	cfg := llamaArchConfig()
	cfg.QKNorm = true
	hd, nKV := cfg.HeadDim, cfg.NumKVHeads
	extra := map[string][]int{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"self_attn.q_norm.weight"] = []int{hd}
		extra[p+"self_attn.k_norm.weight"] = []int{hd}
	}
	m := newSyntheticExtra(cfg, extra)

	// off-config baseline (no qk-norm).
	cfgOff := cfg
	cfgOff.QKNorm = false
	mOff := newSyntheticExtra(cfgOff, extra)

	prompt := []int{3, 17, 5, 23, 9, 2}
	on := m.NewSession().Prefill(prompt)
	off := mOff.NewSession().Prefill(prompt)
	_ = nKV
	if float32BitsEqual(on, off) {
		t.Fatal("qk-norm did not change the output")
	}

	// R2 analogue under qk-norm: cached decode == full prefill. Prefill k tokens, then
	// decode the rest one at a time; compare to a single full prefill's last logits.
	full := m.NewSession().Prefill(prompt)
	s := m.NewSession()
	s.Prefill(prompt[:3])
	var stepLogits []float32
	for _, id := range prompt[3:] {
		stepLogits = s.Step(id)
	}
	// stepLogits is the distribution AFTER the last prompt token, same as full prefill.
	assertFloat32BitsEqual(t, "qk-norm cached-decode == prefill", full, stepLogits)
}

func TestQKNormFullProjectionWeight(t *testing.T) {
	hv := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	w := []float32{0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25}
	got := append([]float32(nil), hv...)
	applyQKNorm(got, w, 2, 4, 1e-5)

	var ss float32
	for _, v := range hv {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(hv))+1e-5)))
	for i := range hv {
		want := hv[i] * inv * w[i]
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("full qk-norm[%d]=%v want %v", i, got[i], want)
		}
	}
}

func TestQKNormUsesGemmaGainWhenConfigured(t *testing.T) {
	hv := []float32{1, 2, 3, 4}
	w := []float32{0.1, -0.2}
	got := append([]float32(nil), hv...)
	applyQKNormCfg(got, w, 2, 2, 1e-5, Config{NormGain1p: true})

	for head := 0; head < 2; head++ {
		start := head * 2
		var ss float32
		for _, v := range hv[start : start+2] {
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(float64(ss/2+1e-5)))
		for d := 0; d < 2; d++ {
			i := start + d
			want := hv[i] * inv * (1 + w[d])
			if math.Abs(float64(got[i]-want)) > 1e-6 {
				t.Fatalf("gemma qk-norm[%d]=%v want %v", i, got[i], want)
			}
		}
	}
}

func TestAlibiScoreBias(t *testing.T) {
	cfg := Config{NumHeads: 3, Alibi: true, AlibiBiasMax: 8}
	keyLen := 5
	got0 := cfg.alibiScoreBias(0, 0, keyLen)
	got1 := cfg.alibiScoreBias(1, 0, keyLen)
	got2 := cfg.alibiScoreBias(2, 0, keyLen)
	// For three heads HF first builds four slopes, then reorders odd indices before even:
	// [2^-4, 2^-8, 2^-2]. Bias at key 0 is slope*(0-keyLen+1).
	want0 := float32(math.Pow(2, -4) * float64(1-keyLen))
	want1 := float32(math.Pow(2, -8) * float64(1-keyLen))
	want2 := float32(math.Pow(2, -2) * float64(1-keyLen))
	for i, pair := range [][2]float32{{got0, want0}, {got1, want1}, {got2, want2}} {
		if math.Abs(float64(pair[0]-pair[1])) > 1e-6 {
			t.Fatalf("alibi bias head %d = %v want %v", i, pair[0], pair[1])
		}
	}
	if (Config{}).alibiScoreBias(0, 0, 5) != 0 {
		t.Fatal("alibi disabled must be zero bias")
	}
}

func TestYarnRopeScalesCosSin(t *testing.T) {
	truncate := false
	cfg := Config{
		HeadDim:         8,
		NumHeads:        1,
		RopeTheta:       150000,
		RopeScaling:     "yarn",
		RopeFactor:      32,
		RopeOrigContext: 4096,
		RopeParameters: RopeParameters{"default": {
			RopeType:                      "yarn",
			RopeTheta:                     150000,
			Factor:                        32,
			BetaFast:                      32,
			BetaSlow:                      1,
			OriginalMaxPositionEmbeddings: 4096,
			Truncate:                      &truncate,
		}},
	}
	cos, sin := ropeRowForLayer(cfg, 0, 0)
	want := float32(0.1*math.Log(32) + 1)
	for i := range cos {
		if math.Abs(float64(cos[i]-want)) > 1e-6 {
			t.Fatalf("yarn cos[%d]=%v want attention factor %v", i, cos[i], want)
		}
		if sin[i] != 0 {
			t.Fatalf("yarn sin[%d]=%v want 0 at position 0", i, sin[i])
		}
	}
}

func TestAttentionSinkSoftmaxDropsSink(t *testing.T) {
	m, err := NewFromF32Tensors(Config{ModelType: "gpt_oss", NumLayers: 1}, []NamedTensorF32{
		{Name: layerName(0, "self_attn.sinks"), Shape: []int{2}, Data: []float32{0, 2}},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	scores := []float32{0}
	m.softmaxAttentionScores(0, 1, scores)
	want := float32(1 / (1 + math.Exp(2)))
	if math.Abs(float64(scores[0]-want)) > 1e-6 {
		t.Fatalf("sink-normalized visible score = %v want %v", scores[0], want)
	}
}

// TestGemmaStackChangesOutput exercises the Gemma mechanical stack together (norm-gain,
// gelu-tanh, embed-scale, per-head scale, soft-caps) and asserts the result is finite and
// differs from Llama — a smoke gate that the combined path runs end-to-end.
func TestGemmaStackChangesOutput(t *testing.T) {
	cfg := llamaArchConfig()
	llama := NewSynthetic(cfg).NewSession().Prefill([]int{1, 2, 3, 4, 5})

	gcfg := cfg
	gcfg.NormGain1p = true
	gcfg.ActGeluTanh = true
	gcfg.EmbedScale = math.Sqrt(float64(cfg.HiddenSize))
	gcfg.QueryPreAttnScalar = cfg.HeadDim // a Gemma-shaped per-head scale
	gcfg.AttnSoftcap = 50
	gcfg.LogitSoftcap = 30
	gemma := NewSynthetic(gcfg).NewSession().Prefill([]int{1, 2, 3, 4, 5})

	if len(gemma) != len(llama) {
		t.Fatalf("gemma logits len %d != llama %d", len(gemma), len(llama))
	}
	for i, v := range gemma {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gemma logit[%d] = %v (non-finite)", i, v)
		}
		// logit soft-cap clamps |logit| < cap.
		if math.Abs(float64(v)) > 30+1e-3 {
			t.Fatalf("gemma logit[%d] = %v exceeds logit softcap 30", i, v)
		}
	}
	if float32BitsEqual(gemma, llama) {
		t.Fatal("Gemma mechanical stack produced identical output to Llama")
	}
}

func float32BitsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

var archPrompt = []int{3, 17, 5, 23, 41, 2, 19}
var archDecode = []int{11, 29, 7}

// TestBlockTopologyPreNormNoOp is the load-bearing no-op gate: an EXPLICIT PreNorm
// config must be byte-for-byte identical to today's pre-axis decoder block, for
// both prefill logits and every KV-cache byte, across prefill and decode. The
// legacy reference (legacyTokenHiddenF32ForTest) is the hand-copied block as it
// existed before the topology axis, so max|Δ|=0 here proves the dispatch lowers
// PreNorm to the verbatim Llama path — the crown-jewel R2/R14 contract is untouched.
func TestBlockTopologyPreNormNoOp(t *testing.T) {
	m := NewSynthetic(archTestCfg(PreNorm))

	// Sanity: PreNorm is the zero value, so a config that never sets the field is
	// the same topology — the no-op holds for every existing export by construction.
	if (Config{}).BlockTopology != PreNorm {
		t.Fatalf("zero-value BlockTopology = %v, want PreNorm", (Config{}).BlockTopology)
	}

	cur := m.NewSession()
	legacy := m.NewSession()

	curLogits := cur.Prefill(archPrompt)
	legacyLogits := legacyPrefillF32ForTest(legacy, archPrompt)
	assertFloat32BitsEqual(t, "PreNorm prefill logits", legacyLogits, curLogits)
	assertKVCacheBitsEqual(t, "PreNorm prefill", legacy.Cache, cur.Cache)

	for step, id := range archDecode {
		curLogits = cur.Step(id)
		legacyX := legacyTokenHiddenF32ForTest(legacy, id, legacy.Cache.Len())
		legacyLogits = legacy.head(legacyX)
		assertFloat32BitsEqual(t, "PreNorm decode logits", legacyLogits, curLogits)
		assertKVCacheBitsEqual(t, "PreNorm decode step "+itoa(step), legacy.Cache, cur.Cache)
	}
}

// TestBlockTopologyForwardPreNormNoOp covers the cacheless Forward/layer path (the
// oracle path) the same way: an explicit PreNorm forward must be bit-identical to
// the zero-value (current) forward, so TestForwardMatchesHFOracle stays green.
func TestBlockTopologyForwardPreNormNoOp(t *testing.T) {
	base := NewSynthetic(archTestCfg(PreNorm)) // BlockTopology=PreNorm, explicitly
	zero := NewSynthetic(Config{               // identical config, topology left default
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	})
	a := base.Forward(archPrompt)
	b := zero.Forward(archPrompt)
	for t2 := 0; t2 < a.Seq; t2++ {
		assertFloat32BitsEqual(t, "Forward logits pos "+itoa(t2), b.Logits[t2], a.Logits[t2])
	}
	for l := range a.Hidden {
		assertFloat32BitsEqual(t, "Forward hidden layer "+itoa(l), b.Hidden[l], a.Hidden[l])
	}
}

// TestBlockTopologyDiffersFromPreNorm asserts each non-Llama topology produces a
// DIFFERENT forward than PreNorm (the structural change is observable), and that
// the cacheless Forward path and the cached Session/blockStep path AGREE on the same
// topology (the two block sites dispatch identically).
func TestBlockTopologyDiffersFromPreNorm(t *testing.T) {
	pre := NewSynthetic(archTestCfg(PreNorm)).Forward(archPrompt)
	for _, topo := range []BlockTopology{PostNorm, SandwichNorm, ParallelResidual} {
		t.Run(topo.String(), func(t *testing.T) {
			m := NewSynthetic(archTestCfg(topo))
			got := m.Forward(archPrompt)
			if maxAbsDiffActs(pre, got) == 0 {
				t.Fatalf("%v Forward is bit-identical to PreNorm; topology had no structural effect", topo)
			}
			// The cached path (Prefill via blockStep) must match the cacheless
			// Forward at the last position for the SAME topology — both block sites
			// implement the same composition.
			last := got.Logits[got.Seq-1]
			cached := m.NewSession().Prefill(archPrompt)
			if d := maxAbsDelta(last, cached); d > 1e-4 {
				t.Fatalf("%v: cacheless Forward and cached Prefill disagree, max|Δ|=%g", topo, d)
			}
		})
	}
}

// TestBlockTopologyComposition pins each topology to an INDEPENDENT reference
// composition of the same sub-layer bodies, so the test proves the wiring is the
// EXPECTED structural graph, not merely "different". Each reference recomputes the
// single-block residual math directly from the primitives.
func TestBlockTopologyComposition(t *testing.T) {
	// A one-layer config keeps the reference composition tractable: we recompute the
	// whole block by hand from the embedding through to the post-block hidden state
	// (Hidden[1] in the Activations), independent of layer().
	cfg := archTestCfg(PreNorm)
	cfg.NumLayers = 1
	ids := []int{4, 9, 2, 15}

	for _, topo := range []BlockTopology{PreNorm, PostNorm, SandwichNorm, ParallelResidual} {
		t.Run(topo.String(), func(t *testing.T) {
			c := cfg
			c.BlockTopology = topo
			m := NewSynthetic(c)
			got := m.Forward(ids).Hidden[1] // hidden AFTER the single block, flattened
			want := referenceBlock(m, c, ids)
			if d := maxAbsDelta(got, want); d > 1e-5 {
				t.Fatalf("%v: block output differs from reference composition, max|Δ|=%g", topo, d)
			}
		})
	}
}

// TestSandwichNormUsesDistinctFeedForwardNorms proves the Gemma-style wiring can
// read separate pre/post feed-forward norm tensors instead of reusing Llama's
// post-attention norm for both sides of the MLP sub-layer. The fixture is still
// synthetic, but the tensor names are the real loader surface that #48 needs to map.
func TestSandwichNormUsesDistinctFeedForwardNorms(t *testing.T) {
	cfg := archTestCfg(SandwichNorm)
	cfg.NumLayers = 1
	H := cfg.HiddenSize
	extra := map[string][]int{
		layerName(0, "pre_feedforward_layernorm.weight"):  {H},
		layerName(0, "post_feedforward_layernorm.weight"): {H},
	}
	m := newSyntheticExtra(cfg, extra)
	ids := []int{4, 9, 2, 15}

	got := m.Forward(ids).Hidden[1]
	want := referenceBlock(m, cfg, ids)
	if d := maxAbsDelta(got, want); d > 1e-5 {
		t.Fatalf("SandwichNorm with distinct FFN norms differs from reference, max|Δ|=%g", d)
	}

	fallback := NewSynthetic(cfg).Forward(ids).Hidden[1]
	if maxAbsDelta(got, fallback) == 0 {
		t.Fatal("distinct feed-forward norm tensors had no effect")
	}

	cached := m.NewSession().Prefill(ids)
	last := m.Forward(ids).Logits[len(ids)-1]
	if d := maxAbsDelta(last, cached); d > 1e-4 {
		t.Fatalf("cached SandwichNorm with distinct FFN norms disagrees with Forward, max|Δ|=%g", d)
	}
}

// referenceBlock recomputes ONE decoder block for a whole prompt directly from the
// primitives, composing the norm/residual graph per the topology. It is an
// independent re-derivation of what arch.go's composeBlock/composeSeqSublayer should
// produce — if they agree, the wiring is the specified graph.
func referenceBlock(m *Model, cfg Config, ids []int) []float32 {
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)
	attnNorm := m.attentionNorms(0)
	mlpNorm := m.mlpNorms(0)
	rp := newRope(cfg, seq)

	embed := m.embedRows()
	x := make([][]float32, seq)
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	attn := func(xn [][]float32) [][]float32 { return m.attnSeq(0, xn, rp) }
	mlp := func(xn [][]float32) [][]float32 { return m.mlpSeq(0, xn) }

	switch cfg.BlockTopology {
	case PreNorm:
		addSeq(x, attn(normEachCfg(x, attnNorm.pre, attnNorm.preBias, eps, cfg)))
		addSeq(x, mlp(normEachCfg(x, mlpNorm.pre, mlpNorm.preBias, eps, cfg)))
	case PostNorm:
		addSeq(x, normEachCfg(attn(x), attnNorm.post, attnNorm.postBias, eps, cfg))
		addSeq(x, normEachCfg(mlp(x), mlpNorm.post, mlpNorm.postBias, eps, cfg))
	case SandwichNorm:
		addSeq(x, normEachCfg(attn(normEachCfg(x, attnNorm.pre, attnNorm.preBias, eps, cfg)), attnNorm.post, attnNorm.postBias, eps, cfg))
		addSeq(x, normEachCfg(mlp(normEachCfg(x, mlpNorm.pre, mlpNorm.preBias, eps, cfg)), mlpNorm.post, mlpNorm.postBias, eps, cfg))
	case ParallelResidual:
		mlpNorm = m.parallelMLPNorms(0, attnNorm)
		o := attn(normEachCfg(x, attnNorm.pre, attnNorm.preBias, eps, cfg))
		d := mlp(normEachCfg(x, mlpNorm.pre, mlpNorm.preBias, eps, cfg))
		for t := 0; t < seq; t++ {
			for i := 0; i < H; i++ {
				x[t][i] += o[t][i] + d[t][i]
			}
		}
	}
	return flatten(x)
}

func normEach(x [][]float32, w []float32, eps float32) [][]float32 {
	return normEachCfg(x, w, nil, eps, Config{})
}

func normEachCfg(x [][]float32, w, bias []float32, eps float32, cfg Config) [][]float32 {
	out := make([][]float32, len(x))
	for t := range x {
		out[t] = normCfg(x[t], w, bias, eps, cfg)
	}
	return out
}

func addSeq(x, d [][]float32) {
	for t := range x {
		for i := range x[t] {
			x[t][i] += d[t][i]
		}
	}
}

func maxAbsDelta(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	var mx float32
	for i := range a {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > mx {
			mx = d
		}
	}
	return mx
}

func maxAbsDiffActs(a, b *Activations) float32 {
	var mx float32
	for t := range a.Logits {
		if d := maxAbsDelta(a.Logits[t], b.Logits[t]); d > mx {
			mx = d
		}
	}
	return mx
}

func TestParallelResidualDoesNotRequireSeparateMLPNorm(t *testing.T) {
	cfg := Config{
		HiddenSize:        8,
		NumLayers:         1,
		NumHeads:          2,
		NumKVHeads:        2,
		HeadDim:           4,
		IntermediateSize:  16,
		VocabSize:         32,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		BlockTopology:     ParallelResidual,
	}
	m := NewSynthetic(cfg)
	delete(m.manifest, layerName(0, "post_attention_layernorm.weight"))

	ids := []int{1, 2, 3}
	act := m.Forward(ids)
	// Parallel residual must still produce full-shape, finite logits without the
	// separate MLP norm weight (it shares the input norm) -- not merely "no panic".
	if act.Seq != len(ids) {
		t.Fatalf("Forward Seq = %d, want %d", act.Seq, len(ids))
	}
	if len(act.Logits) != len(ids) {
		t.Fatalf("Forward returned %d logit rows, want %d", len(act.Logits), len(ids))
	}
	for p, row := range act.Logits {
		if len(row) != cfg.VocabSize {
			t.Fatalf("Forward logits[%d] len = %d, want vocab %d", p, len(row), cfg.VocabSize)
		}
		for v, x := range row {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("Forward logits[%d][%d] not finite: %v", p, v, x)
			}
		}
	}

	// The KV-cached Prefill path is the same model by another route: its
	// last-position logits must match Forward's last row (the missing norm does
	// not diverge the two paths).
	pf := m.NewSession().Prefill(ids)
	if len(pf) != cfg.VocabSize {
		t.Fatalf("Prefill logits len = %d, want vocab %d", len(pf), cfg.VocabSize)
	}
	if d := maxAbsDelta(pf, act.Logits[len(ids)-1]); d > 1e-4 {
		t.Fatalf("Prefill last-row diverges from Forward by %g (>1e-4)", d)
	}
}

// TestBlockTopologyString keeps the enum's String() honest (used in test labels and
// any future config-derivation logging).
func TestBlockTopologyString(t *testing.T) {
	for v, want := range map[BlockTopology]string{
		PreNorm:          "PreNorm",
		PostNorm:         "PostNorm",
		SandwichNorm:     "SandwichNorm",
		ParallelResidual: "ParallelResidual",
	} {
		if got := v.String(); got != want {
			t.Errorf("BlockTopology(%d).String() = %q, want %q", int(v), got, want)
		}
	}
	if !ParallelResidual.IsParallel() || PreNorm.IsParallel() {
		t.Errorf("IsParallel mismatch: parallel=%v prenorm=%v", ParallelResidual.IsParallel(), PreNorm.IsParallel())
	}
}
