package model

import (
	"math"
	"testing"
)

// norm_bias_batched_lanes_test.go — the LayerNorm-WITH-BIAS axis for the batched /
// quantized prefill+decode lanes, the sibling of prefill_batch_norm_bias_test.go.
//
// Why the existing contract tests (TestPrefillEachRectangularMatchesSerial,
// TestBatchedDecodeMatchesSerial, the verify chain gate) could not catch this: they all
// run on an RMSNorm config, which is structurally bias-free. Every norm bias is ABSENT
// there, so a lane that drops the bias is bit-identical to one that applies it and the
// assertion passes vacuously on that axis. But StableLM (PreNorm, LayerNorm WITH a
// learned bias, no MoE/ALiBi/output-gate) is admitted by every gate below —
// batchPreNormFastPathOK (batch.go:109), q8PrefillNeedsTokenLoop (kv.go:825) and
// verifyForwardBatchedOK (verify.go:64) have NO LayerNorm term — while rmsnormCfg
// (arch.go:459) is literally the nil-bias wrapper. So these lanes silently disagreed
// with the per-token path they are contracted to match bit-for-bit, and because the
// pre-attention bias feeds q/k/v, they also built a silently different KV cache.
//
// Every fixture here is weight-free (newSyntheticExtra), deliberately: the norm-bias
// class survived precisely because the checkpoint-backed gates SKIP in CI.

// normBiasArchConfig is a PreNorm LayerNorm family (StableLM-shaped): mean-subtracting
// LayerNorm, so the learned input_layernorm.bias / post_attention_layernorm.bias /
// model.norm.bias are all live weights rather than absent tensors.
func normBiasArchConfig() Config {
	return Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
		LayerNorm: true,
	}
}

// normBiasExtras names the three learned norm biases a LayerNorm family carries.
func normBiasExtras(cfg Config) map[string][]int {
	H := cfg.HiddenSize
	extra := map[string][]int{"model.norm.bias": {H}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"input_layernorm.bias"] = []int{H}
		extra[p+"post_attention_layernorm.bias"] = []int{H}
	}
	return extra
}

// requireNonZeroNormBiases is the NON-VACUITY guard shared by every case here: an
// all-zero bias is indistinguishable from a dropped one, so if the synthetic filler ever
// stops producing non-zero extras these tests must fail loudly rather than quietly stop
// witnessing the bug.
func requireNonZeroNormBiases(t *testing.T, m *Model, extra map[string][]int) {
	t.Helper()
	for name := range extra {
		nonZero := false
		for _, v := range m.tensor(name) {
			if v != 0 {
				nonZero = true
				break
			}
		}
		if !nonZero {
			t.Fatalf("%s is all zero: a dropped bias would be undetectable, test is vacuous", name)
		}
	}
}

func assertBitEqualF32(t *testing.T, what string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: len %d != %d", what, len(want), len(got))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s[%d]: per-token %v != batched %v (norm bias dropped in the batched lane)",
				what, i, want[i], got[i])
		}
	}
}

func assertCachesBitEqual(t *testing.T, label string, cfg Config, want, got *KVCache) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("%s cache len %d != %d", label, want.Len(), got.Len())
	}
	for l := 0; l < cfg.NumLayers; l++ {
		assertBitEqualF32(t, label+" K", want.K[l], got.K[l])
		assertBitEqualF32(t, label+" Kraw", want.Kraw[l], got.Kraw[l])
		assertBitEqualF32(t, label+" V", want.V[l], got.V[l])
	}
	for i := range want.pos {
		if want.pos[i] != got.pos[i] {
			t.Fatalf("%s pos[%d] %d != %d", label, i, want.pos[i], got.pos[i])
		}
	}
}

// TestBatchRectPrefillAndStepNormBias is the LayerNorm-with-bias version of
// TestPrefillEachRectangularMatchesSerial + TestBatchedDecodeMatchesSerial. It pins
// prefillEachRectF32 (batch_prefill.go) and stepBatchF32 (batch_step.go) — the
// shared-weight panel lanes — to the per-user serial Session they are contracted to be
// bit-for-bit identical to. Both hard-passed a nil bias at all three norm sites
// (pre-attention, post-attention, final) while the serial reference passes them.
//
// The logit assertion is what catches any one of the three sites; the KV assertion that
// follows LOCALIZES a regression to the pre-attention norm (input_layernorm.bias feeds
// q/k/v, so dropping it moves the cached K/V, while model.norm.bias moves only the
// logits).
func TestBatchRectPrefillAndStepNormBias(t *testing.T) {
	cfg := normBiasArchConfig()
	extra := normBiasExtras(cfg)
	m := newSyntheticExtra(cfg, extra)

	// Premise guard: this config must actually REACH the batched lanes, or the test proves
	// nothing about production. PrefillEach takes prefillEachRectF32 and StepBatch takes
	// stepBatchF32 exactly when batchRectFastPathOK / batchDecodeFastPathOK hold.
	if !batchRectFastPathOK(cfg, false) {
		t.Fatal("premise broken: this config no longer routes to prefillEachRectF32, so the test is vacuous")
	}
	if !batchDecodeFastPathOK(cfg, false) {
		t.Fatal("premise broken: this config no longer routes to stepBatchF32, so the test is vacuous")
	}
	requireNonZeroNormBiases(t, m, extra)

	V := cfg.VocabSize
	B, P := 4, 7
	if P > batchRectPrefillMaxTokens {
		t.Fatalf("premise broken: P=%d exceeds the rect-prefill cap %d", P, batchRectPrefillMaxTokens)
	}
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		prompts[b] = make([]int, P)
		for i := 0; i < P; i++ {
			prompts[b][i] = (b*41 + i*17 + 5) % V
		}
	}

	// Reference: B independent serial sessions (prefillBatched + blockStep, both bias-aware).
	ref := make([]*Session, B)
	refLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		ref[b] = m.NewSession()
		refLogits[b] = ref[b].Prefill(prompts[b])
	}

	bs := m.NewBatchSession(B)
	batLogits := bs.PrefillEach(prompts)
	for b := 0; b < B; b++ {
		assertBitEqualF32(t, "rect-prefill user logits", refLogits[b], batLogits[b])
		assertCachesBitEqual(t, "rect-prefill user", cfg, ref[b].Cache, bs.Seqs[b].Cache)
	}

	// Decode several steps in lockstep, feeding the SAME deterministic per-user token to
	// both paths, so stepBatchF32's three norm sites are compared against serial Step.
	for s := 0; s < 6; s++ {
		next := make([]int, B)
		for b := 0; b < B; b++ {
			next[b] = (s*53 + b*17 + 1) % V
		}
		for b := 0; b < B; b++ {
			refLogits[b] = ref[b].Step(next[b])
		}
		batLogits = bs.StepBatch(next)
		for b := 0; b < B; b++ {
			assertBitEqualF32(t, "batched-step user logits", refLogits[b], batLogits[b])
		}
	}
	for b := 0; b < B; b++ {
		assertCachesBitEqual(t, "batched-step user", cfg, ref[b].Cache, bs.Seqs[b].Cache)
	}
}

// TestVerifyForwardNormBias pins verifyForwardBatched (verify.go) — the single-pass
// speculative-verify lane — to its own documented contract: "the chain logits are
// bit-identical to serial head(finalNorm(tokenHidden(...)))". Its final norm already went
// through m.finalNorm, but both per-layer norms hard-passed a nil bias, so on a
// LayerNorm family the contract was false. verifyForwardBatchedOK gates only on
// backend/quant/topology, never on the norm kind.
func TestVerifyForwardNormBias(t *testing.T) {
	cfg := normBiasArchConfig()
	extra := normBiasExtras(cfg)
	m := newSyntheticExtra(cfg, extra)
	requireNonZeroNormBiases(t, m, extra)

	V := cfg.VocabSize
	ids := make([]int, 5)
	for i := range ids {
		ids[i] = (i*1299709 + 17) % V
	}

	got := m.NewSession()
	if !verifyForwardBatchedOK(got) {
		t.Fatal("premise broken: this session no longer routes to verifyForwardBatched, so the test is vacuous")
	}
	gotLogits := got.VerifyForward(ids, nil, nil)
	if len(gotLogits) != len(ids) {
		t.Fatalf("VerifyForward returned %d rows, want %d", len(gotLogits), len(ids))
	}

	// Reference: the sequential chain the batched path must reproduce bit-for-bit.
	ref := m.NewSession()
	for i, id := range ids {
		assertBitEqualF32(t, "verify chain logits", ref.Step(id), gotLogits[i])
	}
	assertCachesBitEqual(t, "verify chain", cfg, ref.Cache, got.Cache)
}

// zeroTensorInPlace zeroes a tensor's backing bytes in the model blob. m.tensor is a
// zero-copy view into m.raw, so this makes the named weight present-but-zero — the ONE
// axis the twin-model probes below vary.
func zeroTensorInPlace(m *Model, name string) {
	v := m.tensor(name)
	for i := range v {
		v[i] = 0
	}
}

// newNormBiasTwin builds a byte-identical sibling of newSyntheticExtra(cfg, extra) with
// exactly the named biases zeroed, and PROVES the twin differs on nothing else. That
// "differs on exactly one tensor" guard is what makes the probes below a statement about
// the lane's use of the bias rather than about two unrelated models.
func newNormBiasTwin(t *testing.T, cfg Config, extra map[string][]int, zero ...string) *Model {
	t.Helper()
	base := newSyntheticExtra(cfg, extra)
	twin := newSyntheticExtra(cfg, extra)
	zeroed := map[string]bool{}
	for _, name := range zero {
		zeroTensorInPlace(twin, name)
		zeroed[name] = true
	}
	for name := range base.manifest {
		if zeroed[name] {
			continue
		}
		a, b := base.tensor(name), twin.tensor(name)
		if len(a) != len(b) {
			t.Fatalf("twin tensor %s len %d != %d", name, len(a), len(b))
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("twin differs on %s[%d] (%v vs %v): the probe would not isolate the bias",
					name, i, a[i], b[i])
			}
		}
	}
	return twin
}

// assertFinalNormBiasDelta is the closed-form final-norm probe: layernorm writes
// out[i] = (v-mean)*inv*w[i] and then out[i] += bias[i] (arch.go:496), and the final norm is
// the last op before the returned hidden, so the biased run must equal the zero-bias run PLUS
// the bias in EXACT float32. A lane that drops the bias returns the zero-bias value instead.
//
// It reports through t.Errorf (not Fatalf) and returns after the first mismatch, so a test
// that probes several lanes witnesses EVERY broken one in a single run instead of hiding the
// rest behind the first.
func assertFinalNormBiasDelta(t *testing.T, what string, got, base, bias []float32) {
	t.Helper()
	if len(got) != len(bias) || len(base) != len(bias) {
		t.Errorf("%s: len got=%d base=%d bias=%d, want all equal", what, len(got), len(base), len(bias))
		return
	}
	for i := range got {
		want := base[i] + bias[i]
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Errorf("%s hidden[%d] = %v, want %v (= zero-bias %v + model.norm.bias %v): "+
				"this lane dropped the final-norm bias", what, i, got[i], want, base[i], bias[i])
			return
		}
	}
}

// assertZeroingChanged is the per-layer probe. A per-layer bias propagates through attention
// and the MLP, so its contribution is not closed-form at the output; the decisive statement is
// that zeroing the tensor MOVES the result. A lane that never reads it is unaffected.
func assertZeroingChanged(t *testing.T, what, tensor string, got, zeroed []float32) {
	t.Helper()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(zeroed[i]) {
			return
		}
	}
	t.Errorf("zeroing every layer's %s did not change %s's output: that lane never reads the bias",
		tensor, what)
}

// TestQuantPrefillDecodeNormBias pins the QUANTIZED lanes a LayerNorm family actually
// routes to: prefillBatchedQ (quant_forward.go, reached because q8PrefillNeedsTokenLoop
// has no LayerNorm term) and the SLOW branch of tokenHiddenQ (reached precisely BECAUSE
// q8FastPreNormOK refuses cfg.LayerNorm). Both hand-rolled a final norm with a nil bias
// after otherwise applying every per-layer bias correctly.
//
// The Q8 tile GEMM is not bit-identical to the per-token Q8 GEMV, so this case cannot
// use a cross-lane bit-equality reference. Instead it varies exactly ONE weight — the
// bias under test — and asserts the lane's output moves by exactly the amount that
// weight is defined to contribute. For the FINAL norm the contribution is closed-form:
// layernorm writes out[i] = (v-mean)*inv*w[i] and then out[i] += bias[i] (arch.go:496),
// and the final norm is the last op before the returned hidden, so the biased run must
// equal the zero-bias run PLUS the bias, in exact float32. Dropping the bias makes the
// two runs identical instead — which is the pre-fix behaviour, and fails here.
func TestQuantPrefillDecodeNormBias(t *testing.T) {
	cfg := normBiasArchConfig()
	extra := normBiasExtras(cfg)
	m := newSyntheticExtra(cfg, extra)
	m.Quantize()
	requireNonZeroNormBiases(t, m, extra)

	// Premise guards: quantized Prefill must land in prefillBatchedQ, and quantized decode
	// must land in tokenHiddenQ's slow blockStep branch (never the fast resident loop,
	// which has no final-norm bias site at all).
	if q8PrefillNeedsTokenLoop(cfg) {
		t.Fatal("premise broken: this config no longer routes to prefillBatchedQ, so the test is vacuous")
	}
	probe := m.NewSession()
	probe.Quant = true
	if q8FastDecodeSessionOK(probe, cfg) {
		t.Fatal("premise broken: this config now takes the fast Q8 decode loop, so the tokenHiddenQ probe is vacuous")
	}

	V := cfg.VocabSize
	ids := make([]int, 9)
	for i := range ids {
		ids[i] = (i*1299709 + 17) % V
	}

	// The zero-final-norm-bias twin. Everything upstream of the final norm is identical, so
	// the two hidden states differ by exactly model.norm.bias — if the lane applies it.
	twin := newNormBiasTwin(t, cfg, extra, "model.norm.bias")
	twin.Quantize()
	bias := append([]float32(nil), m.tensor("model.norm.bias")...)

	quantSession := func(mm *Model) *Session {
		s := mm.NewSession()
		s.Quant = true
		return s
	}

	// (1) prefillBatchedQ — the batched quantized prefill's final norm.
	gotPrefill := quantSession(m).prefillBatchedQ(ids)
	basePrefill := quantSession(twin).prefillBatchedQ(ids)
	if len(gotPrefill) != cfg.HiddenSize || len(basePrefill) != cfg.HiddenSize {
		t.Fatalf("prefillBatchedQ hidden len %d/%d, want %d", len(gotPrefill), len(basePrefill), cfg.HiddenSize)
	}
	assertFinalNormBiasDelta(t, "prefillBatchedQ", gotPrefill, basePrefill, bias)

	// (2) tokenHiddenQ slow branch — the quantized decode's final norm, on a fresh cache so
	// both runs see identical KV state. Probed independently of (1): these are two separate
	// call sites in two separate functions, and one must not mask the other.
	gotStep := quantSession(m).tokenHiddenQ(ids[0], 0)
	baseStep := quantSession(twin).tokenHiddenQ(ids[0], 0)
	if len(gotStep) != cfg.HiddenSize || len(baseStep) != cfg.HiddenSize {
		t.Fatalf("tokenHiddenQ hidden len %d/%d, want %d", len(gotStep), len(baseStep), cfg.HiddenSize)
	}
	assertFinalNormBiasDelta(t, "tokenHiddenQ", gotStep, baseStep, bias)
}

// TestResidentQ4KPrefillNormBias pins prefillBatchedQ4K (prefill_q4k.go), the resident-Q4_K
// batched prefill. kv.go:670 routes here on !q8PrefillNeedsTokenLoop, which has no
// LayerNorm term, so a q4_k_m artifact of a biased-LayerNorm family prefills through this
// lane — and all three of its norm sites hard-passed nil.
//
// Fixture shape is dictated by Q4_K: every reduction dim (H, nH*hd, I) must be a multiple
// of qkK=256. The q4kw store is seeded deterministically (fillQ4KW seed 99) so the twin
// below shares byte-identical resident weights.
func TestResidentQ4KPrefillNormBias(t *testing.T) {
	cfg := Config{
		HiddenSize: 256, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, RopeTheta: 10000,
		LayerNorm: true,
	}
	extra := normBiasExtras(cfg)
	if q8PrefillNeedsTokenLoop(cfg) {
		t.Fatal("premise broken: this config no longer routes to prefillBatchedQ4K, so the test is vacuous")
	}

	q4kProjs := func(cfg Config) [][2]any {
		projs := [][2]any{}
		for l := 0; l < cfg.NumLayers; l++ {
			p := layerPrefix(l)
			projs = append(projs,
				[2]any{p + "self_attn.q_proj.weight", cfg.NumHeads * cfg.HeadDim},
				[2]any{p + "self_attn.k_proj.weight", cfg.NumKVHeads * cfg.HeadDim},
				[2]any{p + "self_attn.v_proj.weight", cfg.NumKVHeads * cfg.HeadDim},
				[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
				[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
				[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize},
				[2]any{p + "mlp.down_proj.weight", cfg.HiddenSize},
			)
		}
		return projs
	}

	m := newSyntheticExtra(cfg, extra)
	fillQ4KW(t, m, q4kProjs(cfg), 99)
	requireNonZeroNormBiases(t, m, extra)

	ids := make([]int, 12)
	for i := range ids {
		ids[i] = (i*7 + 3) % cfg.VocabSize
	}
	q4kPrefill := func(mm *Model) []float32 {
		s := mm.NewSession()
		s.Q4K = true
		return append([]float32(nil), s.prefillBatchedQ4K(ids)...)
	}
	got := q4kPrefill(m)
	bias := append([]float32(nil), m.tensor("model.norm.bias")...)

	// Final norm, then each per-layer norm — all three sites probed independently so a
	// regression at one does not hide the other two.
	zeroFinal := newNormBiasTwin(t, cfg, extra, "model.norm.bias")
	fillQ4KW(t, zeroFinal, q4kProjs(cfg), 99)
	assertFinalNormBiasDelta(t, "prefillBatchedQ4K", got, q4kPrefill(zeroFinal), bias)

	for _, suffix := range []string{"input_layernorm.bias", "post_attention_layernorm.bias"} {
		names := make([]string, 0, cfg.NumLayers)
		for l := 0; l < cfg.NumLayers; l++ {
			names = append(names, layerName(l, suffix))
		}
		twin := newNormBiasTwin(t, cfg, extra, names...)
		fillQ4KW(t, twin, q4kProjs(cfg), 99)
		assertZeroingChanged(t, "prefillBatchedQ4K", suffix, got, q4kPrefill(twin))
	}
}

// TestGPTQDecodeFinalNormBias pins tokenHiddenGPTQ (gptq.go), which nothing gates on the
// norm kind at all — kv.go:686 keys only on s.GPTQ, so a GPTQ checkpoint of a biased
// LayerNorm family decodes here. Its blockStep loop applied every per-layer norm bias
// correctly and then hand-rolled a final norm that threw model.norm.bias away.
//
// The projection backend is deliberately left f32-resident (residentKernel prefers the
// exact f32 tensor when it exists, kernel.go:66), because the property under test is the
// final-norm wiring, not the int4 GEMM — and that keeps the fixture weight-free.
func TestGPTQDecodeFinalNormBias(t *testing.T) {
	cfg := normBiasArchConfig()
	extra := normBiasExtras(cfg)
	m := newSyntheticExtra(cfg, extra)
	requireNonZeroNormBiases(t, m, extra)

	gptqHidden := func(mm *Model) []float32 {
		s := mm.NewSession()
		s.GPTQ = true
		return append([]float32(nil), s.tokenHiddenGPTQ(7%cfg.VocabSize, 0)...)
	}
	got := gptqHidden(m)
	if len(got) != cfg.HiddenSize {
		t.Fatalf("tokenHiddenGPTQ hidden len %d, want %d", len(got), cfg.HiddenSize)
	}
	bias := append([]float32(nil), m.tensor("model.norm.bias")...)
	base := gptqHidden(newNormBiasTwin(t, cfg, extra, "model.norm.bias"))
	assertFinalNormBiasDelta(t, "tokenHiddenGPTQ", got, base, bias)
}

// TestQuantPrefillPerLayerNormBias pins the two PER-LAYER norm sites inside
// prefillBatchedQ. Their contribution is not closed-form at the output (it propagates
// through attention and the MLP), so the assertion is the weaker but still decisive one:
// zeroing the bias must CHANGE the result. A lane that never reads the tensor is
// unaffected by zeroing it — which is exactly the pre-fix behaviour, and fails here.
func TestQuantPrefillPerLayerNormBias(t *testing.T) {
	cfg := normBiasArchConfig()
	extra := normBiasExtras(cfg)
	m := newSyntheticExtra(cfg, extra)
	m.Quantize()
	requireNonZeroNormBiases(t, m, extra)
	if q8PrefillNeedsTokenLoop(cfg) {
		t.Fatal("premise broken: this config no longer routes to prefillBatchedQ, so the test is vacuous")
	}

	V := cfg.VocabSize
	ids := make([]int, 9)
	for i := range ids {
		ids[i] = (i*1299709 + 17) % V
	}
	quantPrefill := func(mm *Model) []float32 {
		s := mm.NewSession()
		s.Quant = true
		return append([]float32(nil), s.prefillBatchedQ(ids)...)
	}
	got := quantPrefill(m)

	for _, suffix := range []string{"input_layernorm.bias", "post_attention_layernorm.bias"} {
		names := make([]string, 0, cfg.NumLayers)
		for l := 0; l < cfg.NumLayers; l++ {
			names = append(names, layerName(l, suffix))
		}
		twin := newNormBiasTwin(t, cfg, extra, names...)
		twin.Quantize()
		assertZeroingChanged(t, "prefillBatchedQ", suffix, got, quantPrefill(twin))
	}
}
