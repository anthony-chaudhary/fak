package model

import (
	"math"
	"testing"
)

// proj_bias_batched_lanes_test.go — the PROJECTION-BIAS axis for the batched lanes, the
// sibling of norm_bias_batched_lanes_test.go (which covers the NORM-bias axis).
//
// The tensors under test are self_attn.o_proj.bias and mlp.{gate,up,down}_proj.bias. They
// are presence-driven, not flag-driven: newModel never filters the manifest down to a
// family's required set (tensor_resolver.go is a standalone read-only API — see its header),
// identitySpec already declares self_attn.o_proj.bias as an optional canonical tensor
// (tensor_resolver.go:192), and every per-token site reads them through
// m.addBiasIfPresent (arch.go:203). So ANY checkpoint that physically carries them gets
// them applied by the per-token path with no config knob involved — which is exactly what a
// Llama-architecture export with HF's attention_bias / mlp_bias knobs turned on ships.
//
// Why the existing contract tests could not catch this: TestPrefillBatchedMatchesSerial
// runs on loadFixture's SmolLM2 export and TestBatchedDecodeMatchesSerial /
// TestPrefillEachRectangularMatchesSerial run on a plain NewSynthetic config. Neither
// carries a single projection-bias tensor, so a lane that drops them is bit-identical to one
// that applies them and the assertion passes vacuously on this axis.
//
// The gates have no bias term either: q8PrefillNeedsTokenLoop (kv.go:825),
// batchPreNormFastPathOK (batch.go:109) and verifyForwardBatchedOK (verify.go:64) all key
// on MoE / DenseMLP / ALiBi / topology / RoPE shape and never on whether the checkpoint
// carries a projection bias. A bias-carrying PreNorm gated-MLP model is therefore admitted
// into every batched lane below.
//
// Fixtures are weight-free (newSyntheticExtra), deliberately: this class survived precisely
// because the checkpoint-backed gates SKIP in CI.

// projBiasArchConfig is the plain Llama-shaped PreNorm SwiGLU config every batched f32 lane
// admits — deliberately the same shape TestBatchedDecodeMatchesSerial uses, so the ONLY
// difference from that already-green test is the projection-bias tensors added below.
func projBiasArchConfig() Config {
	return Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
}

// projBiasExtras names the four projection biases a bias-carrying dense gated-MLP layer
// holds: the attention output projection and all three SwiGLU projections.
func projBiasExtras(cfg Config) map[string][]int {
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	extra := map[string][]int{}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"self_attn.o_proj.bias"] = []int{H}
		extra[p+"mlp.gate_proj.bias"] = []int{I}
		extra[p+"mlp.up_proj.bias"] = []int{I}
		extra[p+"mlp.down_proj.bias"] = []int{H}
	}
	return extra
}

// newProjBiasModel builds the fixture and runs the two guards that keep every case here
// non-vacuous: the biases must actually be PRESENT (the per-token path reads them through
// addBiasIfPresent, so an absent tensor is a silent no-op) and they must be NON-ZERO (an
// all-zero bias is indistinguishable from a dropped one).
func newProjBiasModel(t *testing.T, cfg Config) *Model {
	t.Helper()
	extra := projBiasExtras(cfg)
	m := newSyntheticExtra(cfg, extra)
	for name := range extra {
		if !m.has(name) {
			t.Fatalf("%s is absent: the per-token path would skip it too, test is vacuous", name)
		}
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
	return m
}

// assertProjBiasEqualF32 compares two f32 runs bit-for-bit. The message names the class of
// defect this file exists to catch so a future regression reads as what it is.
func assertProjBiasEqualF32(t *testing.T, what string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: len %d != %d", what, len(want), len(got))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s[%d]: reference %v != batched %v "+
				"(a projection bias the per-token path applies was dropped in the batched lane)",
				what, i, want[i], got[i])
		}
	}
}

// assertProjBiasCaches pins the KV cache too. o_proj.bias and the MLP biases both feed the
// residual stream, so layer l's biases move layer l+1's cached K/V — a lane that matched on
// logits but built a different cache would silently break Evict/Clone/prefix reuse.
func assertProjBiasCaches(t *testing.T, label string, cfg Config, want, got *KVCache) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("%s cache len %d != %d", label, want.Len(), got.Len())
	}
	for l := 0; l < cfg.NumLayers; l++ {
		assertProjBiasEqualF32(t, label+" K", want.K[l], got.K[l])
		assertProjBiasEqualF32(t, label+" Kraw", want.Kraw[l], got.Kraw[l])
		assertProjBiasEqualF32(t, label+" V", want.V[l], got.V[l])
	}
	for i := range want.pos {
		if want.pos[i] != got.pos[i] {
			t.Fatalf("%s pos[%d] %d != %d", label, i, want.pos[i], got.pos[i])
		}
	}
}

// projBiasPrompt is a deterministic in-vocabulary token sequence.
func projBiasPrompt(vocab, n, salt int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = (salt*97 + i*1299709 + 17) % vocab
	}
	return p
}

// TestPrefillBatchedProjBias is the load-bearing case: prefillBatched (prefill_batch.go)
// against the per-token tokenHidden loop it declares itself BIT-IDENTICAL to in its own
// file header. The per-token path is the numerics oracle — denseSwiGLU (moe.go:139/140/145)
// adds all three MLP biases and kv.go:573 adds o_proj.bias — so any difference here is
// prefillBatched's.
func TestPrefillBatchedProjBias(t *testing.T) {
	cfg := projBiasArchConfig()
	// Premise guard: this config must actually REACH the batched lane, or the test proves
	// nothing about production. Prefill routes to prefillBatched exactly when
	// q8PrefillNeedsTokenLoop is false (kv.go:720).
	if q8PrefillNeedsTokenLoop(cfg) {
		t.Fatal("premise broken: this config no longer routes to prefillBatched, so the test is vacuous")
	}
	m := newProjBiasModel(t, cfg)

	prompt := projBiasPrompt(cfg.VocabSize, 12, 0)

	// Reference: the per-token forward (blockStep -> denseSwiGLU, and kv.go's o_proj bias).
	ref := m.NewSession()
	var refX []float32
	for _, id := range prompt {
		refX = ref.tokenHidden(id, ref.Cache.Len())
	}
	refLogits := ref.head(refX)

	bat := m.NewSession()
	batLogits := bat.head(bat.prefillBatched(prompt))

	assertProjBiasEqualF32(t, "prefillBatched logits", refLogits, batLogits)
	assertProjBiasCaches(t, "prefillBatched", cfg, ref.Cache, bat.Cache)
}

// TestBatchedDecodeProjBias pins the batched f32 decode (stepBatchF32, batch_step.go)
// against serial Session.Step, the contract TestBatchedDecodeMatchesSerial states.
//
// The prompts are deliberately NON-rectangular (distinct lengths), so PrefillEach takes its
// per-session fallback (batch_prefill.go:19) and BOTH sides enter the decode loop through
// the identical prefill lane. Any mismatch below is therefore the decode step's alone.
func TestBatchedDecodeProjBias(t *testing.T) {
	cfg := projBiasArchConfig()
	if !batchDecodeFastPathOK(cfg, false) {
		t.Fatal("premise broken: this config no longer routes to stepBatchF32, so the test is vacuous")
	}
	m := newProjBiasModel(t, cfg)

	V := cfg.VocabSize
	B := 4
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		prompts[b] = projBiasPrompt(V, 3+b*2, b+1)
	}
	if _, rect := rectangularPrefillLen(prompts); rect {
		t.Fatal("premise broken: prompts are rectangular, so the rect prefill lane would mask the decode lane")
	}

	ref := make([]*Session, B)
	refLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		ref[b] = m.NewSession()
		refLogits[b] = ref[b].Prefill(prompts[b])
	}
	bs := m.NewBatchSession(B)
	batLogits := bs.PrefillEach(prompts)
	for b := 0; b < B; b++ {
		assertProjBiasEqualF32(t, "shared-prefill user logits", refLogits[b], batLogits[b])
	}

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
			assertProjBiasEqualF32(t, "batched-step user logits", refLogits[b], batLogits[b])
		}
	}
	for b := 0; b < B; b++ {
		assertProjBiasCaches(t, "batched-step user", cfg, ref[b].Cache, bs.Seqs[b].Cache)
	}
}

// TestVerifyForwardProjBias pins verifyForwardBatched (verify.go), the single-pass
// speculative-verify lane, to its own documented contract: the chain logits and appended KV
// are "bit-identical to P sequential Session.Step calls" (verify.go:18).
func TestVerifyForwardProjBias(t *testing.T) {
	cfg := projBiasArchConfig()
	m := newProjBiasModel(t, cfg)

	ids := projBiasPrompt(cfg.VocabSize, 5, 3)

	got := m.NewSession()
	if !verifyForwardBatchedOK(got) {
		t.Fatal("premise broken: this session no longer routes to verifyForwardBatched, so the test is vacuous")
	}
	gotLogits := got.VerifyForward(ids, nil, nil)
	if len(gotLogits) != len(ids) {
		t.Fatalf("VerifyForward returned %d rows, want %d", len(gotLogits), len(ids))
	}

	ref := m.NewSession()
	for i, id := range ids {
		assertProjBiasEqualF32(t, "verify chain logits", ref.Step(id), gotLogits[i])
	}
	assertProjBiasCaches(t, "verify chain", cfg, ref.Cache, got.Cache)
}

// TestBatchRectPrefillProjBias is the LANE-VERSUS-LANE witness, and it is the one that makes
// the divergence undeniable without appealing to any oracle: prefillEachRectF32
// (batch_prefill.go) APPLIES these biases while Session.Prefill's prefillBatched does not,
// so today the two batched prefill lanes disagree with EACH OTHER on the same weights.
// Whichever lane one calls correct, they cannot both be.
func TestBatchRectPrefillProjBias(t *testing.T) {
	cfg := projBiasArchConfig()
	if !batchRectFastPathOK(cfg, false) {
		t.Fatal("premise broken: this config no longer routes to prefillEachRectF32, so the test is vacuous")
	}
	m := newProjBiasModel(t, cfg)

	V := cfg.VocabSize
	B, P := 3, 7
	if P > batchRectPrefillMaxTokens {
		t.Fatalf("premise broken: P=%d exceeds the rect-prefill cap %d", P, batchRectPrefillMaxTokens)
	}
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		prompts[b] = projBiasPrompt(V, P, b+11)
	}
	if _, rect := rectangularPrefillLen(prompts); !rect {
		t.Fatal("premise broken: prompts are not rectangular, so prefillEachRectF32 is never entered")
	}

	ref := make([]*Session, B)
	refLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		ref[b] = m.NewSession()
		refLogits[b] = ref[b].Prefill(prompts[b])
	}
	bs := m.NewBatchSession(B)
	batLogits := bs.PrefillEach(prompts)
	for b := 0; b < B; b++ {
		assertProjBiasEqualF32(t, "rect-prefill user logits", refLogits[b], batLogits[b])
		assertProjBiasCaches(t, "rect-prefill user", cfg, ref[b].Cache, bs.Seqs[b].Cache)
	}
}

// TestBatchedDecodeQProjBias covers the Q8_0 multi-user decode (stepBatchQ, batch_step.go),
// the one batched lane where bit-equality with the per-token path is the WRONG oracle: it is
// deliberately not bit-identical to the serial qdot8 decode because the tile GEMM reduces in
// a different lane order (its own header, and TestBatchedDecodeQMatchesF32, say so).
//
// The exact statement that IS available on this axis is that the biases must be LOAD-BEARING.
// Both arms prefill through the SAME model with the biases intact, so their KV caches are
// byte-identical; only then are the bias tensors zeroed in place, and only the decode step
// sees the change. If stepBatchQ reads the biases the two decodes must differ. The pre-fix
// lane never read them — it went straight from ql.oProj / ql.gateProj / ql.upProj /
// ql.downProj to the residual — so the two decodes came out bit-equal.
//
// Zeroing in place is safe here: the biases are 1-D and Model.Quantize only quantizes 2-D
// weights (quant.go:274), so the Q8 weight store is untouched by it.
func TestBatchedDecodeQProjBias(t *testing.T) {
	cfg := projBiasArchConfig()
	if !batchDecodeFastPathOK(cfg, true) {
		t.Fatal("premise broken: this config no longer routes to stepBatchQ, so the test is vacuous")
	}
	m := newProjBiasModel(t, cfg)
	m.Quantize()

	V := cfg.VocabSize
	B := 3
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		prompts[b] = projBiasPrompt(V, 6, b+21)
	}
	next := make([]int, B)
	for b := 0; b < B; b++ {
		next[b] = (b*31 + 5) % V
	}

	// Two independently prefilled Q8 batch sessions over the same model and prompts: same
	// weights, same lane, so their caches and their decode inputs are identical.
	newPrefilled := func() *BatchSession {
		bs := m.NewBatchSession(B)
		bs.SetQuant(true)
		bs.PrefillEach(prompts)
		return bs
	}
	withBias := newPrefilled()
	zeroed := newPrefilled()

	// StepBatch's rows alias a reused per-session buffer, so snapshot before the second run.
	got := withBias.StepBatch(next)
	before := make([][]float32, B)
	for b := 0; b < B; b++ {
		before[b] = append([]float32(nil), got[b]...)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		for _, n := range []string{
			"self_attn.o_proj.bias", "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias",
		} {
			clear(m.tensor(layerName(l, n)))
		}
	}
	after := zeroed.StepBatch(next)

	for b := 0; b < B; b++ {
		same := len(before[b]) == len(after[b])
		for i := range before[b] {
			if math.Float32bits(before[b][i]) != math.Float32bits(after[b][i]) {
				same = false
				break
			}
		}
		if same {
			t.Fatalf("Q8 batched decode user %d: zeroing self_attn.o_proj.bias and "+
				"mlp.{gate,up,down}_proj.bias left the logits bit-identical, so this lane "+
				"never reads them — the per-token Q8 decode (quant_forward.go:312/329/330/337) "+
				"and the Q8 rect prefill (batch_prefill.go:291/323/329) both do", b)
		}
	}
}
