package model

import (
	"encoding/json"
	"math"
	"testing"
)

// syntheticDecodeCfg is the tiny CPU-resident checkpoint shape the cache tests use
// (evict_test.go): real KV/decode wiring, no weights file. Vocab 97 keeps argmax
// fast; the logits are meaningless but the SELECTION path is the production one.
func syntheticDecodeCfg() Config {
	return Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        97,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1, // never stop early; compare the full continuation
	}
}

// TestSampleConstrainedBitExactOff is the load-bearing #929 criterion at the unit
// level: with no bias and no active mask, the constrained sampler returns the EXACT
// token argmaxF32 (the unconstrained greedy selection) would — same value, same
// first-max tie-break — over adversarial logit vectors including ties, negatives,
// and -inf entries. A nil constraint and an empty constraint are both proven no-ops.
func TestSampleConstrainedBitExactOff(t *testing.T) {
	vectors := [][]float32{
		{0},
		{1, 2, 3, 4, 5},
		{5, 4, 3, 2, 1},
		{-1, -2, -3},
		{2, 2, 2, 2}, // all tied -> first wins
		{1, 9, 9, 3}, // tie at the max -> first max wins
		{float32(math.Inf(-1)), 0.5, float32(math.Inf(-1))},
		{-0.0, 0.0, -0.0},
		{1e-7, 2e-7, 1.5e-7},
	}
	history := []int{7, 3}
	for vi, v := range vectors {
		want := argmaxF32(v)
		for _, c := range []*DecodeConstraint{nil, {}, {Bias: LogitBias{}}} {
			if got := sampleConstrained(history, v, c); got != want {
				t.Errorf("vector %d constraint %+v: sampleConstrained=%d, argmaxF32=%d (must be bit-exact-off)", vi, c, got, want)
			}
		}
		// A compiled-in mask that is DORMANT (feature flag off) must also be a no-op.
		t.Setenv("FAK_NATIVE_GUIDED_DECODE", "0")
		dormant := &DecodeConstraint{Mask: AllowedSetMask{0: true}}
		if dormant.Active() {
			t.Fatalf("vector %d: constraint with flag-off mask reports Active()=true", vi)
		}
		if got := sampleConstrained(history, v, dormant); got != want {
			t.Errorf("vector %d: dormant mask changed the token (%d != %d) — flag-off must be bit-exact", vi, got, want)
		}
	}
}

// TestGenerateConstrainedBitExactOff is the #929 acceptance wording exactly: a
// fixed-prompt continuation with the constraint hook compiled-in but INACTIVE,
// diffed against internal/model's current greedy oracle (Session.Generate). nil,
// empty, and flag-off-mask constraints must all reproduce Generate token-for-token.
func TestGenerateConstrainedBitExactOff(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "0") // a compiled-in mask stays dormant
	m := NewSynthetic(syntheticDecodeCfg())
	prompt := []int{3, 17, 5, 23, 41}
	const n = 12

	oracle := m.NewSession().Generate(prompt, n)
	if len(oracle) == 0 {
		t.Fatal("greedy oracle produced no tokens")
	}
	cases := map[string]*DecodeConstraint{
		"nil":          nil,
		"empty":        {},
		"empty-bias":   {Bias: LogitBias{}},
		"dormant-mask": {Mask: AllowedSetMask{1: true, 2: true}}, // flag off => no-op
	}
	for name, c := range cases {
		got := m.NewSession().GenerateConstrained(prompt, n, c)
		if !eq(got, oracle) {
			t.Errorf("%s: GenerateConstrained=%v != greedy oracle %v (bit-exact-off broken)", name, got, oracle)
		}
	}
}

// TestLogitBiasActive witnesses #929 item 1 on a small CPU-resident model: a -100
// bias on the natural argmax winner removes it; a +100 bias forces a reachable
// token. Asserted both at the sampler boundary (full control) and through the real
// GenerateConstrained decode loop.
func TestLogitBiasActive(t *testing.T) {
	// Sampler boundary: total control over the logits.
	logits := []float32{0.1, 0.9, 0.2, 0.3} // natural argmax = index 1
	if w := argmaxF32(logits); w != 1 {
		t.Fatalf("setup: natural argmax = %d, want 1", w)
	}
	// -100 removes the winner -> runner-up (index 3, value 0.3) wins.
	if got := sampleConstrained(nil, logits, &DecodeConstraint{Bias: LogitBias{1: -100}}); got != 3 {
		t.Errorf("logit_bias -100 on the winner: argmax=%d, want runner-up 3", got)
	}
	// +100 forces an otherwise-losing token.
	if got := sampleConstrained(nil, logits, &DecodeConstraint{Bias: LogitBias{0: +100}}); got != 0 {
		t.Errorf("logit_bias +100 on token 0: argmax=%d, want 0", got)
	}
	// Bias beyond the OpenAI bound is clamped, not unbounded — but still decisive.
	if got := sampleConstrained(nil, logits, &DecodeConstraint{Bias: LogitBias{2: +1000}}); got != 2 {
		t.Errorf("clamped +1000 bias on token 2: argmax=%d, want 2", got)
	}

	// Decode loop: a +100 bias forces a chosen first token on the synthetic model.
	m := NewSynthetic(syntheticDecodeCfg())
	natural := m.NewSession().Generate([]int{3, 17, 5}, 1)[0]
	target := (natural + 1) % syntheticDecodeCfg().VocabSize
	forced := m.NewSession().GenerateConstrained([]int{3, 17, 5}, 1, &DecodeConstraint{Bias: LogitBias{target: +100}})
	if forced[0] != target {
		t.Errorf("GenerateConstrained with +100 bias on %d: first token=%d, want forced %d (natural was %d)", target, forced[0], target, natural)
	}
	removed := m.NewSession().GenerateConstrained([]int{3, 17, 5}, 1, &DecodeConstraint{Bias: LogitBias{natural: -100}})
	if removed[0] == natural {
		t.Errorf("GenerateConstrained with -100 bias on the natural winner %d still emitted it", natural)
	}
}

// TestSchemaMaskFlagDefaultsOff witnesses the "flag defaults OFF" half of #929: with
// FAK_NATIVE_GUIDED_DECODE unset/0, a non-nil schema mask is dormant and the decode
// is the unconstrained argmax — a compiled-in constraint never silently fires.
func TestSchemaMaskFlagDefaultsOff(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "0")
	// rawLogits: byte 'X' (88) is the unconstrained winner everywhere.
	raw := make([]float32, 256)
	raw['X'] = 10
	// A mask that would force '{' (123) — but it is dormant with the flag off.
	c := &DecodeConstraint{Mask: AllowedSetMask{'{': true}}
	if c.Active() {
		t.Fatal("constraint with flag-off mask reports Active()=true")
	}
	if got := sampleConstrained(nil, raw, c); got != 'X' {
		t.Errorf("flag off: sampleConstrained=%d, want unconstrained winner 'X'(%d) — mask must be dormant", got, 'X')
	}
}

// TestSchemaMaskActiveDecodesValidJSON is the #929 item-2 witness: under the feature
// flag, a constrained decode CANNOT emit a token that leaves the JSON path, and the
// emitted call parses (a real masked decode, not a stub). It drives the sampler
// boundary step-by-step over a byte vocab with a per-step StepMask compiled from the
// canonical `{"name":...,"arguments":{}}` tool-call shape, with the UNCONSTRAINED
// argmax wired to a path-breaking byte at every step — so the mask, not the model, is
// what keeps the output well-formed.
func TestSchemaMaskActiveDecodesValidJSON(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1")

	target := []byte(`{"name":"x","arguments":{}}`)
	per := make([]map[int]bool, len(target))
	for i, b := range target {
		per[i] = map[int]bool{int(b): true} // step i may only emit target[i]
	}
	mask := &StepMask{PerStep: per}
	c := &DecodeConstraint{Mask: mask}
	if !c.Active() {
		t.Fatal("flag on + non-nil mask: constraint must be Active()")
	}

	// At every step the UNCONSTRAINED argmax is 'X' (88), a path-breaking byte. The
	// mask must override it to the schema-valid byte.
	var out []byte
	history := []int{}
	maskWasLoadBearing := false
	for step := 0; step < len(target); step++ {
		raw := make([]float32, 256)
		raw['X'] = 10 // path-breaking unconstrained winner
		if argmaxF32(raw) != int(target[step]) {
			maskWasLoadBearing = true // without the mask this step would break the path
		}
		tok := sampleConstrained(history, raw, c)
		if !per[step][tok] {
			t.Fatalf("step %d: emitted token %d is outside the allowed set %v — decode left the JSON path", step, tok, per[step])
		}
		out = append(out, byte(tok))
		history = append(history, tok)
	}
	if !maskWasLoadBearing {
		t.Fatal("mask was vacuous: the unconstrained argmax already stayed on the path")
	}

	// The emitted call parses and has the canonical tool-call shape the whole-turn
	// gate receives — well-formed by construction, not parsed-and-repaired. (The
	// adjudication step itself lives in internal/grammar, exercised by the gateway
	// witnesses; here we prove the model sink emits a gate-ready candidate.)
	if string(out) != string(target) {
		t.Fatalf("masked decode = %q, want %q", out, target)
	}
	var call struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(out, &call); err != nil {
		t.Fatalf("masked decode did not parse as JSON: %v", err)
	}
	if call.Name != "x" {
		t.Errorf("parsed tool-call name = %q, want \"x\"", call.Name)
	}
	if len(call.Arguments) == 0 {
		t.Error("parsed tool-call has no arguments object")
	}
}

// TestGenerateConstrainedHonorsMaskInLoop proves the full decode LOOP (not just the
// sampler) honors an active mask: every generated token stays in the allowed set.
func TestGenerateConstrainedHonorsMaskInLoop(t *testing.T) {
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1")
	m := NewSynthetic(syntheticDecodeCfg())
	allowed := AllowedSetMask{5: true, 11: true, 23: true, 42: true}
	out := m.NewSession().GenerateConstrained([]int{3, 17, 5}, 16, &DecodeConstraint{Mask: allowed})
	if len(out) == 0 {
		t.Fatal("constrained decode produced no tokens")
	}
	for i, tok := range out {
		if !allowed[tok] {
			t.Errorf("step %d: token %d not in the allowed set %v — loop ignored the mask", i, tok, allowed)
		}
	}
}
