package model

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// v4quant_admit_test.go — the executable witnesses the #3019 acceptance asks for:
// a Go fixture parses representative V4 tensor metadata WITHOUT downloading weights,
// the admission gate classifies every tensor and asserts its precision, and it FAILS
// CLOSED on an unrecognized FP4 tensor class. See docs/deepseek/v4-fp4-quant-support-plan.md.

// v4FixtureIndex is the shape of testdata/deepseek_v4_tensor_index.json — a metadata
// -only V4 tensor index (names + dtypes + shapes, no weights).
type v4FixtureIndex struct {
	Model   string          `json:"model"`
	Tensors []V4TensorMeta  `json:"tensors"`
	Quant   json.RawMessage `json:"quantization_config"`
}

func loadV4Fixture(t *testing.T) v4FixtureIndex {
	t.Helper()
	path := filepath.Join("testdata", "deepseek_v4_tensor_index.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var idx v4FixtureIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		t.Fatalf("parse fixture %s: %v", path, err)
	}
	if len(idx.Tensors) == 0 {
		t.Fatalf("fixture %s carries no tensors", path)
	}
	return idx
}

// TestV4AdmitFixtureClassifies is acceptance item 1: the metadata fixture parses and
// every tensor admits into its expected class at its expected precision, with the two
// FP4-bearing classes (routed experts, indexer QK) coming in FP4 and the rest FP8/HIGH.
func TestV4AdmitFixtureClassifies(t *testing.T) {
	idx := loadV4Fixture(t)
	rep, err := AdmitV4Checkpoint(idx.Model, idx.Tensors)
	if err != nil {
		t.Fatalf("representative V4 index must admit cleanly, got refusal: %v", err)
	}
	if !rep.OK {
		t.Fatalf("report not OK: %+v", rep)
	}
	if rep.TotalTensors != len(idx.Tensors) {
		t.Fatalf("TotalTensors=%d, want %d", rep.TotalTensors, len(idx.Tensors))
	}
	if rep.Admitted+rep.Skipped != rep.TotalTensors {
		t.Fatalf("admitted(%d)+skipped(%d) != total(%d)", rep.Admitted, rep.Skipped, rep.TotalTensors)
	}

	// The FP4-bearing classes must be exactly the two the plan names, and they must be
	// FP4-only — the property that distinguishes V4 from an all-FP8 checkpoint.
	for _, class := range []V4TensorClass{V4ClassRoutedExpert, V4ClassIndexerQK} {
		if rep.ClassCounts[class] == 0 {
			t.Errorf("fixture should exercise FP4 class %s but count is 0", class)
		}
		if got := rep.ClassPrec[class]; len(got) != 1 || got[0] != string(V4FP4) {
			t.Errorf("class %s precisions = %v, want [FP4] only", class, got)
		}
	}
	// The FP8 weight classes must be FP8-only.
	for _, class := range []V4TensorClass{V4ClassSharedExpert, V4ClassAttention, V4ClassRouter, V4ClassDenseFFN} {
		if rep.ClassCounts[class] == 0 {
			t.Errorf("fixture should exercise FP8 class %s but count is 0", class)
		}
		if got := rep.ClassPrec[class]; len(got) != 1 || got[0] != string(V4FP8) {
			t.Errorf("class %s precisions = %v, want [FP8] only", class, got)
		}
	}
	// Norms / embeddings are kept high precision, never FP4.
	for _, class := range []V4TensorClass{V4ClassNorm, V4ClassEmbedding} {
		for _, p := range rep.ClassPrec[class] {
			if p == string(V4FP4) {
				t.Errorf("high-precision class %s must never be FP4, saw %v", class, rep.ClassPrec[class])
			}
		}
	}
	// The MTP head and the vision tower are the families the loader drops at load; they
	// must be recognized and SKIPPED, never admitted or refused.
	if rep.ClassCounts[V4ClassMTP] == 0 || rep.ClassCounts[V4ClassVision] == 0 {
		t.Errorf("expected both MTP and vision tensors skipped; mtp=%d vision=%d", rep.ClassCounts[V4ClassMTP], rep.ClassCounts[V4ClassVision])
	}
	if rep.Skipped != rep.ClassCounts[V4ClassMTP]+rep.ClassCounts[V4ClassVision] {
		t.Errorf("skipped=%d should equal mtp(%d)+vision(%d)", rep.Skipped, rep.ClassCounts[V4ClassMTP], rep.ClassCounts[V4ClassVision])
	}
}

// TestV4AdmitFailsClosedOnUnknownFP4Class is the headline acceptance property: an FP4
// -typed tensor whose name classifies into NO known FP4-bearing class is refused with
// a typed *UnsupportedFP4TensorError, not silently admitted.
func TestV4AdmitFailsClosedOnUnknownFP4Class(t *testing.T) {
	idx := loadV4Fixture(t)
	// Splice one unrecognized FP4 tensor into an otherwise-valid index.
	poisoned := append([]V4TensorMeta{}, idx.Tensors...)
	poisoned = append(poisoned, V4TensorMeta{
		Name:  "model.layers.7.self_attn.mystery_fp4_proj.weight",
		Dtype: "F4_E2M1",
		Shape: []int{4096, 7168},
	})

	rep, err := AdmitV4Checkpoint(idx.Model, poisoned)
	if err == nil {
		t.Fatalf("an unrecognized FP4 tensor must fail closed, but admission passed: %+v", rep)
	}
	var fp4Err *UnsupportedFP4TensorError
	if !errors.As(err, &fp4Err) {
		t.Fatalf("want *UnsupportedFP4TensorError, got %T: %v", err, err)
	}
	if fp4Err.Why != "unrecognized FP4 tensor class" {
		t.Errorf("Why = %q, want %q", fp4Err.Why, "unrecognized FP4 tensor class")
	}
	if rep.OK || rep.Refusal == nil || rep.Refusal.Disposition != V4Refuse {
		t.Errorf("report should carry the fail-closed refusal, got %+v", rep)
	}
}

// TestV4AdmitFailsClosedOnPrecisionMismatch pins the other half of "fail closed":
// a recognized class whose declared precision is outside its allow-set is refused —
// this is the guard against silently treating V4 as an all-FP8 (or all-FP4) model.
func TestV4AdmitFailsClosedOnPrecisionMismatch(t *testing.T) {
	cases := []struct {
		name string
		meta V4TensorMeta
	}{
		{"routed expert must be FP4 not FP8", V4TensorMeta{
			Name: "model.layers.3.mlp.experts.1.gate_proj.weight", Dtype: "F8_E4M3"}},
		{"attention must be FP8 not FP4", V4TensorMeta{
			Name: "model.layers.3.self_attn.o_proj.weight", Dtype: "F4_E2M1"}},
		{"norm must never be FP4", V4TensorMeta{
			Name: "model.layers.3.input_layernorm.weight", Dtype: "F4_E2M1"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := AdmitV4Checkpoint("t", []V4TensorMeta{tc.meta})
			var fp4Err *UnsupportedFP4TensorError
			if !errors.As(err, &fp4Err) {
				t.Fatalf("want *UnsupportedFP4TensorError, got %T: %v", err, err)
			}
			if fp4Err.Class == "" {
				t.Errorf("a precision-mismatch refusal should name the class it placed the tensor in")
			}
		})
	}
}

// TestV4AdmitFailsClosedOnUnknownDtype: a tensor carrying a dtype tag the loader does
// not recognize is refused rather than waved through at some default precision.
func TestV4AdmitFailsClosedOnUnknownDtype(t *testing.T) {
	_, err := AdmitV4Checkpoint("t", []V4TensorMeta{
		{Name: "model.layers.0.self_attn.o_proj.weight", Dtype: "WAT_9BIT"},
	})
	var fp4Err *UnsupportedFP4TensorError
	if !errors.As(err, &fp4Err) {
		t.Fatalf("want *UnsupportedFP4TensorError for an unknown dtype, got %T: %v", err, err)
	}
}

// TestV4AdmitConsistentWithQuantGate ties the new class map to the REAL loader: every
// tensor this gate admits at a quantized precision (FP4/FP8) must be a weight the
// existing quantize-on-load gate (isQuantWeight) also claims, and every high-precision
// tensor must be one it leaves as f32. This keeps the admission table one description
// of the loader's layout, not a second guess that could drift out of agreement.
func TestV4AdmitConsistentWithQuantGate(t *testing.T) {
	idx := loadV4Fixture(t)
	for _, tm := range idx.Tensors {
		v, err := admitV4Tensor(tm)
		if err != nil {
			t.Fatalf("fixture tensor %s unexpectedly refused: %v", tm.Name, err)
		}
		if v.Disposition != V4Admit {
			continue // SKIP (MTP) tensors are not loaded either way
		}
		quant := isQuantWeight(tm.Name)
		switch v.Precision {
		case V4FP4, V4FP8:
			if !quant {
				t.Errorf("%s admits as %s but isQuantWeight()=false — class map disagrees with loader", tm.Name, v.Precision)
			}
		case V4High:
			if quant {
				t.Errorf("%s admits as HIGH but isQuantWeight()=true — class map disagrees with loader", tm.Name)
			}
		}
	}
}

// TestClassifyV4Tensor is a compact table over the canonical names, so a rename in the
// loader that breaks a class assignment fails here with a precise message.
func TestClassifyV4TensorOfficialFFNNamespace(t *testing.T) {
	tests := []struct {
		name  string
		class V4TensorClass
		ok    bool
	}{
		{"model.layers.0.ffn.experts.0.w1.weight", V4ClassRoutedExpert, true},
		{"model.layers.60.ffn.shared_experts.w1.weight", V4ClassSharedExpert, true},
		{"model.layers.1.ffn.gate.weight", V4ClassRouter, true},
		// Exact dotted components are required; broad "experts" or gate matching
		// would silently admit unknown pinned-checkpoint tensors.
		{"model.layers.0.ffn.expert.0.w1.weight", "", false},
		{"model.layers.0.ffn.experts", "", false},
		{"model.layers.0.ffn.gate.weight.extra", "", false},
		{"model.layers.0.notffn.experts.0.w1.weight", "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := classifyV4Tensor(tc.name)
			if ok != tc.ok || got != tc.class {
				t.Fatalf("classifyV4Tensor(%q) = (%q, %v), want (%q, %v)", tc.name, got, ok, tc.class, tc.ok)
			}
		})
	}
}

func TestClassifyV4Tensor(t *testing.T) {
	cases := []struct {
		name  string
		class V4TensorClass
		ok    bool
	}{
		{"model.layers.3.mlp.experts.12.up_proj.weight", V4ClassRoutedExpert, true},
		{"model.layers.3.mlp.shared_experts.down_proj.weight", V4ClassSharedExpert, true},
		{"model.layers.3.self_attn.indexer.wq_b.weight", V4ClassIndexerQK, true},
		{"model.layers.3.self_attn.indexer.k_norm.weight", V4ClassNorm, true},
		{"model.layers.3.self_attn.kv_b_proj.weight", V4ClassAttention, true},
		{"model.layers.3.mlp.gate.weight", V4ClassRouter, true},
		{"model.layers.0.mlp.down_proj.weight", V4ClassDenseFFN, true},
		{"lm_head.weight", V4ClassHead, true},
		{"model.embed_tokens.weight", V4ClassEmbedding, true},
		{"mtp.head.weight", V4ClassMTP, true},
		{"mtp.0.embed_tokens.weight", V4ClassMTP, true},
		{"model.visual.patch_embed.proj.weight", V4ClassVision, true},
		{"model.layers.3.mlp.gate.e_score_correction_bias", V4ClassNorm, true},
		{"model.layers.7.self_attn.mystery_fp4_proj.weight", "", false},
	}
	for _, tc := range cases {
		got, ok := classifyV4Tensor(tc.name)
		if ok != tc.ok || got != tc.class {
			t.Errorf("classifyV4Tensor(%q) = (%q,%v), want (%q,%v)", tc.name, got, ok, tc.class, tc.ok)
		}
	}
}

// TestV4RetainedMTPFloorsPrecision is the #4353 witness: a RETAINED MTP/draft head
// (RetainMTP set) is floored to a non-FP4 minimum. It REFUSES an FP4/int4 admission —
// which would collapse self-speculation acceptance so the draft head buys nothing — and
// ADMITS at the FP8 floor (and at HIGH above it). With RetainMTP clear the head is still
// dropped (SKIP) exactly as before, so the floor is inert on every non-scaffold load and
// changes no unrelated class. The floor is set by measured draft-acceptance, not by a
// GEMV-cosine proxy; those acceptance numbers are the DGX-gated deferred witness.
func TestV4RetainedMTPFloorsPrecision(t *testing.T) {
	orig := RetainMTP
	defer func() { RetainMTP = orig }()

	const draftHead = "mtp.head.weight"

	// Retained: FP4/int4 is BELOW the floor and must fail closed with a typed refusal
	// that names the MTP class.
	RetainMTP = true
	if v, err := admitV4Tensor(V4TensorMeta{Name: draftHead, Dtype: "F4_E2M1"}); err == nil {
		t.Fatalf("retained MTP head at FP4 must REFUSE (below floor), got %+v", v)
	} else {
		var fp4Err *UnsupportedFP4TensorError
		if !errors.As(err, &fp4Err) {
			t.Fatalf("want *UnsupportedFP4TensorError, got %T: %v", err, err)
		}
		if fp4Err.Class != V4ClassMTP {
			t.Errorf("floor refusal should name class %s, got %q", V4ClassMTP, fp4Err.Class)
		}
	}

	// Retained: FP8 (the floor itself) and HIGH (above it) must ADMIT as the MTP class.
	for _, dtype := range []string{"F8_E4M3", "BF16"} {
		v, err := admitV4Tensor(V4TensorMeta{Name: draftHead, Dtype: dtype})
		if err != nil {
			t.Fatalf("retained MTP head at %s must ADMIT (>= floor), got refusal: %v", dtype, err)
		}
		if v.Disposition != V4Admit {
			t.Errorf("retained MTP head at %s disposition=%s, want ADMIT", dtype, v.Disposition)
		}
		if v.Class != V4ClassMTP {
			t.Errorf("retained MTP head at %s class=%s, want %s", dtype, v.Class, V4ClassMTP)
		}
	}

	// Not retained (the default): the head is dropped regardless of dtype — even an FP4
	// tensor is SKIPPED, never refused, so the floor never bites an unretained checkpoint.
	RetainMTP = false
	v, err := admitV4Tensor(V4TensorMeta{Name: draftHead, Dtype: "F4_E2M1"})
	if err != nil {
		t.Fatalf("unretained MTP head must SKIP, not refuse, got %v", err)
	}
	if v.Disposition != V4Skip {
		t.Errorf("unretained MTP head disposition=%s, want SKIP", v.Disposition)
	}
}

// TestV4AdmitAgreesWithLoaderDrop pins the admission gate to the REAL loader's drop
// predicate: every tensor skipLoadTensor drops (the mtp.* speculative-decode head and
// the model.visual.* multimodal tower) must be SKIPPED by the gate, never refused —
// otherwise a real GLM/DeepSeek checkpoint would trip the fail-closed path on a tensor
// the loader legitimately handles. The two agree by construction on the same prefixes;
// this test is the "one code path with the real loader" guarantee the plan asks for.
func TestV4AdmitAgreesWithLoaderDrop(t *testing.T) {
	orig := RetainMTP
	RetainMTP = false
	defer func() { RetainMTP = orig }()

	glmCfg := Config{ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"}}
	for _, name := range []string{"mtp.0.embed.weight", "mtp.head.weight", "model.visual.encoder.weight"} {
		if !skipLoadTensor(glmCfg, name) {
			t.Fatalf("precondition: loader skipLoadTensor(%q)=false, want true", name)
		}
		v, err := admitV4Tensor(V4TensorMeta{Name: name, Dtype: "F8_E4M3"})
		if err != nil {
			t.Errorf("loader drops %q but admission REFUSED it: %v", name, err)
			continue
		}
		if v.Disposition != V4Skip {
			t.Errorf("loader drops %q but admission disposition=%s, want SKIP", name, v.Disposition)
		}
	}
}
