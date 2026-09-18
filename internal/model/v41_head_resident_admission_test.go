package model

// v41_head_resident_admission_test.go pins issue #13254: the V4.1 forward head
// stage must admit an LM head that is resident in a quant store (kqw for the
// vcruz Q2_K artifact's untied output.weight), not only one present in the f32
// manifest. Before this fix v41ForwardAdmitted and forwardV41's tail keyed on
// m.has("lm_head.weight") - manifest-only - while v41AdmitShape keyed on
// m.manifest[name] for both presence and shape, so a Q2_K-resident head was
// refused with "missing tensor lm_head.weight" even though it was resident.
//
// These tests COMPILE against the parent commit as well as the fix (they use
// only pre-existing API), so the mandatory red-then-green symptom witness can
// build the parent tree and observe the refusal at runtime.
//
// The fail-closed property is retained: a head absent from EVERY store still
// refuses with the named ErrV41ForwardStage, and a genuinely mis-shaped resident
// head still refuses.

import (
	"errors"
	"testing"
)

// v41ResidentHeadConfig is v41FullGeometryConfig with a hidden width that is a
// multiple of the k-quant block (qkK=256) so a Q6_K resident head is constructible.
func v41ResidentHeadConfig(t *testing.T) Config {
	t.Helper()
	cfg := v41FullGeometryConfig(t)
	cfg.HiddenSize = 2 * qkK
	cfg.NumHeads = 1
	cfg.HeadDim = v41KVLoraRank
	cfg.NumKVHeads = 1
	return cfg
}

// v41BuildHeadlessModel is v41BuildFullModel with the f32 lm_head removed: the
// only head left is the resident k-quant one the caller installs into kqw.
func v41BuildHeadlessModel(cfg Config, kvRank int) *Model {
	m := v41BuildFullModel(cfg, kvRank)
	delete(m.manifest, "lm_head.weight")
	return m
}

// TestV41HeadAdmitsResidentKQuantHead is the #13254 positive witness: an untied
// lm_head resident ONLY in kqw (the Q2_K/Q6_K quantized head) must pass the V4.1
// head-stage admission, and the head resolver must point at the resident copy
// (not fall through to the tied-embedding key). On the parent this fails at the
// admission with "missing tensor lm_head.weight".
func TestV41HeadAdmitsResidentKQuantHead(t *testing.T) {
	cfg := v41ResidentHeadConfig(t)
	m := v41BuildHeadlessModel(cfg, v41KVLoraRank)
	m.kqw = map[string]*kQuantTensor{
		"lm_head.weight": q6kTestTensor(cfg.VocabSize, cfg.HiddenSize, 0x51ed),
	}

	if m.has("lm_head.weight") {
		t.Fatal("fixture invariant broken: lm_head.weight is still in the f32 manifest")
	}
	if !m.hasWeight("lm_head.weight") {
		t.Fatal("fixture invariant broken: kqw-resident head not visible to hasWeight")
	}
	// The core-through-line of #13254: admission must pass the head stage.
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("v41ForwardAdmitted refused a kqw-resident head: %v (want admission)", err)
	}
	// The head resolver must name the real (untied) head, not the tied embedding.
	if got := m.headName(); got != "lm_head.weight" {
		t.Fatalf("headName() = %q, want lm_head.weight (resident kqw head unreachable)", got)
	}

}

// TestV41HeadAbsentFromEveryStoreStillRefuses is the retained fail-closed
// property: with neither a manifest head, a tied embedding, nor any resident
// copy, the head stage must still refuse with the named ErrV41ForwardStage.
func TestV41HeadAbsentFromEveryStoreStillRefuses(t *testing.T) {
	cfg := v41ResidentHeadConfig(t)
	m := v41BuildHeadlessModel(cfg, v41KVLoraRank)
	delete(m.manifest, "model.embed_tokens.weight")

	err := m.v41ForwardAdmitted()
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("absent head admission error = %v, want ErrV41ForwardStage", err)
	}
}

// TestV41HeadResidentMisShapeStillRefuses is the retained shape-guard property: a
// resident head whose out/in disagree with the config still refuses, so the fix
// cannot silently mis-read a wrong-shaped head.
func TestV41HeadResidentMisShapeStillRefuses(t *testing.T) {
	cfg := v41ResidentHeadConfig(t)
	m := v41BuildHeadlessModel(cfg, v41KVLoraRank)
	m.kqw = map[string]*kQuantTensor{
		"lm_head.weight": q6kTestTensor(cfg.VocabSize, cfg.HiddenSize, 0x51ee),
	}
	m.kqw["lm_head.weight"].in = cfg.HiddenSize * 2

	err := m.v41ForwardAdmitted()
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("mis-shaped resident head admission error = %v, want ErrV41ForwardStage", err)
	}
}
