package model

import (
	"bytes"
	"testing"
)

// v41_streamed_expert_seam_test.go — the #13275 regression guard for the R5
// streamed-experts arm of the native non-MLA DeepSeek-V4.1 forward.
//
// #13271 (36ce487f1) made the eager batched-expert *emitter*
// (ggufload.batchedExpertCanonicalName) arch-aware, so a resident serve resolves
// the V4 ffn.experts.<e>.{w1,w3,w2} leaves. But the physical strix3 serve with
// --cpu-offload-experts FAK_STREAM_Q4K=1 selects the STREAMED arm, and there the
// seam was never wired:
//
//   - ExpertCheckpointTier.AddShardData indexed every streamed expert under
//     expertName(), which emits the GLM mlp.experts.<e>.<proj>.weight spelling for
//     EVERY arch — so the V4 forward's ffn.experts.<e>.w1.weight lookup missed; and
//   - v41AdmitShape resolved presence from residentShape(), which by construction
//     excludes the checkpoint tier, so a streamed expert was refused by name even
//     when the tier could serve it.
//
// The fix: expertNameArch (arch-aware index key), FusedExpertTensor.Arch threaded
// from cfg.ModelType, ExpertCheckpointTier.Has (index-only presence), and the
// v41AdmitShape resident-miss fallback to the tier. These tests pin all four and
// are RED on the parent commit 36ce487f1.

// v41StreamedTierQ2K builds a streamed tier carrying ONE fused Q2_K routed-expert
// slab (the V4.1 Flash routed-expert quant) under the given arch, enough to
// exercise the index spelling and Has() without any payload being read.
func v41StreamedTierQ2K(t *testing.T, arch, proj string, experts int) *ExpertCheckpointTier {
	t.Helper()
	const rows, cols = qkK, qkK
	nblk := cols / qkK
	stride := int64(rows * nblk * q2kBlockBytes)
	blob := make([]byte, int64(experts)*stride)
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: proj, Arch: arch,
		Quant: ExpertCheckpointQ2K, Offset: 0, Experts: experts, Rows: rows, Cols: cols,
	}}); err != nil {
		t.Fatalf("AddShard over a well-formed Q2_K slab (arch=%q): %v", arch, err)
	}
	return tier
}

// TestV41ExpertNameArchSpelling pins expertNameArch directly against the same
// table ggufload.batchedExpertCanonicalName uses: deepseek41 maps the GGUF
// projection onto the V4 w1/w3/w2 ffn leaves; every other (and the empty) arch
// keeps the historical mlp.experts spelling byte-for-byte.
func TestV41ExpertNameArchSpelling(t *testing.T) {
	cases := []struct {
		arch string
		proj string
		leaf string
		want string
	}{
		{"deepseek41", "gate_proj", "w1", "model.layers.0.ffn.experts.0.w1.weight"},
		{"deepseek41", "up_proj", "w3", "model.layers.0.ffn.experts.0.w3.weight"},
		{"deepseek41", "down_proj", "w2", "model.layers.0.ffn.experts.0.w2.weight"},
		// negative arm: every non-deepseek41 arch keeps the GLM spelling.
		{"", "gate_proj", "gate_proj", "model.layers.0.mlp.experts.0.gate_proj.weight"},
		{"glm_moe_dsa", "up_proj", "up_proj", "model.layers.0.mlp.experts.0.up_proj.weight"},
		{"deepseek2", "down_proj", "down_proj", "model.layers.0.mlp.experts.0.down_proj.weight"},
	}
	for _, c := range cases {
		if got := expertNameArch(c.arch, 0, 0, c.proj+".weight"); got != c.want {
			t.Errorf("expertNameArch(%q,0,0,%q.weight) = %q, want %q", c.arch, c.proj, got, c.want)
		}
	}
	// expertName must stay the default-arch wrapper — byte-identical for callers.
	if got, want := expertName(3, 7, "gate_proj.weight"), "model.layers.3.mlp.experts.7.gate_proj.weight"; got != want {
		t.Errorf("expertName(3,7,gate_proj.weight) = %q, want the unchanged default spelling %q", got, want)
	}
}

// TestExpertCheckpointTierIndexesArchAwareNames is the GAP 2 witness: AddShardData
// must key a deepseek41 slab under the native ffn.experts.<e>.<w> leaf the V4.1
// forward asks for, and an empty/GLM slab under the historical mlp.experts key.
func TestExpertCheckpointTierIndexesArchAwareNames(t *testing.T) {
	v41 := v41StreamedTierQ2K(t, "deepseek41", "gate_proj", 2)
	if !v41.Has("model.layers.0.ffn.experts.0.w1.weight") {
		t.Error("deepseek41 tier does not index model.layers.0.ffn.experts.0.w1.weight — the V4.1 forward would refuse the streamed expert by name")
	}
	if !v41.Has("model.layers.0.ffn.experts.1.w1.weight") {
		t.Error("deepseek41 tier does not index expert 1's ffn.experts.1.w1.weight")
	}
	if v41.Has("model.layers.0.mlp.experts.0.gate_proj.weight") {
		t.Error("deepseek41 tier still indexes the GLM mlp.experts spelling — the loader/forward names would drift")
	}

	glm := v41StreamedTierQ2K(t, "", "gate_proj", 2)
	if !glm.Has("model.layers.0.mlp.experts.0.gate_proj.weight") {
		t.Error("empty-arch tier lost the historical mlp.experts.0.gate_proj.weight key")
	}
	if glm.Has("model.layers.0.ffn.experts.0.w1.weight") {
		t.Error("empty-arch tier indexed a V4 ffn.experts name — the arch gate did not hold")
	}

	// A nil tier (the no-tier default) must answer false, not panic.
	var nilTier *ExpertCheckpointTier
	if nilTier.Has("model.layers.0.ffn.experts.0.w1.weight") {
		t.Error("nil tier Has() = true, want false")
	}
}

// TestV41AdmitShapeFallsBackToStreamedTier is the GAP 1 witness, run through the
// full admission loop. The reduced V4.1 fixture is admitted with its routed
// experts resident in the manifest; once layer-0 expert 0's w1 is moved out of
// every resident store and ONLY a streamed tier carries it, v41ForwardAdmitted
// must still succeed — and must go back to refusing by name when no tier holds it.
func TestV41AdmitShapeFallsBackToStreamedTier(t *testing.T) {
	m := v41ReducedModel(t)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("baseline reduced V4.1 model is not admitted: %v", err)
	}

	name := layerName(0, "ffn.experts.0.w1.weight")
	want := []int{m.Cfg.MoEIntermediateSize, m.Cfg.HiddenSize}

	// The tensor is resident: admission resolves its real shape from the manifest.
	if err := m.v41AdmitShape(name, v41StageMoE, 0, want...); err != nil {
		t.Fatalf("resident %s refused: %v", name, err)
	}

	// Move it out of the resident manifest — the streamed-by-design state.
	delete(m.manifest, name)
	if err := m.v41AdmitShape(name, v41StageMoE, 0, want...); err == nil {
		t.Fatalf("v41AdmitShape admitted %s with no resident store and no tier; the refusal must name the tensor", name)
	}
	if err := m.v41ForwardAdmitted(); err == nil {
		t.Fatalf("v41ForwardAdmitted passed with %s absent from every store — the streamed seam would never fail closed", name)
	}

	// Attach a tier that serves the V4 name: the resident-store miss now falls
	// through to the tier index and admission succeeds.
	m.SetExpertCheckpoint(v41StreamedTierQ2K(t, "deepseek41", "gate_proj", m.Cfg.NumExperts))
	if err := m.v41AdmitShape(name, v41StageMoE, 0, want...); err != nil {
		t.Fatalf("streamed tier did not admit %s: %v", name, err)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("v41ForwardAdmitted refused after the streamed tier served %s: %v", name, err)
	}

	// A name the tier does NOT carry still refuses (presence, not blank admission):
	// the slab carries experts [0,NumExperts), so this name is out of its range.
	absent := layerName(0, "ffn.experts."+itoa(m.Cfg.NumExperts)+".w1.weight")
	delete(m.manifest, absent)
	if err := m.v41AdmitShape(absent, v41StageMoE, 0, want...); err == nil {
		t.Fatalf("v41AdmitShape admitted %s although the tier does not carry it", absent)
	}
}
