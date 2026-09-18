package ggufload

import (
	"os"
	"path/filepath"
	"testing"
)

// v41_streamed_expert_tier_seam_test.go — the ggufload half of the #13275
// regression guard.
//
// FusedExpertTensors describes a checkpoint's batched routed-expert slabs to the
// R5 tier, which indexes each one under a per-expert canonical name. For arch
// deepseek41 that name must be the native V4.1 ffn.experts.<e>.{w1,w3,w2} leaf the
// forward asks for (#13271 wired the eager emitter; #13275 wires the STREAMED
// index). This drives a real deepseek41 GGUF through OpenWeights and asserts the
// arch is threaded onto every descriptor and the assembled tier is keyed on the
// V4 leaves, with the GLM mlp spelling absent.

// v41SeamQ4KPayload fills one batched expert slab with a distinct per-expert byte
// pattern so the descriptor set is not degenerate.
func v41SeamQ4KPayload(t *testing.T, dims []uint64, experts int) []byte {
	t.Helper()
	total := qwen3MoEPayloadBytes(TensorQ4_K, dims)
	if total%blockQ4KBytes != 0 {
		t.Fatalf("fixture payload %d is not a whole number of %d-byte super-blocks", total, blockQ4KBytes)
	}
	per := total / experts
	payload := make([]byte, total)
	for x := 0; x < experts; x++ {
		for i := 0; i < per; i++ {
			payload[x*per+i] = expertPatternByte(x, i)
		}
	}
	return payload
}

// TestDeepSeek41StreamedTierIndexesV4Leaves is the end-to-end seam witness: a
// deepseek41 checkpoint's FusedExpertTensors descriptors carry Arch, and the tier
// built over them answers the V4 ffn.experts.<e>.w1/w3/w2 names the native forward
// admits (and only those).
func TestDeepSeek41StreamedTierIndexesV4Leaves(t *testing.T) {
	const E, I, H = 4, 256, 256
	dimsGate := []uint64{uint64(H), uint64(I), uint64(E)} // [out,in,E]
	dimsDown := []uint64{uint64(I), uint64(H), uint64(E)}

	shard1 := writeDeepSeek41SplitShard(t, 1, 2, 2, true, []splitTensor{
		{name: "blk.0.attn_norm.weight", dims: []uint64{4}, typ: TensorF32, data: f32Payload(1.5, 2.5, 3.5, 4.5)},
	})
	shard2 := writeDeepSeek41SplitShard(t, 2, 2, 2, false, []splitTensor{
		{name: "blk.0.ffn_gate_exps.weight", dims: dimsGate, typ: TensorQ4_K, data: v41SeamQ4KPayload(t, dimsGate, E)},
		{name: "blk.0.ffn_up_exps.weight", dims: dimsGate, typ: TensorQ4_K, data: v41SeamQ4KPayload(t, dimsGate, E)},
		{name: "blk.0.ffn_down_exps.weight", dims: dimsDown, typ: TensorQ4_K, data: v41SeamQ4KPayload(t, dimsDown, E)},
	})

	dir := t.TempDir()
	p1 := filepath.Join(dir, "ds41-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "ds41-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}

	ws, err := OpenWeights(p1)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "deepseek41" {
		t.Fatalf("ModelType = %q, want deepseek41", cfg.ModelType)
	}

	// (a) every descriptor threaded the arch (this is the source-side half of the
	// fix — without it the tier would key the GLM spelling even though the loader
	// emits V4 names).
	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if len(shards) == 0 {
		t.Fatal("no fused expert shards described for a checkpoint with three Q4_K ffn_*_exps slabs")
	}
	var descs int
	for _, sh := range shards {
		for _, f := range sh.Fused {
			descs++
			if f.Arch != "deepseek41" {
				t.Errorf("FusedExpertTensor(%s).Arch = %q, want deepseek41", f.Name, f.Arch)
			}
		}
	}
	if descs != 3 {
		t.Fatalf("described %d fused expert slabs, want 3", descs)
	}

	// (b) the assembled tier answers the native V4 leaves and NOT the GLM spelling.
	tier, err := ws.ExpertCheckpointTier(0)
	if err != nil {
		t.Fatalf("ExpertCheckpointTier: %v", err)
	}
	if tier == nil {
		t.Fatal("ExpertCheckpointTier returned nil for a checkpoint with stageable Q4_K experts")
	}
	for e := 0; e < E; e++ {
		stem := "model.layers.0.ffn.experts." + itoaForTest(e) + "."
		for _, leaf := range []string{"w1", "w3", "w2"} {
			if !tier.Has(stem + leaf + ".weight") {
				t.Errorf("tier does not index %s%s.weight — the V4.1 forward would refuse this streamed expert by name", stem, leaf)
			}
		}
	}
	if tier.Has("model.layers.0.mlp.experts.0.gate_proj.weight") {
		t.Error("tier indexed the GLM mlp.experts.0.gate_proj.weight spelling for deepseek41 — the streamed index would drift from the forward's read")
	}
}
