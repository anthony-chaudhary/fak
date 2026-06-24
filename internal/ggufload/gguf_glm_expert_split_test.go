package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// gguf_glm_expert_split_test.go — the gate for the glm_moe_dsa batched routed-expert splitter
// (glmMoeDsaBatchedExpert + splitGLMMoeDsaExperts in gguf_glm_tensors.go, wired into the
// gguf_weightsource.go load loops). It pins the pure split bit-equal to manual slicing, and proves
// the END-TO-END loader path: a synthetic glm_moe_dsa GGUF whose only routed-expert tensor is one
// [E,out,in] blob loads through F32Tensors into E per-expert canonical 2-D tensors
// (model.layers.<L>.mlp.experts.<e>.<proj>.weight) — the form internal/model's MoE forward reads.

func TestSplitGLMMoeDsaExperts(t *testing.T) {
	const e, out, in = 3, 4, 2
	per := out * in
	data := make([]float32, e*per)
	for i := range data {
		data[i] = float32(i) * 0.5
	}
	got, err := splitGLMMoeDsaExperts(2, "gate_proj", []int{e, out, in}, data)
	if err != nil {
		t.Fatalf("splitGLMMoeDsaExperts: %v", err)
	}
	if len(got) != e {
		t.Fatalf("split produced %d experts, want %d", len(got), e)
	}
	for x := 0; x < e; x++ {
		wantName := "model.layers.2.mlp.experts." + itoaForTest(x) + ".gate_proj.weight"
		if got[x].Name != wantName {
			t.Errorf("expert %d name = %q, want %q", x, got[x].Name, wantName)
		}
		if len(got[x].Shape) != 2 || got[x].Shape[0] != out || got[x].Shape[1] != in {
			t.Errorf("expert %d shape = %v, want [%d %d]", x, got[x].Shape, out, in)
		}
		// bit-equal to the manual contiguous slice along the leading expert axis.
		want := data[x*per : (x+1)*per]
		if len(got[x].Data) != per {
			t.Fatalf("expert %d data len = %d, want %d", x, len(got[x].Data), per)
		}
		for i := 0; i < per; i++ {
			if got[x].Data[i] != want[i] {
				t.Fatalf("expert %d [%d] = %v, want %v (not bit-equal to manual slice)", x, i, got[x].Data[i], want[i])
			}
		}
	}

	// fail-closed: a non-3-D shape or a length mismatch is rejected, not silently mis-split.
	if _, err := splitGLMMoeDsaExperts(0, "up_proj", []int{out, in}, data); err == nil {
		t.Errorf("split of a 2-D shape should fail closed")
	}
	if _, err := splitGLMMoeDsaExperts(0, "up_proj", []int{e, out, in}, data[:per]); err == nil {
		t.Errorf("split with a too-short payload should fail closed")
	}
}

func itoaForTest(x int) string {
	if x == 0 {
		return "0"
	}
	var buf [4]byte
	i := len(buf)
	for x > 0 {
		i--
		buf[i] = byte('0' + x%10)
		x /= 10
	}
	return string(buf[i:])
}

// TestGLMMoeDsaGGUFExpertSplitE2E builds a synthetic glm_moe_dsa GGUF whose sole routed-expert
// tensor is a batched [E,out,in] ffn_gate_exps blob, loads it through the real F32Tensors path, and
// asserts the loader materialized E per-expert canonical tensors bit-equal to the source slices.
func TestGLMMoeDsaGGUFExpertSplitE2E(t *testing.T) {
	const E, I, H = 3, 4, 2 // experts, moe-intermediate (out), hidden (in)
	per := I * H
	gate := make([]float32, E*per)
	for i := range gate {
		gate[i] = float32(100 + i)
	}

	path := filepath.Join(t.TempDir(), "glm_experts.gguf")
	if err := os.WriteFile(path, glmMoeDsaExpertGGUF(E, I, H, gate), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors: %v", err)
	}
	if cfg.ModelType != "glm_moe_dsa" {
		t.Fatalf("ModelType=%q, want glm_moe_dsa", cfg.ModelType)
	}
	byName := map[string]modelTensorForTest{}
	for _, tt := range tensors {
		byName[tt.Name] = modelTensorForTest{shape: tt.Shape, data: tt.Data}
	}
	if len(byName) != E {
		t.Fatalf("loaded %d tensors, want %d per-expert (the batched blob must split 1->E, nothing else)", len(byName), E)
	}
	for x := 0; x < E; x++ {
		name := "model.layers.0.mlp.experts." + itoaForTest(x) + ".gate_proj.weight"
		assertModelTensorForTest(t, byName, name, []int{I, H}, gate[x*per:(x+1)*per])
	}
}

// glmMoeDsaExpertGGUF writes a minimal glm_moe_dsa GGUF: the metadata Config() needs plus one
// batched routed-expert tensor blk.0.ffn_gate_exps.weight. The GGUF stores dims low-to-high, so a
// model-shape [E,out,in] tensor is written with dims {in, out, E}; modelShapeFromGGUFDims reverses
// it back to [E,out,in] for the splitter.
func glmMoeDsaExpertGGUF(E, I, H int, gate []float32) []byte {
	var b bytes.Buffer
	const nKV = 11
	writeMinimalHeader(&b, 1, nKV)
	writeKVString(&b, "general.architecture", "glm_moe_dsa")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "glm_moe_dsa.embedding_length", uint32(H))
	writeKVUint32(&b, "glm_moe_dsa.block_count", 1)
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count", 2)
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count_kv", 1)
	writeKVUint32(&b, "glm_moe_dsa.feed_forward_length", 8)
	writeKVFloat32(&b, "glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVUint32(&b, "glm_moe_dsa.expert_count", uint32(E))
	writeKVUint32(&b, "glm_moe_dsa.expert_used_count", 2)
	writeKVUint32(&b, "glm_moe_dsa.expert_feed_forward_length", uint32(I))
	// dims low-to-high: {in=H, out=I, E}
	writeTensorInfoForTest(&b, "blk.0.ffn_gate_exps.weight", []uint64{uint64(H), uint64(I), uint64(E)}, TensorF32, 0)
	padToAlignment(&b, 32)
	for _, v := range gate {
		writeF32ForTest(&b, v)
	}
	return b.Bytes()
}
