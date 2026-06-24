package ggufload

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_glm_oracle_test.go — the P1 forward-CORRECTNESS rung (closes the "GGUF-loaded argmax ==
// reference argmax" acceptance). A single canonical manifest is the independent reference: a model
// built directly from it (model.NewFromF32Tensors) is compared, forward-for-forward, against the
// model fak loads from a GGUF serialized from the SAME values. They agree bit-for-bit iff the GGUF
// round-trip (inverse name map + dim reversal + expert RE-batching + the loader's split) is
// faithful — so a transposed dim, a swapped expert, or a misnamed tensor that the shape-only loader
// gate cannot see is caught here. This proves fak's glm_moe_dsa GGUF path reconstructs the intended
// model exactly, not just structurally.

func glmOracleArgmax(v []float32) int {
	best, bi := float32(-3.4e38), -1
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// buildGLMOracleFixture returns (canonical manifest, GGUF bytes) from ONE deterministic data source
// for a complete single-MoE-layer glm_moe_dsa model. The manifest is the reference; the GGUF is the
// inverse serialization (canonical->GGUF names, shape reversed to GGUF order, per-expert tensors
// re-batched into one ffn_*_exps blob each).
func buildGLMOracleFixture() ([]model.NamedTensorF32, []byte) {
	const (
		H, V                = 4, 6
		qLora, kvLora       = 6, 4
		qkNope, qkRope, vHd = 2, 2, 2
		nH                  = 2
		idxHeads, idxDim    = 2, 4
		E, I, sharedI       = 3, 8, 8
	)
	qkHead := qkNope + qkRope

	seed := 0
	gen := func(n int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32(seed%89)*0.2 - 8 // bounded, finite, deterministic
			seed++
		}
		return d
	}
	prod := func(s []int) int {
		n := 1
		for _, d := range s {
			n *= d
		}
		return n
	}

	// non-expert canonical tensors paired with their GGUF name; data is shared (a canonical
	// [out,in] is row-major == the GGUF flat payload, only the stored dims order differs).
	type spec struct {
		canon, gguf string
		shape       []int
	}
	specs := []spec{
		{"model.embed_tokens.weight", "token_embd.weight", []int{V, H}},
		{"model.norm.weight", "output_norm.weight", []int{H}},
		{"lm_head.weight", "output.weight", []int{V, H}},
		{"model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight", []int{H}},
		{"model.layers.0.self_attn.q_a_proj.weight", "blk.0.attn_q_a.weight", []int{qLora, H}},
		{"model.layers.0.self_attn.q_a_layernorm.weight", "blk.0.attn_q_a_norm.weight", []int{qLora}},
		{"model.layers.0.self_attn.q_b_proj.weight", "blk.0.attn_q_b.weight", []int{nH * qkHead, qLora}},
		{"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "blk.0.attn_kv_a_mqa.weight", []int{kvLora + qkRope, H}},
		{"model.layers.0.self_attn.kv_a_layernorm.weight", "blk.0.attn_kv_a_norm.weight", []int{kvLora}},
		{"model.layers.0.self_attn.kv_b_proj.weight", "blk.0.attn_kv_b.weight", []int{nH * (qkNope + vHd), kvLora}},
		{"model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight", []int{H, nH * vHd}},
		{"model.layers.0.self_attn.indexer.wq_b.weight", "blk.0.indexer.attn_q_b.weight", []int{idxHeads * idxDim, qLora}},
		{"model.layers.0.self_attn.indexer.wk.weight", "blk.0.indexer.attn_k.weight", []int{idxDim, H}},
		{"model.layers.0.self_attn.indexer.k_norm.weight", "blk.0.indexer.k_norm.weight", []int{idxDim}},
		{"model.layers.0.self_attn.indexer.k_norm.bias", "blk.0.indexer.k_norm.bias", []int{idxDim}},
		{"model.layers.0.self_attn.indexer.weights_proj.weight", "blk.0.indexer.proj.weight", []int{idxHeads, H}},
		{"model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight", []int{H}},
		{"model.layers.0.mlp.gate.weight", "blk.0.ffn_gate_inp.weight", []int{E, H}},
		{"model.layers.0.mlp.gate.e_score_correction_bias", "blk.0.exp_probs_b.bias", []int{E}},
		{"model.layers.0.mlp.shared_experts.gate_proj.weight", "blk.0.ffn_gate_shexp.weight", []int{sharedI, H}},
		{"model.layers.0.mlp.shared_experts.up_proj.weight", "blk.0.ffn_up_shexp.weight", []int{sharedI, H}},
		{"model.layers.0.mlp.shared_experts.down_proj.weight", "blk.0.ffn_down_shexp.weight", []int{H, sharedI}},
	}

	type gt struct {
		name string
		dims []uint64
		data []float32
	}
	var M []model.NamedTensorF32
	var gts []gt
	for _, s := range specs {
		d := gen(prod(s.shape))
		M = append(M, model.NamedTensorF32{Name: s.canon, Shape: append([]int(nil), s.shape...), Data: d})
		dims := make([]uint64, len(s.shape)) // reverse to GGUF (low-to-high) order
		for i := range s.shape {
			dims[i] = uint64(s.shape[len(s.shape)-1-i])
		}
		gts = append(gts, gt{s.gguf, dims, d})
	}
	// experts: E per-expert canonical tensors per projection, re-batched into one [E,out,in] blob.
	experts := []struct {
		proj, gguf string
		out, in    int
	}{
		{"gate_proj", "blk.0.ffn_gate_exps.weight", I, H},
		{"up_proj", "blk.0.ffn_up_exps.weight", I, H},
		{"down_proj", "blk.0.ffn_down_exps.weight", H, I},
	}
	for _, ex := range experts {
		blob := make([]float32, 0, E*ex.out*ex.in)
		for e := 0; e < E; e++ {
			d := gen(ex.out * ex.in)
			M = append(M, model.NamedTensorF32{
				Name:  fmt.Sprintf("model.layers.0.mlp.experts.%d.%s.weight", e, ex.proj),
				Shape: []int{ex.out, ex.in},
				Data:  d,
			})
			blob = append(blob, d...)
		}
		// GGUF dims low-to-high for model-shape [E,out,in] = {in, out, E}.
		gts = append(gts, gt{ex.gguf, []uint64{uint64(ex.in), uint64(ex.out), uint64(E)}, blob})
	}

	// assemble the GGUF.
	align := func(x int) int { return (x + 31) / 32 * 32 }
	var kv bytes.Buffer
	nKV := 0
	ks := func(k, v string) { writeKVString(&kv, k, v); nKV++ }
	ku := func(k string, v uint32) { writeKVUint32(&kv, k, v); nKV++ }
	kf := func(k string, v float32) { writeKVFloat32(&kv, k, v); nKV++ }
	ka := func(k string, vs []string) { writeKVStringArray(&kv, k, vs); nKV++ }
	ks("general.architecture", "glm_moe_dsa")
	ku("general.alignment", 32)
	ku("glm_moe_dsa.embedding_length", H)
	ku("glm_moe_dsa.block_count", 1)
	ku("glm_moe_dsa.attention.head_count", nH)
	ku("glm_moe_dsa.attention.head_count_kv", nH)
	ku("glm_moe_dsa.feed_forward_length", I)
	kf("glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
	kf("glm_moe_dsa.rope.freq_base", 10000)
	ku("glm_moe_dsa.expert_count", E)
	ku("glm_moe_dsa.expert_used_count", 2)
	ku("glm_moe_dsa.expert_feed_forward_length", I)
	ku("glm_moe_dsa.attention.q_lora_rank", qLora)
	ku("glm_moe_dsa.attention.kv_lora_rank", kvLora)
	ku("glm_moe_dsa.attention.qk_nope_head_dim", qkNope)
	ku("glm_moe_dsa.attention.qk_rope_head_dim", qkRope)
	ku("glm_moe_dsa.attention.v_head_dim", vHd)
	ku("glm_moe_dsa.attention.indexer.head_count", idxHeads)
	ku("glm_moe_dsa.attention.indexer.key_length", idxDim)
	ku("glm_moe_dsa.attention.indexer.top_k", 4)
	toks := make([]string, V)
	for i := range toks {
		toks[i] = "t" + itoaForTest(i)
	}
	ka("tokenizer.ggml.tokens", toks)

	var ti bytes.Buffer
	off := 0
	for _, t := range gts {
		writeTensorInfoForTest(&ti, t.name, t.dims, TensorF32, uint64(off))
		off = align(off + len(t.data)*4)
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(gts)), uint64(nKV))
	b.Write(kv.Bytes())
	b.Write(ti.Bytes())
	padToAlignment(&b, 32)
	dataStart := b.Len()
	off = 0
	for _, t := range gts {
		padToLen(&b, dataStart+off)
		for _, v := range t.data {
			writeF32ForTest(&b, v)
		}
		off = align(off + len(t.data)*4)
	}
	return M, b.Bytes()
}

// TestGLMMoeDsaGGUFForwardMatchesReference is the P1 forward-correctness oracle: the GGUF-loaded
// model forwards BIT-FOR-BIT identically to a model built directly from the same canonical tensors.
func TestGLMMoeDsaGGUFForwardMatchesReference(t *testing.T) {
	manifest, ggufBytes := buildGLMOracleFixture()

	path := filepath.Join(t.TempDir(), "glm_oracle.gguf")
	if err := os.WriteFile(path, ggufBytes, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	cfg, _, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors: %v", err)
	}

	refModel, err := model.NewFromF32Tensors(cfg, manifest)
	if err != nil {
		t.Fatalf("reference NewFromF32Tensors: %v", err)
	}
	ggufModel, err := ws.Model()
	if err != nil {
		t.Fatalf("GGUF Model: %v", err)
	}

	ids := []int{0, 1, 2, 3, 4}
	ref := refModel.Forward(ids)
	got := ggufModel.Forward(ids)
	if len(got.Logits) != len(ref.Logits) {
		t.Fatalf("logit rows: gguf=%d ref=%d", len(got.Logits), len(ref.Logits))
	}
	for pos := range ref.Logits {
		if glmOracleArgmax(got.Logits[pos]) != glmOracleArgmax(ref.Logits[pos]) {
			t.Fatalf("pos %d argmax: gguf=%d ref=%d (GGUF round-trip is not faithful)",
				pos, glmOracleArgmax(got.Logits[pos]), glmOracleArgmax(ref.Logits[pos]))
		}
		for i := range ref.Logits[pos] {
			if got.Logits[pos][i] != ref.Logits[pos][i] {
				t.Fatalf("pos %d logit %d: gguf=%v ref=%v (not bit-exact — a value/dim/expert-order round-trip bug)",
					pos, i, got.Logits[pos][i], ref.Logits[pos][i])
			}
		}
	}
}
