package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_glm_full_fixture_test.go — the P1 loader capstone: a COMPLETE single-MoE-layer glm_moe_dsa
// GGUF (every per-layer tensor the native glm_dsa forward consumes — the full MLA stack, the DSA
// indexer incl. k_norm, the router + score-correction bias, the batched routed experts, the shared
// experts, the norms, plus the global embed/norm/lm_head) loads through the real F32Tensors path
// and resolves into the full canonical manifest, with the batched experts split 1->E. It proves the
// glm_moe_dsa loader handles a whole checkpoint's tensor set — no "no canonical mapping" on any
// tensor, no shape error, experts materialized — the loader half of the eventual GGUF-forward
// oracle. (It asserts shapes, not a forward; the forward-vs-oracle is the remaining P1 rung.)

func TestGLMMoeDsaGGUFFullFixtureLoads(t *testing.T) {
	// tiny but structurally complete dims.
	const (
		H        = 4 // hidden
		V        = 5 // vocab
		qLora    = 6
		kvLora   = 4
		qkNope   = 2
		qkRope   = 2
		vHead    = 2
		nH       = 2
		idxHeads = 2
		idxDim   = 4
		E        = 3 // routed experts
		I        = 8 // moe intermediate
		sharedI  = 8 // shared-expert intermediate (MoEIntermediate * shared_count)
	)
	qkHead := qkNope + qkRope // 4

	path := filepath.Join(t.TempDir(), "glm_full.gguf")
	if err := os.WriteFile(path, glmMoeDsaFullGGUF(H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors (a complete glm_moe_dsa GGUF must load every tensor): %v", err)
	}
	if cfg.ModelType != "glm_moe_dsa" {
		t.Fatalf("ModelType=%q, want glm_moe_dsa", cfg.ModelType)
	}
	shapes := map[string][]int{}
	for _, tt := range tensors {
		shapes[tt.Name] = tt.Shape
	}

	want := map[string][]int{
		"model.embed_tokens.weight":                            {V, H},
		"model.norm.weight":                                    {H},
		"lm_head.weight":                                       {V, H},
		"model.layers.0.input_layernorm.weight":                {H},
		"model.layers.0.self_attn.q_a_proj.weight":             {qLora, H},
		"model.layers.0.self_attn.q_a_layernorm.weight":        {qLora},
		"model.layers.0.self_attn.q_b_proj.weight":             {nH * qkHead, qLora},
		"model.layers.0.self_attn.kv_a_proj_with_mqa.weight":   {kvLora + qkRope, H},
		"model.layers.0.self_attn.kv_a_layernorm.weight":       {kvLora},
		"model.layers.0.self_attn.kv_b_proj.weight":            {nH * (qkNope + vHead), kvLora},
		"model.layers.0.self_attn.o_proj.weight":               {H, nH * vHead},
		"model.layers.0.self_attn.indexer.wq_b.weight":         {idxHeads * idxDim, qLora},
		"model.layers.0.self_attn.indexer.wk.weight":           {idxDim, H},
		"model.layers.0.self_attn.indexer.k_norm.weight":       {idxDim},
		"model.layers.0.self_attn.indexer.k_norm.bias":         {idxDim},
		"model.layers.0.self_attn.indexer.weights_proj.weight": {idxHeads, H},
		"model.layers.0.post_attention_layernorm.weight":       {H},
		"model.layers.0.mlp.gate.weight":                       {E, H},
		"model.layers.0.mlp.gate.e_score_correction_bias":      {E},
		"model.layers.0.mlp.shared_experts.gate_proj.weight":   {sharedI, H},
		"model.layers.0.mlp.shared_experts.up_proj.weight":     {sharedI, H},
		"model.layers.0.mlp.shared_experts.down_proj.weight":   {H, sharedI},
	}
	// the batched routed experts must have split 1->E into per-expert tensors.
	for e := 0; e < E; e++ {
		p := "model.layers.0.mlp.experts." + itoaForTest(e) + "."
		want[p+"gate_proj.weight"] = []int{I, H}
		want[p+"up_proj.weight"] = []int{I, H}
		want[p+"down_proj.weight"] = []int{H, I}
	}

	for name, ws := range want {
		got, ok := shapes[name]
		if !ok {
			t.Errorf("canonical tensor %q missing from the loaded manifest", name)
			continue
		}
		if len(got) != len(ws) {
			t.Errorf("%q shape = %v, want %v", name, got, ws)
			continue
		}
		for i := range ws {
			if got[i] != ws[i] {
				t.Errorf("%q shape = %v, want %v", name, got, ws)
				break
			}
		}
	}
	if len(shapes) != len(want) {
		t.Errorf("loaded %d canonical tensors, want exactly %d (the complete glm_moe_dsa layer + globals)", len(shapes), len(want))
	}
}

// TestGLMMoeDsaGGUFFullFixtureForwards is the forward-capability rung: the complete glm_moe_dsa
// GGUF loads via Model() into a real model.Model and its native glm_dsa forward (MLA + DSA indexer
// + MoE routing + shared experts) RUNS to finite logits of the right width — i.e. fak's own engine
// loads a real-architecture glm_moe_dsa checkpoint from GGUF and can serve a forward over it. (This
// proves the loaded model is forward-capable; the argmax-vs-safetensors oracle is the next rung.)
func TestGLMMoeDsaGGUFFullFixtureForwards(t *testing.T) {
	const (
		H, V                = 4, 6
		qLora, kvLora       = 6, 4
		qkNope, qkRope, vHd = 2, 2, 2
		nH                  = 2
		idxHeads, idxDim    = 2, 4
		E, I, sharedI       = 3, 8, 8
	)
	path := filepath.Join(t.TempDir(), "glm_fwd.gguf")
	if err := os.WriteFile(path, glmMoeDsaFullGGUF(H, V, qLora, kvLora, qkNope, qkRope, vHd, nH, idxHeads, idxDim, E, I, sharedI), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel a complete glm_moe_dsa GGUF: %v", err)
	}
	ids := []int{0, 1, 2, 3, 4} // seq >= index_topk(4) so the DSA indexer has enough keys
	act := m.Forward(ids)
	if act == nil || len(act.Logits) != len(ids) {
		t.Fatalf("Forward returned %d logit rows, want %d", lenLogits(act), len(ids))
	}
	for pos, row := range act.Logits {
		if len(row) != V {
			t.Fatalf("logit row %d width = %d, want vocab %d", pos, len(row), V)
		}
		for i, v := range row {
			if isNaNOrInf(v) {
				t.Fatalf("logit[%d][%d] = %v (non-finite) — the glm_dsa forward produced garbage", pos, i, v)
			}
		}
	}
}

func lenLogits(a *model.Activations) int {
	if a == nil {
		return 0
	}
	return len(a.Logits)
}

func isNaNOrInf(v float32) bool {
	return v != v || v > 3.4e38 || v < -3.4e38
}

// glmMoeDsaFullGGUF assembles a complete single-MoE-layer glm_moe_dsa GGUF. GGUF stores dims
// low-to-high, so a canonical [out,in] tensor is written {in,out} and a batched [E,out,in] expert
// blob is written {in,out,E}; modelShapeFromGGUFDims reverses them back. Tensor-data offsets are
// the cumulative payload sizes aligned to general.alignment (32).
func glmMoeDsaFullGGUF(H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI int) []byte {
	return glmMoeDsaFullGGUFTyped(H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI, TensorF32)
}

// glmMoeDsaFullGGUFTyped is glmMoeDsaFullGGUF with the batched routed-expert blobs
// (ffn_{gate,up,down}_exps) written in expertType, so a test can exercise the loader's resident
// raw-quant expert routing. The quant payloads are written as all-zero blocks (valid: dequant to
// 0), which keeps the forward finite while letting the loader take the raw-resident split.
func glmMoeDsaFullGGUFTyped(H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI int, expertType TensorType) []byte {
	// The batched routed-expert blobs take expertType; everything else is F32.
	return glmMoeDsaFullGGUFWithTypes(H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI,
		func(name string) TensorType {
			if strings.Contains(name, "_exps.weight") {
				return expertType
			}
			return TensorF32
		})
}

// glmMoeDsaFullGGUFWithTypes is the generalized fixture: typeOf assigns every tensor's GGUF
// type (return TensorF32 for the default float payload). Lets a test quantize DENSE tensors
// too — e.g. the real UD-Q4_K_M quantizes the dense MLA projections to Q8_0, the layout the
// dense-resident-k-quant loader gate must refuse for glm_moe_dsa.
func glmMoeDsaFullGGUFWithTypes(H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI int, typeOf func(string) TensorType) []byte {
	qkHead := qkNope + qkRope
	type tw struct {
		name string
		dims []uint64
	}
	// dims low-to-high (GGUF order).
	ts := []tw{
		{"token_embd.weight", []uint64{uint64(H), uint64(V)}},
		{"output_norm.weight", []uint64{uint64(H)}},
		{"output.weight", []uint64{uint64(H), uint64(V)}},
		{"blk.0.attn_norm.weight", []uint64{uint64(H)}},
		{"blk.0.attn_q_a.weight", []uint64{uint64(H), uint64(qLora)}},
		{"blk.0.attn_q_a_norm.weight", []uint64{uint64(qLora)}},
		{"blk.0.attn_q_b.weight", []uint64{uint64(qLora), uint64(nH * qkHead)}},
		{"blk.0.attn_kv_a_mqa.weight", []uint64{uint64(H), uint64(kvLora + qkRope)}},
		{"blk.0.attn_kv_a_norm.weight", []uint64{uint64(kvLora)}},
		{"blk.0.attn_kv_b.weight", []uint64{uint64(kvLora), uint64(nH * (qkNope + vHead))}},
		{"blk.0.attn_output.weight", []uint64{uint64(nH * vHead), uint64(H)}},
		{"blk.0.indexer.attn_q_b.weight", []uint64{uint64(qLora), uint64(idxHeads * idxDim)}},
		{"blk.0.indexer.attn_k.weight", []uint64{uint64(H), uint64(idxDim)}},
		{"blk.0.indexer.k_norm.weight", []uint64{uint64(idxDim)}},
		{"blk.0.indexer.k_norm.bias", []uint64{uint64(idxDim)}},
		{"blk.0.indexer.proj.weight", []uint64{uint64(H), uint64(idxHeads)}},
		{"blk.0.ffn_norm.weight", []uint64{uint64(H)}},
		{"blk.0.ffn_gate_inp.weight", []uint64{uint64(H), uint64(E)}},
		{"blk.0.exp_probs_b.bias", []uint64{uint64(E)}},
		{"blk.0.ffn_gate_exps.weight", []uint64{uint64(H), uint64(I), uint64(E)}},
		{"blk.0.ffn_up_exps.weight", []uint64{uint64(H), uint64(I), uint64(E)}},
		{"blk.0.ffn_down_exps.weight", []uint64{uint64(I), uint64(H), uint64(E)}},
		{"blk.0.ffn_gate_shexp.weight", []uint64{uint64(H), uint64(sharedI)}},
		{"blk.0.ffn_up_shexp.weight", []uint64{uint64(H), uint64(sharedI)}},
		{"blk.0.ffn_down_shexp.weight", []uint64{uint64(sharedI), uint64(H)}},
	}
	align := func(x int) int { return (x + 31) / 32 * 32 }
	numValues := func(dims []uint64) int {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		return n
	}
	payloadBytes := func(typ TensorType, n int) int {
		switch typ {
		case TensorQ6_K:
			return n / 256 * blockQ6KBytes
		case TensorQ5_K:
			return n / 256 * blockQ5KBytes
		case TensorQ4_K:
			return n / 256 * blockQ4KBytes
		case TensorIQ3_XXS:
			return n / 256 * blockIQ3XXSBytes
		case TensorIQ4_XS:
			return n / 256 * blockIQ4XSBytes
		case TensorQ8_0:
			return n / 32 * blockQ8_0Bytes
		default: // TensorF32
			return n * 4
		}
	}

	// KV section.
	var kv bytes.Buffer
	nKV := 0
	ks := func(k, v string) { writeKVString(&kv, k, v); nKV++ }
	ku := func(k string, v uint32) { writeKVUint32(&kv, k, v); nKV++ }
	kf := func(k string, v float32) { writeKVFloat32(&kv, k, v); nKV++ }
	ka := func(k string, vs []string) { writeKVStringArray(&kv, k, vs); nKV++ }
	ks("general.architecture", "glm_moe_dsa")
	ku("general.alignment", 32)
	ku("glm_moe_dsa.embedding_length", uint32(H))
	ku("glm_moe_dsa.block_count", 1)
	ku("glm_moe_dsa.attention.head_count", uint32(nH))
	ku("glm_moe_dsa.attention.head_count_kv", uint32(nH))
	ku("glm_moe_dsa.feed_forward_length", uint32(I))
	kf("glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
	kf("glm_moe_dsa.rope.freq_base", 10000)
	ku("glm_moe_dsa.expert_count", uint32(E))
	ku("glm_moe_dsa.expert_used_count", 2)
	ku("glm_moe_dsa.expert_feed_forward_length", uint32(I))
	ku("glm_moe_dsa.attention.q_lora_rank", uint32(qLora))
	ku("glm_moe_dsa.attention.kv_lora_rank", uint32(kvLora))
	ku("glm_moe_dsa.attention.qk_nope_head_dim", uint32(qkNope))
	ku("glm_moe_dsa.attention.qk_rope_head_dim", uint32(qkRope))
	ku("glm_moe_dsa.attention.v_head_dim", uint32(vHead))
	ku("glm_moe_dsa.attention.indexer.head_count", uint32(idxHeads))
	ku("glm_moe_dsa.attention.indexer.key_length", uint32(idxDim))
	ku("glm_moe_dsa.attention.indexer.top_k", 4)
	// vocab size derives from the token list; without it cfg.VocabSize=0 and the LM head
	// produces width-0 logits. V deterministic placeholder tokens.
	toks := make([]string, V)
	for i := range toks {
		toks[i] = "t" + itoaForTest(i)
	}
	ka("tokenizer.ggml.tokens", toks)

	// tensor-info section with cumulative aligned offsets.
	var ti bytes.Buffer
	off := 0
	for _, t := range ts {
		writeTensorInfoForTest(&ti, t.name, t.dims, typeOf(t.name), uint64(off))
		off = align(off + payloadBytes(typeOf(t.name), numValues(t.dims)))
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(ts)), uint64(nKV))
	b.Write(kv.Bytes())
	b.Write(ti.Bytes())
	padToAlignment(&b, 32)
	dataStart := b.Len()
	off = 0
	seed := 0
	for _, t := range ts {
		padToLen(&b, dataStart+off)
		n := numValues(t.dims)
		typ := typeOf(t.name)
		if typ == TensorF32 {
			for i := 0; i < n; i++ {
				writeF32ForTest(&b, float32(seed%97)*0.25-12) // bounded, finite, deterministic
				seed++
			}
		} else {
			// k-quant blob: all-zero super-blocks (valid; dequant to 0) so the loader takes
			// the raw-resident split and the forward stays finite.
			b.Write(bytes.Repeat([]byte{0}, payloadBytes(typ, n)))
		}
		off = align(off + payloadBytes(typ, n))
	}
	return b.Bytes()
}
