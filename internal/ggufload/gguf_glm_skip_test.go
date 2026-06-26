package ggufload

import "testing"

// TestGLMMoeDsaSkipGGUFTensor pins which tensors the glm_moe_dsa loader drops at load: the MTP
// ("nextn") speculative-decoding head and any multimodal vision tower the text forward never
// reads (llama.cpp ignores them too), and NOTHING the forward does read.
func TestGLMMoeDsaSkipGGUFTensor(t *testing.T) {
	skip := []string{
		"blk.78.nextn.eh_proj.weight",
		"blk.78.nextn.enorm.weight",
		"blk.78.nextn.hnorm.weight",
		"blk.78.nextn.shared_head_norm.weight",
		"v.blk.0.attn_q.weight", // a vision tower tensor (if present)
		"mm.input_projection.weight",
	}
	keep := []string{
		"blk.0.attn_k_b.weight",       // the MLA split we MERGE, must not be skipped
		"blk.0.attn_v_b.weight",       //
		"blk.0.attn_q_b.weight",       //
		"blk.0.ffn_gate_exps.weight",  // routed experts the splitter handles
		"blk.0.attn_kv_a_mqa.weight",  //
		"blk.0.attn_norm.weight",      //
		"token_embd.weight",           //
		"output_norm.weight",          //
		"blk.5.ffn_down_shexp.weight", // a shared-expert tensor with no "nextn"
	}
	for _, n := range skip {
		if !glmMoeDsaSkipGGUFTensor(n) {
			t.Errorf("expected to SKIP %q (MTP/vision), but kept", n)
		}
	}
	for _, n := range keep {
		if glmMoeDsaSkipGGUFTensor(n) {
			t.Errorf("expected to KEEP %q, but skipped", n)
		}
	}
}
