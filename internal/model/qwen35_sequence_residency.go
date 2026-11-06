package model

// Qwen35SequencePrefillDeviceResident reports whether a canonical tensor must be DEVICE-RESIDENT
// when the executing backend runs the qwen35-hybrid whole-prompt sequence prefill. It is the
// loader/estimate-side mirror of that kernel's operand audit: the CUDA and Vulkan sequence paths
// validate every operand before dispatch (compute.(*cudaBackend).validateQwen35SequenceRequestLocked)
// and their shared tensor check refuses any managed or host-backed allocation ("must have a live
// default-device allocation; managed memory is refused"). A policy that leaves one of these
// operands out of the device charge therefore admits a plan the kernel fails closed on at warmup.
//
// The audit maps every preflight operand to its canonical name:
//
//	token_embedding        -> model.embed_tokens.weight
//	output_norm            -> model.norm.weight
//	output                 -> lm_head.weight (untied head)
//	input_norm             -> model.layers.N.input_layernorm.weight          (f32, never streamed-folded)
//	post_norm              -> model.layers.N.post_attention_layernorm.weight (f32, never streamed-folded)
//	mlp_gate / mlp_up      -> model.layers.N.mlp.{gate,up}_proj.weight
//	mlp_down               -> model.layers.N.mlp.down_proj.weight
//	gdn_in_qkv / _z / _b   -> model.layers.N.linear_attn.in_proj_{qkv,z,a,b}.weight
//	gdn_out                -> model.layers.N.linear_attn.out_proj.weight
//	gdn_conv / gdn_A_log   -> f32 GDN side tensors (never streamed-folded)
//	gdn_dt_bias / gdn_norm -> f32 GDN side tensors (never streamed-folded)
//	attention_qg / _k      -> model.layers.N.self_attn.{q,k}_proj.weight
//	attention_v / _o       -> model.layers.N.self_attn.{v,o}_proj.weight
//	attention_q/k_norm     -> f32 (never streamed-folded)
//	GDN conv/recurrent state -> durable KV-class allocations, not load-time weights
//
// Every quantized operand is an IsQuantWeight matmul; the two f32 scalars are pinned explicitly
// as belt-and-suspenders: they are not streamed-dense eligible today, and pinning them keeps the
// invariant explicit if eligibility ever widens. Retained MTP heads materialize into
// mtp.layers.N.* names carrying the same suffixes, so they stay pinned too.
func Qwen35SequencePrefillDeviceResident(cfg Config, canon string) bool {
	if !cfg.IsQwen35Hybrid() {
		return false
	}
	switch canon {
	case "model.embed_tokens.weight", "model.norm.weight":
		return true
	}
	return IsQuantWeight(canon)
}
