package model

// Falcon-variant support fence (docs/standards/support-maturity-honesty-fence.md).
//
// "Falcon" is ONE model_type covering THREE architectures. HF's
// FalconAttention._split_heads enumerates them, and they disagree on the memory
// layout of the fused self_attention.query_key_value tensor:
//
//	new_decoder_architecture=true   (Falcon-40B / 180B)
//	    qkv.view(batch, seq, num_kv_heads, num_heads/num_kv_heads + 2, head_dim)
//	    -> INTERLEAVED PER KV GROUP: [g q-heads | 1 k-head | 1 v-head] repeated
//	       num_kv_heads times. The block also carries TWO LayerNorms, ln_attn and
//	       ln_mlp (num_ln_in_parallel_attn=2), and no input_layernorm.
//
//	multi_query=true                (Falcon-7B — THE ONLY IMPLEMENTED VARIANT)
//	    qkv.view(batch, seq, num_heads + 2, head_dim)
//	    -> CONTIGUOUS [num_heads q-heads | 1 shared k-head | 1 shared v-head].
//	       ONE shared input_layernorm feeds both parallel branches.
//
//	neither                          (Falcon-RW-1B / RW-7B)
//	    qkv.view(batch, seq, num_heads, 3, head_dim)
//	    -> INTERLEAVED PER HEAD: [q_h | k_h | v_h] for each head. Also ALiBi
//	       rather than RoPE, and plain MHA (num_kv_heads == num_heads).
//
// fak implements exactly the multi_query one. materialize.go's
// materializeFalconTensors aliases self_attention.query_key_value onto the
// canonical fused name and fused_split.go's splitFusedProjections then cuts it
// into three CONTIGUOUS axis-0 ranges q(nH*hd) | k(nKV*hd) | v(nKV*hd) — which is
// bit-exactly HF's multi_query layout at nKV==1 and is WRONG for either
// interleaved layout. That is witnessed by TestFalconCPUNumericOracle
// (family_falcon_cpu_oracle_test.go), which encodes the multi_query variant and
// only it.
//
// The danger is that the other two variants are SILENT, not loud: both hold
// exactly the same TOTAL row count as the contiguous cut the loader performs
// (RW: 3*nH*hd == nH*hd + nH*hd + nH*hd at nKV==nH; 40B: (nH+2*nKV)*hd both
// ways), so splitOneFused's `out != wantRows` guard passes and the mis-cut
// produces wrong activations with no error at all. Falcon-RW additionally ships
// input_layernorm, so nothing downstream is missing either — it loads clean and
// answers wrong.
//
// refuseUnsupportedFalconVariant turns those two silent divergences into a named
// LOAD-time refusal, the same shape as the #934 UnsupportedArchError next door in
// arch_support.go: a path we have not implemented must REFUSE, not guess.

// FalconVariant names one of the three architectures HF ships behind model_type
// "falcon". Only FalconMultiQuery is implemented by fak's loader + forward.
type FalconVariant string

const (
	// FalconMultiQuery is Falcon-7B: multi_query=true, num_kv_heads==1, ONE shared
	// input_layernorm, CONTIGUOUS [q|k|v] fused-qkv layout. The implemented variant.
	FalconMultiQuery FalconVariant = "multi_query"
	// FalconNewDecoderArch is Falcon-40B / 180B: new_decoder_architecture=true,
	// ln_attn + ln_mlp, fused qkv interleaved per KV group.
	FalconNewDecoderArch FalconVariant = "new_decoder_architecture"
	// FalconRW is Falcon-RW-1B / RW-7B: neither multi_query nor
	// new_decoder_architecture, plain MHA, fused qkv interleaved per head, ALiBi.
	FalconRW FalconVariant = "rw"
)

// UnsupportedFalconVariantError is the typed refusal newModel returns when a
// Falcon checkpoint is one of the two variants whose fused-qkv layout fak's
// contiguous split does not implement. It names the variant, the structural
// witness that identified it, and what IS supported — so an operator reading it
// knows the checkpoint is UNIMPLEMENTED rather than corrupt, which the generic
// "required canonical tensor %q has no source in the manifest" resolver text
// (tensor_resolver.go) does not convey.
type UnsupportedFalconVariantError struct {
	// Variant is the detected, unimplemented Falcon architecture.
	Variant FalconVariant
	// Arch is the checkpoint's declared model_type (cfg.ModelType), "" if absent.
	Arch string
	// Witness is the manifest tensor whose presence (or whose row layout, read
	// together with the head counts) identified the variant.
	Witness string
	// NumHeads / NumKVHeads are the derived attention head counts the detection
	// read, so the refusal can be re-derived from the config by hand.
	NumHeads, NumKVHeads int
}

func (e *UnsupportedFalconVariantError) Error() string {
	arch := e.Arch
	if arch == "" {
		arch = "<unknown>"
	}
	head := "model: unsupported Falcon variant " + string(e.Variant) + " (model_type " + arch + "): "
	supported := " The only implemented Falcon variant is Falcon-7B (multi_query=true, num_kv_heads=1," +
		" one shared input_layernorm, CONTIGUOUS [q|k|v] fused qkv, RoPE) — witnessed by" +
		" TestFalconCPUNumericOracle. Use that variant, or a backend that implements this one."
	switch e.Variant {
	case FalconNewDecoderArch:
		return head + "the checkpoint carries the per-layer two-LayerNorm block (witness tensor " + e.Witness +
			"), which only new_decoder_architecture=true ships (Falcon-40B / 180B)." +
			" HF FalconAttention._split_heads lays that variant's fused self_attention.query_key_value out" +
			" INTERLEAVED PER KV GROUP ([g q-heads | 1 k-head | 1 v-head] repeated num_kv_heads times)," +
			" while fak's loader cuts it into three CONTIGUOUS axis-0 ranges q|k|v." +
			" Both layouts hold (num_heads+2*num_kv_heads)*head_dim rows (here num_heads=" + itoa(e.NumHeads) +
			", num_kv_heads=" + itoa(e.NumKVHeads) + "), so splitFusedProjections' row-count guard cannot tell them" +
			" apart and the load would otherwise succeed and decode wrong; fak maps no ln_attn/ln_mlp either, so" +
			" the forward would then panic on a missing canonical input_layernorm mid-request." + supported
	case FalconRW:
		return head + "num_kv_heads == num_attention_heads (" + itoa(e.NumKVHeads) +
			") on a checkpoint with a fused " + e.Witness +
			", i.e. neither multi_query=true nor new_decoder_architecture=true (Falcon-RW-1B / RW-7B)." +
			" HF FalconAttention._split_heads lays that variant out INTERLEAVED PER HEAD" +
			" (num_heads, 3, head_dim), while fak's loader cuts it into three CONTIGUOUS axis-0 ranges q|k|v." +
			" Both layouts hold exactly 3*num_heads*head_dim rows, so splitFusedProjections' row-count guard" +
			" cannot tell them apart: the checkpoint loads clean and decodes wrong with no error at all." +
			" Falcon-RW is also ALiBi rather than RoPE, and HF folds that bias INTO the logits as" +
			" (scores+alibi)*1/sqrt(head_dim) where fak adds the MPT-convention bias AFTER the scale" +
			" (forward.go attnSeq) — a second divergence this refusal contains." + supported
	default:
		return head + "not implemented." + supported
	}
}

// falconFusedQKVSource returns the first per-layer HF-vocabulary fused attention
// tensor present in the manifest. Its presence is what makes the variant question
// load-bearing: it is the tensor splitFusedProjections is about to cut, and it is
// also the gate materializeFalconTensors itself keys on (a transformer.h.* source
// vocabulary), so a checkpoint without it never reached the falcon path at all.
func falconFusedQKVSource(cfg Config, man map[string]tensorMeta) (string, bool) {
	for l := 0; l < cfg.NumLayers; l++ {
		name := "transformer.h." + itoa(l) + ".self_attention.query_key_value.weight"
		if _, ok := man[name]; ok {
			return name, true
		}
	}
	return "", false
}

// falconDualBlockNorm returns the first per-layer ln_attn / ln_mlp tensor in the
// manifest. That pair is the STRUCTURAL tell for new_decoder_architecture=true:
// HF's FalconDecoderLayer builds ln_attn+ln_mlp when and only when
// new_decoder_architecture is set, and builds input_layernorm otherwise. Keying on
// the tensors rather than on a "new_decoder_architecture" config field means the
// detection still works for a checkpoint (or a GGUF conversion) that does not
// carry the key at all.
func falconDualBlockNorm(cfg Config, man map[string]tensorMeta) (string, bool) {
	for l := 0; l < cfg.NumLayers; l++ {
		p := "transformer.h." + itoa(l) + "."
		for _, suffix := range []string{"ln_attn.weight", "ln_mlp.weight"} {
			if _, ok := man[p+suffix]; ok {
				return p + suffix, true
			}
		}
	}
	return "", false
}

// refuseUnsupportedFalconVariant fails the load with a typed
// *UnsupportedFalconVariantError when a fused-qkv Falcon checkpoint is one of the
// two variants whose layout the contiguous split does not implement.
//
// It must run AFTER materializeFalconTensors (so the falcon path has been taken)
// and BEFORE splitFusedProjections (so the operator gets this named variant
// refusal instead of a silently mis-cut q/k/v).
//
// Both tells are structural or already-derived — no new speculative config flag:
//
//   - new_decoder_architecture: the ln_attn/ln_mlp block-norm pair exists in the
//     manifest. Falcon-7B has neither.
//   - Falcon-RW: num_kv_heads == num_attention_heads. multi_query=true drives
//     NumKVHeads to 1 (config.go deriveConfigAxes) and 40B/180B declare a
//     num_kv_heads strictly below num_attention_heads, so plain MHA on a FUSED
//     Falcon qkv is reachable only through the deriveConfigAxes fallback that
//     fires when neither key selects a KV count — exactly the RW config.
//
// NumHeads<=1 is deliberately NOT refused: at a single head the per-head
// interleaved layout and the contiguous layout are the same bytes, so there is no
// divergence to fence.
func refuseUnsupportedFalconVariant(cfg Config, man map[string]tensorMeta) error {
	src, ok := falconFusedQKVSource(cfg, man)
	if !ok {
		return nil // not a fused-qkv Falcon-vocabulary checkpoint — nothing to fence
	}
	if witness, dual := falconDualBlockNorm(cfg, man); dual {
		return &UnsupportedFalconVariantError{
			Variant: FalconNewDecoderArch, Arch: cfg.ModelType, Witness: witness,
			NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads,
		}
	}
	if cfg.NumHeads > 1 && cfg.NumKVHeads == cfg.NumHeads {
		return &UnsupportedFalconVariantError{
			Variant: FalconRW, Arch: cfg.ModelType, Witness: src,
			NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads,
		}
	}
	return nil
}
