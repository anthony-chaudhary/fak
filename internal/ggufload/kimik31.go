package ggufload

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// kimik31.go — the Kimi-K3.1 ("kimi-k31") GGUF architecture admission: its own
// internal model_type, a config applier for the KDA + Gated-DSA + latent-MoE
// axes, and fail-closed typed refusals naming the offending GGUF key.
//
// SCOPE (ticket 1611-kimi-k31): this file owns arch recognition + header/
// metadata parsing (leaf L1) plus the L1 tail hook into the L2 required-
// tensor-family census (validateKimiK31RequiredTensors, kimik31_tensors.go —
// the GGUF→canonical tensor map, the fail-closed family guard, and the map-
// level routed-expert shard census live there). No tokenizer work (L3), no
// forward path (L5–L8). A loaded kimi-k31 checkpoint therefore admits its
// header AND tensor directory cleanly and stays execution-gated until the
// native K3.1 forward lands; model.ClassifyForwardPath is deliberately NOT
// wired to a K3.1 path (internal/model owns that seam).
//
// WHY DISTINCT FROM "deepseek2": the existing Moonshot alias
// (canonicalGGUFArch, gguf_config.go) collapses every K2/K3 spelling onto
// "deepseek2" because K2/K3 carry NO DSA indexer and ride the plain MLA+MoE
// forward. K3.1 replaces 24 of its 93 full-attention layers with GATED DSA —
// sparse MLA + a top-2048 learned indexer (32 index heads x 128 dim) — and
// keeps 69 KDA linear-attention layers plus one dense layer. Admitting K3.1
// through deepseek2 would silently drop the indexer axis and mis-route the
// hybrid composition, so K3.1 earns its own internal arch "kimik31". The old
// alias stays exact-spelling only: it lists K2/K3 spellings verbatim and does
// not prefix-match, so "kimi-k31" was never captured by it; the explicit case
// below plus the sibling-leak tests pin that precedence.
//
// ARCH SPELLING: the community K3.1 converter registers
// LLM_ARCH_KIMI_K31 = "kimi-k31" (csabakecskemeti/llama.cpp@kimi-k31,
// 46bed7e66a25cba0d0278451c4feb05bc3ff5b9c, fetched 2026-11-07); the underscore
// variant is accepted alongside, matching the kimi-k2/kimi_k2 precedent. Until
// a real kimi-k31 GGUF header is read on the node (study PF-06), this string
// is the planned spelling — a mismatch surfaces as an unrecognized-arch load
// refusal, not a silent mis-admission.
//
// METADATA KEYS: the K3.1 converter inherits every K3 key (llama.cpp master
// src/models/kimi-k3.cpp, K3 arch merged 2026-08-15 as ad1de39) and adds the
// three attention.indexer.* scalars (fork src/models/kimi-k31.cpp). Key
// spellings are pinned against both sources:
//   - MoE axis reuses the deepseek2-convention glmKey* spellings
//     (expert_count / expert_used_count / expert_feed_forward_length /
//     expert_shared_count / leading_dense_block_count / expert_weights_*);
//   - MLA axis reuses the *_mla per-head keys (attention.key_length_mla is the
//     PER-HEAD qk width = qk_nope + qk_rope, exactly as GLM-5.2 stores it);
//   - the DSA indexer reuses the GLM-5.2-VALIDATED attention.indexer.*
//     spellings — the K3.1 fork reads precisely those three keys;
//   - the KDA axis uses the K3-specific ssm.conv_kernel / kda.head_dim /
//     kda.gate_lower_bound spellings;
//   - the per-layer KDA/DSA composition rides the standard per-layer
//     attention.head_count_kv integer array: entry 0 marks a KDA (recurrent)
//     layer, a positive entry a DSA (MLA) layer — the same marker
//     llama.cpp's K3 graph uses (n_head_kv(i) == 0).
//
// DEFERRED AXES (no model.Config field exists yet; owned by later leaves, do
// NOT invent surface here): expert_latent_length (the 3584-wide latent routed
// space), activation.situ_beta / .situ_linear_beta (SiTU-GLU parameters;
// HiddenAct records the family when present), attn_res_block_size (cross-layer
// residual scoring), and expert_gating_func. They are intentionally unread at
// L1.

// archIsKimiK31 reports whether arch is the canonical internal Kimi-K3.1
// architecture. Callers must pass the canonicalGGUFArch-normalized string.
func archIsKimiK31(arch string) bool {
	return arch == "kimik31"
}

// Layer-type tokens applyKimiK31Config writes into cfg.LayerTypes. They are
// deliberately DISTINCT from the qwen35 "linear_attention"/"full_attention"
// vocabulary: IsQwen35Hybrid() fires on any "linear_attention" entry and would
// route a K3.1 checkpoint into the GDN forward, which is not KDA. Unknown
// tokens degrade to cachelayout.LayerKindUnknown (boundary-only), which is the
// correct interim state while the K3.1 forward is unbuilt (L5–L8 replace this
// dispatch with ForwardKimiK31).
const (
	kimiK31LayerKDA = "kda" // Kimi Delta Attention: conv1d + forget gate + delta update + full-rank gate
	kimiK31LayerDSA = "dsa" // Gated DeepSeek Sparse Attention: sparse MLA + top-2048 learned indexer
)

// kimi-k31 metadata keys. The converter namespaces these under the file's own
// "<arch>." prefix (so p == the RAW arch + ".", not "kimik31."), and the
// MoE/MLA/indexer spellings match the deepseek2/glm convention, so the shared
// glmKey* constants are reused where they are identical.
const (
	// ---- MoE FFN axis (deepseek2 convention) --------------------------------
	k31KeyExpertCount        = glmKeyExpertCount        // "expert_count"
	k31KeyExpertUsedCount    = glmKeyExpertUsedCount    // "expert_used_count"
	k31KeyExpertFFNLength    = glmKeyExpertFFNLength    // "expert_feed_forward_length"
	k31KeyExpertSharedCount  = glmKeyExpertSharedCount  // "expert_shared_count"
	k31KeyExpertSharedFFNLen = glmKeyExpertSharedFFNLen // "expert_shared_feed_forward_length"
	k31KeyLeadingDenseBlocks = glmKeyLeadingDenseBlocks // "leading_dense_block_count"
	k31KeyExpertWeightsScale = glmKeyExpertWeightsScale // "expert_weights_scale"
	k31KeyExpertWeightsNorm  = glmKeyExpertWeightsNorm  // "expert_weights_norm"

	// ---- MLA (DeepSeek latent attention) axis -------------------------------
	k31KeyQLoraRank   = glmKeyQLoraRank      // "attention.q_lora_rank"
	k31KeyKVLoraRank  = glmKeyKVLoraRank     // "attention.kv_lora_rank"
	k31KeyQKNopeDim   = glmKeyQKNopeDim      // "attention.qk_nope_head_dim" (explicit override)
	k31KeyQKRopeDim   = glmKeyQKRopeDim      // "attention.qk_rope_head_dim" (explicit override)
	k31KeyVHeadDim    = glmKeyVHeadDim       // "attention.v_head_dim" (explicit override)
	k31KeyKeyLenMLA   = glmKeyKeyLengthMLA   // "attention.key_length_mla" (per-head qk = nope + rope)
	k31KeyValueLenMLA = glmKeyValueLengthMLA // "attention.value_length_mla" (per-head v)
	k31KeyKeyLength   = glmKeyKeyLength      // "attention.key_length" (fallback chain only)
	k31KeyValueLength = glmKeyValueLength    // "attention.value_length" (fallback chain only)

	// ---- Gated DSA learned-indexer axis (GLM-5.2-validated spellings) -------
	k31KeyIndexNHeads  = glmKeyIndexNHeads  // "attention.indexer.head_count"
	k31KeyIndexHeadDim = glmKeyIndexHeadDim // "attention.indexer.key_length"
	k31KeyIndexTopK    = glmKeyIndexTopK    // "attention.indexer.top_k"

	// ---- KDA (Kimi Delta Attention) axis (K3-specific spellings) ------------
	k31KeySSMConvKernel     = "ssm.conv_kernel"      // short causal conv kernel size
	k31KeyKDAHeadDim        = "kda.head_dim"         // per-head KDA width (q/k/v share it)
	k31KeyKDAGateLowerBound = "kda.gate_lower_bound" // decay-gate lower bound (K3 ships -5.0)

	// ---- Cross-layer residual scoring + SiTU-GLU activation ------------------
	k31KeyAttnResBlockSize = "attn_res_block_size"
	k31KeySituBeta         = "activation.situ_beta"
	k31KeySituLinearBeta   = "activation.situ_linear_beta"

	// ---- Per-layer KDA/DSA composition --------------------------------------
	// Standard per-layer head_count_kv integer array: 0 marks a KDA layer, a
	// positive value a DSA layer (llama.cpp K3 recurrent marker).
	k31KeyHeadCountKV = "attention.head_count_kv"
)

// applyKimiK31Config reads the kimi-k31 MoE + MLA + DSA-indexer + KDA axes from
// the file's raw "<arch>." prefix into cfg, mirroring applyDeepSeek41Config's
// contract: best-effort reads (a key present only when positive never clobbers
// a generic value with zero) plus FAIL-LOUD preconditions that name the
// offending GGUF key so a mislabeled or broken artifact refuses at the header,
// not deep in decode.
//
// Required (typed refusal when missing or non-positive):
//   - the DSA indexer trio (attention.indexer.head_count/.key_length/.top_k) —
//     the axis that makes K3.1 what it is; a kimi-k31-branded file without it
//     is a mislabeled K3 or a broken artifact, never a valid K3.1;
//   - expert_count / expert_used_count — K3.1 is MoE-only (896 routed experts
//     over 92 of 93 layers); without them the routed band cannot be sized;
//   - attention.q_lora_rank — the K3.1 DSA indexer reads the q_lora latent
//     (the upstream loader asserts it);
//   - attention.kv_lora_rank — the MLA compressed-KV latent width.
//
// The per-layer KDA/DSA schedule (attention.head_count_kv) is required only
// when the file carries tensors: a header-only fixture (config estimation,
// arch-normalization golden) legitimately omits the tensor directory and must
// not trip it, exactly as the glm_moe_dsa unresolved-schedule guard.
//
// ropeDim is the already-resolved rope.dimension_count, reused as the
// qk_rope_head_dim fallback under the deepseek2 convention.
func applyKimiK31Config(f *File, p string, cfg *model.Config, ropeDim int) error {
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("gguf: kimi-k31 has no layers (block_count=%d)", cfg.NumLayers)
	}
	if cfg.HiddenSize <= 0 {
		return fmt.Errorf("gguf: kimi-k31 has no hidden size (embedding_length=%d)", cfg.HiddenSize)
	}

	// ---- Gated DSA learned-indexer axis (arch-defining; fail-closed) --------
	// Every refusal names the missing key: a kimi-k31 header lacking its
	// indexer is not K3.1, and admitting it would silently ride the
	// no-indexer deepseek2 semantics the old alias encodes.
	for _, item := range []struct {
		key string
		set func(int)
	}{
		{p + k31KeyIndexNHeads, func(v int) { cfg.IndexNHeads = v }},
		{p + k31KeyIndexHeadDim, func(v int) { cfg.IndexHeadDim = v }},
		{p + k31KeyIndexTopK, func(v int) { cfg.IndexTopK = v }},
	} {
		v, err := requireKimiK31Positive(f, item.key)
		if err != nil {
			return err
		}
		item.set(v)
	}

	// ---- MoE FFN axis (arch-defining counts fail-closed; widths best-effort)
	experts, err := requireKimiK31Positive(f, p+k31KeyExpertCount)
	if err != nil {
		return err
	}
	cfg.NumExperts = experts
	topK, err := requireKimiK31Positive(f, p+k31KeyExpertUsedCount)
	if err != nil {
		return err
	}
	cfg.NumExpertsPerTok = topK
	if v := intValueOrZero(f, p+k31KeyExpertFFNLength); v > 0 {
		cfg.MoEIntermediateSize = v
	}
	if v := intValueOrZero(f, p+k31KeyExpertSharedCount); v > 0 {
		cfg.NSharedExperts = v
	}
	if v := intValueOrZero(f, p+k31KeyExpertSharedFFNLen); v > 0 {
		cfg.SharedIntermediateSize = v
	}
	if v := intValueOrZero(f, p+k31KeyLeadingDenseBlocks); v > 0 {
		cfg.FirstKDenseReplace = v
	}
	if v, ok := f.Float64(p + k31KeyExpertWeightsScale); ok {
		cfg.RoutedScalingFactor = v
	}
	if v, ok := f.Bool(p + k31KeyExpertWeightsNorm); ok {
		cfg.NormTopKProb = v
	}

	// ---- MLA (DeepSeek latent attention) axis -------------------------------
	qRank, err := requireKimiK31Positive(f, p+k31KeyQLoraRank)
	if err != nil {
		return err
	}
	cfg.QLoraRank = qRank
	kvRank, err := requireKimiK31Positive(f, p+k31KeyKVLoraRank)
	if err != nil {
		return err
	}
	cfg.KVLoraRank = kvRank
	// qk_rope_head_dim: explicit key, else the resolved rope dimension (the K3
	// converter derives n_rot() from rope.dimension_count).
	cfg.QKRopeHeadDim = intValueOrZero(f, p+k31KeyQKRopeDim)
	if cfg.QKRopeHeadDim == 0 {
		cfg.QKRopeHeadDim = ropeDim
	}
	// qk_nope_head_dim: explicit key, else the per-head MLA key length minus
	// rope. attention.key_length_mla is unambiguously the PER-HEAD qk width
	// (qk_nope + qk_rope), as in GLM-5.2; the unsuffixed attention.key_length
	// is consulted only as the last-resort fallback, exactly as
	// applyGLMMoeDsaConfig chains it.
	cfg.QKNopeHeadDim = intValueOrZero(f, p+k31KeyQKNopeDim)
	if cfg.QKNopeHeadDim == 0 {
		kl := intValueOrZero(f, p+k31KeyKeyLenMLA)
		if kl == 0 {
			kl = intValueOrZero(f, p+k31KeyKeyLength)
		}
		if kl > cfg.QKRopeHeadDim {
			cfg.QKNopeHeadDim = kl - cfg.QKRopeHeadDim
		}
	}
	// v_head_dim: explicit key, else the per-head MLA value length.
	cfg.VHeadDim = intValueOrZero(f, p+k31KeyVHeadDim)
	if cfg.VHeadDim == 0 {
		cfg.VHeadDim = intValueOrZero(f, p+k31KeyValueLenMLA)
		if cfg.VHeadDim == 0 {
			cfg.VHeadDim = intValueOrZero(f, p+k31KeyValueLength)
		}
	}
	// The generic block derived HeadDim from hidden/heads (7168/96 = 74), which
	// is not the MLA head width. Overwrite it with the real per-head qk width
	// when both halves are known, exactly as applyDeepSeek41Config does.
	if cfg.QKNopeHeadDim > 0 && cfg.QKRopeHeadDim > 0 {
		cfg.HeadDim = cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	}

	// ---- KDA (Kimi Delta Attention) axis (best-effort) -----------------------
	if v := intValueOrZero(f, p+k31KeySSMConvKernel); v > 0 {
		cfg.LinearConvKernelDim = v
	}
	if v := intValueOrZero(f, p+k31KeyKDAHeadDim); v > 0 {
		// KDA runs one head width for q/k/v (d_inner = head_dim * n_head), so
		// the key and value state axes share the width.
		cfg.LinearKeyHeadDim = v
		cfg.LinearValueHeadDim = v
	}
	if v, ok := f.Float64(p + k31KeyKDAGateLowerBound); ok {
		cfg.LinearGateLowerBound = v
	}

	// ---- SiTU-GLU activation family (parameters deferred, see file header) --
	if f.hasKey(p+k31KeySituBeta) || f.hasKey(p+k31KeySituLinearBeta) {
		cfg.HiddenAct = "situ"
	}

	// ---- Per-layer KDA/DSA composition --------------------------------------
	// The standard per-layer head_count_kv array: 0 marks a KDA (recurrent)
	// layer, a positive entry a DSA (MLA) layer. The dense-FFN layer is NOT a
	// separate token here — its attention is KDA; its FFN density rides
	// FirstKDenseReplace. A scalar head_count_kv cannot encode the hybrid
	// schedule and is refused below for a file carrying tensors.
	if arr, ok := f.IntArray(p + k31KeyHeadCountKV); ok {
		if len(arr) != cfg.NumLayers {
			return fmt.Errorf("gguf: kimi-k31 %s has %d entries, want block_count=%d layers",
				p+k31KeyHeadCountKV, len(arr), cfg.NumLayers)
		}
		types := make([]string, cfg.NumLayers)
		for l, kv := range arr {
			if kv == 0 {
				types[l] = kimiK31LayerKDA
			} else {
				types[l] = kimiK31LayerDSA
			}
		}
		cfg.LayerTypes = types
	} else if len(f.Tensors) > 0 {
		// Load-bearing guard, mirroring the glm_moe_dsa unresolved-schedule
		// refusal: a file with real weights MUST resolve its per-layer
		// KDA/DSA schedule; a header-only fixture may omit it.
		return fmt.Errorf("gguf: kimi-k31 ships %d tensors but %s is absent: the per-layer KDA/DSA schedule is unresolved (broken or incomplete K3.1 checkpoint)",
			len(f.Tensors), p+k31KeyHeadCountKV)
	}
	// L2 required-tensor-family census (kimik31_tensors.go): a file carrying a
	// tensor directory must carry every arch-defining family, else it is a
	// mislabeled or stripped artifact and the load refuses here naming the
	// first missing family. Header-only fixtures are a valid no-op, so config
	// estimation and arch-normalization goldens stay admitted exactly as at L1.
	return validateKimiK31RequiredTensors(f, cfg)
}

// requireKimiK31Positive reads a required positive integer GGUF key, refusing
// with a typed error that names the key when it is absent, non-integer, zero,
// or out of int range. It is the kimi-k31 fail-closed primitive: every
// arch-defining axis goes through it so the refusal always names the exact
// "<arch>.<key>" the operator must fix.
func requireKimiK31Positive(f *File, key string) (int, error) {
	v, ok := f.Uint64(key)
	if !ok || v == 0 {
		return 0, fmt.Errorf("gguf: kimi-k31 requires %s to be a positive integer; a file branded kimi-k31 without it is not a valid K3.1 header", key)
	}
	if v > uint64(math.MaxInt) {
		return 0, fmt.Errorf("gguf: %s overflows int", key)
	}
	return int(v), nil
}
