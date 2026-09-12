package ggufload

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// deepseek41.go ? the DeepSeek-V4.1-Flash ("deepseek41") GGUF architecture and
// tensor map. This is the HEADER/METADATA + shard-loading slice of the native
// V4.1 track: it recognizes the family, reads its MoE + MLA + indexer +
// compress/hyper-connection/output axes into model.Config, maps the V4 tensor
// suffixes to the canonical HF names a native MLA forward reads, and routes the
// Engram tables to a DEDICATED canonical namespace instead of a generic MLP or
// attention tensor.
//
// SCOPE (deliberate, per issue #12903): config parsing + tensor-name mapping +
// shard loading ONLY. There is no quant kernel here, no streaming/caching, and
// no quality claim for any specific GGUF artifact. model.Config has no Engram
// fields, so the Engram axes stay package-local (DeepSeek41Engram) rather than
// inventing model.Config surface; model.IsDeepSeekV41()/refuseDeepSeekV41Native
// still gate executable native execution until the forward lands.
//
// ARCH SPELLINGS: canonicalGGUFArch (gguf_config.go) normalizes every sibling
// spelling (deepseek4, deepseek-v4, deepseek_v4, deepseekv4, deepseek41,
// deepseek_v41, deepseek-v41, deepseek-v4.1, deepseek_v41_text) onto the single
// internal arch "deepseek41" ? pinning it DISTINCT from "deepseek2"
// (DeepSeek-V2/V3/R1) and from the older "deepseek_v4" 0731 profile.
//
// TENSOR SUFFIXES: llama.cpp already ships a DeepSeek-V4 converter with
// MODEL_ARCH.DEEPSEEK4 = "deepseek4", so the GGUF key/tensor spellings below are
// firm, not guessed. The one exception is Engram (below), which has NO upstream
// converter support and is labelled GUESSED.

// archIsDeepSeek41 reports whether arch is the canonical internal DeepSeek-V4.1
// architecture. Callers must pass the canonicalGGUFArch-normalized string.
func archIsDeepSeek41(arch string) bool {
	return arch == "deepseek41"
}

// deepseek41 metadata keys. The V4 converter namespaces these under the file's
// own "<arch>." prefix (so p == the RAW arch + ".", not "deepseek41."), and the
// MoE/MLA/indexer spellings match the deepseek2/glm convention, so the shared
// glmKey* constants are reused where they are identical.
const (
	ds41KeyExpertSharedCount  = glmKeyExpertSharedCount
	ds41KeyExpertSharedFFNLen = glmKeyExpertSharedFFNLen
	ds41KeyExpertWeightsScale = glmKeyExpertWeightsScale
	ds41KeyExpertWeightsNorm  = glmKeyExpertWeightsNorm

	ds41KeyQLoraRank   = glmKeyQLoraRank
	ds41KeyKVLoraRank  = glmKeyKVLoraRank
	ds41KeyQKNopeDim   = glmKeyQKNopeDim
	ds41KeyQKRopeDim   = glmKeyQKRopeDim
	ds41KeyVHeadDim    = glmKeyVHeadDim
	ds41KeyKeyLenMLA   = glmKeyKeyLengthMLA
	ds41KeyValueLenMLA = glmKeyValueLengthMLA

	ds41KeyIndexNHeads  = glmKeyIndexNHeads
	ds41KeyIndexHeadDim = glmKeyIndexHeadDim
	ds41KeyIndexTopK    = glmKeyIndexTopK

	// V4.1-specific: grouped low-rank output projection + hyper-connection.
	ds41KeyOutputGroupCount = "attention.output_group_count" // o_groups
	ds41KeyOutputLoraRank   = "attention.output_lora_rank"   // o_lora_rank
	ds41KeyCompressRatios   = "attention.compress_ratios"    // u32 array (40 + 3 nextn)
	ds41KeyCompressRopeBase = "attention.compress_rope_freq_base"
	ds41KeyHCCount          = "hyper_connection.count"
	ds41KeyHCSinkhornIters  = "hyper_connection.sinkhorn_iterations"
	ds41KeyHCEpsilon        = "hyper_connection.epsilon"
	ds41KeySlidingWindow    = "attention.sliding_window"

	// Engram metadata. GUESSED ? llama.cpp's deepseek4 converter emits NO Engram
	// tensor or metadata key, so these spellings are fak's own provisional
	// namespace under "<arch>."; reconcile if/when an upstream converter defines
	// them. engram_layer_ids is an i32 array, engram_num_embeddings an i64 array.
	ds41KeyEngramLayerIDs     = "engram_layer_ids"
	ds41KeyEngramNumEmbedding = "engram_num_embeddings"
	ds41KeyEngramMaxNgramSize = "engram_max_ngram_size"
	ds41KeyEngramVocabSize    = "engram_vocab_size"
	ds41KeyEngramNHeads       = "engram_n_heads"
	ds41KeyEngramHeadDim      = "engram_head_dim"
	ds41KeyEngramPadTokenID   = "engram_pad_token_id"
	ds41KeyEngramCompVocab    = "engram_compressed_vocab_size"
)

// deepseek41EngramPrefix is the dedicated canonical root the Engram tables map
// into. Used as a sentinel by deepseek41CanonicalSuffix so CanonicalTensorNameArch
// can substitute the decoder layer index (the suffix map has no layer argument).
const deepseek41EngramPrefix = "model.engram."

// deepseek41EngramLayerPlaceholder is substituted with the decoder layer index
// when a mapped Engram suffix is expanded to its final canonical tensor name.
const deepseek41EngramLayerPlaceholder = "<L>"

// DeepSeek41Engram retains the V4.1 Engram metadata on a loaded File. model.Config
// has no Engram fields, so these axes stay self-contained in ggufload ? the
// header/metadata slice needs them only to validate the tables and route their
// tensors to the dedicated namespace, never to size or dequantize them.
//
// GUESSED spellings (no upstream support ? see the file header): the field set
// mirrors the pinned V4.1 text_config (engram_layer_ids, engram_num_embeddings,
// engram_max_ngram_size, engram_vocab_size, engram_n_heads, engram_head_dim,
// engram_pad_token_id, engram_compressed_vocab_size).
type DeepSeek41Engram struct {
	LayerIDs            []int
	NumEmbeddings       []int
	MaxNgramSize        int
	VocabSize           int
	NHeads              int
	HeadDim             int
	PadTokenID          int
	CompressedVocabSize int
}

// applyDeepSeek41Config reads the deepseek41 MoE + MLA + indexer + compress/
// hyper-connection/grouped-output axes from the file's raw "<arch>." prefix into
// cfg, and retains the Engram metadata on f. It mirrors applyGLMMoeDsaConfig's
// best-effort reads (a key present only when positive never clobbers a generic
// value with zero) so a header-only fixture and a real artifact both resolve.
//
// ropeDim is the already-resolved rope.dimension_count, reused as the
// qk_rope_head_dim fallback under the deepseek2 convention.
//
// FAIL-LOUD (before any large allocation): NumLayers and hidden must be positive;
// and for a file that DECLARES Engram layers, the Engram declaration must be
// internally consistent (one embedding-row count per layer id, every layer id a
// valid decoder index, and positive geometry when a real tensor directory ships
// an Engram table). Each refusal names the offending GGUF key so the failure
// surfaces at the header, not deep in decode.
func applyDeepSeek41Config(f *File, p string, cfg *model.Config, ropeDim int) error {
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("gguf: deepseek41 has no layers (block_count=%d)", cfg.NumLayers)
	}
	if cfg.HiddenSize <= 0 {
		return fmt.Errorf("gguf: deepseek41 has no hidden size (embedding_length=%d)", cfg.HiddenSize)
	}

	// ---- MoE FFN axis (deepseek2 convention) --------------------------------
	applyMoEExpertCounts(f, p, cfg)
	if v := intValueOrZero(f, p+ds41KeyExpertSharedCount); v > 0 {
		cfg.NSharedExperts = v
	}
	if v := intValueOrZero(f, p+ds41KeyExpertSharedFFNLen); v > 0 {
		cfg.SharedIntermediateSize = v
	}
	if v, ok := f.Float64(p + ds41KeyExpertWeightsScale); ok {
		cfg.RoutedScalingFactor = v
	}
	if v, ok := f.Bool(p + ds41KeyExpertWeightsNorm); ok {
		cfg.NormTopKProb = v
	}

	// ---- MLA (DeepSeek latent attention) axis -------------------------------
	if v := intValueOrZero(f, p+ds41KeyQLoraRank); v > 0 {
		cfg.QLoraRank = v
	}
	if v := intValueOrZero(f, p+ds41KeyKVLoraRank); v > 0 {
		cfg.KVLoraRank = v
	}
	cfg.QKRopeHeadDim = intValueOrZero(f, p+ds41KeyQKRopeDim)
	if cfg.QKRopeHeadDim == 0 {
		cfg.QKRopeHeadDim = ropeDim
	}
	cfg.QKNopeHeadDim = intValueOrZero(f, p+ds41KeyQKNopeDim)
	if cfg.QKNopeHeadDim == 0 {
		kl := intValueOrZero(f, p+ds41KeyKeyLenMLA)
		if kl > cfg.QKRopeHeadDim {
			cfg.QKNopeHeadDim = kl - cfg.QKRopeHeadDim
		}
	}
	cfg.VHeadDim = intValueOrZero(f, p+ds41KeyVHeadDim)
	if cfg.VHeadDim == 0 {
		cfg.VHeadDim = intValueOrZero(f, p+ds41KeyValueLenMLA)
	}
	// V4.1's per-head width is qk_nope + qk_rope (448 + 64 = 512). The generic
	// Config derived hidden/heads (80) with no attention.key_length present, so
	// overwrite it with the real MLA head width when both halves are known.
	if cfg.QKNopeHeadDim > 0 && cfg.QKRopeHeadDim > 0 {
		cfg.HeadDim = cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	}

	// ---- DSA-style learned indexer axis -------------------------------------
	if v := intValueOrZero(f, p+ds41KeyIndexNHeads); v > 0 {
		cfg.IndexNHeads = v
	}
	if v := intValueOrZero(f, p+ds41KeyIndexHeadDim); v > 0 {
		cfg.IndexHeadDim = v
	}
	if v := intValueOrZero(f, p+ds41KeyIndexTopK); v > 0 {
		cfg.IndexTopK = v
	}

	// ---- V4 grouped low-rank output + hyper-connection + compression --------
	if v := intValueOrZero(f, p+ds41KeyOutputGroupCount); v > 0 {
		cfg.OGroups = v
	}
	if v := intValueOrZero(f, p+ds41KeyOutputLoraRank); v > 0 {
		cfg.OLoraRank = v
	}
	if v, ok := f.IntArray(p + ds41KeyCompressRatios); ok && len(v) > 0 {
		cfg.CompressRatios = v
	}
	if v := intValueOrZero(f, p+ds41KeyHCCount); v > 0 {
		cfg.HCMult = v
	}
	if v := intValueOrZero(f, p+ds41KeyHCSinkhornIters); v > 0 {
		cfg.HCSinkhornIters = v
	}
	if v, ok := f.Float64(p + ds41KeyHCEpsilon); ok {
		cfg.HCEps = v
	}

	// ---- Engram (GUESSED; validated fail-loud) ------------------------------
	eng := &DeepSeek41Engram{
		LayerIDs:            intArrayOrNil(f, p+ds41KeyEngramLayerIDs),
		NumEmbeddings:       intArrayOrNil(f, p+ds41KeyEngramNumEmbedding),
		MaxNgramSize:        intValueOrZero(f, p+ds41KeyEngramMaxNgramSize),
		VocabSize:           intValueOrZero(f, p+ds41KeyEngramVocabSize),
		NHeads:              intValueOrZero(f, p+ds41KeyEngramNHeads),
		HeadDim:             intValueOrZero(f, p+ds41KeyEngramHeadDim),
		PadTokenID:          intValueOrZero(f, p+ds41KeyEngramPadTokenID),
		CompressedVocabSize: intValueOrZero(f, p+ds41KeyEngramCompVocab),
	}
	if err := validateDeepSeek41Engram(f, p, cfg, eng); err != nil {
		return err
	}
	f.DeepSeek41Engram = eng
	return nil
}

// validateDeepSeek41Engram is the fail-loud Engram guard. It fires only for a
// file that DECLARES Engram layers (non-empty engram_layer_ids): a file with no
// Engram keys is a valid no-op, matching the best-effort contract of the other
// V4 axes. Every refusal names the offending key.
//
// The len(f.Tensors) > 0 guard mirrors applyGLMMoeDsaConfig's indexer guard: a
// header-only fixture (config estimation, arch-normalization golden) legitimately
// declares the Engram axes with no tensor directory and must not trip the
// geometry check. Only a file carrying real weights must resolve positive Engram
// geometry for a shipped table.
func validateDeepSeek41Engram(f *File, p string, cfg *model.Config, eng *DeepSeek41Engram) error {
	if len(eng.LayerIDs) == 0 {
		return nil // no Engram declaration -> nothing to validate
	}
	if len(eng.NumEmbeddings) != len(eng.LayerIDs) {
		return fmt.Errorf("gguf: deepseek41 declares %s=%v but %s has %d entries, want %d",
			p+ds41KeyEngramLayerIDs, eng.LayerIDs, p+ds41KeyEngramNumEmbedding, len(eng.NumEmbeddings), len(eng.LayerIDs))
	}
	for i, id := range eng.LayerIDs {
		if id < 0 || id >= cfg.NumLayers {
			return fmt.Errorf("gguf: deepseek41 %s[%d]=%d is not a valid decoder layer index (block_count=%d)",
				p+ds41KeyEngramLayerIDs, i, id, cfg.NumLayers)
		}
	}
	if len(f.Tensors) > 0 && deepseek41HasEngramTable(f) {
		if eng.NHeads <= 0 {
			return fmt.Errorf("gguf: deepseek41 ships an Engram table tensor but %s=%d is not positive", p+ds41KeyEngramNHeads, eng.NHeads)
		}
		if eng.HeadDim <= 0 {
			return fmt.Errorf("gguf: deepseek41 ships an Engram table tensor but %s=%d is not positive", p+ds41KeyEngramHeadDim, eng.HeadDim)
		}
	}
	return nil
}

// deepseek41HasEngramTable reports whether the tensor directory contains any
// Engram table tensor (blk.<L>.engram_table / engram_key / engram_value). It is
// the presence probe the geometry guard keys on so a declared-but-tensorless
// header does not have to supply positive NHeads/HeadDim.
func deepseek41HasEngramTable(f *File) bool {
	for _, t := range f.Tensors {
		if _, _, suffix, ok := splitDeepSeek41BlkTensor(t.Name); ok {
			if _, isEngram := deepseek41EngramSuffixName(suffix); isEngram {
				return true
			}
		}
	}
	return false
}

// splitDeepSeek41BlkTensor splits "blk.<layer>.<suffix>" into its layer index and
// suffix. It is the deepseek41 front half of the name classifiers; ok=false for
// a non-blk or non-integer-layer name.
func splitDeepSeek41BlkTensor(name string) (layer int, prefix, suffix string, ok bool) {
	l, suffix, ok := parseGLMBlkLayerSuffix(name)
	if !ok {
		return 0, "", "", false
	}
	return l, "blk.", suffix, true
}

// intArrayOrNil reads a GGUF integer metadata array into []int, returning nil when
// the key is absent or malformed. Engram declares engram_layer_ids as an i32
// array and engram_num_embeddings as an i64 array; File.IntArray accepts every
// integer element type, so one reader serves both.
func intArrayOrNil(f *File, key string) []int {
	if v, ok := f.IntArray(key); ok {
		return v
	}
	return nil
}

// deepseek41CanonicalSuffix maps a DeepSeek-V4 per-layer GGUF tensor suffix (after
// "blk.<L>.") to the canonical HF suffix (after "model.layers.<L>.") the native
// MLA forward reads, reusing the glm/deepseek2 conventions where they match.
//
// Engram suffixes map to the DEDICATED canonical root and carry the layer
// placeholder "<L>": deepseek41CanonicalSuffix has no layer argument, so the
// caller (CanonicalTensorNameArch) substitutes the decoder index. They are
// deliberately NOT mapped into any self_attn./mlp. namespace, so an Engram table
// can never fall through to a generic MLP/attention canonical name.
//
// Returns ok=false for anything not V4-specific, so CanonicalTensorNameArch falls
// through to the shared base map (attn_norm, ffn_norm, attn_output, ...). The
// batched routed experts (ffn_gate_exps/up/down) are handled by the loader's 1->E
// splitter BEFORE this 1:1 map, exactly as for glm ? see deepseek41BatchedExpert.
//
// MLA KV-b: V4 uses the SINGLE attn_kv tensor (unlike glm's attn_k_b/attn_v_b
// split), so attn_kv maps straight to self_attn.kv_a_proj_with_mqa.weight and
// deepseek41 is deliberately kept OUT of archUsesMLAMoELayout ? the glm KV-b
// 2->1 merge (glmMoeDsaSplitKVB) must never run for a V4 file.
func deepseek41CanonicalSuffix(suffix string) (string, bool) {
	if name, ok := deepseek41EngramSuffixName(suffix); ok {
		return deepseek41EngramPrefix + deepseek41EngramLayerPlaceholder + "." + name + ".weight", true
	}
	mapped, ok := map[string]string{
		"attn_q_a.weight":         "self_attn.q_a_proj.weight",
		"attn_q_a_norm.weight":    "self_attn.q_a_layernorm.weight",
		"attn_q_b.weight":         "self_attn.q_b_proj.weight",
		"attn_kv.weight":          "self_attn.kv_a_proj_with_mqa.weight",
		"attn_kv_norm.weight":     "self_attn.kv_a_layernorm.weight",
		"attn_output_a.weight":    "self_attn.o_proj_a.weight",
		"attn_output_b.weight":    "self_attn.o_proj_b.weight",
		"indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
		"indexer.proj.weight":     "self_attn.indexer.weights_proj.weight",
		"exp_probs_b.bias":        "mlp.gate.e_score_correction_bias",
		"ffn_gate_shexp.weight":   "mlp.shared_experts.gate_proj.weight",
		"ffn_up_shexp.weight":     "mlp.shared_experts.up_proj.weight",
		"ffn_down_shexp.weight":   "mlp.shared_experts.down_proj.weight",
		// Hyper-connection taps (GUESSED canonical names ? model.Config carries only
		// the HCMult/iters/eps scalars, and the native V4.1 forward is unimplemented;
		// these map into a dedicated per-layer hc. namespace so a real file's
		// hc_attn_*/hc_ffn_* tensors do not hard-fail the shard load).
		"hc_attn_fn":    "hc.attn_fn",
		"hc_attn_base":  "hc.attn_base",
		"hc_attn_scale": "hc.attn_scale",
		"hc_ffn_fn":     "hc.ffn_fn",
		"hc_ffn_base":   "hc.ffn_base",
		"hc_ffn_scale":  "hc.ffn_scale",
	}[suffix]
	return mapped, ok
}

// deepseek41EngramSuffixName maps an Engram tensor suffix to its canonical
// namespace leaf (engram_table / engram_key / engram_value), reporting whether the
// suffix is an Engram table member. The canonical name is composed by
// deepseek41CanonicalSuffix into model.engram.<L>.<leaf>.weight.
func deepseek41EngramSuffixName(suffix string) (string, bool) {
	switch suffix {
	case "engram_table", "engram_key", "engram_value":
		return suffix, true
	}
	return "", false
}

// deepseek41BatchedExpert reports whether a deepseek41 GGUF tensor name is a
// batched routed-expert blob and, if so, its layer and per-expert canonical
// projection. V4 reuses the deepseek2-convention spellings (ffn_gate_exps /
// ffn_up_exps / ffn_down_exps), so this is the shared glm classifier. It exists as
// a named seam so the deepseek41 loader dependency is explicit and witnessed.
func deepseek41BatchedExpert(name string) (layer int, proj string, ok bool) {
	return glmMoeDsaBatchedExpert(name)
}
