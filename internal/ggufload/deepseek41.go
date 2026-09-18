package ggufload

// DeepSeek V4.1 Engram metadata names in this file are adapted from antirez/ds4
// at bd66c402070042bf0a79ad6ece8242de4c93680c (gguf-tools/deepseek41_metadata.py).
//
// MIT License
// Copyright (c) 2026 The ds4.c authors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2023 DeepSeek
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import (
	"fmt"
	"strings"

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
// firm, not guessed. Engram metadata follows antirez/ds4's MIT-licensed
// deepseek41_metadata.py at bd66c402070042bf0a79ad6ece8242de4c93680c.

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
	// V4.1's staged vcruz artifact ships the UNSUFFIXED deepseek2 spellings attention.key_length
	// (per-head qk width, 512 = qk_nope 448 + qk_rope 64) and attention.value_length (per-head v
	// width). They are the fallback when the *_mla spelling is absent â€” the exact chain
	// applyGLMMoeDsaConfig uses. Reading only *_mla left QKNopeHeadDim/VHeadDim at 0, which is
	// how the first physical V4.1 completion died at glmDsaAppendAttentionKV's precondition.
	ds41KeyKeyLength   = glmKeyKeyLength
	ds41KeyValueLength = glmKeyValueLength

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

	// Engram metadata emitted by ds4's converter. The old flat provisional keys
	// remain accepted below as a compatibility fallback, but new files should use
	// this nested namespace and retain the converter-produced hash constants.
	ds41KeyEngramEncoding     = "engram.encoding"
	ds41KeyEngramLayerIDs     = "engram.layer_ids"
	ds41KeyEngramNumEmbedding = "engram.rows"
	ds41KeyEngramCompVocab    = "engram.compressed_vocab_size"
	ds41KeyEngramPadTokenID   = "engram.pad_id"
	ds41KeyEngramTokenMap     = "engram.token_map"
	ds41KeyEngramPrimes       = "engram.primes"
	ds41KeyEngramMultipliers  = "engram.multipliers"

	// vcruz305-dialect nested keys, absent from the ds4 dialect. The ds4 dialect
	// writes encoding/rows/compressed_vocab_size instead of head_count/key_length/
	// max_ngram_size/offsets.
	ds41KeyEngramHeadCount    = "engram.head_count"
	ds41KeyEngramKeyLength    = "engram.key_length"
	ds41KeyEngramMaxNgramSize = "engram.max_ngram_size"
	ds41KeyEngramOffsets      = "engram.offsets"

	// Pre-ds4 fak fixtures used these guessed flat spellings. Read them only when
	// the converter-defined key is absent so real artifact metadata wins.
	ds41LegacyEngramLayerIDs     = "engram_layer_ids"
	ds41LegacyEngramNumEmbedding = "engram_num_embeddings"
	ds41LegacyEngramMaxNgramSize = "engram_max_ngram_size"
	ds41LegacyEngramVocabSize    = "engram_vocab_size"
	ds41LegacyEngramNHeads       = "engram_n_heads"
	ds41LegacyEngramHeadDim      = "engram_head_dim"
	ds41LegacyEngramPadTokenID   = "engram_pad_token_id"
	ds41LegacyEngramCompVocab    = "engram_compressed_vocab_size"
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
type DeepSeek41Engram struct {
	Encoding            string
	LayerIDs            []int
	NumEmbeddings       []int
	MaxNgramSize        int
	VocabSize           int
	NHeads              int
	HeadDim             int
	PadTokenID          int
	CompressedVocabSize int
	TokenMap            []int
	Primes              []int
	Multipliers         []uint64
	Offsets             []int
}

// ds41HasAnyIndexer reports whether ANY decoder layer carries a DSA learned-indexer
// tensor. It is the V4.1 derivation sentinel, deliberately weaker than the
// glm_moe_dsa path's glmLayerHasIndexer(f, 0): V4.1 runs its indexer on a strided
// subset of layers starting at layer 2, so blk.0 legitimately ships NO indexer
// tensors and a layer-0 probe would never fire for the real artifact.
func ds41HasAnyIndexer(f *File, numLayers int) bool {
	for l := 0; l < numLayers; l++ {
		if glmLayerHasIndexer(f, l) {
			return true
		}
	}
	return false
}

// ds41KVLoraRankFromTensors derives the MLA kv_lora_rank from the compressed-KV tensor
// shape when the metadata key is absent. The staged vcruz Q2_K V4.1 artifact ships neither
// attention.kv_lora_rank nor the *_mla dims, but every decoder layer carries
// blk.<L>.attn_kv.weight [H, kvLora] (the single low-rank KV projection, mapped to
// self_attn.kv_a_proj_with_mqa.weight) and blk.<L>.attn_kv_a_norm.weight [kvLora]. The norm
// width is the unambiguous rank; the projection's out-dim is the fallback. Returns 0 when no
// layer carries either, so a header-only fixture keeps deriving from metadata alone.
func ds41KVLoraRankFromTensors(f *File, numLayers int) int {
	if len(f.Tensors) == 0 {
		return 0
	}
	for l := 0; l < numLayers; l++ {
		pfx := fmt.Sprintf("blk.%d.", l)
		for _, t := range f.Tensors {
			if t.Name != pfx+"attn_kv_a_norm.weight" || len(t.Dims) == 0 {
				continue
			}
			if v := int(t.Dims[0]); v > 0 {
				return v
			}
		}
	}
	for l := 0; l < numLayers; l++ {
		pfx := fmt.Sprintf("blk.%d.", l)
		for _, t := range f.Tensors {
			if t.Name != pfx+"attn_kv.weight" || len(t.Dims) < 2 {
				continue
			}
			if v := int(t.Dims[1]); v > 0 {
				return v
			}
		}
	}
	return 0
}

// ds41QBProjOutDim returns the out-dim (last axis) of the first decoder layer's
// blk.<L>.attn_q_b.weight, the MLA query up-projection. That axis equals
// num_heads * (qk_nope_head_dim + qk_rope_head_dim), the artifact's own ground
// truth for the per-head qk width -- the arbiter that disambiguates a
// per-head-semantics attention.key_length from a latent-semantics one
// (issue #13244). Returns 0 when no layer carries the tensor, so a header-only
// fixture keeps deriving from metadata alone without a spurious refusal.
func ds41QBProjOutDim(f *File, numLayers int) int {
	if len(f.Tensors) == 0 {
		return 0
	}
	for l := 0; l < numLayers; l++ {
		pfx := fmt.Sprintf("blk.%d.", l)
		for _, t := range f.Tensors {
			if t.Name != pfx+"attn_q_b.weight" || len(t.Dims) < 2 {
				continue
			}
			if v := int(t.Dims[len(t.Dims)-1]); v > 0 {
				return v
			}
		}
	}
	return 0
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
	// kv_lora_rank: explicit key, else derive from the compressed-KV tensor shape. The
	// staged vcruz Q2_K V4.1 artifact omits attention.kv_lora_rank entirely, but ships
	// blk.<L>.attn_kv.weight [H, kvLora] and blk.<L>.attn_kv_a_norm.weight [kvLora]; a
	// 0 here makes glmDsaAppendAttentionKV fail its preconditions (kvLora=0) and the
	// first physical completion dies at the opaque "glm_moe_dsa attention step failed".
	if v := intValueOrZero(f, p+ds41KeyKVLoraRank); v > 0 {
		cfg.KVLoraRank = v
	} else if v := ds41KVLoraRankFromTensors(f, cfg.NumLayers); v > 0 {
		cfg.KVLoraRank = v
	}
	cfg.QKRopeHeadDim = intValueOrZero(f, p+ds41KeyQKRopeDim)
	if cfg.QKRopeHeadDim == 0 {
		cfg.QKRopeHeadDim = ropeDim
	}
	// qk_nope_head_dim: explicit key, else the per-head MLA key length minus rope.
	//
	// attention.key_length_mla is unambiguously the PER-HEAD qk width. The
	// unsuffixed attention.key_length is DIALECT-DEPENDENT: V4.1's staged vcruz
	// artifact ships the per-head width there (512 = qk_nope 448 + qk_rope 64), but
	// a GLM-style latent-semantics file carries the LATENT key dim (576) under the
	// same name. Subtracting rope from the latent value yields a plausible-but-wrong
	// per-head width, silently mis-shaping the KV cache / kv_b split (issue #13244).
	// The artifact's own attn_q_b.weight out-dim is the arbiter:
	//   out = num_heads * (qk_nope + qk_rope).
	// Admit the unsuffixed fallback only when it agrees with that ground truth;
	// otherwise refuse, naming the offending key, rather than carry a wrong number.
	cfg.QKNopeHeadDim = intValueOrZero(f, p+ds41KeyQKNopeDim)
	if cfg.QKNopeHeadDim == 0 {
		kl := intValueOrZero(f, p+ds41KeyKeyLenMLA)
		if kl == 0 {
			kl = intValueOrZero(f, p+ds41KeyKeyLength)
			if kl > cfg.QKRopeHeadDim {
				if qOut := ds41QBProjOutDim(f, cfg.NumLayers); qOut > 0 && cfg.NumHeads > 0 {
					perHead := qOut / cfg.NumHeads
					if perHead != kl {
						return fmt.Errorf("gguf: deepseek41 %s=%d is %d-wide latent key semantics, not a per-head width "+
							"(blk.0.attn_q_b.weight out=%d / head_count=%d = %d); rename to %s or %s",
							ds41KeyKeyLength, kl, kl, qOut, cfg.NumHeads, perHead, ds41KeyKeyLenMLA, ds41KeyQKNopeDim)
					}
				}
			}
		}
		if kl > cfg.QKRopeHeadDim {
			cfg.QKNopeHeadDim = kl - cfg.QKRopeHeadDim
		}
	}
	// v_head_dim: explicit key, else the per-head value length (value_length_mla, else
	// attention.value_length - the spelling the staged V4.1 artifact actually ships).
	cfg.VHeadDim = intValueOrZero(f, p+ds41KeyVHeadDim)
	if cfg.VHeadDim == 0 {
		cfg.VHeadDim = intValueOrZero(f, p+ds41KeyValueLenMLA)
		if cfg.VHeadDim == 0 {
			cfg.VHeadDim = intValueOrZero(f, p+ds41KeyValueLength)
		}
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

	// The per-layer DSA indexer SCHEDULE. V4.1 runs its lightning indexer on only a
	// strided subset of layers: a "full" layer ships the indexer.* tensors and
	// computes its own sparse top-k; a later layer without them reuses the most
	// recent full layer's selection ("shared"). The artifact carries no
	// indexer_types key, so derive the schedule from tensor presence exactly as the
	// glm_moe_dsa path does (applyGLMMoeDsaConfig). Without this cfg.IndexerTypes
	// stays empty, glmDsaIndexerKind() treats EVERY layer as "full" (its
	// out-of-range default), and the native DSA forward demands indexer.wq_b.weight
	// on the first indexer-less layer at the first completion -- the exact physical
	// strix3 failure.
	//
	// V4.1's strided subset starts at layer 2 (index_source_layer_ids =
	// {2,8,14,20,24,28,32,36}; compress_ratios[0:2] = 0), so layers 0/1 ship NO
	// indexer tensors AND have no preceding full layer to reuse. A leading "shared"
	// layer is unsatisfiable -- dsaIndexShare refuses it (a shared layer's "reuse
	// the previous selection" would read nil), and the GLM DSA band contract
	// forbids starting a band on one. Those prefix layers are therefore "dense":
	// they attend their full causal prefix, exactly the IndexNHeads==0 dense-MLA
	// seam. (This is also why the glm_moe_dsa path's glmLayerHasIndexer(f, 0)
	// sentinel cannot be reused here: blk.0 legitimately ships none.)
	if types, ok := f.StringArray(p + glmKeyIndexerTypes); ok {
		cfg.IndexerTypes = types
	} else if cfg.NumLayers > 0 && len(f.Tensors) > 0 && ds41HasAnyIndexer(f, cfg.NumLayers) {
		// Only derive for a genuine DSA-indexer checkpoint (SOME layer carries the
		// indexer). A file with no indexer tensors at all leaves the schedule empty
		// rather than fabricating one.
		types := make([]string, cfg.NumLayers)
		seenFull := false
		for l := 0; l < cfg.NumLayers; l++ {
			if glmLayerHasIndexer(f, l) {
				types[l] = "full"
				seenFull = true
			} else if seenFull {
				types[l] = "shared"
			} else {
				types[l] = "dense"
			}
		}
		cfg.IndexerTypes = types
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

	// ---- Engram (ds4 converter keys; legacy flat keys are fallback-only) -----
	eng := &DeepSeek41Engram{
		LayerIDs:            intArrayOrNil(f, p+ds41KeyEngramLayerIDs),
		NumEmbeddings:       intArrayOrNil(f, p+ds41KeyEngramNumEmbedding),
		MaxNgramSize:        intValueOrZero(f, p+ds41KeyEngramMaxNgramSize),
		VocabSize:           intValueOrZero(f, p+ds41LegacyEngramVocabSize),
		NHeads:              intValueOrZero(f, p+ds41KeyEngramHeadCount),
		HeadDim:             intValueOrZero(f, p+ds41KeyEngramKeyLength),
		PadTokenID:          intValueOrZero(f, p+ds41KeyEngramPadTokenID),
		CompressedVocabSize: intValueOrZero(f, p+ds41KeyEngramCompVocab),
		TokenMap:            intArrayOrNil(f, p+ds41KeyEngramTokenMap),
		Primes:              intArrayOrNil(f, p+ds41KeyEngramPrimes),
		Multipliers:         uint64ArrayOrNil(f, p+ds41KeyEngramMultipliers),
		Offsets:             intArrayOrNil(f, p+ds41KeyEngramOffsets),
	}
	if _, ok := f.Metadata[p+ds41KeyEngramMaxNgramSize]; !ok {
		eng.MaxNgramSize = max(eng.MaxNgramSize, intValueOrZero(f, p+ds41LegacyEngramMaxNgramSize))
	}
	if _, ok := f.Metadata[p+ds41KeyEngramHeadCount]; !ok {
		eng.NHeads = max(eng.NHeads, intValueOrZero(f, p+ds41LegacyEngramNHeads))
	}
	if _, ok := f.Metadata[p+ds41KeyEngramKeyLength]; !ok {
		eng.HeadDim = max(eng.HeadDim, intValueOrZero(f, p+ds41LegacyEngramHeadDim))
	}
	eng.Encoding, _ = f.String(p + ds41KeyEngramEncoding)
	if _, ok := f.Metadata[p+ds41KeyEngramLayerIDs]; !ok {
		eng.LayerIDs = intArrayOrNil(f, p+ds41LegacyEngramLayerIDs)
	}
	if _, ok := f.Metadata[p+ds41KeyEngramNumEmbedding]; !ok {
		eng.NumEmbeddings = intArrayOrNil(f, p+ds41LegacyEngramNumEmbedding)
	}
	if _, ok := f.Metadata[p+ds41KeyEngramPadTokenID]; !ok {
		eng.PadTokenID = intValueOrZero(f, p+ds41LegacyEngramPadTokenID)
	}
	if _, ok := f.Metadata[p+ds41KeyEngramCompVocab]; !ok {
		eng.CompressedVocabSize = intValueOrZero(f, p+ds41LegacyEngramCompVocab)
	}
	if len(eng.LayerIDs) > 0 && len(eng.Multipliers)%len(eng.LayerIDs) == 0 {
		eng.MaxNgramSize = max(eng.MaxNgramSize, len(eng.Multipliers)/len(eng.LayerIDs))
	}
	if len(eng.LayerIDs) > 0 && eng.MaxNgramSize > 1 {
		eng.NHeads = max(eng.NHeads, len(eng.Primes)/(len(eng.LayerIDs)*(eng.MaxNgramSize-1)))
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
	converterKeys := hasDeepSeek41ConverterEngramMetadata(f, p)
	vcruzKeys := hasDeepSeek41VcruzEngramMetadata(f, p)
	if converterKeys {
		// The two converter dialects write the array keys at different natural
		// widths: ds4 uses U32 arrays; vcruz uses I32 for layer_ids/token_map and
		// U64 for primes/multipliers/offsets. Accept any integer element type so a
		// real vcruz file is not falsely refused as "must be uint32"; still refuse a
		// non-integer element.
		arrayKeys := []string{ds41KeyEngramLayerIDs, ds41KeyEngramTokenMap, ds41KeyEngramPrimes, ds41KeyEngramMultipliers}
		if !vcruzKeys {
			// ds4 declares per-layer row counts; the vcruz dialect does not.
			arrayKeys = append(arrayKeys, ds41KeyEngramNumEmbedding)
		}
		for _, key := range arrayKeys {
			if !metadataArrayHasIntegerElement(f, p+key) {
				return fmt.Errorf("gguf: deepseek41 converter key %s must be an integer array", p+key)
			}
		}
		// offsets is vcruz-only: validate its width only when the key is present.
		if _, ok := f.Metadata[p+ds41KeyEngramOffsets]; ok && !metadataArrayHasIntegerElement(f, p+ds41KeyEngramOffsets) {
			return fmt.Errorf("gguf: deepseek41 converter key %s must be an integer array", p+ds41KeyEngramOffsets)
		}
		// ds4-only scalar keys: validate the width only when the key is present.
		for _, key := range []string{ds41KeyEngramCompVocab, ds41KeyEngramPadTokenID} {
			if value, ok := f.Metadata[p+key]; ok && value.Type != TypeUint32 {
				return fmt.Errorf("gguf: deepseek41 converter key %s must be uint32", p+key)
			}
		}
	}
	if len(eng.LayerIDs) == 0 {
		return nil // no Engram declaration -> nothing to validate
	}
	layerKey, rowsKey := ds41LegacyEngramLayerIDs, ds41LegacyEngramNumEmbedding
	if converterKeys {
		layerKey, rowsKey = ds41KeyEngramLayerIDs, ds41KeyEngramNumEmbedding
	}
	// The vcruz dialect carries no per-layer row counts, so enforce the pairing
	// only for a file that actually declares the rows key.
	if _, hasRows := f.Metadata[p+rowsKey]; hasRows && len(eng.NumEmbeddings) != len(eng.LayerIDs) {
		return fmt.Errorf("gguf: deepseek41 declares %s=%v but %s has %d entries, want %d",
			p+layerKey, eng.LayerIDs, p+rowsKey, len(eng.NumEmbeddings), len(eng.LayerIDs))
	}
	for i, id := range eng.LayerIDs {
		if id < 0 || id >= cfg.NumLayers {
			return fmt.Errorf("gguf: deepseek41 %s[%d]=%d is not a valid decoder layer index (block_count=%d)",
				p+layerKey, i, id, cfg.NumLayers)
		}
	}
	if converterKeys {
		required := []struct {
			key string
			ok  bool
		}{
			{ds41KeyEngramTokenMap, len(eng.TokenMap) > 0},
			{ds41KeyEngramPrimes, len(eng.Primes) > 0},
			{ds41KeyEngramMultipliers, len(eng.Multipliers) > 0},
		}
		if !vcruzKeys {
			// ds4-dialect-only keys; the vcruz dialect writes neither.
			required = append(required,
				struct {
					key string
					ok  bool
				}{ds41KeyEngramEncoding, eng.Encoding != ""},
				struct {
					key string
					ok  bool
				}{ds41KeyEngramCompVocab, eng.CompressedVocabSize > 0},
			)
		}
		for _, item := range required {
			if !item.ok {
				return fmt.Errorf("gguf: deepseek41 declares %s but required converter key %s is missing or malformed", p+ds41KeyEngramLayerIDs, p+item.key)
			}
		}
		if eng.MaxNgramSize < 2 || eng.NHeads <= 0 || len(eng.Primes) != len(eng.LayerIDs)*(eng.MaxNgramSize-1)*eng.NHeads {
			return fmt.Errorf("gguf: deepseek41 %s/%s hash geometry is inconsistent", p+ds41KeyEngramPrimes, p+ds41KeyEngramMultipliers)
		}
	}
	if len(f.Tensors) > 0 && deepseek41HasEngramTable(f) {
		if !converterKeys && eng.NHeads <= 0 {
			return fmt.Errorf("gguf: deepseek41 ships an Engram table tensor but %s=%d is not positive", p+ds41LegacyEngramNHeads, eng.NHeads)
		}
		if !converterKeys && eng.HeadDim <= 0 {
			return fmt.Errorf("gguf: deepseek41 ships an Engram table tensor but %s=%d is not positive", p+ds41LegacyEngramHeadDim, eng.HeadDim)
		}
	}
	return nil
}

// hasDeepSeek41ConverterEngramMetadata treats any key from ds4's nested Engram
// namespace as a converter declaration. This prevents a partial converter
// header from borrowing missing fields from the provisional flat namespace.
func hasDeepSeek41ConverterEngramMetadata(f *File, p string) bool {
	for _, key := range []string{
		ds41KeyEngramEncoding,
		ds41KeyEngramLayerIDs,
		ds41KeyEngramNumEmbedding,
		ds41KeyEngramCompVocab,
		ds41KeyEngramPadTokenID,
		ds41KeyEngramTokenMap,
		ds41KeyEngramPrimes,
		ds41KeyEngramMultipliers,
		ds41KeyEngramHeadCount,
		ds41KeyEngramKeyLength,
		ds41KeyEngramMaxNgramSize,
		ds41KeyEngramOffsets,
	} {
		if _, ok := f.Metadata[p+key]; ok {
			return true
		}
	}
	return false
}

// hasDeepSeek41VcruzEngramMetadata reports whether the file carries a key unique
// to the vcruz305 dialect (head_count / key_length / max_ngram_size / offsets).
// That dialect writes none of ds4's encoding / rows / compressed_vocab_size keys,
// so validation must not require them for a vcruz file.
func hasDeepSeek41VcruzEngramMetadata(f *File, p string) bool {
	for _, key := range []string{
		ds41KeyEngramHeadCount,
		ds41KeyEngramKeyLength,
		ds41KeyEngramMaxNgramSize,
		ds41KeyEngramOffsets,
	} {
		if _, ok := f.Metadata[p+key]; ok {
			return true
		}
	}
	return false
}

// metadataArrayHasIntegerElement reports whether key is a GGUF integer array of
// any element width (u8/u16/u32/u64/i8/i16/i32/i64). It is the dialect-neutral
// predicate the V4.1 Engram guard uses: ds4 writes U32 arrays while vcruz writes
// I32/U64 arrays for the same logical keys.
func metadataArrayHasIntegerElement(f *File, key string) bool {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return false
	}
	for _, item := range items {
		switch item.Type {
		case TypeUint8, TypeUint16, TypeUint32, TypeUint64,
			TypeInt8, TypeInt16, TypeInt32, TypeInt64:
		default:
			return false
		}
	}
	return true
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

// deepseek41EngramTableTensor reports whether a GGUF tensor name is a per-layer
// PACKED Engram table (blk.<L>.engram_embd.weight, and the legacy
// engram_table / engram_key / engram_value spellings). The packed table is NOT a
// matmul weight: the native V4.1 forward reads it row-wise through the bounded
// model.V41EngramRowSource seam (V41EngramQ2KOpen / V41EngramGGUFOpen), never as
// an f32 matrix. The materializing quant loaders must therefore NOT
// eager-dequantize it - on the published vcruz Q2_K checkpoint one table is
// [256, ~384M rows] = ~98.3B elements, i.e. a 366.2 GiB f32 span that OOMs the
// Go runtime (fak#13152). The projection-side Engram tensors
// (engram_wkv/engram_q/engram_k and their forward spellings) are deliberately
// EXCLUDED: those ARE consumed as f32 weights by the forward and must load.
func deepseek41EngramTableTensor(name string) bool {
	_, _, suffix, ok := splitDeepSeek41BlkTensor(name)
	if !ok {
		return false
	}
	leaf, isEngram := deepseek41EngramSuffixName(suffix)
	if !isEngram {
		return false
	}
	switch leaf {
	case "engram_table", "engram_embd":
		return true
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

func uint64ArrayOrNil(f *File, key string) []uint64 {
	v, _ := metadataArray(f, key, valueUint64)
	return v
}

// deepseek41CanonicalSuffix maps a DeepSeek-V4 per-layer GGUF tensor suffix (after
// "blk.<L>.") to the canonical name the NATIVE non-MLA V4.1 forward reads
// (internal/model/v41_forward.go), reusing the glm/deepseek2 conventions only
// where the native forward's admitted names and shapes actually match.
//
// The native-forward contract is authoritative. Its admitted per-layer shapes
// live at internal/model/v41_forward.go:652-715 and the names its layer step
// reads at :862-877 (attn.wq_a/attn.wq_b/attn.wkv/attn.wo_a/attn.wo_b/attn.sink,
// ffn.gate.weight/ffn.gate.e_score_correction_bias, ffn.shared_experts.w1/w3/w2),
// the compressor at :352-358/407-409 (attn.compressor.wkv/wgate/norm), and the
// indexer at :374-383/460-463 (indexer.wq_b/wk/k_norm/weights_proj). Every arm
// below is grounded against those admitted shapes.
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
// The vcruz GGUF converter emits V4.1-Flash as a NON-MLA attention (wkv ->
// head_dim, kv_norm, partial in-place rope), so attn_kv maps to the native
// attn.wkv.weight [kvLatentRank, H], NOT to any MLA kv_a_proj_with_mqa leaf, and
// deepseek41 is deliberately kept OUT of archUsesMLAMoELayout ? the glm KV-b
// 2->1 merge (glmMoeDsaSplitKVB) must never run for a V4 file.
//
// Q/KV NORMS (follow-on, deliberately NOT wired): the reference Attention carries
// self.q_norm = RMSNorm(q_lora_rank) and self.kv_norm = RMSNorm(head_dim), but
// the reduced native forward has no q_norm/kv_norm lookup yet (see the reads at
// v41_forward.go:862-877). The loader still maps those suffixes to NON-COLLIDING
// dedicated leaves (attn.wq_a_norm.weight, attn.kv_norm.weight) so a real file's
// tensors resolve and cannot collide with an admitted projection; wiring them
// into the forward is a separate leaf and MUST NOT be done here.
//
// V4.1 emits attn_kv_a_norm, not glm's attn_kv_norm. Both spellings converge on
// the same attn.kv_norm.weight leaf. The compressor, indexer, and sink suffixes
// also resolve here, which clears only the LOADER's name gate; the reduced
// forward still fails an in-range layer closed at its own admission seam.
func deepseek41CanonicalSuffix(suffix string) (string, bool) {
	if name, ok := deepseek41EngramSuffixName(suffix); ok {
		return deepseek41EngramPrefix + deepseek41EngramLayerPlaceholder + "." + name + ".weight", true
	}
	if name, ok := deepseek41MHCSuffixName(suffix); ok {
		return name, true
	}
	mapped, ok := map[string]string{
		// Native attention projections (v41_forward.go:668-688, read :867-872).
		"attn_q_a.weight":      "attn.wq_a.weight", // admit [QLoraRank, H]
		"attn_q_b.weight":      "attn.wq_b.weight", // admit [nH*hd, QLoraRank]
		"attn_kv.weight":       "attn.wkv.weight",  // admit [kvLatentRank=512, H]
		"attn_output_a.weight": "attn.wo_a.weight", // admit [OLoraRank, qHeadDim]
		"attn_output_b.weight": "attn.wo_b.weight", // admit [H, oDim]
		"attn_sinks.weight":    "attn.sink",        // admit [nH]
		// Per-layer block norms (admit v41_forward.go:653-656, read :879-880).
		// The native non-MLA V4.1 forward reads model.layers.<L>.attn_norm.weight
		// / ffn_norm.weight; the shared base map would rewrite these to the Llama
		// input_layernorm / post_attention_layernorm names, so the forward's
		// manifest admission (v41AdmitShape) would refuse a layer whose real
		// tensor the file DOES carry. Map them here to the exact native names
		// (fak#13255). ffn_norm stays distinct from the Llama post_attention norm
		// for the same reason: the V4 forward reads the pre-MLP norm by that name.
		"attn_norm.weight": "attn_norm.weight",
		"ffn_norm.weight":  "ffn_norm.weight",
		// Norms: NON-COLLIDING dedicated leaves. The native forward reads neither
		// yet; wiring them is a follow-on (see the doc comment above). Both the
		// converter spelling (attn_kv_a_norm) and the glm spelling (attn_kv_norm)
		// converge on attn.kv_norm.weight.
		"attn_q_a_norm.weight":  "attn.wq_a_norm.weight",
		"attn_kv_norm.weight":   "attn.kv_norm.weight",
		"attn_kv_a_norm.weight": "attn.kv_norm.weight",
		// Native lightning indexer (admit :374-383, read :460-463).
		"indexer.attn_q_b.weight": "indexer.wq_b.weight",
		"indexer.attn_k.weight":   "indexer.wk.weight",
		"indexer.k_norm.weight":   "indexer.k_norm.weight",
		"indexer.proj.weight":     "indexer.weights_proj.weight",
		// Native CED/CSA2 compressor (admit :352-358, read :407-409).
		"attn_compressor_gate.weight": "attn.compressor.wgate.weight",
		"attn_compressor_kv.weight":   "attn.compressor.wkv.weight",
		"attn_compressor_norm.weight": "attn.compressor.norm.weight",
		// Native MoE gate + score-correction bias and shared experts
		// (admit :689-703, read :873-877).
		"exp_probs_b.bias":      "ffn.gate.e_score_correction_bias",
		"exp_probs_b_vl.bias":   "ffn.gate.e_score_correction_bias_vl",
		"ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight", // admit [I, H]
		"ffn_up_shexp.weight":   "ffn.shared_experts.w3.weight", // admit [I, H]
		"ffn_down_shexp.weight": "ffn.shared_experts.w2.weight", // admit [H, I]
	}[suffix]
	return mapped, ok
}

// deepseek41EngramSuffixName maps an Engram tensor suffix to its canonical
// namespace leaf, reporting whether the suffix is an Engram table member. The
// canonical name is composed by deepseek41CanonicalSuffix into
// model.engram.<L>.<leaf>.weight.
//
// Two converter dialects emit different suffix sets:
//   - ds4 (antirez): engram_table / engram_key / engram_value (bare, no ".weight").
//   - vcruz305: engram_embd / engram_k / engram_q / engram_wkv (carrying ".weight",
//     the GGUF form), matching the ggml enums ENGRAM_EMBD/K/Q/WKV.
//
// A trailing ".weight" is accepted and stripped uniformly so both dialect forms
// resolve to the same canonical leaf, and neither falls through to a generic
// attention/MLP canonical name.
//
// Projection-side seam. The reference Engram module carries a projection
// `self.wkv` (reference inference/model.py) and two per-HC norms `q_weight` /
// `k_weight`. The reduced native forward consumes those same tensors under ITS
// canonical spellings, which are the loader's contract target:
//
//	Engram wkv projection  -> engram_kv.weight
//	Engram q norm          -> engram_q_norm.weight
//	Engram k norm          -> engram_k_norm.weight
//
// The loader therefore NORMALIZES both the converter spelling (engram_wkv /
// engram_q / engram_k) and the forward spelling (engram_kv / engram_q_norm /
// engram_k_norm) onto that one forward-consumed leaf, so a shard load produces
// exactly the name internal/model/v41_forward.go:295 and
// v41_forward_engram.go:212 look up. Without this the forward's v41AdmitShape
// lookup cannot find the projection and a declared Engram layer refuses on a
// naming mismatch rather than a genuine unsupported-model refusal.
func deepseek41EngramSuffixName(suffix string) (string, bool) {
	leaf := strings.TrimSuffix(suffix, ".weight")
	switch leaf {
	case "engram_table", "engram_key", "engram_value",
		"engram_embd":
		return leaf, true
	// Projection-side tensors: both dialect and forward spellings converge on the
	// single forward-consumed canonical leaf.
	case "engram_wkv", "engram_kv":
		return "engram_kv", true
	case "engram_q", "engram_q_norm":
		return "engram_q_norm", true
	case "engram_k", "engram_k_norm":
		return "engram_k_norm", true
	}
	return "", false
}

// deepseek41MHCSuffixName maps a V4.1 hyper-connection (mHC) coefficient-block
// suffix to the canonical per-layer leaf the reduced native forward both ADMITS
// (v41StageMHC, internal/model/v41_forward.go:659-665) and CONSUMES
// (internal/model/v41_forward.go:864-866):
//
//	mhc_mixes | mhc_mixes.weight | mhc.mixes.weight -> mhc.mixes.weight
//	mhc_base  | mhc_base.weight  | mhc.base         -> mhc.base
//	mhc_scale | mhc_scale.weight | mhc.scale        -> mhc.scale
//
// Two dialects reach the loader: the converter emits the underscore spelling
// (blk.<L>.mhc_base.weight), while an HF-layout file may already carry the
// forward-consumed dotted leaf (blk.<L>.mhc.base). Both normalize onto the SAME
// forward leaf, mirroring deepseek41EngramSuffixName. The leaves stay inside the
// dedicated per-layer mhc. namespace, so an mHC coefficient block can never fall
// through to a generic attention/MLP canonical name.
//
// The staged vcruz305 Q2_K artifact (fak#13258) carries NEITHER of those: it
// emits a PER-SUBLAYER dialect with one mHC coefficient block per sublayer
// (reference: "For each sublayer (Attention / FFN)"). Dumped from the real GGUF
// header at HiddenSize H=5120, hc_mult=4:
//
//	blk.<L>.hc_attn_base.weight  [24]           F32
//	blk.<L>.hc_attn_fn.weight    [HCMult*H, 24] F32  (20480 = 4*H)
//	blk.<L>.hc_attn_scale.weight [3]            F32
//	blk.<L>.hc_ffn_base.weight   [24]           F32
//	blk.<L>.hc_ffn_fn.weight     [HCMult*H, 24] F32
//	blk.<L>.hc_ffn_scale.weight  [3]            F32
//
// The attention trio resolves onto the forward-consumed leaves above
// (hc_attn_base -> mhc.base, hc_attn_scale -> mhc.scale, hc_attn_fn ->
// mhc.mixes.weight); the FFN trio resolves onto distinct, non-colliding
// mhc.ffn_* leaves. Before this arm existed the artifact's hc_attn_*/hc_ffn_*
// suffixes fell through to the (now removed) dead per-layer hc.* namespace, so
// the forward's required mhc.mixes.weight was never populated and admission
// refused by name. This is name resolution only: the [HCMult*H, 24] fn block is
// now consumed by the forward's flattened-four-stream projection, which reads the
// stored [4H, 24] (or logical [24, 4H]) block over the four width-H streams laid
// end to end with a single shared RMS (fak#13258). The reduced fixture's legacy
// [24, H] single-stream geometry is unaffected.
func deepseek41MHCSuffixName(suffix string) (string, bool) {
	leaf := strings.TrimSuffix(suffix, ".weight")
	switch leaf {
	case "mhc_mixes", "mhc.mixes":
		return "mhc.mixes.weight", true
	case "mhc_base", "mhc.base":
		return "mhc.base", true
	case "mhc_scale", "mhc.scale":
		return "mhc.scale", true
	// Per-sublayer dialect (fak#13258): the attention trio reuses the
	// forward-consumed leaves; the FFN trio gets distinct leaves.
	case "hc_attn_base":
		return "mhc.base", true
	case "hc_attn_scale":
		return "mhc.scale", true
	case "hc_attn_fn":
		return "mhc.mixes.weight", true
	case "hc_ffn_base":
		return "mhc.ffn_base", true
	case "hc_ffn_scale":
		return "mhc.ffn_scale", true
	case "hc_ffn_fn":
		return "mhc.ffn_mixes.weight", true
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
