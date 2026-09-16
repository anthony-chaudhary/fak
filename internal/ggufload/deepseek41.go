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
//
// V4.1 emits attn_kv_a_norm, not glm's attn_kv_norm. Both spellings resolve to
// self_attn.kv_a_layernorm.weight. The compressor, indexer, and sink suffixes
// also resolve here, which clears only the LOADER's name gate; the reduced
// forward still fails an in-range layer closed at its own admission seam.
func deepseek41CanonicalSuffix(suffix string) (string, bool) {
	if name, ok := deepseek41EngramSuffixName(suffix); ok {
		return deepseek41EngramPrefix + deepseek41EngramLayerPlaceholder + "." + name + ".weight", true
	}
	mapped, ok := map[string]string{
		"attn_q_a.weight":             "self_attn.q_a_proj.weight",
		"attn_q_a_norm.weight":        "self_attn.q_a_layernorm.weight",
		"attn_q_b.weight":             "self_attn.q_b_proj.weight",
		"attn_kv.weight":              "self_attn.kv_a_proj_with_mqa.weight",
		"attn_kv_norm.weight":         "self_attn.kv_a_layernorm.weight",
		"attn_kv_a_norm.weight":       "self_attn.kv_a_layernorm.weight",
		"attn_output_a.weight":        "self_attn.o_proj_a.weight",
		"attn_output_b.weight":        "self_attn.o_proj_b.weight",
		"indexer.attn_q_b.weight":     "self_attn.indexer.wq_b.weight",
		"indexer.attn_k.weight":       "self_attn.indexer.wk.weight",
		"indexer.k_norm.weight":       "self_attn.indexer.k_norm.weight",
		"indexer.proj.weight":         "self_attn.indexer.weights_proj.weight",
		"attn_compressor_gate.weight": "self_attn.compressor.wgate.weight",
		"attn_compressor_kv.weight":   "self_attn.compressor.wkv.weight",
		"attn_compressor_norm.weight": "self_attn.compressor.norm.weight",
		"attn_sinks.weight":           "attn.attn_sink",
		"exp_probs_b.bias":            "mlp.gate.e_score_correction_bias",
		"exp_probs_b_vl.bias":         "mlp.gate.e_score_correction_bias_vl",
		"ffn_gate_shexp.weight":       "mlp.shared_experts.gate_proj.weight",
		"ffn_up_shexp.weight":         "mlp.shared_experts.up_proj.weight",
		"ffn_down_shexp.weight":       "mlp.shared_experts.down_proj.weight",
		// Hyper-connection taps (GUESSED canonical names ? model.Config carries only
		// the HCMult/iters/eps scalars, and the native V4.1 forward is unimplemented;
		// these map into a dedicated per-layer hc. namespace so a real file's
		// hc_attn_*/hc_ffn_* tensors do not hard-fail the shard load).
		"hc_attn_fn.weight":    "hc.attn_fn.weight",
		"hc_attn_base.weight":  "hc.attn_base.weight",
		"hc_attn_scale.weight": "hc.attn_scale.weight",
		"hc_ffn_fn.weight":     "hc.ffn_fn.weight",
		"hc_ffn_base.weight":   "hc.ffn_base.weight",
		"hc_ffn_scale.weight":  "hc.ffn_scale.weight",
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

// deepseek41BatchedExpert reports whether a deepseek41 GGUF tensor name is a
// batched routed-expert blob and, if so, its layer and per-expert canonical
// projection. V4 reuses the deepseek2-convention spellings (ffn_gate_exps /
// ffn_up_exps / ffn_down_exps), so this is the shared glm classifier. It exists as
// a named seam so the deepseek41 loader dependency is explicit and witnessed.
func deepseek41BatchedExpert(name string) (layer int, proj string, ok bool) {
	return glmMoeDsaBatchedExpert(name)
}
