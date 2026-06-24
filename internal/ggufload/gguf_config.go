package ggufload

import (
	"fmt"
	"io"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Read parses a GGUF header from r — magic, version, the metadata key/value table, and the
// tensor directory — and returns a *File with each tensor's aligned absolute file offset
// resolved. It reads only the header (not the tensor data blob), and errors on a bad magic,
// an unsupported version, or a misaligned tensor offset.
func Read(r io.Reader) (*File, error) {
	rr := &countingReader{r: r}
	magic := make([]byte, 4)
	if err := rr.readFull(magic); err != nil {
		return nil, err
	}
	if string(magic) != Magic {
		return nil, fmt.Errorf("gguf: bad magic %q", string(magic))
	}
	ver, err := rr.u32()
	if err != nil {
		return nil, err
	}
	if ver != Version {
		return nil, fmt.Errorf("gguf: unsupported version %d", ver)
	}
	tensorCount, err := rr.u64()
	if err != nil {
		return nil, err
	}
	kvCount, err := rr.u64()
	if err != nil {
		return nil, err
	}

	meta := make(map[string]Value, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := rr.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: metadata key %d: %w", i, err)
		}
		typ, err := rr.valueType()
		if err != nil {
			return nil, fmt.Errorf("gguf: metadata %s type: %w", key, err)
		}
		v, err := rr.value(typ)
		if err != nil {
			return nil, fmt.Errorf("gguf: metadata %s: %w", key, err)
		}
		meta[key] = v
	}

	tensors := make([]TensorInfo, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := rr.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %d name: %w", i, err)
		}
		nd, err := rr.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %s dims: %w", name, err)
		}
		if nd > 4 {
			return nil, fmt.Errorf("gguf: tensor %s has %d dimensions", name, nd)
		}
		dims := make([]uint64, nd)
		for j := range dims {
			dims[j], err = rr.u64()
			if err != nil {
				return nil, fmt.Errorf("gguf: tensor %s dim %d: %w", name, j, err)
			}
		}
		typ, err := rr.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %s type: %w", name, err)
		}
		off, err := rr.u64()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %s offset: %w", name, err)
		}
		tensors = append(tensors, TensorInfo{Name: name, Dims: dims, Type: TensorType(typ), Offset: off})
	}

	align, err := alignment(meta)
	if err != nil {
		return nil, err
	}
	data := alignOffset(uint64(rr.n), align)
	if data > uint64(math.MaxInt64) {
		return nil, fmt.Errorf("gguf: tensor data offset overflows int64")
	}
	for i := range tensors {
		if tensors[i].Offset%align != 0 {
			return nil, fmt.Errorf("gguf: tensor %s offset %d is not %d-byte aligned", tensors[i].Name, tensors[i].Offset, align)
		}
		if data+tensors[i].Offset > uint64(math.MaxInt64) {
			return nil, fmt.Errorf("gguf: tensor %s file offset overflows int64", tensors[i].Name)
		}
		tensors[i].FileOffset = int64(data + tensors[i].Offset)
	}

	return &File{
		Version:          ver,
		Metadata:         meta,
		Tensors:          tensors,
		Alignment:        align,
		TensorDataOffset: int64(data),
	}, nil
}

// Config derives a model.Config from the file's metadata, reading the architecture-prefixed
// GGUF keys (embedding_length, block_count, head counts, feed_forward_length, ...) and erroring
// when a required key is missing.
func (f *File) Config() (model.Config, error) {
	arch, ok := f.String("general.architecture")
	if !ok || arch == "" {
		return model.Config{}, fmt.Errorf("gguf: missing general.architecture")
	}
	p := arch + "."
	hidden, err := f.requiredInt(p + "embedding_length")
	if err != nil {
		return model.Config{}, err
	}
	layers, err := f.requiredInt(p + "block_count")
	if err != nil {
		return model.Config{}, err
	}
	heads, err := f.requiredInt(p + "attention.head_count")
	if err != nil {
		return model.Config{}, err
	}
	ffn, err := f.requiredInt(p + "feed_forward_length")
	if err != nil {
		return model.Config{}, err
	}
	headDim := hidden / heads
	if v, ok := f.Uint64(p + "attention.key_length"); ok {
		headDim = int(v)
	}
	kvHeads := heads
	if v, ok := f.Uint64(p + "attention.head_count_kv"); ok {
		kvHeads = int(v)
	}
	rms, err := f.requiredFloat(p + "attention.layer_norm_rms_epsilon")
	if err != nil {
		return model.Config{}, err
	}
	theta := 10000.0
	if v, ok := f.Float64(p + "rope.freq_base"); ok {
		theta = v
	}
	ropeDim := headDim
	if v, ok := f.Uint64(p + "rope.dimension_count"); ok {
		ropeDim = int(v)
	}
	vocab := 0
	if toks, ok := f.StringArray("tokenizer.ggml.tokens"); ok {
		vocab = len(toks)
	}
	eos := -1
	if v, ok := f.Uint64("tokenizer.ggml.eos_token_id"); ok {
		eos = int(v)
	}
	cfg := model.Config{
		HiddenSize:            hidden,
		NumLayers:             layers,
		NumHeads:              heads,
		NumKVHeads:            kvHeads,
		HeadDim:               headDim,
		IntermediateSize:      ffn,
		VocabSize:             vocab,
		RMSNormEps:            rms,
		RopeTheta:             theta,
		TieWordEmbeddings:     !f.hasTensor("output.weight") && !f.hasTensor("lm_head.weight"),
		AttentionBias:         f.hasTensor("blk.0.attn_q.bias") || f.hasTensor("blk.0.attn_k.bias") || f.hasTensor("blk.0.attn_v.bias"),
		ModelType:             arch,
		EOSTokenID:            eos,
		MaxPositionEmbeddings: intValueOrZero(f, p+"context_length"),
		HiddenAct:             "silu",
	}
	if ropeDim > 0 && ropeDim < headDim {
		cfg.PartialRotaryFactor = float64(ropeDim) / float64(headDim)
	}
	if arch == "qwen35" || arch == "qwen35moe" {
		if interval, ok := f.Uint64(p + "full_attention_interval"); ok {
			cfg.FullAttentionInterval = int(interval)
		}
		if conv, ok := f.Uint64(p + "ssm.conv_kernel"); ok {
			cfg.LinearConvKernelDim = int(conv)
		}
		if state, ok := f.Uint64(p + "ssm.state_size"); ok {
			cfg.LinearKeyHeadDim = int(state)
			cfg.LinearValueHeadDim = int(state)
		}
		if groups, ok := f.Uint64(p + "ssm.group_count"); ok {
			cfg.LinearNumKeyHeads = int(groups)
		}
		if rank, ok := f.Uint64(p + "ssm.time_step_rank"); ok {
			cfg.LinearNumValueHeads = int(rank)
		} else if inner, ok := f.Uint64(p + "ssm.inner_size"); ok && cfg.LinearValueHeadDim > 0 {
			cfg.LinearNumValueHeads = int(inner) / cfg.LinearValueHeadDim
		}
		cfg.AttnOutputGate = true
		cfg.NormGain1p = true
		cfg.QKNorm = true
		if cfg.FullAttentionInterval > 0 && len(cfg.LayerTypes) == 0 {
			cfg.LayerTypes = make([]string, cfg.NumLayers)
			for l := range cfg.LayerTypes {
				if (l+1)%cfg.FullAttentionInterval == 0 {
					cfg.LayerTypes[l] = "full_attention"
				} else {
					cfg.LayerTypes[l] = "linear_attention"
				}
			}
		}
	}
	if archIsGemma4(arch) {
		if err := applyGemma4Config(f, p, &cfg); err != nil {
			return model.Config{}, err
		}
	}
	if arch == "glm_moe_dsa" {
		applyGLMMoeDsaConfig(f, p, &cfg, ropeDim)
	}
	return cfg, nil
}

// applyGemma4Config derives Google Gemma 4's architecture axes from GGUF metadata into
// cfg. Gemma 4 is GeGLU + sandwich-norm + sqrt(hidden) embed scale + a final-logit
// soft-cap, atop a HETEROGENEOUS per-layer attention geometry: local (sliding) layers
// and global (full) layers carry different head_dim, kv-head counts, RoPE bases, and
// windows, encoded as GGUF arrays. The norm weights are baked (+1) at convert time and
// consumed with plain RMSNorm, so NormGain1p stays false (the safetensors path, which
// reads raw HF weights, is the one that sets it true).
func applyGemma4Config(f *File, p string, cfg *model.Config) error {
	cfg.ActGeluTanh = true
	cfg.BlockTopology = model.SandwichNorm
	cfg.NormGain1p = false
	if cfg.HiddenSize > 0 {
		cfg.EmbedScale = math.Sqrt(float64(cfg.HiddenSize))
	}
	if v, ok := f.Float64(p + "final_logit_softcapping"); ok {
		cfg.LogitSoftcap = v
	}
	if v, ok := f.Float64(p + "attn_logit_softcapping"); ok {
		cfg.AttnSoftcap = v
	}
	if f.hasTensor("blk.0.attn_q_norm.weight") {
		cfg.QKNorm = true
	}
	// Gemma 4 masks image/audio placeholder tokens (a known checkpoint issue) via a
	// final-logit -inf bias; the ids live in the tokenizer metadata.
	if sup, ok := f.IntArray("tokenizer.ggml.suppress_tokens"); ok {
		cfg.SuppressTokens = sup
	}

	n := cfg.NumLayers
	if n <= 0 {
		return fmt.Errorf("gguf: gemma4 has no layers")
	}
	pattern, ok := f.BoolArray(p + "attention.sliding_window_pattern")
	if !ok || len(pattern) < n {
		return fmt.Errorf("gguf: gemma4 attention.sliding_window_pattern missing or short (have %d, want %d)", len(pattern), n)
	}
	kvArr, ok := f.IntArray(p + "attention.head_count_kv")
	if !ok || len(kvArr) < n {
		return fmt.Errorf("gguf: gemma4 attention.head_count_kv missing or short (have %d, want %d)", len(kvArr), n)
	}
	keyLenFull := intValueOrZero(f, p+"attention.key_length")     // global head_dim
	keyLenSWA := intValueOrZero(f, p+"attention.key_length_swa")  // local head_dim
	ropeDimFull := intValueOrZero(f, p+"rope.dimension_count")    // global rotary width
	ropeDimSWA := intValueOrZero(f, p+"rope.dimension_count_swa") // local rotary width
	swaWindow := intValueOrZero(f, p+"attention.sliding_window")
	thetaFull := cfg.RopeTheta // base read rope.freq_base
	if v, ok := f.Float64(p + "rope.freq_base"); ok {
		thetaFull = v
	}
	thetaSWA := 10000.0
	if v, ok := f.Float64(p + "rope.freq_base_swa"); ok {
		thetaSWA = v
	}
	if keyLenSWA == 0 {
		keyLenSWA = keyLenFull
	}
	if ropeDimFull == 0 {
		ropeDimFull = keyLenFull
	}
	if ropeDimSWA == 0 {
		ropeDimSWA = keyLenSWA
	}

	cfg.LayerTypes = make([]string, n)
	cfg.NumKVHeadsPerLayer = make([]int, n)
	cfg.HeadDimPerLayer = make([]int, n)
	cfg.RopeDimPerLayer = make([]int, n)
	cfg.RopeThetaPerLayer = make([]float64, n)
	cfg.Window = make([]int, n)
	for l := 0; l < n; l++ {
		cfg.NumKVHeadsPerLayer[l] = kvArr[l]
		if pattern[l] { // true == sliding / local
			cfg.LayerTypes[l] = "sliding_attention"
			cfg.HeadDimPerLayer[l] = keyLenSWA
			cfg.RopeDimPerLayer[l] = ropeDimSWA
			cfg.RopeThetaPerLayer[l] = thetaSWA
			cfg.Window[l] = swaWindow
		} else { // false == full / global
			cfg.LayerTypes[l] = "full_attention"
			cfg.HeadDimPerLayer[l] = keyLenFull
			cfg.RopeDimPerLayer[l] = ropeDimFull
			cfg.RopeThetaPerLayer[l] = thetaFull
			cfg.Window[l] = -1
		}
	}

	// Representative scalars: the dedicated gemma4 forward uses the per-layer slices, but
	// keep HeadDim/NumKVHeads/GroupSize sane for any shared code that still reads them.
	cfg.HeadDim = keyLenFull
	if kvArr[0] > 0 {
		cfg.NumKVHeads = kvArr[0]
	}
	cfg.RopeTheta = thetaFull
	return nil
}

// GLM-5.2 (model_type "glm_moe_dsa") GGUF metadata keys.
//
// GLM-5.2's architecture is a Mixture-of-Experts FFN over DeepSeek-style
// Multi-head Latent Attention (MLA) plus a learned Dynamic Sparse Attention
// (DSA) indexer. The MoE and MLA metadata mirror llama.cpp's deepseek2.*
// convention (GLM-DSA attention IS DeepSeek MLA + an indexer), so a real
// converter is most likely to spell them this way; the indexer scalars are
// GLM-5.2-specific and have no upstream llama.cpp analogue yet.
//
// PROVISIONAL: no real GLM-5.2 GGUF exists on disk to pin these against, and
// upstream llama.cpp may not yet ship a glm_moe_dsa converter. The spellings
// are collected here as the single source of truth so the deliberate follow-on
// — a golden against a REAL GLM-5.2 GGUF header — only has to re-pin this one
// block. Every key is read relative to the "<arch>." metadata prefix.
const (
	glmKeyExpertCount        = "expert_count"
	glmKeyExpertUsedCount    = "expert_used_count"
	glmKeyExpertFFNLength    = "expert_feed_forward_length"
	glmKeyExpertSharedCount  = "expert_shared_count"
	glmKeyExpertSharedFFNLen = "expert_shared_feed_forward_length"
	glmKeyLeadingDenseBlocks = "leading_dense_block_count"
	glmKeyExpertGroupCount   = "expert_group_count"
	glmKeyExpertGroupUsed    = "expert_group_used_count"
	glmKeyExpertWeightsScale = "expert_weights_scale"
	glmKeyExpertWeightsNorm  = "expert_weights_norm"

	glmKeyQLoraRank   = "attention.q_lora_rank"
	glmKeyKVLoraRank  = "attention.kv_lora_rank"
	glmKeyQKNopeDim   = "attention.qk_nope_head_dim"
	glmKeyQKRopeDim   = "attention.qk_rope_head_dim"
	glmKeyVHeadDim    = "attention.v_head_dim"
	glmKeyKeyLength   = "attention.key_length"
	glmKeyValueLength = "attention.value_length"

	glmKeyIndexNHeads  = "index_n_heads"
	glmKeyIndexHeadDim = "index_head_dim"
	glmKeyIndexTopK    = "index_topk"
	glmKeyIndexerTypes = "indexer_types"
)

// applyGLMMoeDsaConfig derives GLM-5.2's MoE + MLA + DSA-indexer axes from GGUF
// metadata into the model.Config the generic block already populated. It reads
// every key only if present, so it never overwrites a generic value with zero:
// a MoE GLM-5.2 (expert_count>0) and a dense glm_moe_dsa variant (NumExperts==0,
// the synthetic/pipelinegen form) both load correctly. The result mirrors, field
// for field, the model.Config the JSON/safetensors loader already produces for
// the same model (config_test.go TestConfigDerives...), so cfg.isGLMMoeDsa() and
// cfg.IsMoE() fire and the existing native glm_dsa.go forward consumes it.
//
// ropeDim is the already-resolved rope.dimension_count; it is reused as the
// qk_rope_head_dim fallback under the deepseek2 convention (where the rotary
// portion of each latent head equals the global rope dimension).
//
// Scope (deliberate, per the staged native-753B plan): this is config parsing
// ONLY. The GGUF MoE/MLA/indexer TENSOR-name mapping (CanonicalTensorNameArch)
// and the batched-expert splitter are the next two slices; HeadDim semantics for
// MLA are reconciled when the forward wiring lands.
func applyGLMMoeDsaConfig(f *File, p string, cfg *model.Config, ropeDim int) {
	// ---- MoE FFN axis -------------------------------------------------------
	if v := intValueOrZero(f, p+glmKeyExpertCount); v > 0 {
		cfg.NumExperts = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertUsedCount); v > 0 {
		cfg.NumExpertsPerTok = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertFFNLength); v > 0 {
		cfg.MoEIntermediateSize = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertSharedCount); v > 0 {
		cfg.NSharedExperts = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertSharedFFNLen); v > 0 {
		cfg.SharedIntermediateSize = v
	}
	if v := intValueOrZero(f, p+glmKeyLeadingDenseBlocks); v > 0 {
		cfg.FirstKDenseReplace = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertGroupCount); v > 0 {
		cfg.NGroup = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertGroupUsed); v > 0 {
		cfg.TopKGroup = v
	}
	if v, ok := f.Float64(p + glmKeyExpertWeightsScale); ok {
		cfg.RoutedScalingFactor = v
	}
	if v, ok := f.Bool(p + glmKeyExpertWeightsNorm); ok {
		cfg.NormTopKProb = v
	}

	// ---- MLA (DeepSeek latent attention) axis -------------------------------
	if v := intValueOrZero(f, p+glmKeyQLoraRank); v > 0 {
		cfg.QLoraRank = v
	}
	if v := intValueOrZero(f, p+glmKeyKVLoraRank); v > 0 {
		cfg.KVLoraRank = v
	}
	// qk_rope_head_dim: explicit key, else the resolved rope.dimension_count.
	cfg.QKRopeHeadDim = intValueOrZero(f, p+glmKeyQKRopeDim)
	if cfg.QKRopeHeadDim == 0 {
		cfg.QKRopeHeadDim = ropeDim
	}
	// qk_nope_head_dim: explicit key, else attention.key_length - qk_rope_head_dim
	// (deepseek2 stores n_embd_head_k = nope + rope under attention.key_length).
	cfg.QKNopeHeadDim = intValueOrZero(f, p+glmKeyQKNopeDim)
	if cfg.QKNopeHeadDim == 0 {
		if kl := intValueOrZero(f, p+glmKeyKeyLength); kl > cfg.QKRopeHeadDim {
			cfg.QKNopeHeadDim = kl - cfg.QKRopeHeadDim
		}
	}
	// v_head_dim: explicit key, else attention.value_length.
	cfg.VHeadDim = intValueOrZero(f, p+glmKeyVHeadDim)
	if cfg.VHeadDim == 0 {
		cfg.VHeadDim = intValueOrZero(f, p+glmKeyValueLength)
	}

	// ---- DSA learned-indexer axis (GLM-5.2-specific) ------------------------
	if v := intValueOrZero(f, p+glmKeyIndexNHeads); v > 0 {
		cfg.IndexNHeads = v
	}
	if v := intValueOrZero(f, p+glmKeyIndexHeadDim); v > 0 {
		cfg.IndexHeadDim = v
	}
	if v := intValueOrZero(f, p+glmKeyIndexTopK); v > 0 {
		cfg.IndexTopK = v
	}
	if types, ok := f.StringArray(p + glmKeyIndexerTypes); ok {
		cfg.IndexerTypes = types
	}
}

func intValueOrZero(f *File, key string) int {
	if v, ok := f.Uint64(key); ok && v <= uint64(math.MaxInt) {
		return int(v)
	}
	return 0
}
