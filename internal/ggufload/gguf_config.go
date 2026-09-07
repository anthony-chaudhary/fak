package ggufload

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const ggufHeaderBufferSize = 64 << 10

// Read parses a GGUF header from r — magic, version, the metadata key/value table, and the
// tensor directory — and returns a *File with each tensor's aligned absolute file offset
// resolved. It reads only the header (not the tensor data blob), and errors on a bad magic,
// an unsupported version, or a misaligned tensor offset.
func Read(r io.Reader) (*File, error) {
	// Keep countingReader outside the buffer: rr.n is the exact number of header bytes
	// consumed by the parser, independent of any source bytes prefetched by bufio.
	rr := &countingReader{r: bufio.NewReaderSize(r, ggufHeaderBufferSize)}
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
		ModelType:             canonicalGGUFArch(arch),
		Name:                  stringMetaOr(f.Metadata, "general.name", ""),
		EOSTokenID:            eos,
		MaxPositionEmbeddings: intValueOrZero(f, p+"context_length"),
		HiddenAct:             "silu",
	}
	if ropeDim > 0 && ropeDim < headDim {
		cfg.PartialRotaryFactor = float64(ropeDim) / float64(headDim)
	}
	if canonArch := canonicalGGUFArch(arch); canonArch == "qwen35" || canonArch == "qwen35moe" {
		// llama.cpp's Qwen converter appends the MTP decoder block after the target stack,
		// includes it in block_count, and records the split in nextn_predict_layers. fak's
		// target forward must retain the target depth while the optional MTP materializer
		// addresses the trailing block separately. The native Qwen3.8 substrate currently
		// supports exactly one MTP block; reject incompatible metadata instead of silently
		// treating draft layers as target layers.
		if n, ok := f.Uint64(p + "nextn_predict_layers"); ok {
			if n > uint64(math.MaxInt) {
				return model.Config{}, fmt.Errorf("gguf: %snextn_predict_layers overflows int: %d", p, n)
			}
			cfg.NumNextNPredictLayers = int(n)
			if cfg.NumNextNPredictLayers > 1 {
				return model.Config{}, fmt.Errorf("gguf: %snextn_predict_layers=%d is unsupported; want 0 or 1", p, cfg.NumNextNPredictLayers)
			}
			if cfg.NumNextNPredictLayers >= cfg.NumLayers {
				return model.Config{}, fmt.Errorf("gguf: %snextn_predict_layers=%d must be smaller than block_count=%d", p, cfg.NumNextNPredictLayers, cfg.NumLayers)
			}
			cfg.NumLayers -= cfg.NumNextNPredictLayers
		}
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
	if archUsesMLAMoELayout(canonicalGGUFArch(arch)) {
		if err := applyGLMMoeDsaConfig(f, p, &cfg, ropeDim); err != nil {
			return model.Config{}, err
		}
	}
	if arch == "qwen3moe" {
		applyQwen3MoEConfig(f, p, &cfg)
	}
	if canonicalGGUFArch(arch) == "qwen35moe" {
		applyQwen35MoEConfig(f, p, &cfg)
	}
	return cfg, nil
}

// canonicalGGUFArch normalizes a GGUF general.architecture string to fak's internal
// model_type. The real GLM-5.2 (DSA) GGUF — validated against the on-disk community
// Q4_K_M on the lab GPU server, 2026-06-24 — declares general.architecture = "glm-dsa"
// (llama.cpp's LLM_ARCH_GLM_DSA), while fak's native forward and every downstream
// cfg.ModelType check key on "glm_moe_dsa". Map the GGUF spelling to the internal one so
// family detection (isGLMMoeDsa / IsMoE) and the canonical tensor-name branch resolve.
// The metadata-key PREFIX (p) stays the file's own "glm-dsa." — only ModelType normalizes.
//
// DeepSeek-V2/V3/R1 is passed through as "deepseek2" (llama.cpp's LLM_ARCH_DEEPSEEK2,
// which both V2 and V3 declare): it is first-class and honestly labeled — NOT collapsed to
// glm_moe_dsa — while its MLA+MoE layout reuses the glm forward via archUsesMLAMoELayout /
// Config.usesMLAMoELayout. Sibling community spellings (deepseek-v2/-v3, deepseek3) normalize
// to "deepseek2"; the dense DeepSeek-V1 "deepseek" (no MLA) is deliberately NOT normalized.
//
// prism-ml Ternary-Bonsai-27B (epic #4867) is Qwen3.6-27B with the architecture UNCHANGED —
// the same Gated-DeltaNet/SSM hybrid the qwen35 family drives (arch_support.go). Only its
// weights are re-quantized to ternary (Q2_0). The mainline Qwen3.6-27B GGUF already declares
// general.architecture = "qwen35" (gguf_qwen35_nextn_test.go), but the PrismML repack may
// brand its own arch string; normalize the documented Bonsai/Qwen3.6 spellings onto "qwen35"
// so the hybrid config block (LayerTypes/ssm axes) and the recognized-hybrid forward path
// fire instead of the #934 empty-layer_types refusal. The metadata-key PREFIX stays the file's
// own spelling (only ModelType and the hybrid gate normalize), exactly as for glm-dsa above.
//
// Qwen3.8 checkpoints can declare general.architecture = "qwen3.8", "qwen38", "qwen-3.8",
// or "qwen-38". Qwen3.8 shares the hybrid Gated-DeltaNet / SSM architecture family ("qwen35")
// just like Qwen3.5 and Qwen3.6; normalize these spellings onto "qwen35" so (*File).Config()
// sets ModelType = "qwen35", invokes the hybrid GDN configuration block (LayerTypes and ssm.*
// axes), and downstream model.ClassifyForwardPath reaches ForwardQwen35GDN instead of throwing
// UnsupportedArchError (#934).
func canonicalGGUFArch(arch string) string {
	switch arch {
	case "glm-dsa":
		return "glm_moe_dsa"
	case "deepseek-v2", "deepseek-v3", "deepseek3", "deepseekv2", "deepseekv3":
		return "deepseek2"
	case "kimi-k2", "kimi_k2", "kimi2", "kimi-k3", "kimi_k3", "kimi3", "kimi":
		// Kimi K2/K3 (Moonshot) is a scaled-up DeepSeek-V3: same MLA + DeepSeekMoE
		// backbone, no DSA indexer, so the Moonshot-branded spellings collapse onto
		// "deepseek2" and ride the MLA+MoE forward instead of the #934 refusal.
		return "deepseek2"
	case "bonsai", "ternary-bonsai", "qwen3.6", "qwen36", "qwen3.8", "qwen38", "qwen-3.8", "qwen-38":
		return "qwen35"
	}
	return arch
}

// applyMoEExpertCounts reads the three shared MoE expert-axis GGUF scalars — expert count,
// experts-used-per-token, and expert FFN length — into cfg, writing each only when present and
// positive so a generic value is never clobbered by a zero. Shared by the Qwen3-MoE,
// GLM-MoE-DSA, and Gemma 4 config appliers.
func applyMoEExpertCounts(f *File, p string, cfg *model.Config) {
	if v := intValueOrZero(f, p+glmKeyExpertCount); v > 0 {
		cfg.NumExperts = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertUsedCount); v > 0 {
		cfg.NumExpertsPerTok = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertFFNLength); v > 0 {
		cfg.MoEIntermediateSize = v
	}
}

func applyQwen3MoEConfig(f *File, p string, cfg *model.Config) {
	applyMoEExpertCounts(f, p, cfg)
	if cfg.NumExperts > 0 {
		cfg.NormTopKProb = true
	}
}

func applyQwen35MoEConfig(f *File, p string, cfg *model.Config) {
	applyMoEExpertCounts(f, p, cfg)
	if norm, ok := f.Bool(p + glmKeyExpertWeightsNorm); ok {
		cfg.NormTopKProb = norm
	} else if cfg.NumExperts > 0 {
		// Qwen3.5-MoE renormalizes selected router weights even when its GGUF
		// metadata omits the optional expert_weights_norm declaration.
		cfg.NormTopKProb = true
	}
}

// applyGemma4Config derives Google Gemma 4's architecture axes from GGUF metadata into
// cfg. Gemma 4 is GeGLU + sandwich-norm + sqrt(hidden) embed scale + a final-logit
// soft-cap, atop a HETEROGENEOUS per-layer attention geometry: local (sliding) layers
// and global (full) layers carry different head_dim, kv-head counts, RoPE bases, and
// windows, encoded as GGUF arrays. The norm weights are baked (+1) at convert time and
// consumed with plain RMSNorm, so NormGain1p stays false (the safetensors path, which
// reads raw HF weights, is the one that sets it true).
//
// The family also ships in a sparse Mixture-of-Experts shape (HF text_config
// enable_moe_block / num_experts / top_k_experts / moe_intermediate_size — e.g.
// gemma-4-26B-A4B, 128 experts, top-8), so the shared expert-axis scalars are read here
// exactly as the Qwen3-MoE and GLM-MoE-DSA appliers read them (issue #5494). Without that
// read a genuine MoE gemma4 checkpoint resolved with NumExperts == 0 and was silently
// treated as dense — the estimator sized it without the expert FFN (where most of a
// 26B/A4B model's parameters live) and no expert-aware lever had an axis to dispatch on.
// applyMoEExpertCounts writes each field only when the key is present AND positive, so a
// DENSE gemma4 header (no expert_count) still resolves to NumExperts == 0 unchanged.
func applyGemma4Config(f *File, p string, cfg *model.Config) error {
	cfg.ActGeluTanh = true
	cfg.BlockTopology = model.SandwichNorm
	cfg.NormGain1p = false
	applyMoEExpertCounts(f, p, cfg)
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
// converter spells them this way; the indexer scalars are GLM-5.2-specific.
//
// VALIDATED 2026-06-24 against the community GLM-5.2-Q4_K_M GGUF (general.architecture
// "glm-dsa", llama.cpp LLM_ARCH_GLM_DSA) on the lab GPU server: the MLA/MoE keys match the
// deepseek2 convention as guessed, and the indexer scalars were re-pinned to the real
// attention.indexer.* spellings (they were previously the wrong index_* best-guesses).
// Every key is read relative to the file's own "<arch>." metadata prefix ("glm-dsa.").
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
	// GLM-5.2 (DSA) carries SEPARATE per-head MLA dims under *_mla keys: key_length_mla /
	// value_length_mla are the PER-HEAD k/v widths the MLA forward + kv_b_proj use (256 each
	// for GLM-5.2), distinct from the larger attention.key_length / value_length latent dims.
	// Prefer the _mla keys for QKNopeHeadDim/VHeadDim — using key_length (576) / value_length
	// (512) over-sizes the per-head dims and mis-shapes the KV cache + kv_b split.
	glmKeyKeyLengthMLA   = "attention.key_length_mla"
	glmKeyValueLengthMLA = "attention.value_length_mla"

	// VALIDATED against the real GLM-5.2 (glm-dsa) Q4_K_M GGUF on the lab GPU server,
	// 2026-06-24: the DSA-indexer scalars live under attention.indexer.* (the indexer head
	// dim is its key_length). indexer_types has NO key in the real file — it is read only if
	// present, else the per-layer indexer types are derived.
	glmKeyIndexNHeads  = "attention.indexer.head_count"
	glmKeyIndexHeadDim = "attention.indexer.key_length"
	glmKeyIndexTopK    = "attention.indexer.top_k"
	glmKeyIndexerTypes = "indexer_types"

	// DeepSeek-V4 CSA/HCA two-tier compression rates (metadata only; see
	// model.Config.CSACompressionRate / V4 attention seam map, Missing seam #7).
	// No upstream V4 GGUF conversion exists yet, so these key spellings are fak's
	// own under the existing attention.* namespace, to reconcile if/when an
	// upstream converter defines them. GLM-5.2 ships neither key.
	glmKeyCSACompRate = "attention.csa_compression_rate"
	glmKeyHCACompRate = "attention.hca_compression_rate"
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
// It returns an error only for the one fail-loud case below: a checkpoint that
// declares a DSA indexer AND ships a real tensor directory but leaves the
// per-layer schedule unresolvable. Every other axis is read best-effort.
//
// Scope (deliberate, per the staged native-753B plan): this is config parsing
// ONLY. The GGUF MoE/MLA/indexer TENSOR-name mapping (CanonicalTensorNameArch)
// and the batched-expert splitter are the next two slices; HeadDim semantics for
// MLA are reconciled when the forward wiring lands.
func applyGLMMoeDsaConfig(f *File, p string, cfg *model.Config, ropeDim int) error {
	// ---- MoE FFN axis -------------------------------------------------------
	applyMoEExpertCounts(f, p, cfg)
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
	// qk_nope_head_dim: explicit key, else the per-head MLA key length minus rope. GLM-5.2
	// stores the per-head k dim under attention.key_length_mla (256); attention.key_length (576)
	// is the larger latent dim and must NOT be used here. Fall back to key_length only when the
	// _mla key is absent (older/other deepseek2 conversions that store n_embd_head_k there).
	cfg.QKNopeHeadDim = intValueOrZero(f, p+glmKeyQKNopeDim)
	if cfg.QKNopeHeadDim == 0 {
		kl := intValueOrZero(f, p+glmKeyKeyLengthMLA)
		if kl == 0 {
			kl = intValueOrZero(f, p+glmKeyKeyLength)
		}
		if kl > cfg.QKRopeHeadDim {
			cfg.QKNopeHeadDim = kl - cfg.QKRopeHeadDim
		}
	}
	// v_head_dim: explicit key, else the per-head MLA value length (value_length_mla, 256),
	// else attention.value_length. As with the key dim, the _mla variant is the PER-HEAD width.
	cfg.VHeadDim = intValueOrZero(f, p+glmKeyVHeadDim)
	if cfg.VHeadDim == 0 {
		cfg.VHeadDim = intValueOrZero(f, p+glmKeyValueLengthMLA)
		if cfg.VHeadDim == 0 {
			cfg.VHeadDim = intValueOrZero(f, p+glmKeyValueLength)
		}
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

	// ---- DeepSeek-V4 CSA/HCA compression-rate axis (metadata only) ----------
	// V4's two compression tiers. Read best-effort under the existing attention.*
	// namespace; GLM-5.2 ships neither key, so both stay 0 and nothing changes.
	// This parses config only: the co-resident two-plane forward is unbuilt (V4
	// seam map, Missing seams #1-#3) and a distinct deepseek_v4 arch route +
	// tensor family is Missing seam #8. Zero stays a single-plane load.
	if v := intValueOrZero(f, p+glmKeyCSACompRate); v > 0 {
		cfg.CSACompressionRate = v
	}
	if v := intValueOrZero(f, p+glmKeyHCACompRate); v > 0 {
		cfg.HCACompressionRate = v
	}
	// indexer_types: GLM-5.2's DSA "lightning indexer" runs on only a strided subset of
	// layers. A "full" layer computes its own sparse top-k index and carries the indexer.*
	// tensors; a "shared" layer reuses the most recent full layer's selection and has NO
	// indexer tensors in the file. The real GLM-5.2 GGUF ships no indexer_types metadata
	// key, so when it is absent we DERIVE the per-layer schedule from tensor presence:
	// a layer is "full" iff it carries any of its indexer weights, else "shared". Without
	// this cfg.IndexerTypes stays empty, glmDsaIndexerKind() treats every layer as "full"
	// (its out-of-range default), and the forward panics demanding indexer tensors on the
	// first shared layer (blk.3.indexer.* is absent) at the first completion.
	//
	// glmLayerHasIndexer keys on ANY indexer member (not just attn_q_b), so a partially
	// stripped/corrupt full layer classifies "full" and fails LOUD on the missing tensor
	// in the index step rather than silently mis-scheduling as "shared" (which would wrongly
	// reuse a prior layer's top-k). This is identical to the single-sentinel result for a
	// real GLM-5.2 file, where a full layer ships the whole indexer set atomically.
	if types, ok := f.StringArray(p + glmKeyIndexerTypes); ok {
		cfg.IndexerTypes = types
	} else if cfg.NumLayers > 0 && glmLayerHasIndexer(f, 0) {
		// Only derive for a genuine DSA-indexer checkpoint (layer 0 is always full).
		types := make([]string, cfg.NumLayers)
		for l := 0; l < cfg.NumLayers; l++ {
			if glmLayerHasIndexer(f, l) {
				types[l] = "full"
			} else {
				types[l] = "shared"
			}
		}
		cfg.IndexerTypes = types
	}

	// Fail-loud (native-753B hardening #2): a checkpoint that DECLARES a DSA indexer
	// (indexer.head_count > 0) AND ships a real tensor directory — an actual load, not a
	// header-only config probe — MUST resolve a per-layer full/shared schedule of exactly
	// NumLayers entries, whether from the indexer_types key or the tensor-presence
	// derivation above. If it does not, refuse HERE at config parse instead of letting
	// glmDsaIndexerKind() silently default every layer to "full" (its out-of-range case)
	// and the native DSA forward panic on the first shared layer at the first completion —
	// a failure that would otherwise surface far from its cause, deep in decode.
	//
	// The len(f.Tensors) > 0 guard is load-bearing, not incidental: header-only fixtures
	// (config estimation, the arch-normalization golden) legitimately declare the indexer
	// axis with no tensor directory attached, and must NOT trip this. Only a file carrying
	// real weights can be judged "missing its indexer schedule."
	if len(f.Tensors) > 0 && cfg.IndexNHeads > 0 && len(cfg.IndexerTypes) != cfg.NumLayers {
		return fmt.Errorf("gguf: glm_moe_dsa declares a DSA indexer (%s=%d) but its per-layer schedule is unresolved: len(indexer_types)=%d, want block_count=%d — the %q key is absent and no blk.*.indexer.* tensors were found (broken or incomplete DSA checkpoint)",
			p+glmKeyIndexNHeads, cfg.IndexNHeads, len(cfg.IndexerTypes), cfg.NumLayers, p+glmKeyIndexerTypes)
	}
	return nil
}

func stringMetaOr(meta map[string]Value, key, fallback string) string {
	value, ok := meta[key]
	if !ok {
		return fallback
	}
	text, ok := value.Value.(string)
	if !ok || strings.TrimSpace(text) == "" {
		return fallback
	}
	return text
}

func intValueOrZero(f *File, key string) int {
	if v, ok := f.Uint64(key); ok && v <= uint64(math.MaxInt) {
		return int(v)
	}
	return 0
}
