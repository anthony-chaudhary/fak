package model

import (
	"encoding/json"
	"errors"
	"fmt"
)

const (
	DeepSeekV41FlashModelID  = "deepseek-ai/DeepSeek-V4.1-Flash"
	DeepSeekV41FlashRevision = "dba1be0a40aa45a94ad051997016db3960a90277"
)

var (
	ErrV41ConfigAdmission   = errors.New("model: DeepSeek V4.1 Flash config is not admitted")
	ErrV41NativeUnsupported = errors.New("model: DeepSeek V4.1 Flash native forward is not implemented")
)

// DeepSeekV41QuantConfig retains the checkpoint's mixed-precision declaration.
// It is metadata only; it does not enable an FP4 or FP8 execution path.
type DeepSeekV41QuantConfig struct {
	Method          string `json:"quant_method"`
	Activation      string `json:"activation_scheme"`
	WeightBlockSize []int  `json:"weight_block_size"`
	ScaleFormat     string `json:"scale_fmt"`
	ExpertDtype     string `json:"expert_dtype"`
}

// DeepSeekV41AttentionGeometry retains the exact decoder/attention envelope a
// native V4.1 forward consumes. It is metadata only and enables no execution.
type DeepSeekV41AttentionGeometry struct {
	NumLayers           int `json:"num_hidden_layers"`
	HiddenSize          int `json:"hidden_size"`
	NumHeads            int `json:"num_attention_heads"`
	NumKVHeads          int `json:"num_key_value_heads"`
	HeadDim             int `json:"head_dim"`
	QKRopeHeadDim       int `json:"qk_rope_head_dim"`
	QLoraRank           int `json:"q_lora_rank"`
	OLoraRank           int `json:"o_lora_rank"`
	OGroups             int `json:"o_groups"`
	NumExperts          int `json:"n_routed_experts"`
	NSharedExperts      int `json:"n_shared_experts"`
	NumExpertsPerTok    int `json:"num_experts_per_tok"`
	MoEIntermediateSize int `json:"moe_intermediate_size"`
}

// DeepSeekV41Config holds V4.1-only metadata which must not be folded into the
// older deepseek_v4 profile. The slices are copied from the nested text_config.
type DeepSeekV41Config struct {
	WrapperModelType string
	TextModelType    string
	Quantization     DeepSeekV41QuantConfig

	// Attention retains the decoder/attention execution axes as a single typed
	// envelope. It is metadata only: later native leaves read these instead of
	// reaching into the flat Config.
	Attention DeepSeekV41AttentionGeometry

	KVSourceLayerIDs       []int
	IndexSourceLayerIDs    []int
	CandidateSourceLayerID int
	CandidateTopKBlocks    int
	CandidateBlockSize     int

	CompressRatios    []int
	CompressRopeTheta float64
	HCMult            int
	HCSinkhornIters   int
	HCEps             float64

	EngramLayerIDs            []int
	EngramNumEmbeddings       []int
	EngramMaxNgramSize        int
	EngramVocabSize           int
	EngramNHeads              int
	EngramHeadDim             int
	EngramPadTokenID          int
	EngramCompressedVocabSize int

	// Vision and DSpark retain the official checkpoint's fail-closed markers.
	// They are metadata only: their presence turns a config into a vision or
	// DSpark request that the typed Refuse* hooks reject, and neither admits an
	// execution path. See v41_vision.go / v41_dspark.go.
	Vision *DeepSeekV41VisionConfig
	DSpark *DeepSeekV41DSparkConfig
}

// IsDeepSeekV41 reports a V4.1-family identity: either exact wrapper/text
// identity ("deepseek_v41"/"deepseek_v41_text"), a parsed official config which
// additionally retains and validates the pair in DeepSeekV41, or the GGUF
// loader's canonical identity "deepseek41" (canonicalGGUFArch normalizes every
// V4.1 GGUF spelling onto it). The safetensors "deepseek_v41" identity keeps its
// fail-closed gates (refuseDeepSeekV41Native / admitDeepSeekV41Published); the
// GGUF "deepseek41" identity reaches THIS predicate so a loaded GGUF routes to
// the native non-MLA forwardV41 path instead of the GLM-DSA MLA forward. V4.1
// stays distinct from deepseek_v4 because their forward layouts differ.
func (c Config) IsDeepSeekV41() bool {
	return c.DeepSeekV41 != nil || c.ModelType == "deepseek_v41" ||
		c.ModelType == "deepseek_v41_text" || c.ModelType == "deepseek41"
}

type deepSeekV41TextMetadata struct {
	ModelType                 string  `json:"model_type"`
	KVSourceLayerIDs          []int   `json:"kv_source_layer_ids"`
	IndexSourceLayerIDs       []int   `json:"index_source_layer_ids"`
	CandidateSourceLayerID    int     `json:"candidate_source_layer_id"`
	CandidateTopKBlocks       int     `json:"candidate_topk_blocks"`
	CandidateBlockSize        int     `json:"candidate_block_size"`
	CompressRatios            []int   `json:"compress_ratios"`
	CompressRopeTheta         float64 `json:"compress_rope_theta"`
	HCMult                    int     `json:"hc_mult"`
	HCSinkhornIters           int     `json:"hc_sinkhorn_iters"`
	HCEps                     float64 `json:"hc_eps"`
	EngramLayerIDs            []int   `json:"engram_layer_ids"`
	EngramNumEmbeddings       []int   `json:"engram_num_embeddings"`
	EngramMaxNgramSize        int     `json:"engram_max_ngram_size"`
	EngramVocabSize           int     `json:"engram_vocab_size"`
	EngramNHeads              int     `json:"engram_n_heads"`
	EngramHeadDim             int     `json:"engram_head_dim"`
	EngramPadTokenID          int     `json:"engram_pad_token_id"`
	EngramCompressedVocabSize int     `json:"engram_compressed_vocab_size"`
	NumNextNPredictLayers     int     `json:"num_nextn_predict_layers"`
	DSparkBlockSize           int     `json:"dspark_block_size"`
	DSparkNoiseTokenID        int     `json:"dspark_noise_token_id"`
	DSparkTargetLayerIDs      []int   `json:"dspark_target_layer_ids"`
	DSparkMarkovRank          int     `json:"dspark_markov_rank"`
	DSparkNumRoutedExperts    int     `json:"dspark_n_routed_experts"`
	DSparkNumExpertsPerTok    int     `json:"dspark_num_experts_per_tok"`

	NumHiddenLayers     int `json:"num_hidden_layers"`
	HiddenSize          int `json:"hidden_size"`
	NumAttentionHeads   int `json:"num_attention_heads"`
	NumKeyValueHeads    int `json:"num_key_value_heads"`
	HeadDim             int `json:"head_dim"`
	QKRopeHeadDim       int `json:"qk_rope_head_dim"`
	QLoraRank           int `json:"q_lora_rank"`
	OLoraRank           int `json:"o_lora_rank"`
	OGroups             int `json:"o_groups"`
	NRoutedExperts      int `json:"n_routed_experts"`
	NSharedExperts      int `json:"n_shared_experts"`
	NumExpertsPerTok    int `json:"num_experts_per_tok"`
	MoEIntermediateSize int `json:"moe_intermediate_size"`
}

// MarshalJSON reconstructs the V4.1 wrapper from current Config fields. Other
// families keep the existing flat Config encoding.
func (c Config) MarshalJSON() ([]byte, error) {
	if c.DeepSeekV41 == nil {
		return json.Marshal(configAlias(c))
	}
	m := c.DeepSeekV41
	root := c
	root.ModelType = m.WrapperModelType
	root.DeepSeekV41 = nil
	text := c
	text.ModelType = m.TextModelType
	text.DeepSeekV41 = nil
	type textEnvelope struct {
		configAlias
		KVSourceLayerIDs          []int   `json:"kv_source_layer_ids"`
		IndexSourceLayerIDs       []int   `json:"index_source_layer_ids"`
		CandidateSourceLayerID    int     `json:"candidate_source_layer_id"`
		CandidateTopKBlocks       int     `json:"candidate_topk_blocks"`
		CandidateBlockSize        int     `json:"candidate_block_size"`
		CompressRopeTheta         float64 `json:"compress_rope_theta"`
		EngramLayerIDs            []int   `json:"engram_layer_ids"`
		EngramNumEmbeddings       []int   `json:"engram_num_embeddings"`
		EngramMaxNgramSize        int     `json:"engram_max_ngram_size"`
		EngramVocabSize           int     `json:"engram_vocab_size"`
		EngramNHeads              int     `json:"engram_n_heads"`
		EngramHeadDim             int     `json:"engram_head_dim"`
		EngramPadTokenID          int     `json:"engram_pad_token_id"`
		EngramCompressedVocabSize int     `json:"engram_compressed_vocab_size"`
		NumNextNPredictLayers     int     `json:"num_nextn_predict_layers"`
		DSparkBlockSize           int     `json:"dspark_block_size"`
		DSparkNoiseTokenID        int     `json:"dspark_noise_token_id"`
		DSparkTargetLayerIDs      []int   `json:"dspark_target_layer_ids"`
		DSparkMarkovRank          int     `json:"dspark_markov_rank"`
		DSparkNumRoutedExperts    int     `json:"dspark_n_routed_experts"`
		DSparkNumExpertsPerTok    int     `json:"dspark_num_experts_per_tok"`

		NumHiddenLayers     int `json:"num_hidden_layers"`
		HiddenSize          int `json:"hidden_size"`
		NumAttentionHeads   int `json:"num_attention_heads"`
		NumKeyValueHeads    int `json:"num_key_value_heads"`
		HeadDim             int `json:"head_dim"`
		QKRopeHeadDim       int `json:"qk_rope_head_dim"`
		QLoraRank           int `json:"q_lora_rank"`
		OLoraRank           int `json:"o_lora_rank"`
		OGroups             int `json:"o_groups"`
		NRoutedExperts      int `json:"n_routed_experts"`
		NSharedExperts      int `json:"n_shared_experts"`
		NumExpertsPerTok    int `json:"num_experts_per_tok"`
		MoEIntermediateSize int `json:"moe_intermediate_size"`
	}
	// The attention axes are derived from the flat Config (parseDeepSeekV41Metadata
	// rebuilds them from c), so emit them from the same source of truth. Reading
	// m.Attention here would let a hand-built Config emit a document whose nested
	// text_config disagrees with its top-level geometry; reparse would then silently
	// overwrite the caller's Attention. Emitting from text keeps marshal/unmarshal
	// symmetric and the axes genuinely round-trippable.
	nested := textEnvelope{
		configAlias:      configAlias(text),
		KVSourceLayerIDs: m.KVSourceLayerIDs, IndexSourceLayerIDs: m.IndexSourceLayerIDs,
		CandidateSourceLayerID: m.CandidateSourceLayerID, CandidateTopKBlocks: m.CandidateTopKBlocks, CandidateBlockSize: m.CandidateBlockSize,
		CompressRopeTheta: m.CompressRopeTheta,
		EngramLayerIDs:    m.EngramLayerIDs, EngramNumEmbeddings: m.EngramNumEmbeddings,
		EngramMaxNgramSize: m.EngramMaxNgramSize, EngramVocabSize: m.EngramVocabSize,
		EngramNHeads: m.EngramNHeads, EngramHeadDim: m.EngramHeadDim,
		EngramPadTokenID: m.EngramPadTokenID, EngramCompressedVocabSize: m.EngramCompressedVocabSize,
		NumHiddenLayers: text.NumLayers, HiddenSize: text.HiddenSize,
		NumAttentionHeads: text.NumHeads, NumKeyValueHeads: text.NumKVHeads,
		HeadDim: text.HeadDim, QKRopeHeadDim: text.QKRopeHeadDim,
		QLoraRank: text.QLoraRank, OLoraRank: text.OLoraRank, OGroups: text.OGroups,
		NRoutedExperts: text.NumExperts, NSharedExperts: text.NSharedExperts,
		NumExpertsPerTok: text.NumExpertsPerTok, MoEIntermediateSize: text.MoEIntermediateSize,
	}
	if d := m.DSpark; d != nil {
		nested.NumNextNPredictLayers = d.NextNPredictLayers
		nested.DSparkBlockSize = d.BlockSize
		nested.DSparkNoiseTokenID = d.NoiseTokenID
		nested.DSparkTargetLayerIDs = d.TargetLayerIDs
		nested.DSparkMarkovRank = d.MarkovRank
		nested.DSparkNumRoutedExperts = d.NumRoutedExperts
		nested.DSparkNumExpertsPerTok = d.NumExpertsPerTok
	}
	type visionEnvelope struct {
		ModelType         string `json:"model_type"`
		NumHiddenLayers   int    `json:"num_hidden_layers"`
		HiddenSize        int    `json:"hidden_size"`
		NumAttentionHeads int    `json:"num_attention_heads"`
		PatchSize         int    `json:"patch_size"`
		MaxImageTokens    int    `json:"max_image_tokens"`
	}
	type wrapperEnvelope struct {
		configAlias
		Quantization DeepSeekV41QuantConfig `json:"quantization_config"`
		TextConfig   textEnvelope           `json:"text_config"`
		VisionConfig *visionEnvelope        `json:"vision_config,omitempty"`
	}
	var vision *visionEnvelope
	if v := m.Vision; v != nil {
		vision = &visionEnvelope{
			ModelType:         v.ModelType,
			NumHiddenLayers:   v.NumLayers,
			HiddenSize:        v.HiddenSize,
			NumAttentionHeads: v.NumHeads,
			PatchSize:         v.PatchSize,
			MaxImageTokens:    v.MaxImageTokens,
		}
	}
	return json.Marshal(wrapperEnvelope{configAlias(root), m.Quantization, nested, vision})
}

func parseDeepSeekV41Metadata(root, text []byte, c Config) (*DeepSeekV41Config, error) {
	var envelope struct {
		ModelType    string                 `json:"model_type"`
		Quantization DeepSeekV41QuantConfig `json:"quantization_config"`
		VisionConfig *struct {
			ModelType      string `json:"model_type"`
			NumHiddenLayer int    `json:"num_hidden_layers"`
			HiddenSize     int    `json:"hidden_size"`
			NumHeads       int    `json:"num_attention_heads"`
			PatchSize      int    `json:"patch_size"`
			MaxImageTokens int    `json:"max_image_tokens"`
		} `json:"vision_config"`
	}
	if err := json.Unmarshal(root, &envelope); err != nil {
		return nil, err
	}
	var nested deepSeekV41TextMetadata
	if err := json.Unmarshal(text, &nested); err != nil {
		return nil, err
	}
	if envelope.ModelType != "deepseek_v41" && nested.ModelType != "deepseek_v41_text" {
		return nil, nil
	}
	m := &DeepSeekV41Config{
		WrapperModelType:          envelope.ModelType,
		TextModelType:             nested.ModelType,
		Quantization:              envelope.Quantization,
		KVSourceLayerIDs:          append([]int(nil), nested.KVSourceLayerIDs...),
		IndexSourceLayerIDs:       append([]int(nil), nested.IndexSourceLayerIDs...),
		CandidateSourceLayerID:    nested.CandidateSourceLayerID,
		CandidateTopKBlocks:       nested.CandidateTopKBlocks,
		CandidateBlockSize:        nested.CandidateBlockSize,
		CompressRatios:            append([]int(nil), nested.CompressRatios...),
		CompressRopeTheta:         nested.CompressRopeTheta,
		HCMult:                    nested.HCMult,
		HCSinkhornIters:           nested.HCSinkhornIters,
		HCEps:                     nested.HCEps,
		EngramLayerIDs:            append([]int(nil), nested.EngramLayerIDs...),
		EngramNumEmbeddings:       append([]int(nil), nested.EngramNumEmbeddings...),
		EngramMaxNgramSize:        nested.EngramMaxNgramSize,
		EngramVocabSize:           nested.EngramVocabSize,
		EngramNHeads:              nested.EngramNHeads,
		EngramHeadDim:             nested.EngramHeadDim,
		EngramPadTokenID:          nested.EngramPadTokenID,
		EngramCompressedVocabSize: nested.EngramCompressedVocabSize,
		Attention: DeepSeekV41AttentionGeometry{
			NumLayers:           c.NumLayers,
			HiddenSize:          c.HiddenSize,
			NumHeads:            c.NumHeads,
			NumKVHeads:          c.NumKVHeads,
			HeadDim:             c.HeadDim,
			QKRopeHeadDim:       c.QKRopeHeadDim,
			QLoraRank:           c.QLoraRank,
			OLoraRank:           c.OLoraRank,
			OGroups:             c.OGroups,
			NumExperts:          c.NumExperts,
			NSharedExperts:      c.NSharedExperts,
			NumExpertsPerTok:    c.NumExpertsPerTok,
			MoEIntermediateSize: c.MoEIntermediateSize,
		},
	}
	if v := envelope.VisionConfig; v != nil {
		m.Vision = &DeepSeekV41VisionConfig{
			ModelType:      v.ModelType,
			HiddenSize:     v.HiddenSize,
			NumLayers:      v.NumHiddenLayer,
			NumHeads:       v.NumHeads,
			PatchSize:      v.PatchSize,
			MaxImageTokens: v.MaxImageTokens,
		}
	}
	if nested.NumNextNPredictLayers != 0 || nested.DSparkBlockSize != 0 || nested.DSparkNoiseTokenID != 0 ||
		len(nested.DSparkTargetLayerIDs) > 0 || nested.DSparkMarkovRank != 0 ||
		nested.DSparkNumRoutedExperts != 0 || nested.DSparkNumExpertsPerTok != 0 {
		m.DSpark = &DeepSeekV41DSparkConfig{
			NextNPredictLayers: nested.NumNextNPredictLayers,
			BlockSize:          nested.DSparkBlockSize,
			NoiseTokenID:       nested.DSparkNoiseTokenID,
			TargetLayerIDs:     append([]int(nil), nested.DSparkTargetLayerIDs...),
			MarkovRank:         nested.DSparkMarkovRank,
			NumRoutedExperts:   nested.DSparkNumRoutedExperts,
			NumExpertsPerTok:   nested.DSparkNumExpertsPerTok,
		}
	}
	if err := admitDeepSeekV41Published(c, m); err != nil {
		return nil, err
	}
	return m, nil
}

func admitDeepSeekV41Published(c Config, m *DeepSeekV41Config) error {
	wantCompression := []int{
		0, 0,
		2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0,
	}
	checks := []struct {
		name string
		ok   bool
		got  any
	}{
		{"identity", m.WrapperModelType == "deepseek_v41" && m.TextModelType == "deepseek_v41_text", m.WrapperModelType + "/" + m.TextModelType},
		{"decoder geometry", c.NumLayers == 40 && c.HiddenSize == 5120 && c.NumHeads == 64 && c.NumKVHeads == 1 && c.HeadDim == 512, fmt.Sprintf("layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d", c.NumLayers, c.HiddenSize, c.NumHeads, c.NumKVHeads, c.HeadDim)},
		{"attention geometry", c.QKRopeHeadDim == 64 && c.QLoraRank == 1280 && c.OLoraRank == 1024 && c.OGroups == 8, fmt.Sprintf("rope=%d qrank=%d orank=%d groups=%d", c.QKRopeHeadDim, c.QLoraRank, c.OLoraRank, c.OGroups)},
		{"moe geometry", c.NumExperts == 384 && c.NSharedExperts == 1 && c.NumExpertsPerTok == 6 && c.MoEIntermediateSize == 2304, fmt.Sprintf("experts=%d shared=%d topk=%d width=%d", c.NumExperts, c.NSharedExperts, c.NumExpertsPerTok, c.MoEIntermediateSize)},
		{"compression schedule", len(m.CompressRatios) == c.NumLayers+c.NumNextNPredictLayers && v41EqualInts(m.CompressRatios, wantCompression) && m.CompressRopeTheta == 160000, fmt.Sprintf("ratios=%v theta=%g", m.CompressRatios, m.CompressRopeTheta)},
		{"kv sources", v41EqualInts(m.KVSourceLayerIDs, []int{2, 8, 14, 20}), m.KVSourceLayerIDs},
		{"index sources", v41EqualInts(m.IndexSourceLayerIDs, []int{2, 8, 14, 20, 24, 28, 32, 36}), m.IndexSourceLayerIDs},
		{"candidate geometry", m.CandidateSourceLayerID == 20 && m.CandidateTopKBlocks == 2048 && m.CandidateBlockSize == 8, fmt.Sprintf("source=%d topk=%d block=%d", m.CandidateSourceLayerID, m.CandidateTopKBlocks, m.CandidateBlockSize)},
		{"hyperconnection", m.HCMult == 4 && m.HCSinkhornIters == 20 && m.HCEps == 1e-6, fmt.Sprintf("mult=%d iters=%d eps=%g", m.HCMult, m.HCSinkhornIters, m.HCEps)},
		{"engram layers", v41EqualInts(m.EngramLayerIDs, []int{1, 14}) && v41EqualInts(m.EngramNumEmbeddings, []int{384006168, 384016682}), fmt.Sprintf("layers=%v embeddings=%v", m.EngramLayerIDs, m.EngramNumEmbeddings)},
		{"engram geometry", m.EngramMaxNgramSize == 4 && m.EngramVocabSize == 16000000 && m.EngramNHeads == 8 && m.EngramHeadDim == 256 && m.EngramPadTokenID == 2 && m.EngramCompressedVocabSize == 99092, fmt.Sprintf("ngram=%d vocab=%d heads=%d dim=%d pad=%d compressed=%d", m.EngramMaxNgramSize, m.EngramVocabSize, m.EngramNHeads, m.EngramHeadDim, m.EngramPadTokenID, m.EngramCompressedVocabSize)},
		{"quantization", m.Quantization.Method == "fp8" && m.Quantization.Activation == "dynamic" && v41EqualInts(m.Quantization.WeightBlockSize, []int{32, 32}) && m.Quantization.ScaleFormat == "ue8m0" && m.Quantization.ExpertDtype == "fp4", fmt.Sprintf("method=%s activation=%s block=%v scale=%s expert=%s", m.Quantization.Method, m.Quantization.Activation, m.Quantization.WeightBlockSize, m.Quantization.ScaleFormat, m.Quantization.ExpertDtype)},
		{"decoder/attention axes retained", m.Attention == (DeepSeekV41AttentionGeometry{NumLayers: 40, HiddenSize: 5120, NumHeads: 64, NumKVHeads: 1, HeadDim: 512, QKRopeHeadDim: 64, QLoraRank: 1280, OLoraRank: 1024, OGroups: 8, NumExperts: 384, NSharedExperts: 1, NumExpertsPerTok: 6, MoEIntermediateSize: 2304}), fmt.Sprintf("layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d rope=%d qrank=%d orank=%d groups=%d experts=%d shared=%d topk=%d width=%d", m.Attention.NumLayers, m.Attention.HiddenSize, m.Attention.NumHeads, m.Attention.NumKVHeads, m.Attention.HeadDim, m.Attention.QKRopeHeadDim, m.Attention.QLoraRank, m.Attention.OLoraRank, m.Attention.OGroups, m.Attention.NumExperts, m.Attention.NSharedExperts, m.Attention.NumExpertsPerTok, m.Attention.MoEIntermediateSize)},
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("%w: %s=%v", ErrV41ConfigAdmission, check.name, check.got)
		}
	}
	return nil
}

func v41EqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// refuseDeepSeekV41Native is the safetensors "deepseek_v41" load gate: it refuses a
// config that claims the published V4.1 identity without a parsed/validated envelope,
// or whose native execution is still pending. It deliberately does NOT refuse the GGUF
// loader's canonical "deepseek41" identity: that identity has no published envelope and
// routes to the native non-MLA forwardV41 path (IsDeepSeekV41 -> stepV41/prefillV41),
// whose own stage-aware admission ladder (v41ForwardAdmitted) is the fail-closed gate.
// Refusing "deepseek41" here would block the exact staged vcruz GGUF at newModel /
// NewFromF32Tensors before the forward could ever be reached.
func refuseDeepSeekV41Native(c Config) error {
	if c.ModelType == "deepseek41" {
		return nil
	}
	if !c.IsDeepSeekV41() {
		return nil
	}
	return fmt.Errorf("%w: %s@%s metadata is admitted; Engram/shared-KV native execution remains pending", ErrV41NativeUnsupported, DeepSeekV41FlashModelID, DeepSeekV41FlashRevision)
}
