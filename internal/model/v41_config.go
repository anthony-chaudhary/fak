package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	DeepSeekV41FlashModelID  = "deepseek-ai/DeepSeek-V4.1-Flash"
	DeepSeekV41FlashRevision = "dba1be0a40aa45a94ad051997016db3960a90277"
)

var (
	ErrV41ConfigAdmission   = errors.New("model: DeepSeek V4.1 Flash config is not admitted")
	ErrV41NativeUnsupported = errors.New("model: DeepSeek V4.1 Flash native forward is not implemented")
	ErrV41NumericalProfile  = errors.New("model: DeepSeek V4.1 numerical profile is not admitted")
)

// DeepSeekV41NumericalProfileRequest separates the requested target release,
// converted storage and reference-code semantics. It is not an artifact receipt
// or numerical qualification. No explicit profile is executable yet; retaining
// a request prevents it from silently selecting the legacy geometry-based graph.
// The schema version describes this request envelope, not supported arithmetic.
type DeepSeekV41NumericalProfileRequest struct {
	SchemaVersion       int    `json:"schema_version"`
	TargetRepository    string `json:"target_repository"`
	TargetRevision      string `json:"target_revision"`
	StorageRepository   string `json:"storage_repository"`
	StorageRevision     string `json:"storage_revision"`
	StorageFormat       string `json:"storage_format"`
	ReferenceRepository string `json:"reference_repository"`
	ReferenceRevision   string `json:"reference_revision"`
	Semantics           string `json:"semantics"`
}

func v41ValidateNumericalRequest(p *DeepSeekV41NumericalProfileRequest) error {
	if p == nil || p.SchemaVersion != 1 {
		return fmt.Errorf("%w: explicit request requires schema_version 1", ErrV41NumericalProfile)
	}
	for _, field := range []struct{ name, value string }{
		{"target_repository", p.TargetRepository}, {"target_revision", p.TargetRevision},
		{"storage_repository", p.StorageRepository}, {"storage_revision", p.StorageRevision},
		{"storage_format", p.StorageFormat}, {"reference_repository", p.ReferenceRepository},
		{"reference_revision", p.ReferenceRevision}, {"semantics", p.Semantics},
	} {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("%w: explicit request lacks %s", ErrV41NumericalProfile, field.name)
		}
	}
	return nil
}

func v41ParseNumericalRequest(raw json.RawMessage) (*DeepSeekV41NumericalProfileRequest, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var p *DeepSeekV41NumericalProfileRequest
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&p); err != nil {
		return nil, fmt.Errorf("%w: malformed explicit request: %v", ErrV41NumericalProfile, err)
	}
	if err := v41ValidateNumericalRequest(p); err != nil {
		return nil, err
	}
	return p, nil
}

// Only the root may declare the reserved request key. Walk tokens rather than
// decoded maps so a duplicate parent key cannot hide an earlier misplaced
// request. UseNumber preserves unrelated unknown numeric metadata; this check
// neither interprets nor reserializes legacy config values.
func v41CheckNumericalRequestPlacement(raw []byte) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var visit func(root bool) error
	visit = func(root bool) error {
		token, err := d.Token()
		if err != nil {
			return err
		}
		switch token {
		case json.Delim('{'):
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return err
				}
				name, _ := key.(string)
				if !root && strings.EqualFold(name, "fak_v41_numerical_profile") {
					return fmt.Errorf("%w: numerical profile request must be at the config root", ErrV41NumericalProfile)
				}
				if err := visit(false); err != nil {
					return err
				}
			}
		case json.Delim('['):
			for d.More() {
				if err := visit(false); err != nil {
					return err
				}
			}
		default:
			return nil
		}
		_, err = d.Token() // The enclosing json.Unmarshal already validated syntax.
		return err
	}
	return visit(true)
}

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

	// Nil preserves legacy, unqualified arithmetic. An explicit request is
	// serialized at the root as fak_v41_numerical_profile and fails closed
	// before numerical dispatch until that exact contract is qualified.
	NumericalProfile *DeepSeekV41NumericalProfileRequest

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

// MarshalJSON retains canonical GGUF identity and effective V4.1 metadata, or
// reconstructs the official V4.1 wrapper. Other families keep the flat encoding.
func (c Config) MarshalJSON() ([]byte, error) {
	if c.DeepSeekV41 == nil {
		return json.Marshal(configAlias(c))
	}
	m := c.DeepSeekV41
	if m.NumericalProfile != nil {
		if err := v41ValidateNumericalRequest(m.NumericalProfile); err != nil {
			return nil, err
		}
	}
	gguf := c.ModelType == "deepseek41"
	if gguf && (m.WrapperModelType != "" || m.TextModelType != "") {
		return nil, fmt.Errorf("%w: GGUF config carries official wrapper/text identity", ErrV41ConfigAdmission)
	}
	root := c
	root.ModelType = m.WrapperModelType
	root.DeepSeekV41 = nil
	text := c
	text.ModelType = m.TextModelType
	text.DeepSeekV41 = nil
	if gguf {
		text.ModelType = c.ModelType
	}
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
		NumericalProfile *DeepSeekV41NumericalProfileRequest `json:"fak_v41_numerical_profile,omitempty"`
		Quantization     DeepSeekV41QuantConfig              `json:"quantization_config"`
		TextConfig       textEnvelope                        `json:"text_config"`
		VisionConfig     *visionEnvelope                     `json:"vision_config,omitempty"`
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
	if gguf {
		// These are effective execution axes, not an official checkpoint claim.
		// Keep the raw quantization declaration embedded in Config unchanged.
		if m.DSpark == nil {
			nested.NumNextNPredictLayers = c.NumNextNPredictLayers
		}
		return json.Marshal(struct {
			textEnvelope
			NumericalProfile *DeepSeekV41NumericalProfileRequest `json:"fak_v41_numerical_profile,omitempty"`
			VisionConfig     *visionEnvelope                     `json:"vision_config,omitempty"`
		}{nested, m.NumericalProfile, vision})
	}
	return json.Marshal(wrapperEnvelope{configAlias(root), m.NumericalProfile, m.Quantization, nested, vision})
}

func parseDeepSeekV41Metadata(root, text []byte, c Config) (*DeepSeekV41Config, error) {
	var envelope struct {
		NumericalProfile json.RawMessage `json:"fak_v41_numerical_profile"`
		TextConfig       json.RawMessage `json:"text_config"`
		ModelType        string          `json:"model_type"`
		Quantization     json.RawMessage `json:"quantization_config"`
		VisionConfig     *struct {
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
	if err := v41CheckNumericalRequestPlacement(root); err != nil {
		return nil, err
	}
	var nested deepSeekV41TextMetadata
	if err := json.Unmarshal(text, &nested); err != nil {
		return nil, err
	}
	profile, err := v41ParseNumericalRequest(envelope.NumericalProfile)
	if err != nil {
		return nil, err
	}
	gguf := envelope.ModelType == "deepseek41"
	if gguf {
		if len(envelope.TextConfig) != 0 {
			return nil, fmt.Errorf("%w: canonical GGUF JSON must be flat", ErrV41ConfigAdmission)
		}
		// A legacy flat config without effective topology keeps its nil envelope.
		// Once any topology axis is present, require the full serialized set so
		// an omitted candidate source cannot silently become layer zero.
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(root, &fields); err != nil {
			return nil, err
		}
		keys := []string{"kv_source_layer_ids", "index_source_layer_ids", "candidate_source_layer_id", "candidate_topk_blocks", "candidate_block_size"}
		present := false
		for _, key := range keys {
			present = present || len(fields[key]) > 0
		}
		if !present {
			if profile != nil {
				return nil, fmt.Errorf("%w: explicit request requires retained GGUF topology", ErrV41NumericalProfile)
			}
			return nil, nil
		}
		for i, key := range keys {
			if len(fields[key]) == 0 {
				return nil, fmt.Errorf("%w: GGUF effective topology lacks %s", ErrV41ConfigAdmission, key)
			}
			if i >= 2 {
				var value *int
				if err := json.Unmarshal(fields[key], &value); err != nil || value == nil {
					return nil, fmt.Errorf("%w: GGUF effective topology requires integer %s", ErrV41ConfigAdmission, key)
				}
			}
		}
	} else if envelope.ModelType != "deepseek_v41" && nested.ModelType != "deepseek_v41_text" {
		if profile != nil {
			return nil, fmt.Errorf("%w: explicit request requires a V4.1 config", ErrV41NumericalProfile)
		}
		return nil, nil
	}
	m := &DeepSeekV41Config{
		NumericalProfile:          profile,
		WrapperModelType:          envelope.ModelType,
		TextModelType:             nested.ModelType,
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
	if gguf {
		// Cross-check against the GGUF constructor without replacing the decoded
		// values. In particular, a published schedule cannot acquire different
		// sources or candidate geometry merely by passing through JSON.
		want, err := NewDeepSeekV41GGUFConfig(c, V41GGUFTopology{
			KVSourceLayerIDs: m.KVSourceLayerIDs, IndexSourceLayerIDs: m.IndexSourceLayerIDs,
		})
		if err != nil {
			return nil, err
		}
		if !v41EqualInts(m.KVSourceLayerIDs, want.KVSourceLayerIDs) ||
			!v41EqualInts(m.IndexSourceLayerIDs, want.IndexSourceLayerIDs) ||
			m.CandidateSourceLayerID != want.CandidateSourceLayerID ||
			m.CandidateTopKBlocks != want.CandidateTopKBlocks || m.CandidateBlockSize != want.CandidateBlockSize {
			return nil, fmt.Errorf("%w: GGUF effective topology disagrees with its schedule", ErrV41ConfigAdmission)
		}
		m.WrapperModelType, m.TextModelType = "", ""
		return m, nil
	}
	if len(envelope.Quantization) != 0 {
		if err := json.Unmarshal(envelope.Quantization, &m.Quantization); err != nil {
			return nil, err
		}
	}
	if err := admitDeepSeekV41Published(c, m); err != nil {
		return nil, err
	}
	return m, nil
}

func admitDeepSeekV41Published(c Config, m *DeepSeekV41Config) error {
	checks := []struct {
		name string
		ok   bool
		got  any
	}{
		{"identity", m.WrapperModelType == "deepseek_v41" && m.TextModelType == "deepseek_v41_text", m.WrapperModelType + "/" + m.TextModelType},
		{"decoder geometry", c.NumLayers == 40 && c.HiddenSize == 5120 && c.NumHeads == 64 && c.NumKVHeads == 1 && c.HeadDim == 512, fmt.Sprintf("layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d", c.NumLayers, c.HiddenSize, c.NumHeads, c.NumKVHeads, c.HeadDim)},
		{"attention geometry", c.QKRopeHeadDim == 64 && c.QLoraRank == 1280 && c.OLoraRank == 1024 && c.OGroups == 8, fmt.Sprintf("rope=%d qrank=%d orank=%d groups=%d", c.QKRopeHeadDim, c.QLoraRank, c.OLoraRank, c.OGroups)},
		{"moe geometry", c.NumExperts == 384 && c.NSharedExperts == 1 && c.NumExpertsPerTok == 6 && c.MoEIntermediateSize == 2304, fmt.Sprintf("experts=%d shared=%d topk=%d width=%d", c.NumExperts, c.NSharedExperts, c.NumExpertsPerTok, c.MoEIntermediateSize)},
		{"compression schedule", len(m.CompressRatios) == c.NumLayers+c.NumNextNPredictLayers && v41EqualInts(m.CompressRatios, v41PublishedCompressRatios) && m.CompressRopeTheta == v41PublishedCompressRopeTheta, fmt.Sprintf("ratios=%v theta=%g", m.CompressRatios, m.CompressRopeTheta)},
		{"kv sources", v41EqualInts(m.KVSourceLayerIDs, v41PublishedKVSourceLayerIDs), m.KVSourceLayerIDs},
		{"index sources", v41EqualInts(m.IndexSourceLayerIDs, v41PublishedIndexSourceLayerIDs), m.IndexSourceLayerIDs},
		{"candidate geometry", m.CandidateSourceLayerID == v41PublishedCandidateSourceLayerID && m.CandidateTopKBlocks == v41PublishedCandidateTopKBlocks && m.CandidateBlockSize == v41PublishedCandidateBlockSize, fmt.Sprintf("source=%d topk=%d block=%d", m.CandidateSourceLayerID, m.CandidateTopKBlocks, m.CandidateBlockSize)},
		{"hyperconnection", m.HCMult == 4 && m.HCSinkhornIters == 20 && m.HCEps == 1e-6, fmt.Sprintf("mult=%d iters=%d eps=%g", m.HCMult, m.HCSinkhornIters, m.HCEps)},
		{"engram layers", v41EqualInts(m.EngramLayerIDs, []int{1, 14}) && v41EqualInts(m.EngramNumEmbeddings, []int{384006168, 384016682}), fmt.Sprintf("layers=%v embeddings=%v", m.EngramLayerIDs, m.EngramNumEmbeddings)},
		{"engram geometry", m.EngramMaxNgramSize == 4 && m.EngramVocabSize == 16000000 && m.EngramNHeads == 8 && m.EngramHeadDim == 256 && m.EngramPadTokenID == 2 && m.EngramCompressedVocabSize == 99092, fmt.Sprintf("ngram=%d vocab=%d heads=%d dim=%d pad=%d compressed=%d", m.EngramMaxNgramSize, m.EngramVocabSize, m.EngramNHeads, m.EngramHeadDim, m.EngramPadTokenID, m.EngramCompressedVocabSize)},
		{"quantization", m.Quantization.Method == "fp8" && m.Quantization.Activation == "dynamic" && v41EqualInts(m.Quantization.WeightBlockSize, []int{32, 32}) && m.Quantization.ScaleFormat == "ue8m0" && m.Quantization.ExpertDtype == "fp4", fmt.Sprintf("method=%s activation=%s block=%v scale=%s expert=%s", m.Quantization.Method, m.Quantization.Activation, m.Quantization.WeightBlockSize, m.Quantization.ScaleFormat, m.Quantization.ExpertDtype)},
		// The declared position limit is LOAD-BEARING for agent use: it is the
		// window the native planner enforces (InKernelPlanner.ContextWindow) and the
		// capacity the catalog advertises, and the ~150k-token agent envelope the
		// runtime targets depends on it being the published 1M with YaRN factor 16
		// from 65536. Nothing else in this ladder checked it, so a config that
		// silently dropped or shrank the window would be admitted as the published
		// checkpoint while serving a much smaller context.
		{"context window declared", c.MaxPositionEmbeddings == 1048576, fmt.Sprintf("max_position_embeddings=%d", c.MaxPositionEmbeddings)},
		{"rope scaling is yarn x16 from 65536", c.RopeScaling == "yarn" && c.RopeFactor == 16 && c.RopeOrigContext == 65536, fmt.Sprintf("type=%s factor=%g orig=%d", c.RopeScaling, c.RopeFactor, c.RopeOrigContext)},
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
