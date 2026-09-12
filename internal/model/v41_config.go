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

// DeepSeekV41Config holds V4.1-only metadata which must not be folded into the
// older deepseek_v4 profile. The slices are copied from the nested text_config.
type DeepSeekV41Config struct {
	WrapperModelType string
	TextModelType    string
	Quantization     DeepSeekV41QuantConfig

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
}

// IsDeepSeekV41 reports either exact V4.1 wrapper/text identity. Parsed official
// configs additionally retain and validate the pair in DeepSeekV41. V4.1 stays
// distinct from deepseek_v4 because their forward layouts differ.
func (c Config) IsDeepSeekV41() bool {
	return c.DeepSeekV41 != nil || c.ModelType == "deepseek_v41" || c.ModelType == "deepseek_v41_text"
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
	}
	nested := textEnvelope{
		configAlias:      configAlias(text),
		KVSourceLayerIDs: m.KVSourceLayerIDs, IndexSourceLayerIDs: m.IndexSourceLayerIDs,
		CandidateSourceLayerID: m.CandidateSourceLayerID, CandidateTopKBlocks: m.CandidateTopKBlocks, CandidateBlockSize: m.CandidateBlockSize,
		CompressRopeTheta: m.CompressRopeTheta,
		EngramLayerIDs:    m.EngramLayerIDs, EngramNumEmbeddings: m.EngramNumEmbeddings,
		EngramMaxNgramSize: m.EngramMaxNgramSize, EngramVocabSize: m.EngramVocabSize,
		EngramNHeads: m.EngramNHeads, EngramHeadDim: m.EngramHeadDim,
		EngramPadTokenID: m.EngramPadTokenID, EngramCompressedVocabSize: m.EngramCompressedVocabSize,
	}
	type wrapperEnvelope struct {
		configAlias
		Quantization DeepSeekV41QuantConfig `json:"quantization_config"`
		TextConfig   textEnvelope           `json:"text_config"`
	}
	return json.Marshal(wrapperEnvelope{configAlias(root), m.Quantization, nested})
}

func parseDeepSeekV41Metadata(root, text []byte, c Config) (*DeepSeekV41Config, error) {
	var envelope struct {
		ModelType    string                 `json:"model_type"`
		Quantization DeepSeekV41QuantConfig `json:"quantization_config"`
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

func refuseDeepSeekV41Native(c Config) error {
	if !c.IsDeepSeekV41() {
		return nil
	}
	return fmt.Errorf("%w: %s@%s metadata is admitted; Engram/shared-KV native execution remains pending", ErrV41NativeUnsupported, DeepSeekV41FlashModelID, DeepSeekV41FlashRevision)
}
