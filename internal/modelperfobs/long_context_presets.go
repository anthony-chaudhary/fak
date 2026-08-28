package modelperfobs

import "fmt"

// LongContextPresetSchema identifies the dated model-preset schema.
const LongContextPresetSchema = "fak-long-context-model-preset/1"

// LongContextPreset records source-backed model facts separately from bounded
// analytical assumptions and fields that the primary sources leave unknown.
type LongContextPreset struct {
	Schema      string                       `json:"schema"`
	Identity    string                       `json:"identity"`
	AsOfDate    string                       `json:"as_of_date"`
	Sources     []LongContextPresetSource    `json:"sources"`
	Facts       LongContextPresetFacts       `json:"facts"`
	Assumptions LongContextPresetAssumptions `json:"assumptions"`
	Unknowns    []string                     `json:"unknowns"`
}

type LongContextPresetSource struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	Revision string `json:"revision,omitempty"`
}

type LongContextPresetFacts struct {
	TotalParameters          float64  `json:"total_parameters"`
	ActiveParameters         float64  `json:"active_parameters"`
	NGramEmbeddingParameters float64  `json:"ngram_embedding_parameters,omitempty"`
	MaxPositionEmbeddings    uint64   `json:"max_position_embeddings"`
	HiddenLayers             uint64   `json:"hidden_layers"`
	TokenAttentionLayers     uint64   `json:"full_attention_layers"`
	LayerPattern             []string `json:"layer_pattern"`
	WeightDType              string   `json:"weight_dtype"`
}

type LongContextPresetAssumptions struct {
	WeightBits       ClosedRange `json:"weight_bits"`
	MetadataOverhead ClosedRange `json:"metadata_overhead_fraction"`
	KVBytesPerToken  ClosedRange `json:"kv_bytes_per_token"`
	KVTrafficBounds  string      `json:"kv_traffic_bounds"`
}

// LongContextScenario is a deterministic request vector generated from a
// dated preset. Input is ready for the generic estimator.
type LongContextScenario struct {
	PresetIdentity     string                    `json:"preset_identity"`
	ContextTokens      uint64                    `json:"context_tokens"`
	PrefillDecodeRatio uint64                    `json:"prefill_decode_ratio"`
	Input              LongContextEstimatorInput `json:"input"`
}

// DatedLongContextPresets returns only released identities supported by the
// official sources captured on 2026-08-28. The slice is newly allocated.
func DatedLongContextPresets() []LongContextPreset {
	return []LongContextPreset{qwen38FlashNextPreset(), glm53FlashPreset()}
}

func qwen38FlashNextPreset() LongContextPreset {
	return LongContextPreset{
		Schema: LongContextPresetSchema, Identity: "Qwen3.8-Flash-Next", AsOfDate: "2026-08-28",
		Sources: []LongContextPresetSource{
			{Name: "Qwen release blog", URL: "https://qwen.ai/blog?id=qwen3.8-flash-next"},
			{Name: "Qwen official model card", URL: "https://huggingface.co/Qwen/Qwen3.8-Flash-Next", Revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540"},
			{Name: "Qwen official config", URL: "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json", Revision: "de4b8e4d43b917e7706784d8bb445c9af86a3540"},
		},
		Facts: LongContextPresetFacts{
			TotalParameters: 125e9, ActiveParameters: 6e9, NGramEmbeddingParameters: 51e9, MaxPositionEmbeddings: 262_144,
			HiddenLayers: 48, TokenAttentionLayers: 12,
			LayerPattern: []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
			WeightDType:  "bfloat16",
		},
		Assumptions: LongContextPresetAssumptions{
			WeightBits: ClosedRange{16, 16}, MetadataOverhead: ClosedRange{0.02, 0.08},
			// Lower: only 12 documented full-attention layers retain token-indexed
			// BF16 K+V for 2 KV heads x 256 dimensions. Upper: all 48 layers
			// charged as that same full-attention KV shape. Linear-attention state
			// and implementation workspaces are not claimed to follow either end.
			KVBytesPerToken: ClosedRange{24_576, 98_304},
			KVTrafficBounds: "analytical lower: documented full-attention layers only; analytical upper: every layer charged as full-attention KV; both are assumptions, not measured fak-native traffic",
		},
		Unknowns: []string{
			"fak-native materialized linear-attention recurrent-state and workspace bytes",
			"fak-native realized KV/cache traffic after kernel fusion, paging, and reuse",
			"whether the separately reported 51B n-gram embedding parameters are fully resident and how their lookup cost maps to activated-parameter FLOPs",
			"artifact metadata overhead outside raw BF16 weights",
		},
	}
}

func glm53FlashPreset() LongContextPreset {
	return LongContextPreset{
		Schema: LongContextPresetSchema, Identity: "GLM-5.3-Flash", AsOfDate: "2026-08-28",
		Sources: []LongContextPresetSource{
			{Name: "Z.ai GLM-5.3-Flash release blog", URL: "https://z.ai/blog/glm-5.3-flash"},
			{Name: "Z.ai official BF16 model card", URL: "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16", Revision: "f12e0fe1f6b2ea274c11a569582edfd99d993c5e"},
			{Name: "Z.ai official BF16 config", URL: "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/f12e0fe1f6b2ea274c11a569582edfd99d993c5e/config.json", Revision: "f12e0fe1f6b2ea274c11a569582edfd99d993c5e"},
		},
		Facts: LongContextPresetFacts{
			TotalParameters: 320e9, ActiveParameters: 18e9, MaxPositionEmbeddings: 1_048_576,
			HiddenLayers: 45, TokenAttentionLayers: 11,
			LayerPattern: []string{"linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"},
			WeightDType:  "bfloat16",
		},
		Assumptions: LongContextPresetAssumptions{
			WeightBits: ClosedRange{16, 16}, MetadataOverhead: ClosedRange{0.02, 0.08},
			// The config documents MLA ranks and sparse layers, but not a
			// fak-native per-token cache layout. Use a deliberately wide interval:
			// lower charges one BF16 kv_lora_rank vector in each sparse layer;
			// upper charges conventional BF16 K+V for every head in every layer.
			KVBytesPerToken: ClosedRange{11_264, 2_949_120},
			KVTrafficBounds: "analytical lower: 11 sparse-attention layers x BF16 kv_lora_rank; analytical upper: conventional BF16 K+V for 64 heads x 256 dimensions across all 45 layers; both are assumptions, not measured fak-native traffic",
		},
		Unknowns: []string{
			"fak-native materialized MLA, sparse-index, linear-attention state, and workspace bytes",
			"whether the deployed fak-native cache stores compressed or expanded sparse-attention states",
			"fak-native realized KV/cache traffic after kernel fusion, paging, and reuse",
			"artifact metadata overhead outside raw BF16 weights",
		},
	}
}

// LongContextScenarios applies the preset's model fields to a caller-provided
// hardware/runtime template and emits the acceptance matrix deterministically.
func LongContextScenarios(p LongContextPreset, runtime LongContextEstimatorInput) ([]LongContextScenario, error) {
	if p.Schema != LongContextPresetSchema {
		return nil, fmt.Errorf("unsupported long-context preset schema %q", p.Schema)
	}
	contexts := [...]uint64{35_000, 64_000, 128_000, 200_000}
	ratios := [...]uint64{200, 300}
	out := make([]LongContextScenario, 0, len(contexts)*len(ratios))
	for _, contextTokens := range contexts {
		if contextTokens > p.Facts.MaxPositionEmbeddings {
			return nil, fmt.Errorf("context %d exceeds %s maximum %d", contextTokens, p.Identity, p.Facts.MaxPositionEmbeddings)
		}
		for _, ratio := range ratios {
			in := runtime
			in.TotalParameters = p.Facts.TotalParameters + p.Facts.NGramEmbeddingParameters
			in.ActiveParameters = p.Facts.ActiveParameters
			in.WeightBits = p.Assumptions.WeightBits
			in.MetadataOverhead = p.Assumptions.MetadataOverhead
			in.KVBytesPerToken = p.Assumptions.KVBytesPerToken
			in.ResidentContextTokens = contextTokens
			in.DecodeTokens = contextTokens / (ratio + 1)
			in.PrefillTokens = ratio * in.DecodeTokens
			if _, err := EstimateLongContextEnvelope(in); err != nil {
				return nil, fmt.Errorf("%s context=%d ratio=%d:1: %w", p.Identity, contextTokens, ratio, err)
			}
			out = append(out, LongContextScenario{PresetIdentity: p.Identity, ContextTokens: contextTokens, PrefillDecodeRatio: ratio, Input: in})
		}
	}
	return out, nil
}
