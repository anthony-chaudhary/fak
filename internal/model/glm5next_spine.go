package model

import (
	"encoding/json"
	"fmt"
)

const (
	glm5NextModelType    = "glm5_next"
	glm5NextArchitecture = "Glm5NextForConditionalGeneration"
)

// GLM5NextUnsupportedError is returned before forward selection for the exact
// GLM-5.3-Flash family. fak recognizes this envelope, but its KDA/DSA/indexer/mHC
// execution path is not implemented by the native engine yet.
type GLM5NextUnsupportedError struct {
	ModelType    string
	Architecture string
}

func (e *GLM5NextUnsupportedError) Error() string {
	return fmt.Sprintf("fak-native: recognized %s/%s checkpoint, but native execution is unsupported (KDA/DSA/indexer/mHC kernels are not implemented)", e.ModelType, e.Architecture)
}

type glm5NextIdentity struct {
	ModelType         string   `json:"model_type"`
	Architectures     []string `json:"architectures"`
	Dtype             string   `json:"dtype"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	MaxPositionEmbeds int      `json:"max_position_embeddings"`
	LayerTypes        []string `json:"layer_types"`
	MLPLayerTypes     []string `json:"mlp_layer_types"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKVHeads        int      `json:"num_key_value_heads"`
	NRoutedExperts    int      `json:"n_routed_experts"`
	NSharedExperts    int      `json:"n_shared_experts"`
	ExpertsPerToken   int      `json:"num_experts_per_tok"`
	MoEIntermediate   int      `json:"moe_intermediate_size"`
	FirstDenseLayers  int      `json:"first_k_dense_replace"`
	QLoraRank         int      `json:"q_lora_rank"`
	KVLoraRank        int      `json:"kv_lora_rank"`
	QKNopeHeadDim     int      `json:"qk_nope_head_dim"`
	VHeadDim          int      `json:"v_head_dim"`
	IndexHeadDim      int      `json:"index_head_dim"`
	IndexHeads        int      `json:"index_n_heads"`
	IndexTopK         int      `json:"index_topk"`
	IndexKPool        int      `json:"index_kpool"`
	MHC               bool     `json:"mhc"`
	HCMult            int      `json:"hc_mult"`
	LinearAttn        struct {
		NumHeads            int   `json:"num_heads"`
		HeadDim             int   `json:"head_dim"`
		ShortConvKernelSize int   `json:"short_conv_kernel_size"`
		KDALayers           []int `json:"kda_layers"`
		FullAttnLayers      []int `json:"full_attn_layers"`
	} `json:"linear_attn_config"`
}

type glm5NextEnvelope struct {
	ModelType     string           `json:"model_type"`
	Architectures []string         `json:"architectures"`
	TextConfig    glm5NextIdentity `json:"text_config"`
	VisionConfig  struct {
		ModelType                  string `json:"model_type"`
		Depth                      int    `json:"depth"`
		HiddenSize                 int    `json:"hidden_size"`
		IntermediateSize           int    `json:"intermediate_size"`
		NumHeads                   int    `json:"num_heads"`
		ImageSize                  int    `json:"image_size"`
		PatchSize                  int    `json:"patch_size"`
		SpatialMergeSize           int    `json:"spatial_merge_size"`
		TemporalPatchSize          int    `json:"temporal_patch_size"`
		OutHiddenSize              int    `json:"out_hidden_size"`
		ProjectionIntermediateSize int    `json:"projection_intermediate_size"`
	} `json:"vision_config"`
	QuantizationConfig struct {
		Method           string `json:"quant_method"`
		Format           string `json:"fmt"`
		ActivationScheme string `json:"activation_scheme"`
	} `json:"quantization_config"`
}

func isExactGLM5NextConfig(raw []byte) bool {
	var envelope glm5NextEnvelope
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return false
	}
	text := envelope.TextConfig
	if envelope.ModelType != glm5NextModelType || !containsGLM5NextString(envelope.Architectures, glm5NextArchitecture) ||
		text.ModelType != "glm5_next_text" || text.Dtype != "bfloat16" || text.NumHiddenLayers != 45 ||
		text.MaxPositionEmbeds != 1048576 || text.HiddenSize != 4096 || text.IntermediateSize != 12288 ||
		text.NumAttentionHeads != 64 || text.NumKVHeads != 64 || text.NRoutedExperts != 288 ||
		text.NSharedExperts != 1 || text.ExpertsPerToken != 8 || text.MoEIntermediate != 2048 ||
		text.FirstDenseLayers != 3 || text.QLoraRank != 1536 || text.KVLoraRank != 512 ||
		text.QKNopeHeadDim != 256 || text.VHeadDim != 256 || text.IndexHeadDim != 128 ||
		text.IndexHeads != 32 || text.IndexTopK != 2048 || text.IndexKPool != 4 || !text.MHC || text.HCMult != 4 ||
		text.LinearAttn.NumHeads != 64 || text.LinearAttn.HeadDim != 128 || text.LinearAttn.ShortConvKernelSize != 4 ||
		!glm5NextCadence(text.LayerTypes, text.LinearAttn.KDALayers, text.LinearAttn.FullAttnLayers) ||
		!glm5NextMLPCadence(text.MLPLayerTypes) {
		return false
	}
	vision := envelope.VisionConfig
	quant := envelope.QuantizationConfig
	isVisionValid := vision.ModelType == "glm5_next_vision" && vision.Depth == 24 && vision.HiddenSize == 1024 &&
		vision.IntermediateSize == 4096 && vision.NumHeads == 16 && vision.ImageSize == 448 &&
		vision.PatchSize == 14 && vision.SpatialMergeSize == 2 && vision.TemporalPatchSize == 2 &&
		vision.OutHiddenSize == 4096 && vision.ProjectionIntermediateSize == 10240
	if !isVisionValid {
		return false
	}
	isFP8 := quant.Method == "fp8" && quant.Format == "e4m3" && quant.ActivationScheme == "dynamic"
	isBF16 := quant.Method == "" && quant.Format == "" && quant.ActivationScheme == ""
	return isFP8 || isBF16
}

func glm5NextCadence(layerTypes []string, kda, full []int) bool {
	if len(layerTypes) != 45 || len(kda) != 34 || len(full) != 11 {
		return false
	}
	ki, fi := 0, 0
	for layer, typ := range layerTypes {
		wantFull := layer%4 == 3
		if wantFull {
			if typ != "deepseek_sparse_attention" || fi >= len(full) || full[fi] != layer {
				return false
			}
			fi++
		} else {
			if typ != "linear_attention" || ki >= len(kda) || kda[ki] != layer {
				return false
			}
			ki++
		}
	}
	return ki == len(kda) && fi == len(full)
}

func glm5NextMLPCadence(types []string) bool {
	if len(types) != 45 {
		return false
	}
	for i, typ := range types {
		want := "sparse"
		if i < 3 {
			want = "dense"
		}
		if typ != want {
			return false
		}
	}
	return true
}

func containsGLM5NextString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

// GLM5NextConfig holds the verified architecture parameters for GLM-5.3-Flash.
type GLM5NextConfig struct {
	NumHiddenLayers   int
	HiddenSize        int
	IntermediateSize  int
	NumAttentionHeads int
	NumKVHeads        int
	NRoutedExperts    int
	NSharedExperts    int
	ExpertsPerToken   int
	MoEIntermediate   int
	FirstDenseLayers  int
	QLoraRank         int
	KVLoraRank        int
	QKNopeHeadDim     int
	VHeadDim          int
	IndexHeadDim      int
	IndexHeads        int
	IndexTopK         int
	IndexKPool        int
	MHC               bool
	HCMult            int
	KDANumHeads       int
	KDAHeadDim        int
	ConvWindowSize    int
	KDALayers         []int
	FullAttnLayers    []int
}

// DefaultGLM5NextConfig returns the canonical pinned architecture geometry for GLM-5.3-Flash.
func DefaultGLM5NextConfig() GLM5NextConfig {
	kda := make([]int, 0, 34)
	full := make([]int, 0, 11)
	for l := 0; l < 45; l++ {
		if l%4 == 3 {
			full = append(full, l)
		} else {
			kda = append(kda, l)
		}
	}
	return GLM5NextConfig{
		NumHiddenLayers:   45,
		HiddenSize:        4096,
		IntermediateSize:  12288,
		NumAttentionHeads: 64,
		NumKVHeads:        64,
		NRoutedExperts:    288,
		NSharedExperts:    1,
		ExpertsPerToken:   8,
		MoEIntermediate:   2048,
		FirstDenseLayers:  3,
		QLoraRank:         1536,
		KVLoraRank:        512,
		QKNopeHeadDim:     256,
		VHeadDim:          256,
		IndexHeadDim:      128,
		IndexHeads:        32,
		IndexTopK:         2048,
		IndexKPool:        4,
		MHC:               true,
		HCMult:            4,
		KDANumHeads:       64,
		KDAHeadDim:        128,
		ConvWindowSize:    4,
		KDALayers:         kda,
		FullAttnLayers:    full,
	}
}

// IsKDALayer reports whether layer is a KDA linear-attention layer.
func (c GLM5NextConfig) IsKDALayer(layer int) bool {
	if layer < 0 || layer >= c.NumHiddenLayers {
		return false
	}
	return layer%4 != 3
}

// IsDSALayer reports whether layer is a Decoupled Sparse Attention layer.
func (c GLM5NextConfig) IsDSALayer(layer int) bool {
	if layer < 0 || layer >= c.NumHiddenLayers {
		return false
	}
	return layer%4 == 3
}

// IsDenseMLPLayer reports whether layer uses dense MLP rather than MoE (layers 0..2).
func (c GLM5NextConfig) IsDenseMLPLayer(layer int) bool {
	if layer < 0 || layer >= c.NumHiddenLayers {
		return false
	}
	return layer < c.FirstDenseLayers
}

// IsSparseMoELayer reports whether layer uses 288-expert routed MoE (layers 3..44).
func (c GLM5NextConfig) IsSparseMoELayer(layer int) bool {
	if layer < 0 || layer >= c.NumHiddenLayers {
		return false
	}
	return layer >= c.FirstDenseLayers
}

// IsGLM5NextKDALayer reports whether layer is a GLM5Next linear attention layer on Config.
func (c Config) IsGLM5NextKDALayer(layer int) bool {
	if !c.GLM5Next {
		return false
	}
	return layer >= 0 && layer < 45 && layer%4 != 3
}

// IsGLM5NextDSALayer reports whether layer is a GLM5Next sparse attention layer on Config.
func (c Config) IsGLM5NextDSALayer(layer int) bool {
	if !c.GLM5Next {
		return false
	}
	return layer >= 0 && layer < 45 && layer%4 == 3
}

// IsGLM5NextDenseMLP reports whether layer is one of the initial dense MLP layers on Config.
func (c Config) IsGLM5NextDenseMLP(layer int) bool {
	if !c.GLM5Next {
		return false
	}
	return layer >= 0 && layer < 3
}

// IsGLM5NextSparseMoE reports whether layer is a routed MoE layer on Config.
func (c Config) IsGLM5NextSparseMoE(layer int) bool {
	if !c.GLM5Next {
		return false
	}
	return layer >= 3 && layer < 45
}
