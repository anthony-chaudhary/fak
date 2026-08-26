package model

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
)

const (
	qwen4ExpModelType    = "qwen4_exp_text"
	qwen4ExpEnvelopeType = "qwen4_exp"
	qwen4ExpArchitecture = "Qwen4ExpForConditionalGeneration"
)

type Qwen4ExpMTPConfig struct {
	Hybrid          bool     `json:"hybrid"`
	LayerTypes      []string `json:"layer_types"`
	NumHiddenLayers int      `json:"num_hidden_layers"`
}

type Qwen4ExpFlashNextConfig struct {
	ModelType                    string
	Architecture                 string
	NumHiddenLayers              int
	FullAttentionInterval        int
	LayerTypes                   []string
	NumExperts                   int
	NumExpertsPerToken           int
	SharedExpertIntermediateSize int
	IndexerBudget                int
	MambaSSMDType                string
	NgramSize                    int
	MaxPositionEmbeddings        int
	MTP                          Qwen4ExpMTPConfig
}

type qwen4ExpEnvelope struct {
	Architectures []string           `json:"architectures"`
	ModelType     string             `json:"model_type"`
	TextConfig    qwen4ExpTextConfig `json:"text_config"`
}

type qwen4ExpTextConfig struct {
	ModelType                    string            `json:"model_type"`
	NumHiddenLayers              int               `json:"num_hidden_layers"`
	FullAttentionInterval        int               `json:"full_attention_interval"`
	LayerTypes                   []string          `json:"layer_types"`
	NumExperts                   int               `json:"num_experts"`
	NumExpertsPerToken           int               `json:"num_experts_per_tok"`
	SharedExpertIntermediateSize int               `json:"shared_expert_intermediate_size"`
	IndexerBudget                int               `json:"indexer_budget"`
	MambaSSMDType                string            `json:"mamba_ssm_dtype"`
	NgramSize                    int               `json:"ngram_size"`
	MaxPositionEmbeddings        int               `json:"max_position_embeddings"`
	MTP                          Qwen4ExpMTPConfig `json:"mtp"`
}

func LoadQwen4ExpFlashNextConfig(path string) (Qwen4ExpFlashNextConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Qwen4ExpFlashNextConfig{}, err
	}
	var raw qwen4ExpEnvelope
	if err := json.Unmarshal(data, &raw); err != nil {
		return Qwen4ExpFlashNextConfig{}, fmt.Errorf("qwen4exp config: %w", err)
	}
	if len(raw.Architectures) != 1 {
		return Qwen4ExpFlashNextConfig{}, fmt.Errorf("qwen4exp config: require exactly one architecture")
	}
	if raw.ModelType != qwen4ExpEnvelopeType || raw.TextConfig.ModelType != qwen4ExpModelType || raw.Architectures[0] != qwen4ExpArchitecture {
		return Qwen4ExpFlashNextConfig{}, fmt.Errorf("qwen4exp config: unsupported identity %q/%q/%q", raw.ModelType, raw.TextConfig.ModelType, raw.Architectures[0])
	}
	t := raw.TextConfig
	cfg := Qwen4ExpFlashNextConfig{t.ModelType, raw.Architectures[0], t.NumHiddenLayers, t.FullAttentionInterval, t.LayerTypes, t.NumExperts, t.NumExpertsPerToken, t.SharedExpertIntermediateSize, t.IndexerBudget, t.MambaSSMDType, t.NgramSize, t.MaxPositionEmbeddings, t.MTP}
	return cfg, cfg.Validate()
}

func (c Qwen4ExpFlashNextConfig) Validate() error {
	if c.ModelType != qwen4ExpModelType || c.Architecture != qwen4ExpArchitecture {
		return fmt.Errorf("qwen4exp config: identity must be exact; got %q/%q", c.ModelType, c.Architecture)
	}
	if c.NumHiddenLayers != 48 || c.FullAttentionInterval != 4 || len(c.LayerTypes) != 48 {
		return fmt.Errorf("qwen4exp config: require 48-layer 3:1 cadence")
	}
	for i, kind := range c.LayerTypes {
		want := "linear_attention"
		if (i+1)%4 == 0 {
			want = "full_attention"
		}
		if kind != want {
			return fmt.Errorf("qwen4exp config: layer %d = %q, want %q", i, kind, want)
		}
	}
	if c.NumExperts != 512 || c.NumExpertsPerToken != 10 || c.SharedExpertIntermediateSize != 640 {
		return fmt.Errorf("qwen4exp config: require 512/top-10/shared-640 MoE")
	}
	if c.IndexerBudget != 2048 || c.MambaSSMDType != "float32" || c.NgramSize != 3 || c.MaxPositionEmbeddings != 262144 {
		return fmt.Errorf("qwen4exp config: architecture-defining field mismatch")
	}
	if !c.MTP.Hybrid || c.MTP.NumHiddenLayers != 1 || len(c.MTP.LayerTypes) != 1 || c.MTP.LayerTypes[0] != "full_attention" {
		return fmt.Errorf("qwen4exp config: require hybrid one-layer full-attention MTP")
	}
	return nil
}

type Qwen4ExpTensor struct {
	Shape []int  `json:"shape"`
	Shard string `json:"shard"`
	DType string `json:"dtype"`
}
type Qwen4ExpTensorInventory struct {
	Source  string                    `json:"source"`
	Tensors map[string]Qwen4ExpTensor `json:"tensors"`
}

func LoadQwen4ExpTensorInventory(path string) (Qwen4ExpTensorInventory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Qwen4ExpTensorInventory{}, err
	}
	var inventory Qwen4ExpTensorInventory
	if err := json.Unmarshal(data, &inventory); err != nil {
		return inventory, fmt.Errorf("qwen4exp tensor inventory: %w", err)
	}
	return inventory, nil
}

var qwen4ExpRequiredTensors = map[string][]int{
	"model.language_model.embed_tokens.weight": {248320, 2560},
	"lm_head.weight": {248320, 2560},
	"model.language_model.layers.0.linear_attn.A_log":                       {48},
	"model.language_model.layers.0.linear_attn.in_proj_qkv.weight":          {10240, 2560},
	"model.language_model.layers.0.mlp.gate.weight":                         {512, 2560},
	"model.language_model.layers.0.mlp.shared_expert.up_proj.weight":        {640, 2560},
	"model.language_model.layers.3.self_attn.q_proj.weight":                 {12288, 2560},
	"model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight":  {640, 2560},
	"model.language_model.layers.47.self_attn.indexer.index_qk_proj.weight": {640, 2560},
	"model.language_model.layers.47.mlp.experts.gate_up_proj":               {512, 1280, 2560},
	"model.language_model.layers.47.mlp.shared_expert.up_proj.weight":       {640, 2560},
	"mtp.layers.0.self_attn.q_proj.weight":                                  {12288, 2560},
	"mtp.layers.0.mlp.experts.gate_up_proj":                                 {512, 1280, 2560},
	"mtp.pre_fc_norm_hidden.weight":                                         {10240},
}

func ValidateQwen4ExpTensorInventory(cfg Qwen4ExpFlashNextConfig, inventory Qwen4ExpTensorInventory) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if !strings.Contains(inventory.Source, "Qwen/Qwen3.8-Flash-Next@f5d08274") {
		return fmt.Errorf("qwen4exp tensor inventory: unpinned source %q", inventory.Source)
	}
	for name, shape := range qwen4ExpRequiredTensors {
		tensor, ok := inventory.Tensors[name]
		if !ok {
			return fmt.Errorf("qwen4exp tensor inventory: missing %q", name)
		}
		if !reflect.DeepEqual(tensor.Shape, shape) {
			return fmt.Errorf("qwen4exp tensor inventory: %q shape %v, want %v", name, tensor.Shape, shape)
		}
		if tensor.DType != "BF16" {
			return fmt.Errorf("qwen4exp tensor inventory: %q dtype %q, want BF16", name, tensor.DType)
		}
		if tensor.Shard == "" {
			return fmt.Errorf("qwen4exp tensor inventory: %q has no shard", name)
		}
	}
	return nil
}

type UnsupportedQwen4ExpExecutionError struct{ Engine, ModelType, Architecture string }

func (e *UnsupportedQwen4ExpExecutionError) Error() string {
	return fmt.Sprintf("%s: unsupported execution for %s/%s: native Flash-Next layers are not implemented", e.Engine, e.ModelType, e.Architecture)
}
func (c Qwen4ExpFlashNextConfig) NativeExecution() error {
	if err := c.Validate(); err != nil {
		return err
	}
	return &UnsupportedQwen4ExpExecutionError{Engine: "fak-native", ModelType: c.ModelType, Architecture: c.Architecture}
}
