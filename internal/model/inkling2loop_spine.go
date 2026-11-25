package model

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
)

// Inkling 2 Mini (Thinking Machines, model_type "inkling2_loop") is a looped MoE:
// 24 stored layers in four attention-residual blocks of six, where the middle
// two blocks (layers 6..17) run three passes per token with gated cross-pass
// messages, so a token makes 48 layer visits. None of fak's forward paths
// implement the loop, attention residuals, relative-bias attention or the
// short convolutions; the generic Config decode flattens text_config into
// something that looks like an ordinary MoE, so without this spine the
// checkpoint would classify onto a generic path and decode garbage.
const (
	inkling2LoopEnvelopeType = "inkling2_loop"
	inkling2LoopTextType     = "inkling2_loop_text"
	inkling2LoopArchitecture = "Inkling2LoopForConditionalGeneration"
)

// Inkling2LoopOSSRoutes names the upstream engines that run inkling2_loop at
// the revisions where looped support first landed.
var Inkling2LoopOSSRoutes = []string{
	"llama.cpp@29b52d398b (GGUF, src/models/inkling2-loop.cpp)",
	"vllm@ab18334a4f (HF safetensors, vllm/models/inkling2/nvidia/loop.py)",
	"sglang PR #51926 (HF safetensors)",
}

// Inkling2LoopPasses is the loop schedule from text_config.loop_config. CoreEnd
// is exclusive.
type Inkling2LoopPasses struct {
	CoreStart int  `json:"core_start"`
	CoreEnd   int  `json:"core_end"`
	NumPasses int  `json:"num_passes"`
	GateRank  int  `json:"gate_rank"`
	IsVarloop bool `json:"is_varloop"`
}

type Inkling2LoopMTPConfig struct {
	NumHiddenLayers int   `json:"num_hidden_layers"`
	BlockSize       int   `json:"block_size"`
	TargetLayerIDs  []int `json:"target_layer_ids"`
}

// Inkling2LoopConfig is the pinned identity and geometry of an inkling2_loop
// checkpoint. One spine covers both released checkpoints: Inkling-2-Mini
// (fixed three passes) and TML-d24-varloop-research (is_varloop, trained on
// one to five passes, three by default).
type Inkling2LoopConfig struct {
	ModelType              string
	TextModelType          string
	Architecture           string
	HiddenSize             int
	NumHiddenLayers        int
	NumAttentionHeads      int
	NumKVHeads             int
	HeadDim                int
	VocabSize              int
	UnpaddedVocabSize      int
	LocalLayerIDs          []int
	SlidingWindow          int
	SconvKernelSize        int
	LayersPerAttnResBlock  int
	NumRoutedExperts       int
	NumExpertsPerTok       int
	NumSharedExperts       int
	ExpertIntermediateSize int
	Loop                   Inkling2LoopPasses
	MTP                    Inkling2LoopMTPConfig
	QuantMethod            string
	QuantAlgo              string
}

type inkling2LoopEnvelope struct {
	Architectures []string `json:"architectures"`
	ModelType     string   `json:"model_type"`
	TextConfig    struct {
		ModelType              string             `json:"model_type"`
		HiddenSize             int                `json:"hidden_size"`
		NumHiddenLayers        int                `json:"num_hidden_layers"`
		NumAttentionHeads      int                `json:"num_attention_heads"`
		NumKVHeads             int                `json:"num_key_value_heads"`
		HeadDim                int                `json:"head_dim"`
		VocabSize              int                `json:"vocab_size"`
		UnpaddedVocabSize      int                `json:"unpadded_vocab_size"`
		LocalLayerIDs          []int              `json:"local_layer_ids"`
		SlidingWindow          int                `json:"sliding_window_size"`
		SconvKernelSize        int                `json:"sconv_kernel_size"`
		LayersPerAttnResBlock  int                `json:"n_layers_per_attn_res_block"`
		NumRoutedExperts       int                `json:"n_routed_experts"`
		NumExpertsPerTok       int                `json:"num_experts_per_tok"`
		NumSharedExperts       int                `json:"n_shared_experts"`
		ExpertIntermediateSize int                `json:"intermediate_size"`
		Loop                   Inkling2LoopPasses `json:"loop_config"`
	} `json:"text_config"`
	MTP   Inkling2LoopMTPConfig `json:"mtp_config"`
	Quant struct {
		Method string `json:"quant_method"`
		Algo   string `json:"quant_algo"`
	} `json:"quantization_config"`
}

// LoadInkling2LoopConfig reads and validates an inkling2_loop config.json.
func LoadInkling2LoopConfig(path string) (Inkling2LoopConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Inkling2LoopConfig{}, err
	}
	return ParseInkling2LoopConfig(data)
}

// ParseInkling2LoopConfig decodes and validates inkling2_loop config.json bytes.
func ParseInkling2LoopConfig(data []byte) (Inkling2LoopConfig, error) {
	var raw inkling2LoopEnvelope
	if err := json.Unmarshal(data, &raw); err != nil {
		return Inkling2LoopConfig{}, fmt.Errorf("inkling2_loop config: %w", err)
	}
	if len(raw.Architectures) != 1 {
		return Inkling2LoopConfig{}, fmt.Errorf("inkling2_loop config: require exactly one architecture, got %d", len(raw.Architectures))
	}
	t := raw.TextConfig
	cfg := Inkling2LoopConfig{
		ModelType:              raw.ModelType,
		TextModelType:          t.ModelType,
		Architecture:           raw.Architectures[0],
		HiddenSize:             t.HiddenSize,
		NumHiddenLayers:        t.NumHiddenLayers,
		NumAttentionHeads:      t.NumAttentionHeads,
		NumKVHeads:             t.NumKVHeads,
		HeadDim:                t.HeadDim,
		VocabSize:              t.VocabSize,
		UnpaddedVocabSize:      t.UnpaddedVocabSize,
		LocalLayerIDs:          t.LocalLayerIDs,
		SlidingWindow:          t.SlidingWindow,
		SconvKernelSize:        t.SconvKernelSize,
		LayersPerAttnResBlock:  t.LayersPerAttnResBlock,
		NumRoutedExperts:       t.NumRoutedExperts,
		NumExpertsPerTok:       t.NumExpertsPerTok,
		NumSharedExperts:       t.NumSharedExperts,
		ExpertIntermediateSize: t.ExpertIntermediateSize,
		Loop:                   t.Loop,
		MTP:                    raw.MTP,
		QuantMethod:            raw.Quant.Method,
		QuantAlgo:              raw.Quant.Algo,
	}
	return cfg, cfg.Validate()
}

// Validate refuses any drift from the released geometry. The loop core must
// sit on attention-residual block boundaries, because the cross-pass message
// reads whole blocks.
func (c Inkling2LoopConfig) Validate() error {
	if c.ModelType != inkling2LoopEnvelopeType || c.TextModelType != inkling2LoopTextType || c.Architecture != inkling2LoopArchitecture {
		return fmt.Errorf("inkling2_loop config: unsupported identity %q/%q/%q", c.ModelType, c.TextModelType, c.Architecture)
	}
	if c.HiddenSize != 2304 || c.NumHiddenLayers != 24 || c.NumAttentionHeads != 24 || c.NumKVHeads != 8 || c.HeadDim != 128 {
		return fmt.Errorf("inkling2_loop config: require hidden 2304, 24 layers, 24q/8kv heads of 128")
	}
	if c.VocabSize != 201024 || c.UnpaddedVocabSize != 200058 || c.SlidingWindow != 1024 || c.SconvKernelSize != 4 {
		return fmt.Errorf("inkling2_loop config: architecture-defining field mismatch")
	}
	if c.LayersPerAttnResBlock != 6 {
		return fmt.Errorf("inkling2_loop config: require 6-layer attention-residual blocks, got %d", c.LayersPerAttnResBlock)
	}
	if !reflect.DeepEqual(c.LocalLayerIDs, inkling2LoopExpectedLocalLayers(c.NumHiddenLayers, c.LayersPerAttnResBlock)) {
		return fmt.Errorf("inkling2_loop config: local_layer_ids %v, want every layer except the last of each block", c.LocalLayerIDs)
	}
	if c.NumRoutedExperts != 128 || c.NumExpertsPerTok != 8 || c.NumSharedExperts != 0 || c.ExpertIntermediateSize != 1152 {
		return fmt.Errorf("inkling2_loop config: require 128 routed/top-8/no-shared MoE with 1152 expert width")
	}
	l := c.Loop
	if l.CoreStart != 6 || l.CoreEnd != 18 || l.NumPasses != 3 || l.GateRank != 64 {
		return fmt.Errorf("inkling2_loop config: loop_config %+v, want core [6,18) x3 passes, gate rank 64", l)
	}
	if l.CoreStart%c.LayersPerAttnResBlock != 0 || l.CoreEnd%c.LayersPerAttnResBlock != 0 {
		return fmt.Errorf("inkling2_loop config: loop core [%d,%d) is not block aligned", l.CoreStart, l.CoreEnd)
	}
	if c.MTP.NumHiddenLayers != 3 || c.MTP.BlockSize != 8 {
		return fmt.Errorf("inkling2_loop config: require 3-layer block-8 MTP draft head")
	}
	if n := len(c.MTP.TargetLayerIDs); n == 0 || c.MTP.TargetLayerIDs[n-1] != c.LayerVisits() {
		return fmt.Errorf("inkling2_loop config: MTP target_layer_ids %v must end at the %d-visit output", c.MTP.TargetLayerIDs, c.LayerVisits())
	}
	if c.QuantMethod != "modelopt" || c.QuantAlgo != "NVFP4" {
		return fmt.Errorf("inkling2_loop config: require modelopt NVFP4 quantization, got %q/%q", c.QuantMethod, c.QuantAlgo)
	}
	return nil
}

func inkling2LoopExpectedLocalLayers(layers, block int) []int {
	out := make([]int, 0, layers)
	for i := 0; i < layers; i++ {
		if (i+1)%block != 0 {
			out = append(out, i)
		}
	}
	return out
}

// LayerVisits is the number of layer applications per token: prelude and coda
// once, the core NumPasses times.
func (c Inkling2LoopConfig) LayerVisits() int {
	core := c.Loop.CoreEnd - c.Loop.CoreStart
	return c.NumHiddenLayers - core + core*c.Loop.NumPasses
}

// IsLooped reports whether stored layer runs once per pass.
func (c Inkling2LoopConfig) IsLooped(layer int) bool {
	return layer >= c.Loop.CoreStart && layer < c.Loop.CoreEnd
}

// ExpertPrecision is the released per-layer expert format: NVFP4 everywhere
// except the last four layers, which were trained and shipped in MXFP8.
func (c Inkling2LoopConfig) ExpertPrecision(layer int) string {
	if layer >= c.NumHiddenLayers-4 {
		return "MXFP8"
	}
	return "NVFP4"
}

type Inkling2LoopTensor struct {
	Shape []int  `json:"shape"`
	DType string `json:"dtype"`
}

type Inkling2LoopTensorInventory struct {
	Source  string                        `json:"source"`
	Tensors map[string]Inkling2LoopTensor `json:"tensors"`
}

func LoadInkling2LoopTensorInventory(path string) (Inkling2LoopTensorInventory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Inkling2LoopTensorInventory{}, err
	}
	var inv Inkling2LoopTensorInventory
	if err := json.Unmarshal(data, &inv); err != nil {
		return inv, fmt.Errorf("inkling2_loop tensor inventory: %w", err)
	}
	return inv, nil
}

var inkling2LoopRequiredTensors = map[string]Inkling2LoopTensor{
	"model.llm.embed.weight":                           {[]int{201024, 2304}, "BF16"},
	"model.llm.unembed.weight":                         {[]int{201024, 2304}, "BF16"},
	"model.llm.attn_res.output_query":                  {[]int{2304}, "F32"},
	"model.llm.attn_res.attn_queries.10":               {[]int{3, 2304}, "F32"},
	"model.llm.middle_cycle_cross_pass_queries":        {[]int{24, 2304}, "F32"},
	"model.llm.middle_cycle_cross_pass_kv_queries":     {[]int{24, 2304}, "F32"},
	"model.llm.layers.0.attn.wq_du.weight":             {[]int{3072, 2304}, "BF16"},
	"model.llm.layers.0.attn.wk_dv.weight":             {[]int{1024, 2304}, "BF16"},
	"model.llm.layers.0.attn.wr_du.weight":             {[]int{384, 2304}, "BF16"},
	"model.llm.layers.0.attn.k_sconv.weight":           {[]int{1024, 1, 4}, "BF16"},
	"model.llm.layers.5.attn.k_sconv.static_w":         {[]int{4, 1024}, "BF16"},
	"model.llm.layers.6.controllers.attn.gain_weight":  {[]int{2304, 64}, "BF16"},
	"model.llm.layers.17.controllers.attn.gain_weight": {[]int{2304, 64}, "BF16"},
	"model.llm.layers.0.mlp.gate.weight":               {[]int{128, 2304}, "BF16"},
	"model.llm.layers.0.mlp.experts.w13_weight":        {[]int{128, 2304, 1152}, "U8"},
	"model.llm.layers.0.mlp.experts.w13_weight_scale":  {[]int{128, 2304, 144}, "F8_E4M3"},
	"model.llm.layers.19.mlp.experts.w13_weight":       {[]int{128, 2304, 1152}, "U8"},
	"model.llm.layers.20.mlp.experts.w13_weight":       {[]int{128, 2304, 2304}, "F8_E4M3"},
	"model.llm.layers.20.mlp.experts.w13_weight_scale": {[]int{128, 2304, 72}, "U8"},
	"model.llm.layers.23.mlp.experts.w2_weight":        {[]int{128, 2304, 1152}, "F8_E4M3"},
}

// ValidateInkling2LoopTensorInventory checks a pinned header inventory against
// the geometry, including the NVFP4/MXFP8 split: NVFP4 experts pack two values
// per U8 with F8_E4M3 block scales, MXFP8 experts are F8_E4M3 with U8 (E8M0)
// scales.
func ValidateInkling2LoopTensorInventory(cfg Inkling2LoopConfig, inv Inkling2LoopTensorInventory) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if !strings.Contains(inv.Source, "thinkingmachines/Inkling-2-Mini@b85c3cfa") {
		return fmt.Errorf("inkling2_loop tensor inventory: unpinned source %q", inv.Source)
	}
	for name, want := range inkling2LoopRequiredTensors {
		got, ok := inv.Tensors[name]
		if !ok {
			return fmt.Errorf("inkling2_loop tensor inventory: missing %q", name)
		}
		if !reflect.DeepEqual(got.Shape, want.Shape) || got.DType != want.DType {
			return fmt.Errorf("inkling2_loop tensor inventory: %q = %s%v, want %s%v", name, got.DType, got.Shape, want.DType, want.Shape)
		}
	}
	for name, got := range inv.Tensors {
		var layer int
		if _, err := fmt.Sscanf(name, "model.llm.layers.%d.mlp.experts.w13_weight", &layer); err != nil || !strings.HasSuffix(name, ".mlp.experts.w13_weight") {
			continue
		}
		want := map[string]string{"NVFP4": "U8", "MXFP8": "F8_E4M3"}[cfg.ExpertPrecision(layer)]
		if got.DType != want {
			return fmt.Errorf("inkling2_loop tensor inventory: layer %d experts %s, want %s for %s", layer, got.DType, want, cfg.ExpertPrecision(layer))
		}
	}
	return nil
}

// UnsupportedInkling2LoopExecutionError is the typed refusal for running an
// inkling2_loop checkpoint on fak's native forward.
type UnsupportedInkling2LoopExecutionError struct {
	Engine       string
	ModelType    string
	Architecture string
	// Backend is the compute backend that was asked to run it; empty means the
	// generic native forward, which has no looped path on any backend.
	Backend string
}

func (e *UnsupportedInkling2LoopExecutionError) Error() string {
	where := "the generic native forward"
	if e.Backend != "" {
		where = "the " + e.Backend + " backend"
	}
	return fmt.Sprintf("%s: unsupported execution for %s/%s on %s: fak's looped forward runs on the %s backend from an inkling2_loop GGUF (fak serve --engine inkernel --backend %s --gguf <file>) and on Apple-Silicon Metal in a darwin/arm64 cgo build (fak serve --%s --gguf <file>); otherwise serve it on %s",
		e.Engine, e.ModelType, e.Architecture, where, Inkling2LoopBackend, Inkling2LoopBackend, Inkling2LoopMetalBackend, strings.Join(Inkling2LoopOSSRoutes, ", "))
}

// NativeExecution validates the config and returns the typed refusal.
func (c Inkling2LoopConfig) NativeExecution() error {
	if err := c.Validate(); err != nil {
		return err
	}
	return &UnsupportedInkling2LoopExecutionError{Engine: "fak-native", ModelType: c.ModelType, Architecture: c.Architecture}
}

// IsInkling2Loop reports whether a generically decoded Config is an
// inkling2_loop checkpoint. The wrapper model_type overlays text_config, so
// either spelling identifies the family.
func (c Config) IsInkling2Loop() bool {
	return c.ModelType == inkling2LoopEnvelopeType || c.ModelType == inkling2LoopTextType
}

func refuseInkling2LoopNative(c Config) error {
	if !c.IsInkling2Loop() {
		return nil
	}
	return &UnsupportedInkling2LoopExecutionError{Engine: "fak-native", ModelType: c.ModelType, Architecture: inkling2LoopArchitecture}
}
