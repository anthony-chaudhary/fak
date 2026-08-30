package model

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const qwen35MTPEngine = "fak-native"

// qwen35MTPRequiredTensors is the complete one-layer MTP head shipped by the
// Qwen3.5-text-family Qwen3.8 checkpoint. Eligibility is feature-based: fak does
// not infer a release name that config.json does not carry.
var qwen35MTPRequiredTensors = [...]string{
	"mtp.fc.weight",
	"mtp.pre_fc_norm_embedding.weight",
	"mtp.pre_fc_norm_hidden.weight",
	"mtp.norm.weight",
	"mtp.layers.0.input_layernorm.weight",
	"mtp.layers.0.post_attention_layernorm.weight",
	"mtp.layers.0.self_attn.k_norm.weight",
	"mtp.layers.0.self_attn.k_proj.weight",
	"mtp.layers.0.self_attn.o_proj.weight",
	"mtp.layers.0.self_attn.q_norm.weight",
	"mtp.layers.0.self_attn.q_proj.weight",
	"mtp.layers.0.self_attn.v_proj.weight",
	"mtp.layers.0.mlp.down_proj.weight",
	"mtp.layers.0.mlp.gate_proj.weight",
	"mtp.layers.0.mlp.up_proj.weight",
}

// Qwen35MTPIncompleteError reports an MTP namespace that is present but cannot
// represent the checkpoint's complete one-layer draft head. It is a load-time
// refusal, not permission to switch to another inference engine.
type Qwen35MTPIncompleteError struct {
	Missing []string
}

func (e *Qwen35MTPIncompleteError) Error() string {
	return fmt.Sprintf("model: incomplete qwen3.8 MTP head: missing %s; disable MTP retention to use ordinary fak-native target decode", strings.Join(e.Missing, ", "))
}

// Qwen35MTPArtifactError reports retained Qwen3.8 MTP metadata or storage that
// cannot safely drive the native draft head. Kind is stable for programmatic
// handling; Field/Tensor and Want/Got make the refusal actionable. The only
// downgrade it advertises is ordinary fak-native target decode.
type Qwen35MTPArtifactError struct {
	Kind   string
	Field  string
	Tensor string
	Want   string
	Got    string
}

func (e *Qwen35MTPArtifactError) Error() string {
	where := e.Field
	if e.Tensor != "" {
		where = "tensor " + e.Tensor
	}
	if where == "" {
		where = "artifact"
	}
	return fmt.Sprintf("model: qwen3.8 MTP %s %s: got %s, want %s; disable MTP retention to use ordinary fak-native target decode", e.Kind, where, e.Got, e.Want)
}

// Qwen35MTPMode is the conservative admission result for the native MTP head.
// Enabled means the retained checkpoint substrate is complete and not explicitly
// disabled, so NewQwen35MTPForward may bind the native one-layer draft head.
type Qwen35MTPMode struct {
	Eligible bool
	Enabled  bool
	Reason   string
	Engine   string
}

func (c Config) isQwen35TextFamily() bool {
	if c.ModelType == "qwen3_5_text" || c.ModelType == "qwen3_5_moe_text" {
		return true
	}
	return c.IsQwen35Hybrid()
}

func qwen35MTPMissing(man map[string]tensorMeta) (present bool, missing []string) {
	for name := range man {
		if strings.HasPrefix(name, "mtp.") {
			present = true
			break
		}
	}
	for _, name := range qwen35MTPRequiredTensors {
		if _, ok := man[name]; !ok {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	return present, missing
}

func validateQwen35MTPManifest(cfg Config, man map[string]tensorMeta) error {
	if !cfg.isQwen35TextFamily() || cfg.NumMTPLayers() == 0 {
		return nil
	}
	present, missing := qwen35MTPMissing(man)
	if present && len(missing) != 0 {
		return &Qwen35MTPIncompleteError{Missing: missing}
	}
	return nil
}

// validateQwen35MTPLoadArtifact is the load-time admission fence shared by the
// safetensors path and loader-neutral source-format path used by GGUF. It runs
// only when retention was explicitly requested. An absent MTP declaration and
// namespace remains a supported ordinary-target checkpoint; once either is
// present, metadata, namespace, dtype, shape, and packed storage must all match
// the exact one-layer Qwen3.8 head or loading fails closed.
func validateQwen35MTPLoadArtifact(cfg Config, man map[string]tensorMeta, raw []byte) error {
	if !RetainMTP || !cfg.isQwen35TextFamily() {
		return nil
	}

	present, missing := qwen35MTPMissing(man)
	declared := cfg.MTPNumHiddenLayers != 0 || cfg.NumNextNPredictLayers != 0
	if !present && !declared {
		return nil
	}
	if cfg.MTPNumHiddenLayers < 0 || cfg.NumNextNPredictLayers < 0 {
		return qwen35MTPArtifactError("malformed-metadata", "MTP depth", "non-negative", fmt.Sprintf("mtp_num_hidden_layers=%d num_nextn_predict_layers=%d", cfg.MTPNumHiddenLayers, cfg.NumNextNPredictLayers))
	}
	if cfg.MTPNumHiddenLayers > 0 && cfg.NumNextNPredictLayers > 0 && cfg.MTPNumHiddenLayers != cfg.NumNextNPredictLayers {
		return qwen35MTPArtifactError("malformed-metadata", "MTP depth", "matching depth fields", fmt.Sprintf("mtp_num_hidden_layers=%d num_nextn_predict_layers=%d", cfg.MTPNumHiddenLayers, cfg.NumNextNPredictLayers))
	}
	if depth := cfg.NumMTPLayers(); depth != 1 {
		return qwen35MTPArtifactError("incompatible-metadata", "MTP depth", "1", fmt.Sprint(depth))
	}
	if cfg.MTPUseDedicatedEmbeddings {
		return qwen35MTPArtifactError("incompatible-metadata", "mtp_use_dedicated_embeddings", "false (shared target embeddings)", "true")
	}
	if len(missing) != 0 {
		return &Qwen35MTPIncompleteError{Missing: missing}
	}

	expected, err := qwen35MTPExpectedShapes(cfg)
	if err != nil {
		return err
	}
	for name := range man {
		if !strings.HasPrefix(name, "mtp.") {
			continue
		}
		if _, ok := expected[name]; !ok {
			return qwen35MTPArtifactError("incompatible-layout", "", "the exact supported one-layer namespace", name)
		}
	}
	for _, name := range qwen35MTPRequiredTensors {
		meta := man[name]
		wantShape := expected[name]
		if !strings.EqualFold(meta.Dtype, "f32") {
			return qwen35MTPArtifactTensorError("incompatible-dtype", name, "F32 after source decoding", meta.Dtype)
		}
		if !sameIntShape(meta.Shape, wantShape) {
			return qwen35MTPArtifactTensorError("incompatible-shape", name, fmt.Sprint(wantShape), fmt.Sprint(meta.Shape))
		}
		wantBytes, ok := qwen35MTPF32Bytes(wantShape)
		if !ok {
			return qwen35MTPArtifactTensorError("incompatible-shape", name, "shape with an addressable F32 payload", fmt.Sprint(wantShape))
		}
		if meta.Nbytes != wantBytes || meta.Offset < 0 || meta.Offset > len(raw)-meta.Nbytes {
			return qwen35MTPArtifactTensorError("malformed-storage", name, fmt.Sprintf("%d bytes inside model payload", wantBytes), fmt.Sprintf("offset=%d nbytes=%d payload=%d", meta.Offset, meta.Nbytes, len(raw)))
		}
	}
	return nil
}

func qwen35MTPExpectedShapes(cfg Config) (map[string][]int, error) {
	h, hd := cfg.HiddenSize, cfg.HeadDim
	if h <= 0 || hd <= 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.IntermediateSize <= 0 {
		got := fmt.Sprintf("hidden=%d head_dim=%d heads=%d kv_heads=%d intermediate=%d", h, hd, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize)
		return nil, qwen35MTPArtifactError("malformed-metadata", "model dimensions", "positive hidden/head/attention/intermediate dimensions", got)
	}
	qOut := cfg.NumHeads * hd
	if cfg.AttnOutputGate {
		qOut *= 2
	}
	return map[string][]int{
		"mtp.fc.weight":                                {h, 2 * h},
		"mtp.pre_fc_norm_embedding.weight":             {h},
		"mtp.pre_fc_norm_hidden.weight":                {h},
		"mtp.norm.weight":                              {h},
		"mtp.layers.0.input_layernorm.weight":          {h},
		"mtp.layers.0.post_attention_layernorm.weight": {h},
		"mtp.layers.0.self_attn.q_norm.weight":         {hd},
		"mtp.layers.0.self_attn.k_norm.weight":         {hd},
		"mtp.layers.0.self_attn.q_proj.weight":         {qOut, h},
		"mtp.layers.0.self_attn.k_proj.weight":         {cfg.NumKVHeads * hd, h},
		"mtp.layers.0.self_attn.v_proj.weight":         {cfg.NumKVHeads * hd, h},
		"mtp.layers.0.self_attn.o_proj.weight":         {h, cfg.NumHeads * hd},
		"mtp.layers.0.mlp.gate_proj.weight":            {cfg.IntermediateSize, h},
		"mtp.layers.0.mlp.up_proj.weight":              {cfg.IntermediateSize, h},
		"mtp.layers.0.mlp.down_proj.weight":            {h, cfg.IntermediateSize},
	}, nil
}

func qwen35MTPF32Bytes(shape []int) (int, bool) {
	n := 1
	for _, dim := range shape {
		if dim <= 0 || n > math.MaxInt/dim {
			return 0, false
		}
		n *= dim
	}
	if n > math.MaxInt/4 {
		return 0, false
	}
	return n * 4, true
}

func qwen35MTPArtifactError(kind, field, want, got string) *Qwen35MTPArtifactError {
	return &Qwen35MTPArtifactError{Kind: kind, Field: field, Want: want, Got: got}
}

func qwen35MTPArtifactTensorError(kind, tensor, want, got string) *Qwen35MTPArtifactError {
	return &Qwen35MTPArtifactError{Kind: kind, Tensor: tensor, Want: want, Got: got}
}

// Qwen35MTPAdmission reports whether a loaded manifest contains the exact native
// Qwen3.5-family MTP substrate. Complete eligible heads default on unless disabled.
// Ineligible or disabled configurations stay on the ordinary fak-native target;
// there is deliberately no external-runtime fallback state.
func qwen35MTPAdmission(cfg Config, man map[string]tensorMeta, disabled bool) (Qwen35MTPMode, error) {
	mode := Qwen35MTPMode{Engine: qwen35MTPEngine}
	if !cfg.isQwen35TextFamily() {
		mode.Reason = "not-qwen3.5-text-family"
		return mode, nil
	}
	if cfg.NumMTPLayers() != 1 {
		mode.Reason = "unsupported-mtp-depth"
		return mode, nil
	}
	if cfg.MTPUseDedicatedEmbeddings {
		mode.Reason = "dedicated-mtp-embeddings-unsupported"
		return mode, nil
	}
	present, missing := qwen35MTPMissing(man)
	if present && len(missing) != 0 {
		return mode, &Qwen35MTPIncompleteError{Missing: missing}
	}
	if !present {
		mode.Reason = "mtp-tensors-not-retained"
		return mode, nil
	}
	mode.Eligible = true
	if disabled {
		mode.Reason = "explicitly-disabled"
		return mode, nil
	}
	mode.Enabled = true
	mode.Reason = "eligible-default-on"
	return mode, nil
}

// Qwen35MTPMode reports this loaded model's native MTP substrate state. The
// disabled argument is an explicit runtime rollback to ordinary target decode.
func (m *Model) Qwen35MTPMode(disabled bool) (Qwen35MTPMode, error) {
	if m == nil {
		return Qwen35MTPMode{Engine: qwen35MTPEngine, Reason: "nil-model"}, nil
	}
	return qwen35MTPAdmission(m.Cfg, m.manifest, disabled)
}
