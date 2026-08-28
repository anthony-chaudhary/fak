package model

import (
	"fmt"
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
	return fmt.Sprintf("model: incomplete qwen3.5-family MTP head: missing %s", strings.Join(e.Missing, ", "))
}

// Qwen35MTPMode is the conservative admission result for the native MTP head.
// Enabled means the retained checkpoint substrate is complete and not explicitly
// disabled. It does not claim that draft forward execution is wired yet.
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
