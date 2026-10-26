package computetrace

import (
	"strconv"
	"strings"
)

// TorchStageMap classifies module outputs for one engine/HF architecture pair.
// Layered keys are module suffixes after model.layers.<n>.
type TorchStageMap struct {
	Engine, Architecture string
	Layered, Pre, Post   map[string]string
	Unmapped             map[string]string
}

// TorchStageMapFor returns a fresh map so callers cannot mutate the registry.
func TorchStageMapFor(engine, arch string) (TorchStageMap, bool) {
	if engine != "sglang" && engine != "vllm" {
		return TorchStageMap{}, false
	}
	m := TorchStageMap{Engine: engine, Architecture: arch, Layered: map[string]string{}, Pre: map[string]string{}, Post: map[string]string{}, Unmapped: map[string]string{}}
	switch arch {
	case "LlamaForCausalLM":
		m.Layered["output"] = "residual_streams"
		m.Layered["self_attn.o_proj.output"] = "attention_projected"
		m.Layered["mlp.output"] = "moe_sum"
	case "DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM":
		attn, mlp := "self_attn", "mlp"
		if engine == "vllm" {
			attn, mlp = "attn", "ffn"
		}
		m.Layered = map[string]string{"output": "residual_streams", attn + ".q_norm.output": "q_latent", attn + ".kv_norm.output": "kv_latent", attn + ".wo_b.output": "attention_projected", mlp + ".output": "moe_sum"}
		m.Unmapped[attn+".wq_b.output"] = "column-parallel shards require concatenation"
		m.Unmapped[attn+".wo_a.output"] = "column-parallel shards require concatenation"
	default:
		return TorchStageMap{}, false
	}
	return m, true
}

func (m TorchStageMap) classify(name string) (stage string, layer int, reason string) {
	if !strings.HasPrefix(name, "non_intrusive__") {
		return "", 0, "core field has no activation-stage contract"
	}
	path := strings.TrimPrefix(name, "non_intrusive__")
	if stage = m.Pre[path]; stage != "" {
		return stage, -1, ""
	}
	if stage = m.Post[path]; stage != "" {
		return stage, llamaCppPostLayer, ""
	}
	parts := strings.Split(path, ".")
	if len(parts) < 4 || parts[0] != "model" || parts[1] != "layers" {
		return "", 0, "module has no registered stage"
	}
	n, e := strconv.Atoi(parts[2])
	if e != nil || n < 0 {
		return "", 0, "module has no decoder-layer index"
	}
	suffix := strings.Join(parts[3:], ".")
	if strings.HasSuffix(suffix, ".output.0") {
		suffix = strings.TrimSuffix(suffix, ".0")
	}
	if suffix == "output.0" {
		suffix = "output"
	}
	if stage = m.Layered[suffix]; stage != "" {
		return stage, n, ""
	}
	if reason = m.Unmapped[suffix]; reason != "" {
		return "", n, reason
	}
	return "", n, "module output has no registered stage"
}
