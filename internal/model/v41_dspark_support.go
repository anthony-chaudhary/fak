package model

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Support-GGUF names and invariants adapted from antirez/ds4
// b4e3faf5262b21db94ec309cf321640ff1eed460, gguf-tools/deepseek41_dspark.py.
// MIT License: Copyright (c) 2026 The ds4.c authors;
// Copyright (c) 2023-2026 The ggml authors; Copyright (c) 2023 DeepSeek.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions: The above copyright notice and this
// permission notice shall be included in all copies or substantial portions of
// the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
// EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// V41DSparkGGUFHeader is a header-only storage declaration. Tensor Shape uses
// source-axis order (the reverse of GGUF dimensions). No field certifies source
// weight equivalence: lossless F16 expansion and MXFP4 repacking require an
// independent original-weight audit, which this header cannot supply.
type V41DSparkGGUFHeader struct {
	Architecture, TargetArchitecture, SourceConfig, WeightEncoding string
	TargetLayerIDs                                                 []int
	Parameters                                                     map[string]int
	Tensors                                                        []V41TensorRef
}

// ValidateV41DSparkGGUFHeader checks the supported three-stage artifact header;
// success is neither weight-provenance approval nor execution admission.
func ValidateV41DSparkGGUFHeader(h V41DSparkGGUFHeader) (Config, error) {
	fail := func(reason string) (Config, error) {
		return Config{}, fmt.Errorf("%w: support GGUF %s", ErrV41DSparkArtifact, reason)
	}
	if h.Architecture != "deepseek41-dspark" || h.TargetArchitecture != "deepseek41" || h.WeightEncoding != "native MXFP4; lossless F16/F32 expansion" {
		return fail("architecture or weight encoding mismatch")
	}
	if len(h.SourceConfig) == 0 || len(h.SourceConfig) > 1<<20 {
		return fail("source config must be present and at most 1 MiB")
	}
	var source struct {
		ModelType string `json:"model_type"`
		Text      struct {
			Layers int `json:"num_hidden_layers"`
			Window int `json:"sliding_window"`
		} `json:"text_config"`
		Quant struct {
			Block []int `json:"weight_block_size"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal([]byte(h.SourceConfig), &source); err != nil || source.ModelType != "deepseek_v41" || source.Text.Layers < 1 || source.Text.Layers > 4096 || !slices.Equal(source.Quant.Block, []int{32, 32}) {
		return fail("invalid bounded V4.1 source config or native FP8 block shape")
	}
	// Config.UnmarshalJSON normally overlays wrapper fields onto text_config.
	// This donor takes geometry only from text_config, so refuse unexpected
	// wrapper axes before invoking that parser rather than validating one
	// geometry and silently constructing another.
	var wrapper map[string]json.RawMessage
	if err := json.Unmarshal([]byte(h.SourceConfig), &wrapper); err != nil {
		return fail("invalid source wrapper")
	}
	wrapperKeys := map[string]bool{"architectures": true, "model_type": true, "dtype": true, "torch_dtype": true, "transformers_version": true, "bos_token_id": true, "eos_token_id": true, "pad_token_id": true, "image_token_id": true, "quantization_config": true, "text_config": true, "vision_config": true}
	for key := range wrapper {
		if !wrapperKeys[key] {
			return fail("unsupported source wrapper field: " + key)
		}
	}
	var cfg Config
	if err := json.Unmarshal([]byte(h.SourceConfig), &cfg); err != nil || !cfg.IsDeepSeekV41() || !cfg.HasDeepSeekV41DSpark() {
		return fail("source config has no valid DSpark declaration")
	}
	d := cfg.DeepSeekV41.DSpark
	if len(d.TargetLayerIDs) != 3 || !slices.Equal(h.TargetLayerIDs, d.TargetLayerIDs) || d.BlockSize > source.Text.Window {
		return fail("target layers or draft window mismatch")
	}
	want := map[string]int{"block_size": d.BlockSize, "markov_rank": d.MarkovRank, "noise_token_id": d.NoiseTokenID, "n_routed_experts": d.NumRoutedExperts, "num_experts_per_tok": d.NumExpertsPerTok}
	if len(h.Parameters) != len(want) {
		return fail("missing or unknown scalar metadata")
	}
	for k, v := range want {
		if got, ok := h.Parameters[k]; !ok || got != v {
			return fail("scalar metadata disagrees with source config: " + k)
		}
	}
	// Existing native inventory supplies checked geometry and original roles.
	// Its stage=3/block=5/hc=4 constraints intentionally remain stricter than
	// the converter's generic block<=window check.
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		return Config{}, err
	}
	expected, err := dsparkGGUFSpecs(inv, d.NumRoutedExperts)
	if err != nil {
		return Config{}, err
	}
	if len(h.Tensors) != len(expected) {
		return fail("incomplete or extra tensor inventory")
	}
	seen := make(map[string]bool, len(h.Tensors))
	for _, t := range h.Tensors {
		spec, ok := expected[t.Name]
		if !ok || seen[t.Name] || !slices.Equal(t.Shape, spec.Shape) || !slices.Contains(spec.AllowedDtypes, t.Dtype) {
			return fail("unclaimed, duplicate, or incompatible tensor: " + t.Name)
		}
		seen[t.Name] = true
	}
	return cfg, nil
}

func dsparkGGUFSpecs(inv *V41DSparkInventory, experts int) (map[string]V41DSparkTensorSpec, error) {
	out := make(map[string]V41DSparkTensorSpec)
	roles := append(inv.Names(), inv.optional...)
	renames := map[string]string{
		"attn.attn_sink": "attn_sinks.weight", "attn.wq_a.weight": "attn_q_a.weight", "attn.wq_b.weight": "attn_q_b.weight",
		"attn.q_norm.weight": "attn_q_a_norm.weight", "attn.wkv.weight": "attn_kv.weight", "attn.kv_norm.weight": "attn_kv_a_norm.weight",
		"attn.wo_a.weight": "attn_output_a.weight", "attn.wo_b.weight": "attn_output_b.weight", "ffn.gate.weight": "ffn_gate_inp.weight",
		"ffn.gate.e_score_correction_bias": "exp_probs_b.bias", "ffn.gate.e_score_correction_bias_vl": "exp_probs_b_vl.bias",
		"markov_head.embed.weight": "markov_head.markov_w1.weight", "markov_head.head.weight": "markov_head.markov_w2.weight",
	}
	for _, name := range roles {
		spec, _ := inv.Lookup(name)
		parts := strings.SplitN(name, ".", 3)
		tail := parts[2]
		dest := tail
		spec.AllowedDtypes = []string{"F32"}
		if strings.HasPrefix(tail, "ffn.experts.") {
			x := strings.Split(tail, ".")
			if x[2] != "0" {
				continue
			}
			part := map[string]string{"w1": "gate", "w2": "down", "w3": "up"}[x[3]]
			if part == "" || len(spec.LogicalShape) != 2 || spec.LogicalShape[1]%32 != 0 {
				return nil, fmt.Errorf("%w: support expert geometry", ErrV41DSparkArtifact)
			}
			dest = "ffn_" + part + "_exps.weight"
			spec.Shape = append([]int{experts}, spec.LogicalShape...)
			spec.AllowedDtypes = []string{"MXFP4"}
		} else {
			if mapped, ok := renames[tail]; ok {
				dest = mapped
			}
			if strings.HasPrefix(tail, "hc_") {
				dest = tail + ".weight"
			}
			if strings.HasPrefix(tail, "ffn.shared_experts.") {
				x := strings.Split(tail, ".")
				part := map[string]string{"w1": "gate", "w2": "down", "w3": "up"}[x[2]]
				dest = "ffn_" + part + "_shexp.weight"
			}
			if slices.Contains(inv.byName[name].AllowedDtypes, "F8_E4M3") || strings.HasPrefix(tail, "markov_head.") {
				spec.AllowedDtypes = []string{"F16", "F32"}
			}
		}
		if _, ok := checkedShapeProduct(spec.Shape...); !ok {
			return nil, fmt.Errorf("%w: support shape overflow", ErrV41DSparkArtifact)
		}
		spec.Name = "mtp." + strconv.Itoa(spec.Stage) + "." + dest
		if _, exists := out[spec.Name]; exists {
			return nil, fmt.Errorf("%w: duplicate support mapping", ErrV41DSparkArtifact)
		}
		out[spec.Name] = spec
	}
	return out, nil
}
