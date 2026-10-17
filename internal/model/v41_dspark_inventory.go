package model

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ErrV41DSparkArtifact rejects an unknown or malformed bundled draft artifact.
// Artifact admission is distinct from runtime admission: this code never relaxes
// ErrV41DSparkUnsupported or enables a forward/verify path.
var ErrV41DSparkArtifact = errors.New("model: invalid DeepSeek V4.1 DSpark artifact")

// V41DSparkTensorSpec is a storage contract, not a claim that a remote tensor
// header was inspected. An opened shard supplies the actual dtype and original
// source name. BF16/F32 norms are retained as declared, never silently cast.
type V41DSparkTensorSpec struct {
	Name          string
	Class         V41TensorClass
	Stage         int
	Shape         []int
	LogicalShape  []int
	AllowedDtypes []string
	PackedWeight  bool
	ScaleShape    []int
}

// V41DSparkInventory is separate from V41Inventory. In particular the target's
// expert count and quantization do not determine the draft's storage format.
// The original bundled DSpark experts are MXFP4 (E2M1/I8, E8M0 per 32 K), even
// in NVIDIA's NVFP4 target export. No NVFP4 draft representation is admitted.
type V41DSparkInventory struct {
	byName map[string]V41DSparkTensorSpec
	names  []string
}

// DeepSeekV41DSparkInventory describes the three pretrained mtp.* stages using
// the original checkpoint namespace. The adapter shares target embed/head at
// runtime; it must not invent a separate Qwen-style verifier head here.
//
// Architecture sources (MIT model/reference, Apache-2.0 vLLM):
// https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
// https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py
// https://github.com/vllm-project/vllm/blob/763e3f7c911308fa609fb9437642929efe955798/vllm/models/deepseek_v41/nvidia/dspark.py
// The pinned FAK artifact identity remains DeepSeekV41FlashRevision. Individual
// artifact headers/hashes must still be captured by the hardware intake receipt.
func DeepSeekV41DSparkInventory(c Config) (*V41DSparkInventory, error) {
	if !c.IsDeepSeekV41() || c.DeepSeekV41 == nil || c.DeepSeekV41.DSpark == nil {
		return nil, fmt.Errorf("%w: V4.1 DSpark metadata is required", ErrV41DSparkArtifact)
	}
	d := c.DeepSeekV41.DSpark
	if d.NextNPredictLayers != 3 || d.BlockSize != 5 || c.DeepSeekV41.HCMult != 4 ||
		len(d.TargetLayerIDs) == 0 || len(d.TargetLayerIDs) > c.NumLayers ||
		d.NumRoutedExperts < 1 || d.NumRoutedExperts > 128 ||
		d.NumExpertsPerTok < 1 || d.NumExpertsPerTok > d.NumRoutedExperts ||
		d.MarkovRank < 1 || d.MarkovRank > c.HiddenSize ||
		d.NoiseTokenID < 0 || d.NoiseTokenID >= c.VocabSize {
		return nil, fmt.Errorf("%w: unsupported DSpark geometry", ErrV41DSparkArtifact)
	}
	seen := map[int]bool{}
	for _, layer := range d.TargetLayerIDs {
		if layer < 0 || layer >= c.NumLayers || seen[layer] {
			return nil, fmt.Errorf("%w: invalid target tap %d", ErrV41DSparkArtifact, layer)
		}
		seen[layer] = true
	}
	for _, axis := range []int{c.HiddenSize, c.NumHeads, c.HeadDim, c.QLoraRank, c.OLoraRank, c.OGroups, c.MoEIntermediateSize, c.VocabSize} {
		if axis <= 0 {
			return nil, fmt.Errorf("%w: non-positive tensor geometry", ErrV41DSparkArtifact)
		}
	}
	if c.HiddenSize%32 != 0 || c.MoEIntermediateSize%32 != 0 || c.NumHeads%c.OGroups != 0 {
		return nil, fmt.Errorf("%w: incompatible MXFP4/grouped attention geometry", ErrV41DSparkArtifact)
	}
	qWidth, ok1 := checkedShapeProduct(c.NumHeads, c.HeadDim)
	oWidth, ok2 := checkedShapeProduct(c.OGroups, c.OLoraRank)
	mainWidth, ok3 := checkedShapeProduct(c.HiddenSize, len(d.TargetLayerIDs))
	hcWidth, ok4 := checkedShapeProduct(c.HiddenSize, 4)
	if !ok1 || !ok2 || !ok3 || !ok4 || c.HiddenSize > int(^uint(0)>>1)-d.MarkovRank {
		return nil, fmt.Errorf("%w: tensor geometry overflow", ErrV41DSparkArtifact)
	}
	inv := &V41DSparkInventory{byName: map[string]V41DSparkTensorSpec{}}
	add := func(name string, class V41TensorClass, stage int, dtypes []string, shape ...int) error {
		if _, ok := checkedShapeProduct(shape...); !ok {
			return fmt.Errorf("%w: %s shape overflow", ErrV41DSparkArtifact, name)
		}
		inv.byName[name] = V41DSparkTensorSpec{Name: name, Class: class, Stage: stage,
			Shape: shape, LogicalShape: append([]int(nil), shape...), AllowedDtypes: dtypes}
		inv.names = append(inv.names, name)
		return nil
	}
	plain := func(name string, class V41TensorClass, stage int, shape ...int) error {
		return add(name, class, stage, []string{"BF16", "F32"}, shape...)
	}
	dense := func(name string, class V41TensorClass, stage, out, in int) error {
		if err := add(name, class, stage, []string{"F8_E4M3"}, out, in); err != nil {
			return err
		}
		spec := inv.byName[name]
		spec.ScaleShape = []int{(out-1)/32 + 1, (in-1)/32 + 1}
		inv.byName[name] = spec
		return nil
	}
	for stage := 0; stage < 3; stage++ {
		p := "mtp." + strconv.Itoa(stage) + "."
		for _, role := range []struct {
			name string
			o, i int
		}{
			{"attn.wq_a.weight", c.QLoraRank, c.HiddenSize},
			{"attn.wq_b.weight", qWidth, c.QLoraRank},
			{"attn.wkv.weight", c.HeadDim, c.HiddenSize},
			// Stored wo_a projects each group's heads, not the model hidden width.
			{"attn.wo_a.weight", oWidth, qWidth / c.OGroups},
			{"attn.wo_b.weight", c.HiddenSize, oWidth},
		} {
			if err := dense(p+role.name, V41ClassAttentionDense, stage, role.o, role.i); err != nil {
				return nil, err
			}
		}
		for _, role := range []struct {
			name string
			dim  int
		}{
			{"attn_norm.weight", c.HiddenSize}, {"ffn_norm.weight", c.HiddenSize},
			{"attn.q_norm.weight", c.QLoraRank}, {"attn.kv_norm.weight", c.HeadDim},
		} {
			if err := plain(p+role.name, V41ClassNorm, stage, role.dim); err != nil {
				return nil, err
			}
		}
		if err := add(p+"attn.attn_sink", V41ClassAttentionDense, stage, []string{"F32"}, c.NumHeads); err != nil {
			return nil, err
		}
		for _, sublayer := range []string{"attn", "ffn"} {
			for _, role := range []struct {
				name  string
				shape []int
			}{
				{"fn", []int{24, hcWidth}}, {"base", []int{24}}, {"scale", []int{3}},
			} {
				if err := add(p+"hc_"+sublayer+"_"+role.name, V41ClassNorm, stage, []string{"F32"}, role.shape...); err != nil {
					return nil, err
				}
			}
		}
		if err := plain(p+"ffn.gate.weight", V41ClassRouter, stage, d.NumRoutedExperts, c.HiddenSize); err != nil {
			return nil, err
		}
		if err := add(p+"ffn.gate.e_score_correction_bias", V41ClassRouterBias, stage, []string{"F32"}, d.NumRoutedExperts); err != nil {
			return nil, err
		}
		for _, role := range []struct {
			name string
			o, i int
		}{
			{"w1", c.MoEIntermediateSize, c.HiddenSize}, {"w2", c.HiddenSize, c.MoEIntermediateSize},
			{"w3", c.MoEIntermediateSize, c.HiddenSize},
		} {
			if err := dense(p+"ffn.shared_experts."+role.name+".weight", V41ClassSharedExpert, stage, role.o, role.i); err != nil {
				return nil, err
			}
			for expert := 0; expert < d.NumRoutedExperts; expert++ {
				name := p + "ffn.experts." + strconv.Itoa(expert) + "." + role.name + ".weight"
				if err := add(name, V41ClassRoutedExpert, stage, []string{"I8"}, role.o, role.i/2); err != nil {
					return nil, err
				}
				spec := inv.byName[name]
				spec.LogicalShape, spec.PackedWeight = []int{role.o, role.i}, true
				spec.ScaleShape = []int{role.o, role.i / 32}
				inv.byName[name] = spec
			}
		}
	}
	if err := dense("mtp.0.main_proj.weight", V41ClassAttentionDense, 0, c.HiddenSize, mainWidth); err != nil {
		return nil, err
	}
	for _, name := range []string{"mtp.0.main_norm.weight", "mtp.2.norm.weight"} {
		stage := 0
		if strings.HasPrefix(name, "mtp.2.") {
			stage = 2
		}
		if err := plain(name, V41ClassNorm, stage, c.HiddenSize); err != nil {
			return nil, err
		}
	}
	for _, name := range []string{"mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight"} {
		if err := add(name, V41ClassHead, 2, []string{"BF16"}, c.VocabSize, d.MarkovRank); err != nil {
			return nil, err
		}
	}
	if err := add("mtp.2.confidence_head.proj.weight", V41ClassHead, 2, []string{"BF16"}, 1, c.HiddenSize+d.MarkovRank); err != nil {
		return nil, err
	}
	sort.Strings(inv.names)
	return inv, nil
}

// Names returns required canonical weight names, without scale siblings or
// target-owned embedding/head aliases. It does not imply those weights exist.
func (inv *V41DSparkInventory) Names() []string {
	if inv == nil {
		return nil
	}
	return append([]string(nil), inv.names...)
}

// Lookup resolves only a required canonical weight role. Returned slices are
// copied so an intake consumer cannot mutate another consumer's contract.
func (inv *V41DSparkInventory) Lookup(name string) (V41DSparkTensorSpec, bool) {
	if inv == nil {
		return V41DSparkTensorSpec{}, false
	}
	spec, ok := inv.byName[name]
	spec.Shape = append([]int(nil), spec.Shape...)
	spec.LogicalShape = append([]int(nil), spec.LogicalShape...)
	spec.AllowedDtypes = append([]string(nil), spec.AllowedDtypes...)
	spec.ScaleShape = append([]int(nil), spec.ScaleShape...)
	return spec, ok
}

// v41DSparkCanonicalName applies only the explicit original-export aliases
// documented in DeepSeek's convert.py. It never uses suffix/fuzzy matching.
func v41DSparkCanonicalName(name string) (canonical string, draft bool) {
	name = strings.TrimPrefix(name, "model.")
	if !strings.HasPrefix(name, "mtp.") {
		return "", false
	}
	parts := strings.Split(name, ".")
	if len(parts) < 3 {
		return name, true
	}
	stage, err := strconv.Atoi(parts[1])
	if err != nil || stage < 0 || strconv.Itoa(stage) != parts[1] {
		return name, true
	}
	if parts[2] == "self_attn" {
		parts[2] = "attn"
	} else if parts[2] == "mlp" {
		parts[2] = "ffn"
	}
	if parts[len(parts)-1] == "weight_scale_inv" {
		parts[len(parts)-1] = "scale"
	}
	if len(parts) == 5 && parts[2] == "ffn" && parts[3] == "gate" && parts[4] == "bias" {
		parts[4] = "e_score_correction_bias"
	}
	return strings.Join(parts, "."), true
}
