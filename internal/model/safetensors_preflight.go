package model

// safetensors_preflight.go — the header-only preflight plan for sharded and single-file
// safetensors checkpoints (ticket P-23 / #14005). It answers, without reading a single tensor
// byte, what LoadSafetensorsQuantDir would install: which index entries are missing from their
// declared shard, which header tensors no index claims, how the source dtypes census, which
// F8_E4M3/I8 weights lost their scale sibling, which dtypes no host decode path accepts, and —
// per tensor — the loader role that consumes it and the load-time representation that role
// installs. Role routing reuses the loader's own predicates in the loader's own order
// (quantizeNamedTensorsInto), so the plan cannot drift from what a load installs; admission
// rejects a V4.1 config before any role runs, so for V4.1 the plan uses the V4.1 role
// predicates instead (v41DenseFP8RoleForName, parseV41ExpertQuantWeightName).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// SafetensorsPreflight roles, in the loader's own consumption order.
const (
	// SafetensorsRoleDroppedMTP: Flash mtp.* tensors dropped before any pairing.
	SafetensorsRoleDroppedMTP = "dropped-mtp"
	// SafetensorsRoleDroppedHashTable: Flash layers.0-2 tid2eid tables that stay in the
	// indexed source for range reads.
	SafetensorsRoleDroppedHashTable = "dropped-hash-table"
	// SafetensorsRoleDroppedSource: GLM/MiniMax/Qwen3.5 sidecars (vision tower, MTP head
	// without retention) dropped at the source-name gate.
	SafetensorsRoleDroppedSource = "dropped-source"
	// SafetensorsRoleCompressedHead: the compressed-tensors FP8 lm_head pair.
	SafetensorsRoleCompressedHead = "compressed-tensors-head"
	// SafetensorsRoleV4DenseFP8: DeepSeek-V4-Flash dense E4M3 weight + E8M0 .scale pair.
	SafetensorsRoleV4DenseFP8 = "v4-dense-fp8"
	// SafetensorsRoleV41DenseFP8: V4.1 dense E4M3 weight + E8M0 .scale pair.
	SafetensorsRoleV41DenseFP8 = "v41-dense-fp8"
	// SafetensorsRoleV41Expert: V4.1 routed-expert packed I8 (E2M1 nibbles) weight.
	SafetensorsRoleV41Expert = "v41-expert"
	// SafetensorsRoleV41ExpertScale: the F8_E8M0 .scale sibling of a V4.1 routed expert.
	SafetensorsRoleV41ExpertScale = "v41-expert-scale"
	// SafetensorsRoleMXFP4: U8 _blocks + _scales MXFP4 pair.
	SafetensorsRoleMXFP4 = "mxfp4"
	// SafetensorsRoleFP8BlockScale: F8_E4M3 .weight + F32 _scale_inv block-scale pair.
	SafetensorsRoleFP8BlockScale = "fp8-block-scale"
	// SafetensorsRoleSkippedBias: gpt_neox U8 attention biases the loader skips.
	SafetensorsRoleSkippedBias = "skipped-bias"
	// SafetensorsRolePlain: the ordinary decode tail (BF16/F16/F32 or an unpaired dtype).
	SafetensorsRolePlain = "plain"
)

// Load-time representations a role installs.
const (
	// SafetensorsRepNone: the tensor is dropped and never materialized.
	SafetensorsRepNone = "none"
	// SafetensorsRepSourceRange: the payload stays in the indexed source for range reads.
	SafetensorsRepSourceRange = "source-range"
	// SafetensorsRepQ8: a Q8_0 resident tensor is installed at load.
	SafetensorsRepQ8 = "q8_0"
	// SafetensorsRepF32Host: the tensor decodes to f32 and stays host-resident.
	SafetensorsRepF32Host = "f32-host"
	// SafetensorsRepV41Expert: the packed I8+F8_E8M0 pair is preserved verbatim for the
	// device GEMV (no full-checkpoint dequantization).
	SafetensorsRepV41Expert = "packed-i8+f8_e8m0"
)

// SafetensorsPreflightInput names the checkpoint directory and an optional HF config override.
type SafetensorsPreflightInput struct {
	// Dir holds model.safetensors.index.json + shards, or a single model.safetensors.
	Dir string
	// HFConfig, when non-empty, overrides <Dir>/config.json.
	HFConfig []byte
}

// SafetensorsPreflightTensor is one tensor's census row: its source dtype, the loader role
// that consumes it, and the load-time representation that role installs.
type SafetensorsPreflightTensor struct {
	Name           string
	Dtype          string
	Role           string
	Representation string
}

// SafetensorsScaleGap names one F8_E4M3/I8 weight whose scale sibling is incomplete: Sibling
// is the expected name and Dtype the dtype actually found there ("" when absent).
type SafetensorsScaleGap struct {
	Weight  string
	Sibling string
	Dtype   string
}

// SafetensorsPreflight is the header-only view of a quant load. AdmissionErr carries the
// typed admission refusal (nil when admitted); the census always continues past it, so a
// refused config still reports every other seam.
type SafetensorsPreflight struct {
	Config       Config
	AdmissionErr error
	// Missing: index-declared tensors absent from their declared shard's header.
	Missing []string
	// Orphans: header tensors no index entry claims.
	Orphans []string
	// DTypes censuses source dtype strings across every header tensor.
	DTypes map[string]int
	// ScaleGaps: F8_E4M3/I8 weights missing (or mistyping) their scale sibling.
	ScaleGaps []SafetensorsScaleGap
	// HostGaps keys "host_dequant(<dtype>)" to the tensors no host decode path accepts.
	HostGaps map[string][]string
	// RoleErrors keys a tensor name to the loader-role metadata failure a load would hit.
	RoleErrors map[string]string
	// Tensors lists every header tensor once, sorted by name.
	Tensors []SafetensorsPreflightTensor
}

// BuildSafetensorsPreflight runs the header-only plan over dir. The only errors it returns
// are I/O and config-parse failures; everything else lands in the report so one run names
// every seam, matching ggufload.BuildManifestPreflight's aggregate-instead-of-stop contract.
func BuildSafetensorsPreflight(in SafetensorsPreflightInput) (*SafetensorsPreflight, error) {
	raw := in.HFConfig
	if len(raw) == 0 {
		configPath := filepath.Join(in.Dir, "config.json")
		b, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("safetensors: read config %s: %w", configPath, err)
		}
		raw = b
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("safetensors: parse config: %w", err)
	}
	if cfg.ModelType == "" || cfg.HiddenSize <= 0 || cfg.NumLayers <= 0 {
		return nil, fmt.Errorf("safetensors: config is missing model_type/hidden_size/num_hidden_layers")
	}

	rep := &SafetensorsPreflight{
		Config:     cfg,
		DTypes:     map[string]int{},
		HostGaps:   map[string][]string{},
		RoleErrors: map[string]string{},
	}
	if _, err := admitQuantLoad(cfg, nil); err != nil {
		rep.AdmissionErr = err
	}

	idxPath := filepath.Join(in.Dir, "model.safetensors.index.json")
	var (
		shards    []string
		weightMap map[string]string
	)
	if _, statErr := os.Stat(idxPath); statErr == nil {
		var err error
		shards, weightMap, err = safetensorsIndexShards(idxPath)
		if err != nil {
			return nil, err
		}
	} else {
		single := filepath.Join(in.Dir, "model.safetensors")
		if _, err := os.Stat(single); err != nil {
			return nil, fmt.Errorf("safetensors: %s has neither model.safetensors.index.json nor model.safetensors", in.Dir)
		}
		shards = []string{"model.safetensors"}
	}

	type shardHeader struct {
		name string
		hdr  map[string]json.RawMessage
	}
	headers := make([]shardHeader, 0, len(shards))
	byShard := map[string]map[string]json.RawMessage{}
	for _, shard := range shards {
		sf, err := openSafetensorsFile(filepath.Join(in.Dir, shard))
		if err != nil {
			return nil, err
		}
		hdr := sf.hdr
		_ = sf.Close()
		headers = append(headers, shardHeader{name: shard, hdr: hdr})
		byShard[shard] = hdr
	}

	byName := map[string]stEntry{}
	seen := map[string]bool{}
	for _, h := range headers {
		for name, rawMsg := range h.hdr {
			if seen[name] {
				continue
			}
			var e stEntry
			if err := json.Unmarshal(rawMsg, &e); err != nil {
				return nil, fmt.Errorf("safetensors: entry %s in %s: %w", name, h.name, err)
			}
			seen[name] = true
			byName[name] = e
			rep.DTypes[e.Dtype]++
		}
	}

	if weightMap != nil {
		for name, shard := range weightMap {
			if _, ok := byShard[shard][name]; !ok {
				rep.Missing = append(rep.Missing, name)
			}
		}
		sort.Strings(rep.Missing)
		for name := range byName {
			if _, ok := weightMap[name]; !ok {
				rep.Orphans = append(rep.Orphans, name)
			}
		}
		sort.Strings(rep.Orphans)
	}

	names := make([]string, 0, len(byName))
	for n := range byName {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, name := range names {
		e := byName[name]
		t := SafetensorsPreflightTensor{Name: name, Dtype: e.Dtype}
		t.Role, t.Representation = rep.routeRole(name, e, byName)
		rep.Tensors = append(rep.Tensors, t)
	}
	return rep, nil
}

// routeRole assigns one tensor its loader role and load-time representation, walking the
// same decision points quantizeNamedTensorsInto walks, in the same order. It records host
// gaps, scale gaps and role errors on the receiver as it goes.
func (rep *SafetensorsPreflight) routeRole(name string, e stEntry, byName map[string]stEntry) (role, representation string) {
	cfg := rep.Config
	switch {
	case isDeepSeekV4FlashProfile(cfg) && strings.HasPrefix(name, "mtp."):
		return SafetensorsRoleDroppedMTP, SafetensorsRepNone
	case isDeepSeekV4FlashProfile(cfg) && isDeepSeekV4FlashLazyHashTable(name):
		return SafetensorsRoleDroppedHashTable, SafetensorsRepSourceRange
	}

	if cfg.IsDeepSeekV41() {
		if stem, projection, err := parseV41ExpertQuantWeightName(name); err == nil {
			rep.checkV41ExpertPair(name, stem+projection+".scale", projection, e, byName)
			return SafetensorsRoleV41Expert, SafetensorsRepV41Expert
		}
		if strings.HasSuffix(name, ".scale") {
			owner := strings.TrimSuffix(name, ".scale") + ".weight"
			if _, _, err := parseV41ExpertQuantWeightName(owner); err == nil {
				return SafetensorsRoleV41ExpertScale, SafetensorsRepV41Expert
			}
		}
		if r, ok, err := v41DenseFP8RoleForName(name, cfg); err != nil {
			rep.RoleErrors[name] = err.Error()
			return SafetensorsRoleV41DenseFP8, ""
		} else if ok {
			rep.checkV41DensePair(name, r, e, byName)
			return SafetensorsRoleV41DenseFP8, SafetensorsRepQ8
		}
	}

	if enabled, err := compressedTensorsLMHeadEnabled(cfg); err == nil && enabled && strings.HasPrefix(name, "lm_head.") {
		switch name {
		case "lm_head.weight", "lm_head.weight_scale":
			return SafetensorsRoleCompressedHead, SafetensorsRepQ8
		default:
			rep.RoleErrors[name] = "unsupported compressed-tensors head companion " + name
			return SafetensorsRoleCompressedHead, ""
		}
	}

	if r, ok, err := v4DenseFP8RoleForName(name, cfg); err != nil {
		rep.RoleErrors[name] = err.Error()
		return SafetensorsRoleV4DenseFP8, ""
	} else if ok {
		rep.checkV4DensePair(name, r, e, byName)
		return SafetensorsRoleV4DenseFP8, SafetensorsRepQ8
	}

	if strings.HasSuffix(name, "_blocks") {
		base := strings.TrimSuffix(name, "_blocks")
		if _, ok := byName[base+"_scales"]; ok {
			representation := SafetensorsRepF32Host
			if DecodedFloatGoesToQ8(cfg, base, e.Shape) {
				representation = SafetensorsRepQ8
			}
			return SafetensorsRoleMXFP4, representation
		}
		// An unpaired _blocks falls through to the plain decoder, which rejects U8:
		// record the host gap exactly as the load would surface it.
		rep.addHostGap(name, e.Dtype)
		return SafetensorsRolePlain, ""
	}

	if strings.HasSuffix(name, ".weight") && e.Dtype == "F8_E4M3" {
		scaleName := name + "_scale_inv"
		if _, ok := byName[scaleName]; ok {
			// The direct and decoded install paths differ only in speed, not in the
			// installed representation: both land in q8w for a rank-2 matmul projection.
			canonical, keep := quantSourceTensorName(cfg, name)
			representation := SafetensorsRepF32Host
			if keep && isQuantWeight(canonical) && len(e.Shape) == 2 {
				representation = SafetensorsRepQ8
			}
			return SafetensorsRoleFP8BlockScale, representation
		}
		rep.ScaleGaps = append(rep.ScaleGaps, SafetensorsScaleGap{Weight: name, Sibling: scaleName})
		rep.addHostGap(name, e.Dtype)
		return SafetensorsRolePlain, ""
	}

	if skipSafetensorsTensor(name, e) {
		return SafetensorsRoleSkippedBias, SafetensorsRepNone
	}

	switch e.Dtype {
	case "BF16", "F16", "F32":
		resolved, keep := quantSourceTensorNameWithRetention(cfg, name, false)
		if !keep {
			return SafetensorsRoleDroppedSource, SafetensorsRepNone
		}
		representation := SafetensorsRepF32Host
		if DecodedFloatGoesToQ8(cfg, resolved, e.Shape) {
			representation = SafetensorsRepQ8
		}
		return SafetensorsRolePlain, representation
	default:
		rep.addHostGap(name, e.Dtype)
		return SafetensorsRolePlain, ""
	}
}

func (rep *SafetensorsPreflight) addHostGap(name, dtype string) {
	key := "host_dequant(" + dtype + ")"
	rep.HostGaps[key] = append(rep.HostGaps[key], name)
}

// checkV41ExpertPair validates one V4.1 routed-expert (weight, scale) pair header-only:
// the scale sibling must exist, be F8_E8M0, and both shapes must match the pinned profile.
func (rep *SafetensorsPreflight) checkV41ExpertPair(weight, scaleName, projection string, weightEntry stEntry, byName map[string]stEntry) {
	if weightEntry.Dtype != "I8" {
		rep.RoleErrors[weight] = fmt.Sprintf("V4.1 %s dtype %q, want I8", weight, weightEntry.Dtype)
		return
	}
	scale, ok := byName[scaleName]
	if !ok {
		rep.ScaleGaps = append(rep.ScaleGaps, SafetensorsScaleGap{Weight: weight, Sibling: scaleName})
		return
	}
	if scale.Dtype != "F8_E8M0" {
		rep.ScaleGaps = append(rep.ScaleGaps, SafetensorsScaleGap{Weight: weight, Sibling: scaleName, Dtype: scale.Dtype})
		return
	}
	if _, ok := selectV41ExpertQuantSpec(projection, weightEntry.Shape, scale.Shape); !ok {
		rep.RoleErrors[weight] = fmt.Sprintf("V4.1 %s shape %v with %s shape %v does not match a supported V4.1 profile", weight, weightEntry.Shape, scaleName, scale.Shape)
	}
}

// checkV41DensePair validates one V4.1 dense FP8 pair header-only. The weight visit pins the
// published geometry and the F8_E8M0 .scale sibling; the scale visit only pins the sibling
// dtype, since the owner weight's walk carries the shape check.
func (rep *SafetensorsPreflight) checkV41DensePair(name string, r v41DenseFP8Role, e stEntry, byName map[string]stEntry) {
	if strings.HasSuffix(name, ".scale") {
		if e.Dtype != "F8_E8M0" {
			rep.RoleErrors[name] = fmt.Sprintf("V4.1 dense scale %s dtype %q, want F8_E8M0", name, e.Dtype)
		}
		return
	}
	if len(e.Shape) != 2 || e.Shape[0] != r.shape[0] || e.Shape[1] != r.shape[1] {
		rep.RoleErrors[name] = fmt.Sprintf("V4.1 dense tensor %s shape %v, want published shape %v", name, e.Shape, r.shape[:])
		return
	}
	if strings.HasSuffix(name, ".weight") {
		want := "F8_E4M3"
		if e.Dtype != want {
			rep.RoleErrors[name] = fmt.Sprintf("V4.1 dense weight %s dtype %q, want %s", name, e.Dtype, want)
			return
		}
		scaleName := strings.TrimSuffix(name, ".weight") + ".scale"
		scale, ok := byName[scaleName]
		if !ok {
			rep.ScaleGaps = append(rep.ScaleGaps, SafetensorsScaleGap{Weight: name, Sibling: scaleName})
			return
		}
		if scale.Dtype != "F8_E8M0" {
			rep.ScaleGaps = append(rep.ScaleGaps, SafetensorsScaleGap{Weight: name, Sibling: scaleName, Dtype: scale.Dtype})
		}
	}
}

// checkV4DensePair is the DeepSeek-V4-Flash twin of checkV41DensePair.
func (rep *SafetensorsPreflight) checkV4DensePair(name string, r v4DenseFP8Role, e stEntry, byName map[string]stEntry) {
	if strings.HasSuffix(name, ".scale") {
		if e.Dtype != "F8_E8M0" {
			rep.RoleErrors[name] = fmt.Sprintf("V4 dense scale %s dtype %q, want F8_E8M0", name, e.Dtype)
		}
		return
	}
	if len(e.Shape) != 2 || e.Shape[0] != r.shape[0] || e.Shape[1] != r.shape[1] {
		rep.RoleErrors[name] = fmt.Sprintf("V4 dense tensor %s shape %v, want published shape %v", name, e.Shape, r.shape[:])
		return
	}
	if strings.HasSuffix(name, ".weight") {
		if e.Dtype != "F8_E4M3" {
			rep.RoleErrors[name] = fmt.Sprintf("V4 dense weight %s dtype %q, want F8_E4M3", name, e.Dtype)
			return
		}
		scaleName := strings.TrimSuffix(name, ".weight") + ".scale"
		scale, ok := byName[scaleName]
		if !ok {
			rep.ScaleGaps = append(rep.ScaleGaps, SafetensorsScaleGap{Weight: name, Sibling: scaleName})
			return
		}
		if scale.Dtype != "F8_E8M0" {
			rep.ScaleGaps = append(rep.ScaleGaps, SafetensorsScaleGap{Weight: name, Sibling: scaleName, Dtype: scale.Dtype})
		}
	}
}

// DecodedFloatGoesToQ8 reports whether a decoded float-source tensor (BF16/F16/F32, or the
// decoded form of a paired MXFP4/FP8 tensor named by its base) becomes a Q8_0 resident weight
// under the quant load path rather than staying f32 on the host. It mirrors the decision
// points of quantizeDecodedFloatTensorIntoWithRetention header-only: the fused qkv/gate_up
// split installs q8w parts, the gpt-oss and mixtral MoE source forms install aliased q8w
// tensors, and anything else falls through to the plain-matmul gate. canon is the canonical
// (post-remap) tensor name; shape is the source shape as declared in the shard header.
func DecodedFloatGoesToQ8(cfg Config, canon string, shape []int) bool {
	layer, suffix, ok := parseLayerTensorSuffix(canon)
	if !ok || layer < 0 || layer >= cfg.NumLayers {
		return isQuantWeight(canon) && len(shape) == 2
	}
	switch suffix {
	case suffixQKVProj, suffixGateUpProj:
		return true
	}
	if cfg.NumExperts > 0 {
		switch suffix {
		case "mlp.router.weight",
			"mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj.weight",
			"mlp.experts.down_proj", "mlp.experts.down_proj.weight",
			"block_sparse_moe.gate.weight":
			return true
		}
		if rest, cut := strings.CutPrefix(suffix, "block_sparse_moe.experts."); cut {
			expertText, weightSuffix, dot := strings.Cut(rest, ".")
			if dot {
				if _, err := strconv.Atoi(expertText); err == nil {
					switch weightSuffix {
					case "w1.weight", "w2.weight", "w3.weight":
						return true
					}
				}
			}
		}
	}
	return isQuantWeight(canon) && len(shape) == 2
}
