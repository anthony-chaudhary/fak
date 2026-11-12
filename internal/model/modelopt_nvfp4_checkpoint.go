package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrModelOptNVFP4Config reports a ModelOpt quantization declaration fak does
// not admit: a non-NVFP4/non-MIXED_PRECISION algorithm, a group outside the
// NVFP4 (4-bit group-16) / MXFP8 (8-bit group-32) envelope, or a mixed group
// whose weights and input_activations specs disagree.
var ErrModelOptNVFP4Config = errors.New("model: unsupported ModelOpt quantization config")

// ErrModelOptMixedGroupUnsupported reports a detected ModelOpt mixed-precision
// (NVFP4 + MXFP8) checkpoint: fak detects and validates the declaration, but
// mixed-precision execution is not wired yet (fak#14364, fak#14365), so the
// loader refuses with this error before any tensor is quantized or resident.
var ErrModelOptMixedGroupUnsupported = errors.New("model: ModelOpt mixed-precision group not executable (execution is not wired yet: fak#14364, fak#14365)")

// ErrModelOptNVFP4Checkpoint reports a sharded ModelOpt NVFP4 checkpoint whose
// layout the loader refuses (split triples, duplicate canonical names, missing
// shards).
var ErrModelOptNVFP4Checkpoint = errors.New("model: invalid ModelOpt NVFP4 checkpoint")

// ModelOptMXFP8BlockSize is the block size of a ModelOpt MXFP8 group (8-bit
// group-32, E8M0 scales), the second kind admitted in a MIXED_PRECISION
// declaration alongside NVFP4's compute.NVFP4BlockSize.
const ModelOptMXFP8BlockSize = 32

// ModelOptMixedGroupKind names the quantization format of one ModelOpt
// mixed-precision config group.
type ModelOptMixedGroupKind string

const (
	ModelOptMixedGroupNVFP4 ModelOptMixedGroupKind = "NVFP4"
	ModelOptMixedGroupMXFP8 ModelOptMixedGroupKind = "MXFP8"
)

// ModelOptMixedQuantSpec is one weights or input_activations quant spec.
type ModelOptMixedQuantSpec struct {
	NumBits   int
	GroupSize int
	Dynamic   bool
}

// ModelOptMixedGroupSpec is one ModelOpt mixed-precision config group: the
// format it quantizes (kind), how weights and optionally activations are
// quantized, and how many layer targets it covers.
type ModelOptMixedGroupSpec struct {
	Name    string
	Kind    ModelOptMixedGroupKind
	Weights ModelOptMixedQuantSpec
	Inputs  *ModelOptMixedQuantSpec // nil when the group declares no input_activations
	Targets int
}

// ModelOptMixedConfig is the parsed summary of a ModelOpt MIXED_PRECISION
// declaration (config.json quant_method "modelopt_mixed" config_groups, or
// hf_quant_config.json quant_algo "MIXED_PRECISION").
type ModelOptMixedConfig struct {
	QuantAlgo       string
	Groups          []ModelOptMixedGroupSpec // sorted by group name
	QuantizedLayers int
}

// ModelOptNVFP4Config is the admitted subset of a ModelOpt NVFP4 declaration.
// A mixed NVFP4+MXFP8 declaration sets Mixed (and leaves GroupSize 0): fak
// detects and validates it, but the loader refuses execution until
// fak#14364/#14365 wire mixed-precision kernels.
type ModelOptNVFP4Config struct {
	QuantAlgo      string
	GroupSize      int
	ExcludeModules []string
	Mixed          *ModelOptMixedConfig // non-nil for MIXED_PRECISION declarations
}

type modelOptQuantDecl struct {
	QuantMethod    string   `json:"quant_method"`
	QuantAlgo      string   `json:"quant_algo"`
	GroupSize      int      `json:"group_size"`
	ExcludeModules []string `json:"exclude_modules"`
}

// modelOptQuantSpecDecl is one weights/input_activations spec inside a ModelOpt
// mixed-precision group or quantized_layers entry.
type modelOptQuantSpecDecl struct {
	NumBits   int    `json:"num_bits"`
	GroupSize int    `json:"group_size"`
	Dynamic   bool   `json:"dynamic"`
	Type      string `json:"type"`
	Strategy  string `json:"strategy"`
}

// modelOptMixedGroupDecl is one config_groups entry of a ModelOpt
// MIXED_PRECISION declaration (nvidia/Kimi-K3.1-Pro-NVFP4 shape).
type modelOptMixedGroupDecl struct {
	Weights          *modelOptQuantSpecDecl `json:"weights"`
	InputActivations *modelOptQuantSpecDecl `json:"input_activations"`
	Targets          []string               `json:"targets"`
}

// modelOptLayerQuantDecl is one quantized_layers entry of a MIXED_PRECISION
// declaration.
type modelOptLayerQuantDecl struct {
	Weights          *modelOptQuantSpecDecl `json:"weights"`
	InputActivations *modelOptQuantSpecDecl `json:"input_activations"`
}

// modelOptMixedDecl is the full ModelOpt quantization declaration superset: the
// plain NVFP4 fields (embedded, so plain checkpoints parse unchanged) plus the
// mixed-precision config_groups/quantized_layers/ignore payload.
type modelOptMixedDecl struct {
	modelOptQuantDecl
	ConfigGroups    map[string]modelOptMixedGroupDecl `json:"config_groups"`
	QuantizedLayers map[string]modelOptLayerQuantDecl `json:"quantized_layers"`
	Ignore          []string                          `json:"ignore"`
}

// ReadModelOptNVFP4Config detects a ModelOpt quantization declaration in dir. It
// reads both forms vLLM's ModelOpt quantization config accepts: config.json
// quantization_config {"quant_method":"modelopt","quant_algo":"NVFP4"} and the
// hf_quant_config.json {"producer":{"name":"modelopt"},"quantization":{...}}.
// A mixed NVFP4+MXFP8 declaration (config.json quant_method "modelopt_mixed"
// with config_groups, or quant_algo "MIXED_PRECISION") also detects: ok is true
// with Mixed summarizing the config groups. ok is false when neither file
// declares ModelOpt quantization.
func ReadModelOptNVFP4Config(dir string) (cfg ModelOptNVFP4Config, ok bool, err error) {
	var decl *modelOptMixedDecl
	if b, rerr := os.ReadFile(filepath.Join(dir, "config.json")); rerr == nil {
		var doc struct {
			QuantizationConfig *modelOptMixedDecl `json:"quantization_config"`
		}
		if err := json.Unmarshal(b, &doc); err != nil {
			return cfg, false, fmt.Errorf("modelopt config.json: %w", err)
		}
		if q := doc.QuantizationConfig; q != nil &&
			(strings.EqualFold(q.QuantMethod, "modelopt") || strings.EqualFold(q.QuantMethod, "modelopt_mixed")) {
			decl = q
		}
	}
	if b, rerr := os.ReadFile(filepath.Join(dir, "hf_quant_config.json")); rerr == nil {
		var doc struct {
			Producer struct {
				Name string `json:"name"`
			} `json:"producer"`
			Quantization *modelOptMixedDecl `json:"quantization"`
		}
		if err := json.Unmarshal(b, &doc); err != nil {
			return cfg, false, fmt.Errorf("modelopt hf_quant_config.json: %w", err)
		}
		if doc.Quantization != nil && strings.EqualFold(doc.Producer.Name, "modelopt") {
			if decl != nil && !strings.EqualFold(decl.QuantAlgo, doc.Quantization.QuantAlgo) {
				return cfg, false, fmt.Errorf("%w: config.json quant_algo %q disagrees with hf_quant_config.json %q",
					ErrModelOptNVFP4Config, decl.QuantAlgo, doc.Quantization.QuantAlgo)
			}
			if decl == nil {
				decl = doc.Quantization
			}
		}
	}
	if decl == nil {
		return cfg, false, nil
	}
	if strings.EqualFold(decl.QuantAlgo, "MIXED_PRECISION") || decl.ConfigGroups != nil {
		return readModelOptMixedConfig(decl)
	}
	if !strings.EqualFold(decl.QuantAlgo, "NVFP4") {
		return cfg, true, fmt.Errorf("%w: quant_algo %q (only NVFP4 is admitted)", ErrModelOptNVFP4Config, decl.QuantAlgo)
	}
	if decl.GroupSize != 0 && decl.GroupSize != compute.NVFP4BlockSize {
		return cfg, true, fmt.Errorf("%w: group_size %d (NVFP4 requires %d)", ErrModelOptNVFP4Config, decl.GroupSize, compute.NVFP4BlockSize)
	}
	return ModelOptNVFP4Config{
		QuantAlgo:      "NVFP4",
		GroupSize:      compute.NVFP4BlockSize,
		ExcludeModules: append([]string(nil), decl.ExcludeModules...),
	}, true, nil
}

// readModelOptMixedConfig summarizes a MIXED_PRECISION declaration. Groups come
// from config.json config_groups; an hf-only mixed checkpoint (no config.json
// config_groups) still detects, with empty Groups, and is refused at load.
// The plain top-level GroupSize check does not apply: mixed groups carry their
// own per-group sizes (NVFP4 16, MXFP8 32).
func readModelOptMixedConfig(decl *modelOptMixedDecl) (ModelOptNVFP4Config, bool, error) {
	groups, err := parseModelOptMixedGroups(decl.ConfigGroups)
	if err != nil {
		return ModelOptNVFP4Config{}, true, err
	}
	return ModelOptNVFP4Config{
		QuantAlgo:      "MIXED_PRECISION",
		ExcludeModules: append([]string(nil), decl.ExcludeModules...),
		Mixed: &ModelOptMixedConfig{
			QuantAlgo:       "MIXED_PRECISION",
			Groups:          groups,
			QuantizedLayers: len(decl.QuantizedLayers),
		},
	}, true, nil
}

// modelOptMixedGroupKind maps a ModelOpt quant spec onto the mixed-precision
// group kind fak admits: 4-bit group-16 is NVFP4, 8-bit group-32 is MXFP8;
// anything else (including a non-float quant type) is refused.
func modelOptMixedGroupKind(spec modelOptQuantSpecDecl) (ModelOptMixedGroupKind, error) {
	if spec.Type != "" && !strings.EqualFold(spec.Type, "float") {
		return "", fmt.Errorf("%w: quant type %q (only float is admitted)", ErrModelOptNVFP4Config, spec.Type)
	}
	switch {
	case spec.NumBits == 4 && spec.GroupSize == compute.NVFP4BlockSize:
		return ModelOptMixedGroupNVFP4, nil
	case spec.NumBits == 8 && spec.GroupSize == ModelOptMXFP8BlockSize:
		return ModelOptMixedGroupMXFP8, nil
	default:
		return "", fmt.Errorf("%w: %d-bit group-%d spec (only 4-bit group-%d NVFP4 and 8-bit group-%d MXFP8 are admitted)",
			ErrModelOptNVFP4Config, spec.NumBits, spec.GroupSize, compute.NVFP4BlockSize, ModelOptMXFP8BlockSize)
	}
}

// parseModelOptMixedGroups validates config_groups into a per-group summary
// sorted by group name. A group's kind comes from its weights spec;
// input_activations, when present, must resolve to the same kind.
func parseModelOptMixedGroups(groups map[string]modelOptMixedGroupDecl) ([]ModelOptMixedGroupSpec, error) {
	names := make([]string, 0, len(groups))
	for name := range groups {
		names = append(names, name)
	}
	sort.Strings(names)
	specs := make([]ModelOptMixedGroupSpec, 0, len(names))
	for _, name := range names {
		g := groups[name]
		if g.Weights == nil {
			return nil, fmt.Errorf("%w: config group %q declares no weights spec", ErrModelOptNVFP4Config, name)
		}
		kind, err := modelOptMixedGroupKind(*g.Weights)
		if err != nil {
			return nil, fmt.Errorf("config group %q weights: %w", name, err)
		}
		var inputs *ModelOptMixedQuantSpec
		if g.InputActivations != nil {
			inKind, err := modelOptMixedGroupKind(*g.InputActivations)
			if err != nil {
				return nil, fmt.Errorf("config group %q input_activations: %w", name, err)
			}
			if inKind != kind {
				return nil, fmt.Errorf("%w: config group %q mixes %s weights with %s input_activations", ErrModelOptNVFP4Config, name, kind, inKind)
			}
			inputs = &ModelOptMixedQuantSpec{NumBits: g.InputActivations.NumBits, GroupSize: g.InputActivations.GroupSize, Dynamic: g.InputActivations.Dynamic}
		}
		specs = append(specs, ModelOptMixedGroupSpec{
			Name:    name,
			Kind:    kind,
			Weights: ModelOptMixedQuantSpec{NumBits: g.Weights.NumBits, GroupSize: g.Weights.GroupSize, Dynamic: g.Weights.Dynamic},
			Inputs:  inputs,
			Targets: len(g.Targets),
		})
	}
	return specs, nil
}

// modelOptMixedKindLabel renders one group kind with its bit/group envelope.
func modelOptMixedKindLabel(kind ModelOptMixedGroupKind) string {
	switch kind {
	case ModelOptMixedGroupNVFP4:
		return fmt.Sprintf("%s (4-bit group-%d)", kind, compute.NVFP4BlockSize)
	case ModelOptMixedGroupMXFP8:
		return fmt.Sprintf("%s (8-bit group-%d)", kind, ModelOptMXFP8BlockSize)
	default:
		return string(kind)
	}
}

// refuseModelOptMixed builds the typed load-time refusal for a detected ModelOpt
// mixed-precision checkpoint, naming every group kind execution would need.
// Detection and validation succeeded; mixed-precision execution is not wired
// yet (fak#14364, fak#14365). Only the two admitted kinds can appear: the
// parser refuses every other envelope before a summary exists.
func refuseModelOptMixed(mixed *ModelOptMixedConfig) error {
	if len(mixed.Groups) == 0 {
		return fmt.Errorf("%w: MIXED_PRECISION declaration with no parsable config groups", ErrModelOptMixedGroupUnsupported)
	}
	type kindTally struct{ groups, targets int }
	tally := map[ModelOptMixedGroupKind]*kindTally{}
	for _, g := range mixed.Groups {
		t := tally[g.Kind]
		if t == nil {
			t = &kindTally{}
			tally[g.Kind] = t
		}
		t.groups++
		t.targets += g.Targets
	}
	parts := make([]string, 0, len(tally))
	for _, kind := range []ModelOptMixedGroupKind{ModelOptMixedGroupNVFP4, ModelOptMixedGroupMXFP8} {
		if t := tally[kind]; t != nil {
			plural := "groups"
			if t.groups == 1 {
				plural = "group"
			}
			parts = append(parts, fmt.Sprintf("%d %s %s over %d targets", t.groups, modelOptMixedKindLabel(kind), plural, t.targets))
		}
	}
	return fmt.Errorf("%w: %s", ErrModelOptMixedGroupUnsupported, strings.Join(parts, " + "))
}

// qwen35CanonicalWeightName maps a checkpoint tensor name onto the canonical
// name the forward resolves, mirroring materializeQwen35Tensors for tensors that
// never enter the f32 manifest. keep is false for tensors a text load drops.
func qwen35CanonicalWeightName(cfg Config, name string) (canonical string, keep bool) {
	if !cfg.IsQwen35Hybrid() {
		return name, true
	}
	const lm = "model.language_model."
	switch {
	case strings.HasPrefix(name, lm):
		name = "model." + name[len(lm):]
	case strings.HasPrefix(name, "model.visual."):
		if !RetainVision {
			return "", false
		}
	case strings.HasPrefix(name, "mtp."):
		if !RetainMTP {
			return "", false
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		p := layerPrefix(l)
		switch name {
		case p + "self_attn.qkv_proj.weight":
			return p + "linear_attn.in_proj_qkv.weight", true
		case p + "self_attn.q_gate_proj.weight":
			return p + "linear_attn.in_proj_z.weight", true
		}
	}
	return name, true
}

// ModelOptNVFP4Checkpoint is a sharded ModelOpt NVFP4 checkpoint directory
// exposed through compute.WeightSource under canonical (Qwen3.5/3.8-normalized)
// tensor names. Each packed triple must live in one shard.
type ModelOptNVFP4Checkpoint struct {
	sources map[string]*ModelOptNVFP4Source // shard file -> source
	byName  map[string]modelOptNVFP4Ref     // canonical weight name -> location
	aux     map[string]bool                 // checkpoint names owned by NVFP4 triples
}

type modelOptNVFP4Ref struct {
	shard string
	base  string // checkpoint base name (no ".weight")
}

// OpenModelOptNVFP4Checkpoint opens dir's model.safetensors.index.json shards
// (or a single model.safetensors) and indexes every NVFP4 triple by canonical
// name. Tensors are read lazily by Weight.
func OpenModelOptNVFP4Checkpoint(dir string, cfg Config, be compute.Backend) (*ModelOptNVFP4Checkpoint, error) {
	shards, err := modelOptShardFiles(dir)
	if err != nil {
		return nil, err
	}
	ck := &ModelOptNVFP4Checkpoint{
		sources: map[string]*ModelOptNVFP4Source{},
		byName:  map[string]modelOptNVFP4Ref{},
		aux:     map[string]bool{},
	}
	present := map[string]string{} // checkpoint tensor name -> shard
	for _, shard := range shards {
		src, err := OpenModelOptNVFP4Source(filepath.Join(dir, shard), be)
		if err != nil {
			ck.Close()
			return nil, fmt.Errorf("modelopt shard %s: %w", shard, err)
		}
		ck.sources[shard] = src
		for name := range src.file.hdr {
			if name != "__metadata__" {
				present[name] = shard
			}
		}
	}
	bases := map[string]bool{}
	for name := range present {
		for _, suffix := range []string{".weight_scale_2", ".weights_scaling_factor_2"} {
			if strings.HasSuffix(name, suffix) {
				bases[strings.TrimSuffix(name, suffix)] = true
			}
		}
	}
	for base := range bases {
		siblings := []string{base + ".weight"}
		for _, suffix := range []string{".weight_scale", ".weight_scale_2", ".weights_scaling_factor", ".weights_scaling_factor_2", ".input_scale"} {
			if _, ok := present[base+suffix]; ok {
				siblings = append(siblings, base+suffix)
			}
		}
		shard, ok := present[base+".weight"]
		if !ok {
			ck.Close()
			return nil, fmt.Errorf("%w: %s has scales but no packed weight", ErrModelOptNVFP4Checkpoint, base)
		}
		for _, s := range siblings {
			if present[s] != shard {
				ck.Close()
				return nil, fmt.Errorf("%w: %s is split across shards %s and %s", ErrModelOptNVFP4Checkpoint, base, shard, present[s])
			}
			ck.aux[s] = true
		}
		canonical, keep := qwen35CanonicalWeightName(cfg, base+".weight")
		if !keep {
			continue
		}
		if prior, dup := ck.byName[canonical]; dup {
			ck.Close()
			return nil, fmt.Errorf("%w: %s and %s both map to %s", ErrModelOptNVFP4Checkpoint, prior.base, base, canonical)
		}
		ck.byName[canonical] = modelOptNVFP4Ref{shard: shard, base: base}
	}
	return ck, nil
}

func modelOptShardFiles(dir string) ([]string, error) {
	idxPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(idxPath); err != nil {
		if _, serr := os.Stat(filepath.Join(dir, "model.safetensors")); serr != nil {
			return nil, fmt.Errorf("%w: %s has neither model.safetensors.index.json nor model.safetensors", ErrModelOptNVFP4Checkpoint, dir)
		}
		return []string{"model.safetensors"}, nil
	}
	shards, _, err := safetensorsIndexShards(idxPath)
	if err != nil {
		return nil, err
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("%w: empty weight_map in %s", ErrModelOptNVFP4Checkpoint, idxPath)
	}
	return shards, nil
}

// Names returns the canonical NVFP4 weight names in sorted order.
func (c *ModelOptNVFP4Checkpoint) Names() []string {
	names := make([]string, 0, len(c.byName))
	for name := range c.byName {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// CheckpointBase returns the checkpoint base name behind a canonical weight.
func (c *ModelOptNVFP4Checkpoint) CheckpointBase(canonical string) (string, bool) {
	ref, ok := c.byName[canonical]
	return ref.base, ok
}

// Weight implements compute.WeightSource over canonical names.
func (c *ModelOptNVFP4Checkpoint) Weight(name string, want compute.Dtype) (compute.Tensor, error) {
	ref, ok := c.byName[name]
	if !ok {
		return compute.Tensor{}, fmt.Errorf("%w: no NVFP4 tensor %s", ErrModelOptNVFP4Checkpoint, name)
	}
	return c.sources[ref.shard].Weight(ref.base, want)
}

// ownsCheckpointTensor reports whether a checkpoint tensor belongs to an NVFP4
// triple (packed weight, block/global scales, activation input_scale) and so must
// not be widened into the f32 manifest.
func (c *ModelOptNVFP4Checkpoint) ownsCheckpointTensor(name string) bool { return c.aux[name] }

// Close releases every shard.
func (c *ModelOptNVFP4Checkpoint) Close() error {
	var first error
	for _, src := range c.sources {
		if err := src.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
