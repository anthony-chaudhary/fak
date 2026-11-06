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
// not admit (a non-NVFP4 algorithm or a non-16 group size).
var ErrModelOptNVFP4Config = errors.New("model: unsupported ModelOpt quantization config")

// ErrModelOptNVFP4Checkpoint reports a sharded ModelOpt NVFP4 checkpoint whose
// layout the loader refuses (split triples, duplicate canonical names, missing
// shards).
var ErrModelOptNVFP4Checkpoint = errors.New("model: invalid ModelOpt NVFP4 checkpoint")

// ModelOptNVFP4Config is the admitted subset of a ModelOpt NVFP4 declaration.
type ModelOptNVFP4Config struct {
	QuantAlgo      string
	GroupSize      int
	ExcludeModules []string
}

type modelOptQuantDecl struct {
	QuantMethod    string   `json:"quant_method"`
	QuantAlgo      string   `json:"quant_algo"`
	GroupSize      int      `json:"group_size"`
	ExcludeModules []string `json:"exclude_modules"`
}

// ReadModelOptNVFP4Config detects a ModelOpt NVFP4 declaration in dir. It reads
// both forms vLLM's ModelOpt quantization config accepts: config.json
// quantization_config {"quant_method":"modelopt","quant_algo":"NVFP4"} and the
// hf_quant_config.json {"producer":{"name":"modelopt"},"quantization":{...}}.
// ok is false when neither file declares ModelOpt quantization.
func ReadModelOptNVFP4Config(dir string) (cfg ModelOptNVFP4Config, ok bool, err error) {
	var decl *modelOptQuantDecl
	if b, rerr := os.ReadFile(filepath.Join(dir, "config.json")); rerr == nil {
		var doc struct {
			QuantizationConfig *modelOptQuantDecl `json:"quantization_config"`
		}
		if err := json.Unmarshal(b, &doc); err != nil {
			return cfg, false, fmt.Errorf("modelopt config.json: %w", err)
		}
		if q := doc.QuantizationConfig; q != nil && strings.EqualFold(q.QuantMethod, "modelopt") {
			decl = q
		}
	}
	if b, rerr := os.ReadFile(filepath.Join(dir, "hf_quant_config.json")); rerr == nil {
		var doc struct {
			Producer struct {
				Name string `json:"name"`
			} `json:"producer"`
			Quantization *modelOptQuantDecl `json:"quantization"`
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
