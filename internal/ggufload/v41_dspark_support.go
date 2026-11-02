package ggufload

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func (f *File) isV41DSparkGGUF() bool {
	arch, _ := f.String("general.architecture")
	if arch == "deepseek41-dspark" {
		return true
	}
	for key := range f.Metadata {
		if strings.HasPrefix(key, "dspark.") {
			return true
		}
	}
	return false
}

// configV41DSparkGGUF never returns a runnable target config. This is the
// actual pre-load consumer of the header contract; original-weight equivalence
// is not observable here and execution remains disabled even for a valid header.
func (f *File) configV41DSparkGGUF() (model.Config, error) {
	fail := func(reason string) (model.Config, error) {
		return model.Config{}, fmt.Errorf("%w: support GGUF %s", model.ErrV41DSparkArtifact, reason)
	}
	h := model.V41DSparkGGUFHeader{Parameters: map[string]int{}}
	for key, dst := range map[string]*string{"general.architecture": &h.Architecture, "dspark.target_architecture": &h.TargetArchitecture, "dspark.source_config": &h.SourceConfig, "dspark.weight_encoding": &h.WeightEncoding} {
		v, ok := f.String(key)
		if !ok {
			return fail("missing or non-string " + key)
		}
		*dst = v
	}
	keys := map[string]bool{"target_architecture": true, "source_config": true, "weight_encoding": true, "target_layer_ids": true}
	for _, key := range []string{"block_size", "markov_rank", "noise_token_id", "n_routed_experts", "num_experts_per_tok"} {
		v, ok := f.lookup("dspark." + key)
		if !ok || v.Type != TypeUint32 {
			return fail("missing or non-u32 dspark." + key)
		}
		n, ok := valueUint64(v)
		if !ok || n > uint64(int(^uint(0)>>1)) {
			return fail("scalar range dspark." + key)
		}
		h.Parameters[key] = int(n)
		keys[key] = true
	}
	for key := range f.Metadata {
		if strings.HasPrefix(key, "dspark.") && !keys[strings.TrimPrefix(key, "dspark.")] {
			return fail("unknown metadata " + key)
		}
	}
	v, ok := f.lookup("dspark.target_layer_ids")
	if !ok || v.Type != TypeArray {
		return fail("missing target array")
	}
	items, ok := v.Value.([]Value)
	if !ok || len(items) != 3 {
		return fail("three target layers required")
	}
	for _, item := range items {
		n, ok := valueUint64(item)
		if !ok || item.Type != TypeUint32 || n > uint64(int(^uint(0)>>1)) {
			return fail("invalid u32 target layer")
		}
		h.TargetLayerIDs = append(h.TargetLayerIDs, int(n))
	}
	if len(f.Tensors) > 256 {
		return fail("tensor count exceeds supported three-stage inventory")
	}
	for _, t := range f.Tensors {
		if len(t.Dims) < 1 || len(t.Dims) > 3 {
			return fail("tensor rank")
		}
		r := model.V41TensorRef{Name: t.Name, Dtype: t.Type.String(), Shape: make([]int, len(t.Dims))}
		for i, n := range t.Dims {
			if n == 0 || n > uint64(int(^uint(0)>>1)) {
				return fail("tensor dimension range")
			}
			r.Shape[len(t.Dims)-1-i] = int(n)
		}
		h.Tensors = append(h.Tensors, r)
	}
	cfg, err := model.ValidateV41DSparkGGUFHeader(h)
	if err != nil {
		return model.Config{}, err
	}
	return model.Config{}, fmt.Errorf("support GGUF header contract valid; original-weight losslessness is unverified: %w", cfg.RefuseDeepSeekV41DSpark())
}
