package model

import "fmt"

// splitBatchedMoEExperts materializes HF Mixtral/Qwen-MoE packed expert tensors
// into the per-expert canonical names the runtime uses. HF exports:
//
//	model.layers.L.mlp.experts.gate_up_proj  [E, 2I, H]
//	model.layers.L.mlp.experts.down_proj     [E, H, I]
//
// The runtime names:
//
//	model.layers.L.mlp.experts.e.gate_proj.weight [I, H]
//	model.layers.L.mlp.experts.e.up_proj.weight   [I, H]
//	model.layers.L.mlp.experts.e.down_proj.weight [H, I]
//
// Every output manifest row is a zero-copy byte-range view into the source blob.
func splitBatchedMoEExperts(cfg Config, man map[string]tensorMeta) error {
	if cfg.NumExperts <= 0 {
		return nil
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if err := splitBatchedMoEGateUp(cfg, man, l); err != nil {
			return err
		}
		if err := splitBatchedMoEDown(cfg, man, l); err != nil {
			return err
		}
	}
	return nil
}

func splitBatchedMoEGateUp(cfg Config, man map[string]tensorMeta, layer int) error {
	name, meta, ok := firstTensor(man,
		layerName(layer, "mlp.experts.gate_up_proj"),
		layerName(layer, "mlp.experts.gate_up_proj.weight"),
	)
	if !ok {
		return nil
	}
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	wantShape := []int{E, 2 * I, H}
	if !sameShape(meta.Shape, wantShape) {
		return fmt.Errorf("model: MoE tensor %s has shape %v, want %v", name, meta.Shape, wantShape)
	}
	wantBytes := E * 2 * I * H * 4
	if meta.Nbytes != wantBytes {
		return fmt.Errorf("model: MoE tensor %s has %d bytes, shape %v f32 implies %d",
			name, meta.Nbytes, meta.Shape, wantBytes)
	}
	expertStride := 2 * I * H * 4
	partBytes := I * H * 4
	for e := 0; e < E; e++ {
		gateName := expertName(layer, e, "gate_proj.weight")
		upName := expertName(layer, e, "up_proj.weight")
		if _, exists := man[gateName]; exists {
			return fmt.Errorf("model: cannot split %s: component %s already present", name, gateName)
		}
		if _, exists := man[upName]; exists {
			return fmt.Errorf("model: cannot split %s: component %s already present", name, upName)
		}
		base := meta.Offset + e*expertStride
		man[gateName] = tensorMeta{Dtype: meta.Dtype, Shape: []int{I, H}, Offset: base, Nbytes: partBytes}
		man[upName] = tensorMeta{Dtype: meta.Dtype, Shape: []int{I, H}, Offset: base + partBytes, Nbytes: partBytes}
	}
	delete(man, name)
	return nil
}

func splitBatchedMoEDown(cfg Config, man map[string]tensorMeta, layer int) error {
	name, meta, ok := firstTensor(man,
		layerName(layer, "mlp.experts.down_proj"),
		layerName(layer, "mlp.experts.down_proj.weight"),
	)
	if !ok {
		return nil
	}
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	wantShape := []int{E, H, I}
	if !sameShape(meta.Shape, wantShape) {
		return fmt.Errorf("model: MoE tensor %s has shape %v, want %v", name, meta.Shape, wantShape)
	}
	partBytes := H * I * 4
	wantBytes := E * partBytes
	if meta.Nbytes != wantBytes {
		return fmt.Errorf("model: MoE tensor %s has %d bytes, shape %v f32 implies %d",
			name, meta.Nbytes, meta.Shape, wantBytes)
	}
	for e := 0; e < E; e++ {
		downName := expertName(layer, e, "down_proj.weight")
		if _, exists := man[downName]; exists {
			return fmt.Errorf("model: cannot split %s: component %s already present", name, downName)
		}
		man[downName] = tensorMeta{
			Dtype:  meta.Dtype,
			Shape:  []int{H, I},
			Offset: meta.Offset + e*partBytes,
			Nbytes: partBytes,
		}
	}
	delete(man, name)
	return nil
}

func firstTensor(man map[string]tensorMeta, names ...string) (string, tensorMeta, bool) {
	for _, name := range names {
		if meta, ok := man[name]; ok {
			return name, meta, true
		}
	}
	return "", tensorMeta{}, false
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
