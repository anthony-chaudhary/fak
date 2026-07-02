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

// requireMoESplitShape validates a packed MoE tensor's shape and f32 byte
// count before a zero-copy split.
func requireMoESplitShape(name string, meta tensorMeta, wantShape []int) error {
	if !sameShape(meta.Shape, wantShape) {
		return fmt.Errorf("model: MoE tensor %s has shape %v, want %v", name, meta.Shape, wantShape)
	}
	wantBytes := 4
	for _, d := range wantShape {
		wantBytes *= d
	}
	if meta.Nbytes != wantBytes {
		return fmt.Errorf("model: MoE tensor %s has %d bytes, shape %v f32 implies %d",
			name, meta.Nbytes, meta.Shape, wantBytes)
	}
	return nil
}

// refuseSplitCollision errors when a zero-copy split's destination component
// name is already present in the manifest.
func refuseSplitCollision(man map[string]tensorMeta, srcName string, dstNames ...string) error {
	for _, dst := range dstNames {
		if _, exists := man[dst]; exists {
			return fmt.Errorf("model: cannot split %s: component %s already present", srcName, dst)
		}
	}
	return nil
}

func splitBatchedMoEGateUp(cfg Config, man map[string]tensorMeta, layer int) error {
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	expertStride := 2 * I * H * 4
	partBytes := I * H * 4
	return materializeFusedExpertTensor(man, E, []int{E, 2 * I, H}, requireMoESplitShape,
		func(name string, meta tensorMeta, e int) error {
			gateName := expertName(layer, e, "gate_proj.weight")
			upName := expertName(layer, e, "up_proj.weight")
			if err := refuseSplitCollision(man, name, gateName, upName); err != nil {
				return err
			}
			base := meta.Offset + e*expertStride
			man[gateName] = tensorMeta{Dtype: meta.Dtype, Shape: []int{I, H}, Offset: base, Nbytes: partBytes}
			man[upName] = tensorMeta{Dtype: meta.Dtype, Shape: []int{I, H}, Offset: base + partBytes, Nbytes: partBytes}
			return nil
		},
		layerName(layer, "mlp.experts.gate_up_proj"),
		layerName(layer, "mlp.experts.gate_up_proj.weight"),
	)
}

func splitBatchedMoEDown(cfg Config, man map[string]tensorMeta, layer int) error {
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	partBytes := H * I * 4
	return materializeFusedExpertTensor(man, E, []int{E, H, I}, requireMoESplitShape,
		func(name string, meta tensorMeta, e int) error {
			downName := expertName(layer, e, "down_proj.weight")
			if err := refuseSplitCollision(man, name, downName); err != nil {
				return err
			}
			man[downName] = tensorMeta{
				Dtype:  meta.Dtype,
				Shape:  []int{H, I},
				Offset: meta.Offset + e*partBytes,
				Nbytes: partBytes,
			}
			return nil
		},
		layerName(layer, "mlp.experts.down_proj"),
		layerName(layer, "mlp.experts.down_proj.weight"),
	)
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
