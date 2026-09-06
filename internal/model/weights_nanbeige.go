package model

import (
	"fmt"
	"strings"
)

// ValidateNanbeigeWeights verifies that the manifest contains the full set of 22 physical
// block tensors with expected Nanbeige4.2 dimensions (HiddenSize: 3072, HeadDim: 128,
// NumHeads: 48, NumKVHeads: 8) and all required normalization tensors (input_layernorm,
// post_attention_layernorm) without invalid or duplicate weight allocations.
func ValidateNanbeigeWeights(cfg Config, man map[string]tensorMeta) error {
	if !cfg.IsNanbeige() {
		return nil
	}
	if cfg.NumLayers != 22 {
		return &UnsupportedNanbeigeVariantError{
			Reason:      fmt.Sprintf("num_hidden_layers must be 22, got %d", cfg.NumLayers),
			NumLayers:   cfg.NumLayers,
			NumLoops:    cfg.NumLoops,
			HeadDim:     cfg.HeadDim,
			SharedCache: cfg.SharedCache,
		}
	}
	numLoops := cfg.NumLoops
	if numLoops == 0 {
		numLoops = 2
	}
	if numLoops != 2 {
		return &UnsupportedNanbeigeVariantError{
			Reason:      fmt.Sprintf("num_loops must be 2, got %d", numLoops),
			NumLayers:   cfg.NumLayers,
			NumLoops:    numLoops,
			HeadDim:     cfg.HeadDim,
			SharedCache: cfg.SharedCache,
		}
	}
	if cfg.HeadDim != 128 {
		return &UnsupportedNanbeigeVariantError{
			Reason:      fmt.Sprintf("head_dim must be 128, got %d", cfg.HeadDim),
			NumLayers:   cfg.NumLayers,
			NumLoops:    numLoops,
			HeadDim:     cfg.HeadDim,
			SharedCache: cfg.SharedCache,
		}
	}

	wantQ := []int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize}
	wantKV := []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}
	wantNorm := []int{cfg.HiddenSize}

	for l := 0; l < cfg.NumLayers; l++ {
		// Normalization tensors
		inNormName := layerName(l, "input_layernorm.weight")
		inNorm, ok := man[inNormName]
		if !ok {
			return fmt.Errorf("model: Nanbeige layer %d missing required normalization tensor %s", l, inNormName)
		}
		if !sameShape(inNorm.Shape, wantNorm) {
			return fmt.Errorf("model: Nanbeige layer %d %s has shape %v, want %v", l, inNormName, inNorm.Shape, wantNorm)
		}

		postNormName := layerName(l, "post_attention_layernorm.weight")
		postNorm, ok := man[postNormName]
		if !ok {
			return fmt.Errorf("model: Nanbeige layer %d missing required normalization tensor %s", l, postNormName)
		}
		if !sameShape(postNorm.Shape, wantNorm) {
			return fmt.Errorf("model: Nanbeige layer %d %s has shape %v, want %v", l, postNormName, postNorm.Shape, wantNorm)
		}

		// Attention Q/K/V tensors
		qName := layerName(l, "self_attn.q_proj.weight")
		qMeta, ok := man[qName]
		if !ok {
			return fmt.Errorf("model: Nanbeige layer %d missing required attention tensor %s", l, qName)
		}
		if !sameShape(qMeta.Shape, wantQ) {
			return fmt.Errorf("model: Nanbeige layer %d %s has shape %v, want %v", l, qName, qMeta.Shape, wantQ)
		}

		kName := layerName(l, "self_attn.k_proj.weight")
		kMeta, ok := man[kName]
		if !ok {
			return fmt.Errorf("model: Nanbeige layer %d missing required attention tensor %s", l, kName)
		}
		if !sameShape(kMeta.Shape, wantKV) {
			return fmt.Errorf("model: Nanbeige layer %d %s has shape %v, want %v", l, kName, kMeta.Shape, wantKV)
		}

		vName := layerName(l, "self_attn.v_proj.weight")
		vMeta, ok := man[vName]
		if !ok {
			return fmt.Errorf("model: Nanbeige layer %d missing required attention tensor %s", l, vName)
		}
		if !sameShape(vMeta.Shape, wantKV) {
			return fmt.Errorf("model: Nanbeige layer %d %s has shape %v, want %v", l, vName, vMeta.Shape, wantKV)
		}

		if oName := layerName(l, "self_attn.o_proj.weight"); man != nil {
			if oMeta, ok := man[oName]; ok {
				wantO := []int{cfg.HiddenSize, cfg.NumHeads * cfg.HeadDim}
				if !sameShape(oMeta.Shape, wantO) {
					return fmt.Errorf("model: Nanbeige layer %d %s has shape %v, want %v", l, oName, oMeta.Shape, wantO)
				}
			}
		}
	}

	if normMeta, ok := man["model.norm.weight"]; ok {
		if !sameShape(normMeta.Shape, wantNorm) {
			return fmt.Errorf("model: Nanbeige model.norm.weight has shape %v, want %v", normMeta.Shape, wantNorm)
		}
	}

	return nil
}

// MaterializeNanbeigeSharedWeights maps the 22 physical block tensors across recurrent loops
// without duplicating weight allocation in memory. Layers across loops (e.g. layers 22-43 in loop 1)
// reference the exact same underlying physical weights in the manifest.
func MaterializeNanbeigeSharedWeights(cfg Config, man map[string]tensorMeta) error {
	if !cfg.IsNanbeige() {
		return nil
	}
	if err := ValidateNanbeigeWeights(cfg, man); err != nil {
		return err
	}

	numLoops := cfg.NumLoops
	if numLoops == 0 {
		numLoops = 2
	}
	if numLoops <= 1 {
		return nil
	}

	type aliasEntry struct {
		dst           string
		meta          tensorMeta
		physicalLayer int
	}
	var aliases []aliasEntry

	for name, meta := range man {
		for l := 0; l < cfg.NumLayers; l++ {
			srcPrefix := layerPrefix(l)
			if strings.HasPrefix(name, srcPrefix) {
				suffix := strings.TrimPrefix(name, srcPrefix)
				for loop := 1; loop < numLoops; loop++ {
					logicalLayer := loop*cfg.NumLayers + l
					dstName := layerPrefix(logicalLayer) + suffix
					aliases = append(aliases, aliasEntry{dst: dstName, meta: meta, physicalLayer: l})
				}
				break
			}
		}
	}

	for _, a := range aliases {
		if existing, exists := man[a.dst]; exists {
			if existing.Offset != a.meta.Offset || existing.Nbytes != a.meta.Nbytes {
				return fmt.Errorf("model: duplicate weight allocation detected for %s (must share physical layer %d weights at offset %d)", a.dst, a.physicalLayer, a.meta.Offset)
			}
			continue
		}
		man[a.dst] = a.meta
	}

	return nil
}

func materializeNanbeigeSharedWeights(cfg Config, man map[string]tensorMeta) error {
	return MaterializeNanbeigeSharedWeights(cfg, man)
}

// MapNanbeigeSharedWeights is an alias for MaterializeNanbeigeSharedWeights.
func MapNanbeigeSharedWeights(cfg Config, man map[string]tensorMeta) error {
	return MaterializeNanbeigeSharedWeights(cfg, man)
}

// NanbeigePhysicalLayer returns the physical block index (0 to 21) corresponding to a logical layer.
// Returns -1 for negative indices.
func (m *Model) NanbeigePhysicalLayer(logicalLayer int) int {
	if logicalLayer < 0 {
		return -1
	}
	if !m.Cfg.IsNanbeige() || m.Cfg.NumLayers <= 0 {
		return logicalLayer
	}
	return logicalLayer % m.Cfg.NumLayers
}

// NanbeigeLogicalLayers returns the total logical layer count (NumLayers * NumLoops).
func (c Config) NanbeigeLogicalLayers() int {
	if !c.IsNanbeige() {
		return c.NumLayers
	}
	loops := c.NumLoops
	if loops <= 0 {
		loops = 2
	}
	return c.NumLayers * loops
}

// NanbeigeNumLogicalLayers returns the total logical layer count (NumLayers * NumLoops).
func (m *Model) NanbeigeNumLogicalLayers() int {
	return m.Cfg.NanbeigeLogicalLayers()
}
