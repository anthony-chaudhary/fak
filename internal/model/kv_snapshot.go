package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/kvcache"
)

// CaptureNVFP4Snapshot quantizes a dense FP32 KVCache into a model-agnostic
// block-16 NVFP4 kvcache.Snapshot for 7x-compressed prefix storage or transfer.
// It returns overflow=true if any K/V element exceeds NVFP4 block range or is non-finite.
func CaptureNVFP4Snapshot(c *KVCache, tokens []int) (*kvcache.Snapshot, bool, error) {
	if c == nil {
		return nil, false, fmt.Errorf("model: nil KVCache")
	}
	nTok := c.Len()
	if len(tokens) != 0 && len(tokens) != nTok {
		return nil, false, fmt.Errorf("model: tokens len %d != KVCache len %d", len(tokens), nTok)
	}
	desc := KVCacheLayoutDesc(c.cfg, KVPrecisionNVFP4, nTok)
	if err := desc.Validate(); err != nil {
		return nil, false, err
	}
	if len(c.K) != c.cfg.NumLayers || len(c.V) != c.cfg.NumLayers {
		return nil, false, fmt.Errorf("model: K/V layer count mismatch (%d/%d != %d)", len(c.K), len(c.V), c.cfg.NumLayers)
	}

	layers := make([]kvcache.SnapshotLayer, c.cfg.NumLayers)
	var anyOverflow bool
	for l := 0; l < c.cfg.NumLayers; l++ {
		kc, ks, okK, err := kvcache.QuantizeBlock16NVFP4(c.K[l], c.cfg.NumKVHeads, c.cfg.HeadDim, nTok)
		if err != nil {
			return nil, false, fmt.Errorf("model: quantize K layer %d: %w", l, err)
		}
		vc, vs, okV, err := kvcache.QuantizeBlock16NVFP4(c.V[l], c.cfg.NumKVHeads, c.cfg.HeadDim, nTok)
		if err != nil {
			return nil, false, fmt.Errorf("model: quantize V layer %d: %w", l, err)
		}
		if okK || okV {
			anyOverflow = true
		}
		layers[l] = kvcache.SnapshotLayer{
			Index:   l,
			KCodes:  kc,
			VCodes:  vc,
			KScales: ks,
			VScales: vs,
		}
	}

	tokCopy := make([]int, len(tokens))
	copy(tokCopy, tokens)
	posCopy := make([]int, len(c.pos))
	copy(posCopy, c.pos)

	return &kvcache.Snapshot{
		Layout:    desc,
		Tokens:    tokCopy,
		Positions: posCopy,
		Layers:    layers,
	}, anyOverflow, nil
}

// RestoreNVFP4Snapshot decodes a block-16 NVFP4 kvcache.Snapshot into a
// dense FP32 KVCache compatible with cfg.
func RestoreNVFP4Snapshot(snap *kvcache.Snapshot, cfg Config) (*KVCache, error) {
	if snap == nil {
		return nil, fmt.Errorf("model: nil snapshot")
	}
	if snap.Layout.Precision != kvcache.PrecisionNVFP4 {
		return nil, fmt.Errorf("model: snapshot precision %q != nvfp4", snap.Layout.Precision)
	}
	if snap.Layout.Layers != cfg.NumLayers || snap.Layout.KVHeads != cfg.NumKVHeads || snap.Layout.HeadDim != cfg.HeadDim {
		return nil, fmt.Errorf("model: snapshot geometry (%d,%d,%d) != config (%d,%d,%d)",
			snap.Layout.Layers, snap.Layout.KVHeads, snap.Layout.HeadDim,
			cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim)
	}
	nTok := len(snap.Tokens)
	if len(snap.Positions) > nTok {
		nTok = len(snap.Positions)
	}
	c := NewKVCache(cfg)
	for l := 0; l < cfg.NumLayers; l++ {
		sl := snap.Layers[l]
		k, err := kvcache.DequantizeBlock16NVFP4(sl.KCodes, sl.KScales, cfg.NumKVHeads, cfg.HeadDim, nTok)
		if err != nil {
			return nil, fmt.Errorf("model: dequantize K layer %d: %w", l, err)
		}
		v, err := kvcache.DequantizeBlock16NVFP4(sl.VCodes, sl.VScales, cfg.NumKVHeads, cfg.HeadDim, nTok)
		if err != nil {
			return nil, fmt.Errorf("model: dequantize V layer %d: %w", l, err)
		}
		c.K[l] = k
		c.Kraw[l] = append([]float32(nil), k...)
		c.V[l] = v
	}
	if len(snap.Positions) > 0 {
		c.pos = append([]int(nil), snap.Positions...)
	} else {
		c.pos = make([]int, nTok)
		for i := range c.pos {
			c.pos[i] = i
		}
	}
	return c, nil
}
