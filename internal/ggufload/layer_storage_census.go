package ggufload

import (
	"fmt"
	"math"
)

// LayerStorageCensus describes header-derived logical loader storage, not physical
// device allocations, startup peaks, workspace, or an executable placement.
type LayerStorageCensus struct {
	Layers    []int64
	Embedding int64
	Head      int64
}

func addLayerStorageBytes(dst *int64, n uint64) error {
	if n > math.MaxInt64 || *dst > math.MaxInt64-int64(n) {
		return fmt.Errorf("gguf: layer storage census overflow")
	}
	*dst += int64(n)
	return nil
}

// CensusQ4KLayerStorage groups the existing transformed-storage estimate by
// source layer, embedding and head. Tied packed tables have ambiguous ownership
// and are rejected; a separate tied Q8 head copy is charged to the head.
func (s *WeightSource) CensusQ4KLayerStorage(opts ...Q4KLoadOption) (LayerStorageCensus, error) {
	if s == nil || s.File == nil {
		return LayerStorageCensus{}, fmt.Errorf("gguf: layer storage census requires a weight source")
	}
	cfg, err := s.File.Config()
	if err != nil {
		return LayerStorageCensus{}, err
	}
	if cfg.NumLayers <= 0 || cfg.NumLayers > 65536 {
		return LayerStorageCensus{}, fmt.Errorf("gguf: layer storage census layer count out of bounds")
	}
	c := LayerStorageCensus{Layers: make([]int64, cfg.NumLayers)}
	_, _, err = s.walkQ4KLoadStorage(opts, func(st q4kTensorStorage) error {
		if st.streamedBounded || st.qwenMTP {
			return fmt.Errorf("%w: layer census excludes streamed or MTP storage", ErrQ4KLoadEstimateUnsupported)
		}
		var dst *int64
		if layer, _, ok := parseGLMBlkLayerSuffix(st.info.Name); ok {
			if layer < 0 || layer >= len(c.Layers) {
				return fmt.Errorf("%w: layer outside census", ErrQ4KLoadEstimateUnsupported)
			}
			dst = &c.Layers[layer]
		} else {
			switch st.info.Name {
			case "token_embd.weight":
				if cfg.TieWordEmbeddings && st.tiedHeadQ8 == 0 {
					return fmt.Errorf("%w: tied embedding ownership", ErrQ4KLoadEstimateUnsupported)
				}
				dst = &c.Embedding
			case "output.weight", "output_norm.weight":
				dst = &c.Head
			default:
				return fmt.Errorf("%w: unclassified tensor ownership %s", ErrQ4KLoadEstimateUnsupported, st.info.Name)
			}
		}
		if err := addLayerStorageBytes(dst, st.bytes); err != nil {
			return err
		}
		return addLayerStorageBytes(&c.Head, st.tiedHeadQ8)
	})
	if err != nil {
		return LayerStorageCensus{}, err
	}
	return c, nil
}
