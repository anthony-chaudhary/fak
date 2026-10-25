package ggufload

import "fmt"

// LayerWeightCensus is the GGUF file's tensor payload bytes grouped the way the
// multi-device band planner (model.PlanDeviceBands) charges them. File bytes are an
// upper bound on device bytes: a backend that repacks or dequantizes can only change a
// tensor's resident size, which the planner's headroom absorbs.
type LayerWeightCensus struct {
	// Layers holds the payload bytes of every blk.N.* tensor for N in [0, numLayers).
	Layers []int64 `json:"layers"`
	// EmbedBytes is token_embd.weight, charged to the first band.
	EmbedBytes int64 `json:"embed_bytes"`
	// HeadBytes is output.weight plus output_norm.weight, charged to the last band. With
	// no output.weight the head is tied, and the last band holds its own embedding copy.
	HeadBytes int64 `json:"head_bytes"`
	TiedHead  bool  `json:"tied_head"`
	// OtherBytes is everything else (blk.N beyond numLayers such as an MTP layer, vision
	// or projector tensors). The planner does not charge it.
	OtherBytes int64 `json:"other_bytes"`
}

// CensusLayerWeights sums each tensor's payload bytes into the per-layer census. It
// fails closed on a tensor whose payload size cannot be computed.
func CensusLayerWeights(f *File, numLayers int) (LayerWeightCensus, error) {
	if f == nil || numLayers <= 0 {
		return LayerWeightCensus{}, fmt.Errorf("gguf: layer census needs a file and numLayers > 0 (got %d)", numLayers)
	}
	c := LayerWeightCensus{Layers: make([]int64, numLayers)}
	var outputNorm int64
	haveHead := false
	for _, t := range f.Tensors {
		n, err := tensorPayloadBytes(t)
		if err != nil {
			return LayerWeightCensus{}, fmt.Errorf("gguf: layer census: %w", err)
		}
		b := int64(n)
		if layer, _, ok := parseGLMBlkLayerSuffix(t.Name); ok {
			if layer >= 0 && layer < numLayers {
				c.Layers[layer] += b
			} else {
				c.OtherBytes += b
			}
			continue
		}
		switch t.Name {
		case "token_embd.weight":
			c.EmbedBytes += b
		case "output.weight":
			c.HeadBytes += b
			haveHead = true
		case "output_norm.weight":
			outputNorm += b
		default:
			c.OtherBytes += b
		}
	}
	if !haveHead {
		c.TiedHead = true
		c.HeadBytes = c.EmbedBytes
	}
	c.HeadBytes += outputNorm
	return c, nil
}
