package model

import (
	"fmt"
	"strings"
)

// Qwen35StateLayout is the single canonical description of Qwen recurrent
// conv + delta-rule state geometry for one request unit at one linear layer,
// derived once from Config plus an explicit state dtype. The preallocated
// state bank and the recurrent capacity manager both consume it so allocation,
// admission, and byte accounting share one derivation.
type Qwen35StateLayout struct {
	StateDtype                string `json:"state_dtype"`
	ElementBytes              int    `json:"element_bytes"`
	NumLinearLayers           int    `json:"num_linear_layers"`
	ConvDim                   int    `json:"conv_dim"`
	ConvKernel                int    `json:"conv_kernel"`
	ConvElementsPerLayer      int    `json:"conv_elements_per_layer"`
	RecurrentElementsPerLayer int    `json:"recurrent_elements_per_layer"`
}

// NewQwen35StateLayout derives the canonical recurrent state layout from model
// geometry and an explicit state dtype ("f32", "f16", "bf16"). Empty stateDtype
// defaults to "f32". Unsupported dtypes or degenerate geometry are rejected so
// no consumer silently allocates a zero-sized or wrong-width state bank.
func NewQwen35StateLayout(cfg Config, stateDtype string) (Qwen35StateLayout, error) {
	normalized := strings.ToLower(strings.TrimSpace(stateDtype))
	if normalized == "" {
		normalized = "f32"
	}

	var elemBytes int
	switch normalized {
	case "f32":
		elemBytes = 4
	case "f16", "bf16":
		elemBytes = 2
	default:
		return Qwen35StateLayout{}, fmt.Errorf("unsupported state dtype %q: must be f32, f16, or bf16", stateDtype)
	}

	numLinearLayers, convDim, kernel, convElements, recurrentElements := deriveQwen35StateGeometry(cfg)

	if numLinearLayers <= 0 || convElements <= 0 || recurrentElements <= 0 {
		return Qwen35StateLayout{}, fmt.Errorf("unsupported recurrent geometry: %d linear layers, %d conv elements, %d recurrent elements",
			numLinearLayers, convElements, recurrentElements)
	}

	return Qwen35StateLayout{
		StateDtype:                normalized,
		ElementBytes:              elemBytes,
		NumLinearLayers:           numLinearLayers,
		ConvDim:                   convDim,
		ConvKernel:                kernel,
		ConvElementsPerLayer:      convElements,
		RecurrentElementsPerLayer: recurrentElements,
	}, nil
}

// deriveQwen35StateGeometry is the single derivation of the Qwen recurrent
// conv + delta-rule element geometry. Both the preallocated state bank and the
// recurrent capacity manager consume it via NewQwen35StateLayout so their
// allocation and byte accounting can never drift apart.
func deriveQwen35StateGeometry(cfg Config) (numLinearLayers, convDim, kernel, convElementsPerLayer, recurrentElementsPerLayer int) {
	numLinearLayers = 0
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			numLinearLayers++
		}
	}
	if numLinearLayers == 0 {
		if cfg.IsQwen35Hybrid() {
			numLinearLayers = cfg.NumLayers
		} else {
			numLinearLayers = 1
		}
	}

	nK, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	kernel = cfg.LinearConvKernelDim
	if kernel <= 1 {
		kernel = 4
	}

	// Overflow-guard the element counts with checkedMul; on overflow the affected
	// count is left at 0 so NewQwen35StateLayout's degenerate-geometry rejection
	// fires instead of silently wrapping to a bogus (possibly positive) size.
	convElementsPerLayer = 0
	if v, ok := checkedMul(int64(kernel-1), int64(convDim)); ok {
		convElementsPerLayer = int(v)
	}
	recurrentElementsPerLayer = 0
	if v, ok := checkedMul(int64(nV), int64(kHd), int64(vHd)); ok && v > 0 {
		recurrentElementsPerLayer = int(v)
	} else if v, ok := checkedMul(int64(nK), int64(kHd), int64(vHd)); ok && v > 0 {
		recurrentElementsPerLayer = int(v)
	}
	return numLinearLayers, convDim, kernel, convElementsPerLayer, recurrentElementsPerLayer
}

// ConvBytesPerLayer returns the conv-state bytes for one linear layer.
func (l Qwen35StateLayout) ConvBytesPerLayer() int64 {
	return int64(l.ConvElementsPerLayer * l.ElementBytes)
}

// RecurrentBytesPerLayer returns the delta-rule recurrent-state bytes for one linear layer.
func (l Qwen35StateLayout) RecurrentBytesPerLayer() int64 {
	return int64(l.RecurrentElementsPerLayer * l.ElementBytes)
}

// BytesPerLayer returns the combined conv + recurrent state bytes for one linear layer.
func (l Qwen35StateLayout) BytesPerLayer() int64 {
	return l.ConvBytesPerLayer() + l.RecurrentBytesPerLayer()
}

// BytesPerUnit returns the combined state bytes for one request unit across all linear layers.
func (l Qwen35StateLayout) BytesPerUnit() int64 {
	return l.BytesPerLayer() * int64(l.NumLinearLayers)
}

// TotalBytes returns the state bytes for maxUnits request units.
func (l Qwen35StateLayout) TotalBytes(maxUnits int) int64 {
	return l.BytesPerUnit() * int64(maxUnits)
}
