package ggufload

// This file closes the last clause of the GGUF-leaf scope (MODEL-ARCH-SEAM.md §4): a
// GGUF file is "expose[d] via the compute.WeightSource interface". The WeightSource type
// above already dequants every block to f32 and normalizes GGUF tensor names to the
// canonical HF-Llama names internal/model consumes; ComputeSource is the thin adapter
// that hands those same dequantized weights to a compute.Backend by name, as the
// compute.WeightSource seam (Weight(name, want) (compute.Tensor, error)) requires.
//
// Default behavior matches the leaf's default: blocks (including Q4_K_M super-blocks)
// are dequantized to f32 on load, so the returned Tensor is always an F32 host tensor.
// `want` is honored only for F32 (the dequant currency); any other request is refused
// rather than silently downcast — the direct-Q8_0-block path is a separately gated
// stretch that must not ride this default seam.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ComputeSource adapts a GGUF WeightSource to compute.WeightSource. It serves weights by
// their CANONICAL (HF-Llama) name — the same names model.NewFromF32Tensors registers —
// applying the identical rotary-unpermute normalization F32Tensors applies, so a backend
// fetching "model.layers.N.self_attn.q_proj.weight" gets the unpermuted q/k a from-disk
// HF checkpoint would have provided.
type ComputeSource struct {
	be      compute.Backend
	ws      *WeightSource
	cfg     model.Config
	byCanon map[string]string // canonical name -> raw GGUF tensor name
}

// compile-time proof ComputeSource implements the seam.
var _ compute.WeightSource = (*ComputeSource)(nil)

// NewComputeSource builds the adapter, reading the config once (needed for the rotary
// unpermute) and indexing every GGUF tensor by its canonical name. be owns the tensors
// the adapter returns; pass compute.Default() for the reference CPU backend.
func NewComputeSource(be compute.Backend, ws *WeightSource) (*ComputeSource, error) {
	if be == nil {
		return nil, fmt.Errorf("gguf: nil compute backend")
	}
	if ws == nil {
		return nil, fmt.Errorf("gguf: nil weight source")
	}
	cfg, err := ws.File.Config()
	if err != nil {
		return nil, err
	}
	byCanon := make(map[string]string, len(ws.File.Tensors))
	for _, info := range ws.File.Tensors {
		canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			continue // skip non-weight tensors (e.g. unmapped extras); Weight reports a miss
		}
		if prior, dup := byCanon[canon]; dup {
			return nil, fmt.Errorf("gguf: canonical name %s maps to both %s and %s", canon, prior, info.Name)
		}
		byCanon[canon] = info.Name
	}
	return &ComputeSource{be: be, ws: ws, cfg: cfg, byCanon: byCanon}, nil
}

// Weight returns the dequantized-to-f32 weight named `name` (a canonical HF-Llama name)
// as a host compute.Tensor owned by the adapter's backend. Only want==compute.F32 is
// served: dequant-to-f32 is the GGUF leaf's default, and the direct-Q8_0 path is a
// separately gated stretch, so any non-F32 request is refused rather than approximated.
func (s *ComputeSource) Weight(name string, want compute.Dtype) (compute.Tensor, error) {
	if want != compute.F32 {
		return compute.Tensor{}, fmt.Errorf("gguf: ComputeSource serves f32 only, got %s", want)
	}
	raw, ok := s.byCanon[name]
	if !ok {
		return compute.Tensor{}, fmt.Errorf("gguf: missing weight %s", name)
	}
	info, _ := s.ws.Tensor(raw)
	shape, err := modelShapeFromGGUFDims(raw, info.Dims)
	if err != nil {
		return compute.Tensor{}, err
	}
	data, _, err := s.ws.TensorF32(raw)
	if err != nil {
		return compute.Tensor{}, err
	}
	data, err = normalizeCanonicalTensorData(name, data, s.cfg)
	if err != nil {
		return compute.Tensor{}, err
	}
	return compute.NewF32(s.be, shape, data), nil
}

// Config returns the model.Config parsed from the GGUF KV-metadata (the same config the
// adapter uses for rotary normalization), so a caller can size a KV cache / forward pass
// from the one source.
func (s *ComputeSource) Config() model.Config { return s.cfg }
