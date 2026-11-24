package engine

import "github.com/anthony-chaudhary/fak/internal/model"

// LoopedBlockShape describes a middle-block looped model: Prelude and Coda
// layers run once per token, Loop layers run once per recurrent pass. Only the
// looped block carries per-pass KV, so a request at depth L stores
// Prelude+Coda+L*Loop layers of KV per token, not L full stacks.
type LoopedBlockShape struct {
	PreludeLayers int
	LoopLayers    int
	CodaLayers    int
}

// LoopedBlockFromCoreRange maps the core_start/core_end layout a looped
// checkpoint declares (core = [coreStart, coreEnd) of layers) onto a shape.
func LoopedBlockFromCoreRange(layers, coreStart, coreEnd int) LoopedBlockShape {
	return LoopedBlockShape{
		PreludeLayers: coreStart,
		LoopLayers:    coreEnd - coreStart,
		CodaLayers:    layers - coreEnd,
	}
}

// LoopedBlockFromInkling2Loop maps a loaded inkling2_loop checkpoint onto a
// shape. A checkpoint that never repeats its core (Passes <= 1) or declares a
// core outside [0, Layers) yields the zero shape, keeping full-stack accounting.
func LoopedBlockFromInkling2Loop(hp model.Inkling2LoopHParams) LoopedBlockShape {
	if hp.Passes <= 1 || hp.CoreStart < 0 || hp.CoreStart >= hp.CoreEnd || hp.CoreEnd > hp.Layers {
		return LoopedBlockShape{}
	}
	return LoopedBlockFromCoreRange(hp.Layers, hp.CoreStart, hp.CoreEnd)
}

// Declared reports whether the shape describes a real looped block; the zero
// value keeps the legacy full-stack-per-pass accounting.
func (s LoopedBlockShape) Declared() bool {
	return s.LoopLayers > 0 && s.PreludeLayers >= 0 && s.CodaLayers >= 0
}

// PhysicalLayers is the layer count of one unrolled pass.
func (s LoopedBlockShape) PhysicalLayers() int {
	return s.PreludeLayers + s.LoopLayers + s.CodaLayers
}

// LayerVisits is the number of layer evaluations (and KV-bearing layer slots)
// one token costs at the given loop depth.
func (s LoopedBlockShape) LayerVisits(loops int) int {
	if loops < 1 {
		loops = 1
	}
	return s.PreludeLayers + s.CodaLayers + loops*s.LoopLayers
}
