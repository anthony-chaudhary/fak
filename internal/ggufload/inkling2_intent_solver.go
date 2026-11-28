package ggufload

import (
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ResolveInkling2ServeIntent reconciles a declarative ServeIntent against hp
// and hw into a concrete ResolvedPlan using the exact resident VRAM accounting
// of inkling2loop_resident.go.
func ResolveInkling2ServeIntent(hp model.Inkling2LoopHParams, intent ServeIntent, hw HardwareEnvelope) (ResolvedPlan, error) {
	return ResolveServeIntent(inkling2MemoryBudget{hp: hp}, intent, hw)
}
