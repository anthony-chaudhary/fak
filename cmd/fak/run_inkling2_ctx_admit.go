package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/memgate"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2AdmitResidentCtx sizes a resident Metal session's global cache before its
// admission is priced. An explicit FAK_INKLING2_CTX is kept as it is. Otherwise the
// context is the largest that fits the memory left after the weights, the same fit the
// driver applies on CUDA, and it is pinned so the driver builds exactly what was priced.
func inkling2AdmitResidentCtx(path string, f *ggufload.File, arch string) (int, error) {
	if !ggufload.Inkling2ResidentCtxAuto(os.LookupEnv) {
		return inkling2ResidentCtx(os.LookupEnv(inkling2ResidentCtxEnv)), nil
	}
	shape, err := inkling2ResidentShapeOf(f, arch)
	if err != nil {
		return 0, err
	}
	base, err := inkling2RunResidentMemory(path, f, arch, 0)
	if err != nil {
		return 0, err
	}
	mem, err := memgate.ReadMemory()
	if err != nil {
		return 0, fmt.Errorf("Inkling 2: cannot read host memory to size the resident context: %w", err)
	}
	ctx := inkling2AdmittedCtx(shape, mem.AvailableBytes-base.SteadyBytes)
	if ctx == 0 {
		return 0, fmt.Errorf("Inkling 2: %d bytes available after weights hold no resident context; set %s to a smaller value",
			mem.AvailableBytes-base.SteadyBytes, inkling2ResidentCtxEnv)
	}
	ggufload.Inkling2ResidentCtxPin(ctx)
	return ctx, nil
}

// inkling2AdmittedCtx is the shared fit applied to the admission shape: the largest
// global context whose cache rows fit free bytes.
func inkling2AdmittedCtx(s inkling2ResidentShape, free int64) int {
	hp := model.Inkling2LoopHParams{
		Layers:     len(s.Local),
		Passes:     s.Passes,
		CoreStart:  s.CoreStart,
		CoreEnd:    s.CoreEnd,
		Heads:      s.Heads,
		KVHeads:    s.KVHeads,
		HeadDim:    s.HeadDim,
		LocalLayer: s.Local,
	}
	return ggufload.Inkling2ResidentFitCtx(hp, free)
}
