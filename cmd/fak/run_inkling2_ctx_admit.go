package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/memgate"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2ResidentCtxFloor is the context a resident session is always given when the
// device can physically hold the weights plus that cache, however loaded the host is.
const inkling2ResidentCtxFloor = 4096

// inkling2ResidentDeviceShare is the part of unified RAM the GPU may keep resident,
// Metal's recommended working set on Apple silicon. Live wired pages are not used for
// the ceiling: they are sampled before the GPU lease, while a previous holder may still
// have its own model wired.
const inkling2ResidentDeviceShare = 0.75

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
	ctx, err := inkling2ResidentCtxForHost(shape, mem, base.SteadyBytes)
	if err != nil {
		return 0, err
	}
	ggufload.Inkling2ResidentCtxPin(ctx)
	return ctx, nil
}

// inkling2ResidentCtxForHost picks the resident context for a host sample and the
// session's ctx-0 plan. The weights are mapped file pages the kernel evicts under
// pressure, so live available memory below the weight size means a busy page cache, not
// a device that cannot run: below the floor the question becomes whether the GPU's share
// of physical RAM holds the weights plus a floor-sized cache.
func inkling2ResidentCtxForHost(s inkling2ResidentShape, mem memgate.Memory, plan int64) (int, error) {
	floor := min(inkling2ResidentCtxFloor, inkling2ResidentCtxMax)
	if ctx := inkling2AdmittedCtx(s, mem.AvailableBytes-plan); ctx >= floor {
		return ctx, nil
	}
	device := int64(float64(mem.TotalBytes) * inkling2ResidentDeviceShare)
	if fit := inkling2AdmittedCtx(s, device-plan); fit >= floor {
		return floor, nil
	}
	return 0, fmt.Errorf("Inkling 2: the GPU share (%d bytes) of %d bytes of RAM cannot hold the %d-byte resident plan plus a %d-token context; set %s to a smaller value",
		device, mem.TotalBytes, plan, floor, inkling2ResidentCtxEnv)
}

// inkling2AdmittedCtx is the shared fit applied to the admission shape: the largest
// global context whose cache rows fit free bytes.
func inkling2AdmittedCtx(s inkling2ResidentShape, free int64) int {
	if free <= 0 {
		return 0
	}
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
	return min(ggufload.Inkling2ResidentFitCtx(hp, free), ggufload.Inkling2AutoCtxCeiling)
}
