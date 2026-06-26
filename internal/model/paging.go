package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// paging.go — pagedKernel, the first honest "paged to device on demand" primitive of the
// native-753B Pillar-4 (CPU/NVMe offload) track. The resident HAL (weightHAL / halW in hal.go)
// keeps a weight on the device for the WHOLE session — the "VRAM fits the model" assumption a
// 753B GLM-5.2 breaks. A paged weight instead lives on the device only for a SINGLE op: upload
// (page IN) → compute → free (page OUT), so VRAM holds just the active weight. That is the lever
// a >VRAM model needs, and it is honest about what "offload" means — today's --n-cpu-moe path is
// compute-placement (the expert stays host-resident), not tensor paging; this is the primitive
// that actually moves a weight onto the device on demand and frees it after.
//
// The contract this primitive pins, proven against the cpu-ref backend (where Upload/Free are the
// identity/no-op but the lifecycle and the counter are exact) and bit-equal to the resident MatMul:
//   - the GEMM is BIT-EQUAL to keeping the weight resident (same Upload + MatMul; only the
//     residency LIFETIME differs), so paging changes memory behavior, not numerics;
//   - pageIn counts each page-in, and because nothing is cached, N ops page in N times (a resident
//     cache would page in once and reuse — pageIn==1 for N ops). That distinction IS the proof the
//     weight is paged, not retained.
//
// Scope (honest): this is the STANDALONE primitive. Wiring it into the session weight HAL — a paged
// twin of weightHALQ4K/weightHALQ8 that bypasses halW so a memory-lean GLM-5.2 streams experts
// per-layer instead of holding them resident, with async/pinned H2D and a per-weight VRAM ring — is
// the next Pillar-4 step (P4 Async expert streaming). It lands the observable upload→compute→free
// rung that step builds on. It is NOT yet on the live serve path and does NOT claim a >VRAM serve.
type pagedKernel struct {
	be     compute.Backend
	pageIn int // observable: weights paged in (uploaded fresh) by this kernel; never cached
}

func newPagedKernel(be compute.Backend) *pagedKernel {
	if be == nil {
		be = compute.Default()
	}
	return &pagedKernel{be: be}
}

// matMul pages an f32 weight [out,in] onto the device, runs y = w·x, materializes the host result,
// then FREES the weight — the device holds the weight only across this one op. The result is copied
// out BEFORE the free so it never aliases the (possibly freed) weight buffer on a real device
// backend. Bit-equal to be.MatMul(resident(w), x); increments pageIn once per call (each call pages
// in afresh — the kernel caches nothing, which is the whole point).
func (p *pagedKernel) matMul(shape []int, w []float32, x compute.Tensor) []float32 {
	wt := uploadHostF32Class(p.be, shape, w, compute.MemoryOffload, "paged-weight")
	p.pageIn++
	y := p.be.MatMul(wt, x)
	out := append([]float32(nil), p.be.Read(y)...) // materialize before page-out
	p.be.Free(wt)                                  // page OUT: the weight leaves the device
	return out
}
