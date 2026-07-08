package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// paging_ring.go — pagedRing, the bounded per-weight resident cache that turns pagedKernel's
// page-EVERY-op primitive into the "per-weight VRAM ring" the native-753B Pillar-4 "async expert
// streaming" step needs (issue #2726, epic #2722 — Mac CPU+RAM+SSD offload serving): hold as many
// paged weights device-resident as a weight-byte budget allows, reuse a resident handle on a hit,
// and page a cold weight in on a miss — evicting the coldest UNPINNED weight (LRU) when admitting
// the newcomer would exceed the budget.
//
// It is the middle tier between the two weight placements that exist today. splitKernel
// (moe_offload.go, the --n-cpu-moe split) keeps ALL experts HOST-resident forever — no device
// residency at all. pagedKernel (paging.go) uploads AND frees a weight on EVERY op — the device
// holds exactly one weight and nothing is cached. The ring is the bounded working set between them:
// a hot set of weights resident on the device, cold ones streamed on demand — the Tier-1 GPU ring
// of the SOTA 3-tier (GPU / pinned-CPU / SSD) expert cache.
//
// Policy REUSE, not reinvention: the byte budget, the LRU victim choice, the pinned-exemption and
// the all-or-nothing admit are ALL polymodel.Pool's — the SAME proven policy internal/residency's
// Manager reuses for whole models, here bound to per-WEIGHT device handles instead. So every
// polymodel invariant its witness suite asserts (used<=budget, pinned-never-evicted,
// admit-unchanged-on-error) holds here by construction; the ring adds only the name->handle binding
// and the Upload-on-miss / Free-on-evict lifecycle.
//
// Correctness contract (pinned by paging_ring_test.go against the cpu-ref backend, where Upload/Free
// are no-ops but the lifecycle and the counters are exact):
//   - a ring matMul is BIT-EQUAL to a resident MatMul whether the weight was a HIT (resident handle
//     reused) or a MISS (freshly uploaded) — only the residency LIFETIME differs, never the math;
//   - pageIn counts each cold upload, hit counts each resident reuse, evict counts each page-out
//     (LRU or explicit); used()<=budget() always, so the device footprint is bounded;
//   - a weight evicted under budget pressure pages IN again on its next use — it is not silently
//     lost, distinguishing a ring (bounded residency) from splitKernel (unbounded host residency).
//
// Scope (honest, matching pagedKernel + residency.Manager): a STANDALONE primitive, OFF the live
// serve path, no async/pinned H2D yet. It stages f32, Q8_0, or Q4_K weights (matMul / matMulQ8 /
// matMulQ4K), so the ring accounts the QUANTIZED residency a memory-lean MoE host actually streams
// (#3174 R1) rather than an f32 expansion of it — but wiring it into the session weight HAL as the
// paged twin of weightHALQ4K/weightHALQ8 (so a real model streams experts per-layer under a device
// budget) is still the follow-on rung of #2726. This lands the bounded-residency lifecycle + the
// witness that rung builds on. It moves no bytes and links nothing new into the default binary.
type pagedRing struct {
	be   compute.Backend
	pool *polymodel.Pool
	// resident is the name->live device weight handle map; it is kept in exact lockstep with pool
	// (a name is in one iff it is in the other), so pool's residency bookkeeping governs the handles.
	resident map[polymodel.ModelID]compute.Tensor

	pageIn int // cold uploads (a miss that was admitted)
	hit    int // resident reuses (a handle served without upload)
	evict  int // page-outs (LRU victims dropped during admit to stay within budget)
}

// newPagedRing returns a ring over be with the given resident weight-byte budget. A nil backend
// uses compute.Default() (the cpu-ref backend); a negative budget is clamped to 0 by polymodel
// (every admit then pages straight back out), matching pagedKernel/NewPool.
func newPagedRing(be compute.Backend, budgetBytes int64) *pagedRing {
	if be == nil {
		be = compute.Default()
	}
	return &pagedRing{
		be:       be,
		pool:     polymodel.NewPool(budgetBytes),
		resident: map[polymodel.ModelID]compute.Tensor{},
	}
}

// uploadStaged stages a host source tensor onto be as dtype under the MemoryOffload class — the same
// routing uploadHostF32Class uses (UploadClass when the backend honors placement, plain Upload
// otherwise), but dtype-general so the ring can stage a RESIDENT quantized weight (Q8_0 / Q4_K)
// exactly as the Session weightHALQ8/weightHALQ4K staged builders do. On the cpu-ref backend (no
// UploadClass) it falls through to be.Upload(src, dtype) — the path the quantized HAL and its
// witnesses already pin.
func uploadStaged(be compute.Backend, src compute.Tensor, dtype compute.Dtype, site string) compute.Tensor {
	if b, ok := be.(classedUploadBackend); ok {
		return b.UploadClass(src, dtype, compute.MemoryOffload, site)
	}
	return be.Upload(src, dtype)
}

// matMulStaged is the dtype-general ring core shared by the f32 matMul and the quantized matMulQ8 /
// matMulQ4K twins. It runs y = w·x for the named weight, keeping the uploaded device handle resident
// under the ring budget. On a HIT (name already resident) the handle is reused and Touched (no
// upload); on a MISS mk builds the host source (f32 / Q8_0 / Q4_K, exactly as the weightHAL staged
// builders do), it is uploaded as dtype (page-in) and admitted to the pool — evicting the coldest
// UNPINNED residents to stay within budget, whose handles are Freed — and retained. Bit-equal to a
// resident Upload(mk(), dtype)+MatMul either way; only the residency lifetime differs. weightBytes is
// the RESIDENT footprint the budget accounts (caller-supplied, exactly as polymodel.Model.WeightBytes)
// — the QUANTIZED size for a quantized weight (Q8_0 ~1.125 B/weight, Q4_K ~0.56), so a ring sized for
// N f32 experts holds several times as many quantized ones. Returns nil — paging nothing and leaving
// the resident set unchanged — when the weight alone exceeds the budget (polymodel.ErrTooLarge) or
// fits only by dropping a pinned resident (ErrPinnedNoRoom); the caller then falls back to a per-op
// paged (pagedKernel) or host GEMM, exactly as an over-budget weight would.
func (r *pagedRing) matMulStaged(name string, mk func() compute.Tensor, dtype compute.Dtype, x compute.Tensor, weightBytes int64, pinned bool) []float32 {
	id := polymodel.ModelID(name)
	if wt, ok := r.resident[id]; ok {
		r.pool.Touch(id)
		r.hit++
		return append([]float32(nil), r.be.Read(r.be.MatMul(wt, x))...)
	}
	// Miss: build + upload the weight, then admit it under the budget. Admit is all-or-nothing: on
	// error the pool is unchanged, so page the just-uploaded handle straight back out and defer.
	wt := uploadStaged(r.be, mk(), dtype, "paged-ring-weight")
	evicted, err := r.pool.Admit(polymodel.Model{ID: id, WeightBytes: weightBytes, Pinned: pinned})
	if err != nil {
		r.be.Free(wt)
		return nil
	}
	r.pageIn++
	for _, vid := range evicted {
		if vh, ok := r.resident[vid]; ok {
			r.be.Free(vh) // page the LRU victim out; its device storage is released
			delete(r.resident, vid)
			r.evict++
		}
	}
	r.resident[id] = wt
	return append([]float32(nil), r.be.Read(r.be.MatMul(wt, x))...)
}

// matMul runs y = w·x for the named f32 weight [out,in] under the ring budget — the original ring
// path, now a thin f32 specialization of matMulStaged. See matMulStaged for the hit/miss/evict
// lifecycle and the nil-on-unadmittable contract.
func (r *pagedRing) matMul(name string, shape []int, w []float32, x compute.Tensor, weightBytes int64, pinned bool) []float32 {
	return r.matMulStaged(name, func() compute.Tensor {
		return compute.NewF32(compute.Default(), append([]int(nil), shape...), w)
	}, compute.F32, x, weightBytes, pinned)
}

// matMulQ8 is the Q8_0 twin of matMul: it stages qt as a RESIDENT Q8_0 weight under the ring budget
// and runs the quantized GEMM the cpu-ref / cuda Q8 kernel serves — the path past the ring's former
// f32-only scope (#3174 R1: a MoE expert-weight cache holds the QUANTIZED experts a memory-lean host
// actually streams, ~1.125 B/weight, not their f32 expansion). qt is the same prequantized source
// the Session weightHALQ8 stages; weightBytes is its resident Q8 footprint. Bit-equal to a resident
// Q8 GEMM on a hit or a miss.
func (r *pagedRing) matMulQ8(name string, qt *q8Tensor, x compute.Tensor, weightBytes int64, pinned bool) []float32 {
	return r.matMulStaged(name, func() compute.Tensor {
		return compute.NewQ8(compute.Default(), []int{qt.out, qt.in}, qt.q, qt.d, qBlk)
	}, compute.Q8_0, x, weightBytes, pinned)
}

// matMulQ4K is the Q4_K twin of matMul: it stages qt (raw GGUF super-blocks) as a RESIDENT Q4_K
// weight under the ring budget and runs the dequant-fused Q4_K GEMM (k_q4k_gemm on cuda, the Q4_K
// MatMul on cpu-ref) — the ~0.56 B/weight residency that lets a 753B GLM-5.2's expert majority ride
// a bounded device ring. qt is the same source the Session weightHALQ4K stages; weightBytes is its
// resident Q4_K footprint. Bit-equal to a resident Q4_K GEMM on a hit or a miss.
func (r *pagedRing) matMulQ4K(name string, qt *q4kTensor, x compute.Tensor, weightBytes int64, pinned bool) []float32 {
	return r.matMulStaged(name, func() compute.Tensor {
		return compute.NewQ4K(compute.Default(), []int{qt.out, qt.in}, qt.raw)
	}, compute.Q4_K, x, weightBytes, pinned)
}

// isResident reports whether the named weight currently holds a device handle in the ring.
func (r *pagedRing) isResident(name string) bool {
	_, ok := r.resident[polymodel.ModelID(name)]
	return ok
}

// residentCount is the number of weights holding a device handle (== the pool's resident count).
func (r *pagedRing) residentCount() int { return len(r.resident) }

// used / budget expose the polymodel byte accounting: used() <= budget() always.
func (r *pagedRing) used() int64   { return r.pool.Used() }
func (r *pagedRing) budget() int64 { return r.pool.Budget() }
