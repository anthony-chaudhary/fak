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
// Scope: it stages f32, Q8_0, or Q4_K weights (matMul / matMulQ8 / matMulQ4K), so the ring accounts
// the QUANTIZED residency a memory-lean MoE host actually streams (#3174 R1) rather than an f32
// expansion of it. The follow-on rung #2726 named — wiring it into the session weight HAL as the
// bounded twin of weightHALQ4K/weightHALQ8, so a real model streams ROUTED EXPERTS under a device
// budget — landed as #5611 (epic #5606 R0) via stage() + expert_ring_hal.go; it is opt-in per
// session (Session.ExpertRingBytes > 0) and inert at the default 0. Still no async/pinned H2D: a
// miss uploads synchronously (the prefetch seam is #5614).
type pagedRing struct {
	be   compute.Backend
	pool *polymodel.Pool
	// resident is the name->live device weight handle map; it is kept in exact lockstep with pool
	// (a name is in one iff it is in the other), so pool's residency bookkeeping governs the handles.
	resident map[polymodel.ModelID]compute.Tensor

	pageIn int // cold uploads (a miss that was admitted)
	hit    int // resident reuses (a handle served without upload)
	evict  int // page-outs (LRU victims dropped during admit to stay within budget)

	// peak is the high-water mark of pool.Used() — the largest device weight footprint the ring ever
	// held. It is the boundedness WITNESS: peak <= budget across an arbitrary access sequence is what
	// distinguishes a ring from the unbounded halW memoizer, and it cannot be recovered after the fact
	// from used() (which only reports the instantaneous set).
	peak int64

	// holds counts the live hold() spans per weight (see hold/release). A weight with a live hold is
	// PINNED in the pool for the span, so admitting a later weight cannot evict a handle its caller is
	// still using — the invariant a multi-weight op (one expert's gate/up/down) depends on.
	holds map[polymodel.ModelID]int
	// heldPins records the ids whose Pinned bit hold() itself set, so release() restores only those and
	// never clears a pin the durable pin-set owns.
	heldPins map[polymodel.ModelID]bool

	// pins is the online-learning resident pin-set (expert_warmpins.go): the workload-personalized
	// hot-set warm-started from the summed cross-session usage histogram and drifted between turns by
	// RepinPass. nil until WarmStartPins seeds it — a ring that was never warm-started has no pin-set,
	// so isExpertPinned reports false and RepinPass is a no-op. R2/#5613 gave it its live-path
	// consumer: weightHALStagedBounded computes matMulStaged's `pinned` from it per routed expert
	// (expert_ring_pins.go), so the pool's pinned-never-evicted invariant now protects the workload's
	// hot set instead of nothing.
	pins *ExpertPinSet
	// turn is THIS turn's routed-expert usage, folded in by observeExpert and consumed by
	// Session.ExpertRingEndTurn (which decays the standing heat, folds this in, repins, and dumps).
	// nil alongside a nil pins, so a ring that was never warm-started observes nothing.
	turn *ExpertUsageHistogram
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
	wt, ok := r.stage(name, mk, dtype, weightBytes, pinned)
	if !ok {
		return nil
	}
	return append([]float32(nil), r.be.Read(r.be.MatMul(wt, x))...)
}

// stage is the RESIDENCY half of matMulStaged with the GEMM removed: it makes the named weight
// device-resident under the ring budget and hands back the live handle, so a caller that wants to
// run its OWN ops against the weight (the session weight HAL, whose expertSwiGLUHAL issues three
// MatMuls and a SwiGLU itself) gets the bounded residency without the ring dictating the math.
// matMulStaged is stage + MatMul, so the two share one lifecycle by construction and the bit-equality
// the ring witnesses for matMulStaged transfers to every stage() caller: on a HIT the resident handle
// is reused and Touched, on a MISS mk builds the host source, it is uploaded as dtype (page-in) and
// admitted — evicting the coldest UNPINNED residents, whose handles are Freed — and retained.
//
// Returns ok=false, having paged nothing and left the resident set unchanged, when the weight is
// unadmittable (polymodel.ErrTooLarge — it alone exceeds the budget; or ErrPinnedNoRoom — it fits
// only by dropping a pinned resident). The caller then falls back to its own unbounded/host path,
// exactly as matMulStaged's nil return means.
//
// The returned handle is valid only until the NEXT stage/matMul on the same ring may evict it. A
// caller holding several handles at once must hold() each for the span it uses them (see hold).
func (r *pagedRing) stage(name string, mk func() compute.Tensor, dtype compute.Dtype, weightBytes int64, pinned bool) (compute.Tensor, bool) {
	id := polymodel.ModelID(name)
	if wt, ok := r.resident[id]; ok {
		r.pool.Touch(id)
		r.hit++
		return wt, true
	}
	// Miss: build + upload the weight, then admit it under the budget. Admit is all-or-nothing: on
	// error the pool is unchanged, so page the just-uploaded handle straight back out and defer.
	wt := uploadStaged(r.be, mk(), dtype, "paged-ring-weight")
	evicted, err := r.pool.Admit(polymodel.Model{ID: id, WeightBytes: weightBytes, Pinned: pinned})
	if err != nil {
		r.be.Free(wt)
		return compute.Tensor{}, false
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
	if used := r.pool.Used(); used > r.peak {
		r.peak = used
	}
	return wt, true
}

// hold protects an already-staged weight from eviction for the span of a multi-weight computation,
// and release ends that span. They exist because one MoE expert is THREE weights (gate/up/down) used
// together: without a hold, staging `up` under a tight budget could evict `gate` and Free a handle
// the caller is about to MatMul against — a use-after-free the unbounded halW memoizer could never
// produce. Holds nest (a second hold on the same weight only increments), so overlapping spans are safe.
//
// The mechanism is polymodel's own documented recipe for changing a resident descriptor — Evict then
// re-Admit with Pinned set — not a new policy: a held weight is exactly a pinned resident, so it
// inherits the pool's proven pinned-never-evicted invariant. release restores ONLY pins hold itself
// set (heldPins), so a durable pin from the warm-start pin-set survives a hold/release cycle. A hold
// on a weight the ring does not hold (an unadmittable one that fell back to the caller's own path) is
// a no-op, so callers need not branch on whether staging was served by the ring.
func (r *pagedRing) hold(name string) {
	id := polymodel.ModelID(name)
	if r.holds == nil {
		r.holds = map[polymodel.ModelID]int{}
	}
	r.holds[id]++
	if r.holds[id] > 1 {
		return // already held; the first hold owns the pin
	}
	m, ok := r.pool.Get(id)
	if !ok || m.Pinned {
		return // not ring-resident, or already pinned by the durable pin-set — nothing to set or restore
	}
	m.Pinned = true
	r.repin(id, m)
	if r.heldPins == nil {
		r.heldPins = map[polymodel.ModelID]bool{}
	}
	r.heldPins[id] = true
}

// release ends one hold span (see hold). The last release of a weight hold() pinned clears that pin,
// returning it to the LRU victim pool.
func (r *pagedRing) release(name string) {
	id := polymodel.ModelID(name)
	n := r.holds[id]
	if n <= 0 {
		return
	}
	if n > 1 {
		r.holds[id] = n - 1
		return
	}
	delete(r.holds, id)
	if !r.heldPins[id] {
		return
	}
	delete(r.heldPins, id)
	if m, ok := r.pool.Get(id); ok {
		m.Pinned = false
		r.repin(id, m)
	}
}

// repin rewrites a resident descriptor in place via polymodel's Evict-then-Admit recipe (the pool
// documents the descriptor as immutable otherwise). Re-admitting bytes the pool just released cannot
// exceed a budget they already fit, so the Admit cannot fail; if it ever did, the handle is paged out
// so `resident` and `pool` stay in the exact lockstep the rest of the ring assumes.
func (r *pagedRing) repin(id polymodel.ModelID, m polymodel.Model) {
	r.pool.Evict(id)
	if _, err := r.pool.Admit(m); err != nil {
		if h, live := r.resident[id]; live {
			r.be.Free(h)
			delete(r.resident, id)
		}
		delete(r.holds, id)
		delete(r.heldPins, id)
	}
}

// freeAll pages every resident weight out and drops the pool accounting to zero — the teardown a ring
// owner runs at close so no device handle outlives the session that staged it.
func (r *pagedRing) freeAll() {
	if r == nil {
		return
	}
	for id, t := range r.resident {
		r.be.Free(t)
		delete(r.resident, id)
		r.pool.Evict(id)
	}
	r.holds, r.heldPins = nil, nil
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

// peakUsed is the high-water mark of used() over the ring's life — the number a boundedness witness
// asserts against budget(), since used() alone cannot show what the footprint reached in between.
func (r *pagedRing) peakUsed() int64 { return r.peak }
