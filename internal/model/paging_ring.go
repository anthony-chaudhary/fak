package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// paging_ring.go — pagedRing, the bounded per-weight resident cache: hold as many paged weights
// device-resident as a weight-byte budget allows, reuse a resident handle on a hit, and page a cold
// weight in on a miss, evicting the coldest unpinned weight when the budget would be exceeded.
//
// It is the middle tier between the two weight placements that bracket it: splitKernel (moe_offload.go)
// keeps all experts host-resident forever, and pagedKernel (paging.go) uploads and frees a weight on
// every op. The ring is the bounded working set between them — the Tier-1 GPU ring of the 3-tier
// (GPU / pinned-CPU / SSD) expert cache.
//
// Policy reuse, not reinvention: the byte budget, LRU victim choice, pinned exemption, and
// all-or-nothing admit are all polymodel.Pool's, bound here to per-weight device handles, so every
// polymodel invariant holds by construction; the ring adds only the name->handle binding and the
// Upload-on-miss / Free-on-evict lifecycle.
//
// Correctness contract: a ring matMul is bit-equal to a resident MatMul whether the weight was a hit
// or a miss — only the residency lifetime differs, never the math; a healthy ring stays within budget; an
// evicted weight pages in again on its next use. A terminal refill keeps its detached bytes charged,
// even if a later control shrinks the budget. Stages f32, Q8_0, or Q4_K weights, accounting the
// quantized residency a memory-lean MoE host actually streams. Wired into the session weight HAL via
// stage() + expert_ring_hal.go, opt-in per session (Session.ExpertRingBytes > 0).
//
// Async H2D: a backend advertising compute.AsyncUploader stages through UploadAsync and the ring
// holds the Fence in pending until the weight is demanded. A demand stage always settles its fence
// before returning a handle, so callers keep a synchronous contract; only the activated-set
// prefetch returns with a transfer in flight, which is what makes overlap measurable (awaitStaged).
// A backend without the extension takes the synchronous path with a nil fence.
type pagedRing struct {
	be   compute.Backend
	pool *polymodel.Pool
	// resident is kept in exact lockstep with pool (a name is in one iff it is in the other).
	resident map[polymodel.ModelID]compute.Tensor
	// An optional eager refill owns a detached slot until checked publication.
	// A terminal record retains/quarantines its bytes and blocks further staging.
	refill *expertRingRefill

	pageIn int // cold uploads (a miss that was admitted)
	hit    int // resident reuses (a handle served without upload)
	evict  int // page-outs (LRU victims dropped during admit to stay within budget)

	// Reconciliation pair: every stage() call ends in exactly one of hit / pageIn / refused, so
	// hit + pageIn + refused == lookups is an identity an operator surface can check. Counted
	// independently of the outcome counters on purpose.
	lookups int
	refused int
	// Total device bytes moved by cold uploads; bytes-per-token cannot be recovered from pageIn
	// alone because routed-expert projections differ in size and quantization.
	pageInBytes int64

	// In-flight transfer Fences not yet awaited. Nothing may read, multiply, or Free a handle
	// while its id is still here.
	pending map[polymodel.ModelID]*expertRingTransfer
	// Split the awaited fences by whether the transfer had already landed when demanded: the
	// clock-free overlap meter.
	asyncOverlapped int
	asyncWaited     int
	// Live execution scope this ring registered a pending-fence drain on; only while live may an
	// owner-capable backend issue async.
	drainScope *backendExecutionScope

	// High-water mark of pool.Used(): the boundedness witness, unrecoverable from used() alone.
	peak int64

	// Scan-resistant stable set (expert_ring_stable.go): cold admits are pinned while their bytes
	// fit stableCap, so a layer-cyclic sweep larger than the ring cannot flush them. 0 = off.
	stableCap   int64
	stableBytes int64

	// Live hold() spans per weight; a held weight is pinned in the pool for the span, so a later
	// admit cannot evict a handle its caller is still using.
	holds map[polymodel.ModelID]int
	// Ids whose Pinned bit hold() itself set, so release() restores only those and never clears a
	// pin the durable pin-set owns.
	heldPins map[polymodel.ModelID]bool

	// Online-learning resident pin-set (expert_warmpins.go): the workload hot-set protected by the
	// pool's pinned-never-evicted invariant. nil until WarmStartPins seeds it.
	pins *ExpertPinSet
	// This turn's routed-expert usage, consumed by Session.ExpertRingEndTurn. nil alongside a nil pins.
	turn *ExpertUsageHistogram
	// Online hot-set learner (expert_hot_set.go), built lazily with the pin-set and only when
	// Session.ExpertHotSetHysteresis > 0; seeds the durable pin-set from learned activation
	// statistics. nil on every default session.
	hotSet *ExpertHotSetLearner

	// Victim ranking among unpinned residents (expert_ring_policy.go). Zero value (LRU) keeps
	// polymodel's coldestUnpinned choice; value-aware picks victims by decaying heat and evicts
	// them before Admit.
	policy ExpertRingEvictPolicy
	// Starts at one so zero remains an uninitialized/not-a-ring value in exported receipts.
	policyGeneration uint64
	// Per-weight decaying access counter (value-aware ranking) and recency tie-break clock. heat is
	// nil under LRU and retains an evicted weight's count as ghost heat.
	heat     map[polymodel.ModelID]int
	lastUse  map[polymodel.ModelID]uint64
	clock    uint64
	accesses int

	// Ordered routed-expert access sequence for the offline regret gauge; the histogram above is
	// aggregate and loses the order Belady needs. traceDropped marks a truncated window.
	trace        []ExpertAccessTraceEvent
	traceDropped int

	// Marks activated-set prefetch stagings as hints rather than demands: they take recency but
	// earn no heat, so the policy ranks on what was used, not what was speculated.
	prefetching bool
	// Coverage meter: of the experts the router activated, how many the budget could actually hold.
	prefetched       int
	activatedExperts int
	activatedCovered int

	// Cross-agent owner when this ring is the residency of a (model, device) pair rather than one
	// conversation; nil on a private per-session ring (the default).
	//
	// The methods in this file never lock, under any owner. Serialization happens at the
	// session-level entry points (Session.ringEnter), leaving one place to audit and no lock
	// ordering to get wrong — see expert_ring_shared.go's header.
	shared *SharedExpertRing
}

// dropResidency closes a weight's residency span in the cross-agent ledger, called wherever a
// handle leaves resident, so the next page-in opens a new span with a new payer.
func (r *pagedRing) dropResidency(id polymodel.ModelID) {
	if r.shared != nil {
		r.shared.endResidency(id)
	}
}

// newPagedRing returns a ring over be with the given resident weight-byte budget. A nil backend
// uses compute.Default() (the cpu-ref backend); a negative budget is clamped to 0 by polymodel
// matching pagedKernel/NewPool.
func newPagedRing(be compute.Backend, budgetBytes int64) *pagedRing {
	if be == nil {
		be = compute.Default()
	}
	return &pagedRing{
		be:               be,
		pool:             polymodel.NewPool(budgetBytes),
		resident:         map[polymodel.ModelID]compute.Tensor{},
		policyGeneration: 1,
	}
}

// uploadStaged stages a host source tensor onto be as dtype under the MemoryOffload class
// (UploadClass when the backend honors placement, plain Upload otherwise), dtype-general so the
// ring can stage a resident quantized weight exactly as the Session weight HAL builders do.
func uploadStaged(be compute.Backend, src compute.Tensor, dtype compute.Dtype, site string) compute.Tensor {
	if b, ok := be.(classedUploadBackend); ok {
		return b.UploadClass(src, dtype, compute.MemoryOffload, site)
	}
	return be.Upload(src, dtype)
}

// uploadStagedAsync is uploadStaged over compute.AsyncUploader when the backend offers one,
// returning the in-flight handle plus its Fence; a backend without the extension falls through to
// the synchronous path with a nil Fence, which awaitStaged treats as already-landed.
//
// Two guards keep an async transfer from escaping what owns it:
//   - CLASS: a placement-aware backend goes async only through ClassedAsyncUploader, which books
//     the allocation under MemoryOffload; a plain UploadAsync would leave it unlabelled and corrupt
//     the offload accounting that sizes the ring.
//   - DOMAIN: an undemanded prefetch fence can outlive the enclosing forward, so on a backend with
//     a recording-domain owner the transfer goes async only when domainJoined: the ring has
//     registered a drain on the live execution scope that settles every pending fence before the
//     domain mutex is released.
func uploadStagedAsync(be compute.Backend, src compute.Tensor, dtype compute.Dtype, site string, domainJoined bool) (compute.Tensor, compute.Fence) {
	if _, owned := be.(compute.ForwardExecutionOwnerBackend); owned && !domainJoined {
		return uploadStaged(be, src, dtype, site), nil
	}
	if _, classed := be.(classedUploadBackend); classed {
		if cb, ok := be.(compute.ClassedAsyncUploader); ok {
			return cb.UploadAsyncClass(src, dtype, compute.MemoryOffload, site)
		}
		return uploadStaged(be, src, dtype, site), nil
	}
	if ab, ok := be.(compute.AsyncUploader); ok {
		return ab.UploadAsync(src, dtype)
	}
	return uploadStaged(be, src, dtype, site), nil
}

// domainJoined reports whether a transfer issued now is covered by a drain registered on the
// currently live execution scope. A released scope reads false, so a stale arming is inert.
func (r *pagedRing) domainJoined() bool {
	return r.drainScope != nil && r.drainScope.live
}

// settlePendingTransfers waits out every in-flight transfer without booking the overlap meter: a
// prefetch still undemanded when its forward ends computed nothing. The weights stay resident; the
// next demand is an ordinary hit.
func (r *pagedRing) settlePendingTransfers() {
	r.drainScope = nil
	r.drainTransfers()
	r.requireRefillIdle()
}

// matMulStaged is the dtype-general ring core shared by the f32 matMul and the quantized
// matMulQ8 / matMulQ4K twins: it runs y = w·x for the named weight, keeping the uploaded device
// handle resident under the ring budget. On a hit the handle is reused and Touched; on a miss mk
// builds the host source, it is uploaded (page-in) and admitted, evicting the coldest unpinned
// residents to stay within budget. Bit-equal to a resident Upload+MatMul either way; only the
// residency lifetime differs. weightBytes is the resident (quantized) footprint the budget
// accounts. Returns nil — paging nothing — when the weight alone exceeds the budget
// (polymodel.ErrTooLarge) or fits only by dropping a pinned resident (ErrPinnedNoRoom); the caller
// then falls back to a per-op paged or host GEMM.
func (r *pagedRing) matMulStaged(name string, mk func() compute.Tensor, dtype compute.Dtype, x compute.Tensor, weightBytes int64, pinned bool) []float32 {
	wt, ok := r.stage(name, mk, dtype, weightBytes, pinned)
	if !ok {
		return nil
	}
	return append([]float32(nil), r.be.Read(r.be.MatMul(wt, x))...)
}

// stage is the residency half of matMulStaged with the GEMM removed: it makes the named weight
// device-resident under the ring budget and hands back the live handle, so a caller that wants to
// run its own ops against the weight (the session weight HAL) gets the bounded residency without
// the ring dictating the math. matMulStaged is stage + MatMul, so the two share one lifecycle by
// construction.
//
// Returns ok=false, having paged nothing, when the weight is unadmittable (polymodel.ErrTooLarge or
// ErrPinnedNoRoom). The returned handle is valid only until the next stage/matMul on the same ring
// may evict it; a caller holding several handles at once must hold() each for the span it uses them.
func (r *pagedRing) stage(name string, mk func() compute.Tensor, dtype compute.Dtype, weightBytes int64, pinned bool) (compute.Tensor, bool) {
	t, err := r.stageWithError(name, mk, dtype, weightBytes, pinned)
	if expertRingIsTransferError(err) {
		panic(err) // A failed copy is not a budget refusal permitting a fallback upload.
	}
	return t, err == nil
}

func (r *pagedRing) stageWithError(name string, mk func() compute.Tensor, dtype compute.Dtype, weightBytes int64, pinned bool) (compute.Tensor, error) {
	id := polymodel.ModelID(name)
	r.lookups++ // booked before anything is decided, so it counts every exit
	booked := false
	defer func() {
		if !booked {
			r.refused++
			if r.shared != nil {
				r.shared.noteRefusal()
			}
		}
	}()
	r.requireRefillIdle()
	r.noteAccess(id) // recency clock always, decaying heat under a value-aware policy
	// A demand is an agent asking for bytes, so the prefetch's hints are excluded from the ledger.
	if r.shared != nil && !r.prefetching {
		r.shared.noteDemand()
	}
	if wt, ok := r.resident[id]; ok {
		// A hit on a prefetched weight is where overlap is observed; skip the fence while
		// prefetching so re-hinting an in-flight weight does not block the hint path.
		if !r.prefetching {
			if err := r.awaitStaged(id); err != nil {
				return compute.Tensor{}, err
			}
		}
		r.pool.Touch(id)
		r.hit++
		booked = true
		r.noteDemandServe(id)
		return wt, nil
	}
	if p := r.pending[id]; p != nil {
		return compute.Tensor{}, p.refused(compute.ErrOwnedTransferUnknown)
	}
	// Refuse before building the host source or uploading: a post-upload refusal would briefly
	// allocate the very weight the budget excludes.
	if weightBytes > r.pool.Budget() {
		return compute.Tensor{}, polymodel.ErrTooLarge
	}
	// Reuse the pool's snapshot check rather than duplicating its pinned/held feasibility
	// rule. This is no reservation: the final Admit still rechecks.
	stable := !pinned && r.stableAdmit(weightBytes)
	if stable {
		pinned = true
	}
	if r.pool.CheckAdmission(polymodel.Model{ID: id, WeightBytes: weightBytes, Pinned: pinned}) == polymodel.ErrPinnedNoRoom {
		return compute.Tensor{}, polymodel.ErrPinnedNoRoom
	}
	// Build + upload, then admit. Admit is all-or-nothing: on error the pool is unchanged, so page
	// the just-uploaded handle straight back out. Under a non-default victim policy the ring makes
	// room itself first (evictForPolicy), so Admit fits and polymodel's LRU choice never applies.
	source := mk()
	m := polymodel.Model{ID: id, WeightBytes: weightBytes, Pinned: pinned}
	if wt, reused := r.tryRefill(source, m, dtype); reused {
		r.noteStableAdmit(stable, weightBytes)
		r.pageIn++
		booked = true
		r.pageInBytes += weightBytes // Reusing storage does not remove H2D traffic.
		r.noteDemandServe(id)
		return wt, nil
	}
	wt, fence := uploadStagedAsync(r.be, source, dtype, "paged-ring-weight", r.domainJoined())
	if err := r.retainStaged(id, source, wt, fence); err != nil {
		return compute.Tensor{}, err
	}
	if err := r.settleAdmissionVictims(m); err != nil {
		if cleanup := r.rollbackStaged(id, wt); cleanup != nil {
			return compute.Tensor{}, cleanup
		}
		return compute.Tensor{}, err
	}
	r.evictForPolicy(weightBytes)
	evicted, err := r.pool.Admit(m)
	if err != nil {
		if cleanup := r.rollbackStaged(id, wt); cleanup != nil {
			return compute.Tensor{}, cleanup
		}
		return compute.Tensor{}, err
	}
	r.noteStableAdmit(stable, weightBytes)
	r.pageIn++
	booked = true
	r.pageInBytes += weightBytes
	if r.shared != nil {
		// The payer is booked even for a prefetch: a later agent served off that span is
		// coalescing whether the span opened on a demand or on a guess.
		r.shared.notePageIn(id, weightBytes)
	}
	// Admission owns this tensor even if retiring a previous allocation panics.
	// Install it before callbacks so a scope drain cannot mistake it for rollback.
	r.resident[id] = wt
	if used := r.pool.Used(); used > r.peak {
		r.peak = used
	}
	for _, vid := range evicted {
		if vh, ok := r.resident[vid]; ok {
			// A victim may still be in flight; settling the fence first avoids freeing storage a
			// transfer is writing into, and nothing computed against it, so no overlap is booked.
			r.discardStaged(vid)
			r.be.Free(vh)
			delete(r.resident, vid)
			r.dropResidency(vid)
			r.evict++
		}
	}
	// A demand miss must hand back a settled handle; only the prefetch path may return with the
	// transfer still in flight.
	if !r.prefetching {
		if err := r.awaitStaged(id); err != nil {
			return compute.Tensor{}, err
		}
	}
	r.noteDemandServe(id)
	return wt, nil
}

func (r *pagedRing) noteDemandServe(id polymodel.ModelID) {
	if r.shared != nil && !r.prefetching {
		r.shared.noteServe(id)
	}
}

// awaitStaged makes a staged weight visible to subsequent ops and books the overlap. The Done()
// poll happens before the Wait, and that ordering is the whole meter: a fence already satisfied
// when demanded moved its bytes underneath other work; counting the two is clock-free. Idempotent
// and safe on a weight never staged asynchronously.
func (r *pagedRing) awaitStaged(id polymodel.ModelID) error {
	return r.settleStaged(id, true)
}

// discardStaged settles an in-flight transfer whose weight is about to be dropped, without
// booking it in the overlap meter: nothing ever computed against it, so counting it either way
// would bias the fraction the meter exists to report.
func (r *pagedRing) discardStaged(id polymodel.ModelID) {
	if err := r.settleStaged(id, false); err != nil {
		panic(err)
	}
}

// stageInFlight makes the named weight device-resident without settling its transfer, issuing the
// upload back to back with its siblings so the host->device copies overlap the compute that
// follows. It is stage minus the demand-path await: the caller settles the fence by demanding the
// weight through the ordinary path. Otherwise the semantics are exactly stage's, with the
// prefetching flag set for the duration so the transfer is booked as a hint, not a demand.
func (r *pagedRing) stageInFlight(name string, mk func() compute.Tensor, dtype compute.Dtype, weightBytes int64, pinned bool) (compute.Tensor, bool) {
	wasPrefetching := r.prefetching
	r.prefetching = true
	defer func() { r.prefetching = wasPrefetching }()
	return r.stage(name, mk, dtype, weightBytes, pinned)
}

// hold protects an already-staged weight from eviction for the span of a multi-weight computation,
// and release ends that span. One MoE expert is three weights (gate/up/down) used together: without
// a hold, staging `up` under a tight budget could evict `gate` and Free a handle the caller is about
// to MatMul against — a use-after-free the unbounded halW memoizer could never produce. Holds nest.
//
// The mechanism is polymodel's Evict-then-re-Admit-with-Pinned recipe: a held weight is exactly a
// pinned resident, inheriting the pool's pinned-never-evicted invariant. release restores only pins
// hold itself set (heldPins), so a durable pin from the warm-start pin-set survives. A hold on a
// weight the ring does not hold is a no-op.
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

// release ends one hold span; the last release of a weight hold() pinned clears that pin, returning
// it to the LRU victim pool.
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
// documents the descriptor as immutable otherwise). If the Admit ever failed, the handle is paged
// out so resident and pool stay in lockstep.
func (r *pagedRing) repin(id polymodel.ModelID, m polymodel.Model) {
	// Normal holds change only Pinned and must not prematurely settle prefetch.
	// A different identity/size can fail after removal, so settle it first.
	if old, ok := r.pool.Get(id); !ok || m.ID != id || m.WeightBytes != old.WeightBytes {
		r.discardStaged(id)
	}
	r.pool.Evict(id)
	if _, err := r.pool.Admit(m); err != nil {
		if h, live := r.resident[id]; live {
			r.discardStaged(id)
			r.be.Free(h)
			delete(r.resident, id)
			r.dropResidency(id)
		}
		delete(r.holds, id)
		delete(r.heldPins, id)
	}
}

// freeAll pages every resident weight out and drops the pool accounting to zero — the teardown a
// ring owner runs at close so no device handle outlives the session that staged it.
func (r *pagedRing) freeAll() {
	if r == nil {
		return
	}
	r.requireRefillIdle() // No optimistic Free or retry of an uncertain refill.
	r.drainTransfers()
	for id, t := range r.resident {
		r.discardStaged(id) // never tear down storage a transfer is still writing into
		r.be.Free(t)
		delete(r.resident, id)
		r.dropResidency(id)
		r.pool.Evict(id)
	}
	r.pending = nil
	r.holds, r.heldPins = nil, nil
	r.stableBytes = 0
}

// matMul runs y = w·x for the named f32 weight under the ring budget — a thin f32 specialization
// of matMulStaged.
func (r *pagedRing) matMul(name string, shape []int, w []float32, x compute.Tensor, weightBytes int64, pinned bool) []float32 {
	return r.matMulStaged(name, func() compute.Tensor {
		return compute.NewF32(compute.Default(), append([]int(nil), shape...), w)
	}, compute.F32, x, weightBytes, pinned)
}

// matMulQ8 is the Q8_0 twin of matMul: it stages qt as a resident Q8_0 weight under the ring
// budget and runs the quantized GEMM, accounting the quantized footprint (~1.125 B/weight) rather
// than an f32 expansion.
func (r *pagedRing) matMulQ8(name string, qt *q8Tensor, x compute.Tensor, weightBytes int64, pinned bool) []float32 {
	return r.matMulStaged(name, func() compute.Tensor {
		return compute.NewQ8(compute.Default(), []int{qt.out, qt.in}, qt.q, qt.d, qBlk)
	}, compute.Q8_0, x, weightBytes, pinned)
}

// matMulQ4K is the Q4_K twin of matMul: it stages qt (raw GGUF super-blocks) as a resident Q4_K
// weight under the ring budget and runs the dequant-fused Q4_K GEMM — the ~0.56 B/weight residency
// that lets a large MoE's expert majority ride a bounded device ring.
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

// used includes a detached refill's charged storage until checked publication.
// A terminal ring retains that charge even if another control shrinks its budget.
func (r *pagedRing) used() int64 {
	n := r.pool.Used()
	if r.refill != nil && r.refill.reserved {
		n += r.refill.bytes
	}
	return n
}
func (r *pagedRing) budget() int64 { return r.pool.Budget() }

// peakUsed is the high-water mark of used() over the ring's life — the number a boundedness
// witness asserts against budget().
func (r *pagedRing) peakUsed() int64 { return r.peak }
