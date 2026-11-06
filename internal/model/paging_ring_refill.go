package model

import (
	"reflect"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// Adapted from vLLM fa3b9d1377219247060d3c8b49eae199c9e7e154,
// vllm/model_executor/offloader/prefetch.py (Apache-2.0, Copyright contributors
// to the vLLM project): reusable geometry-matched slots and completion before
// reuse. FAK keeps its actual admission policy, not vLLM's modulo layer pool.
//
// expertRingRefillBackend is an optional structural provider contract. Merely
// implementing it does not enable reuse: OffloadRefillEnabled must reflect an
// explicit operator opt-in. This initial consumer admits a single exact-size
// victim in a private ring under its already-live complete execution domain.
// Shared rings, multiple victims and unowned calls retain the incumbent path.
//
// OffloadRefillEnabled and CanRefillOffload are side-effect-free, including a negative answer. It checks
// exact bytes, dtype/layout/shape/quant metadata, host source extent/non-alias,
// device identity, independently owned MemoryOffload storage, native readiness,
// and absence of cache sharing, managed storage and graph dependencies.
//
// RefillOffload rechecks native eligibility and retires the OLD backend handle
// before its first destructive write. Complete with nil error alone certifies
// visible bytes, no remaining transfer accesses, and a NEW handle over the SAME
// destination allocation. It never frees that allocation. Any other result is
// terminal here, not permission to retry or fall back. The provider must root
// and quarantine source staging and destination BEFORE any effect/callback,
// including on panic/Goexit, independently of this ring and its domain mutex.
// This contract neither adapts an unchecked Fence nor grants graph admission.
type expertRingRefillBackend interface {
	OffloadRefillEnabled() bool
	CanRefillOffload(destination, source compute.Tensor, bytes int64) bool
	RefillOffload(destination, source compute.Tensor, bytes int64) (compute.Tensor, compute.OwnedTransferResult)
}

// One in-progress or terminal transaction. reserved charges a detached victim
// until successful publication; an unknown outcome cannot make its bytes vanish
// from used(). Provider native retention is still required, not implied by Go roots.
type expertRingRefill struct {
	id                               polymodel.ModelID
	source, destination, replacement compute.Tensor
	bytes                            int64
	reserved                         bool
	failure                          error
}

func (r *pagedRing) requireRefillIdle() {
	if p := r.refill; p != nil {
		// A reentrant call is also terminal, even if its caller catches this panic.
		if p.failure == nil {
			p.failure = compute.ErrOwnedTransferUnknown
		}
		panic(&expertRingTransferError{id: p.id, err: p.failure})
	}
}

// Handle identity is independent of tensor metadata. A copied Quant descriptor
// must not disguise the old Buffer as a new backend handle. Opaque value buffers
// do not provide the stable pointer identity this optional consumer requires.
func expertRingRefillSameBuffer(a, b compute.Tensor) bool {
	x, y := reflect.ValueOf(a.Buf()), reflect.ValueOf(b.Buf())
	return x.IsValid() && y.IsValid() && x.Kind() == reflect.Pointer && y.Kind() == reflect.Pointer &&
		!x.IsNil() && !y.IsNil() && x.Type() == y.Type() && x.Pointer() == y.Pointer()
}

func expertRingRefillGeometry(a, b compute.Tensor) bool {
	if a.Dtype != b.Dtype || a.Layout != compute.RowMajor || b.Layout != a.Layout ||
		len(a.Shape) != 2 || !slices.Equal(a.Shape, b.Shape) || a.Shape[0] <= 0 || a.Shape[1] <= 0 ||
		a.Buf() == nil || b.Buf() == nil {
		return false
	}
	if a.Dtype == compute.F32 {
		return a.Quant == nil && b.Quant == nil
	}
	// Raw K/IQ blocks may carry all quantization in their byte stream. A provider
	// must still recognize the specific format; side-channel scales never qualify.
	if !a.Dtype.Quantized() || a.Quant == nil || b.Quant == nil {
		return false
	}
	x, y := a.Quant, b.Quant
	return len(x.Scale) == 0 && len(y.Scale) == 0 && len(x.ZeroPoint) == 0 && len(y.ZeroPoint) == 0 &&
		x.Block == y.Block && x.Axis == y.Axis && x.Bits == y.Bits &&
		x.Symmetric == y.Symmetric && x.StaticAct == y.StaticAct
}

func (r *pagedRing) refillVictim(m polymodel.Model) (polymodel.ModelID, compute.Tensor, bool) {
	if m.WeightBytes <= 0 || m.WeightBytes <= r.pool.Budget()-r.pool.Used() {
		return "", compute.Tensor{}, false
	}
	ids, err := r.pool.AdmissionVictims(m)
	if err != nil {
		return "", compute.Tensor{}, false
	}
	if r.policy != ExpertRingEvictLRU {
		var ok bool
		ids, ok = r.victimsByHeat(r.pool.Budget() - m.WeightBytes)
		if !ok {
			return "", compute.Tensor{}, false
		}
	}
	// A single equal-size replacement needs no extra victim cleanup and never
	// changes the incumbent ranking to find a more convenient reusable allocation.
	if len(ids) != 1 {
		return "", compute.Tensor{}, false
	}
	id := ids[0]
	old, ok := r.pool.Get(id)
	t, live := r.resident[id]
	return id, t, ok && live && !old.Pinned && r.holds[id] == 0 && old.WeightBytes == m.WeightBytes
}

// tryRefill returns false only before selecting destructive reuse. Every selected
// failure panics with a transfer error: the portable routed HAL otherwise treats
// a returned stageWithError error as permission to upload permanent residency.
func (r *pagedRing) tryRefill(source compute.Tensor, m polymodel.Model, dtype compute.Dtype) (out compute.Tensor, used bool) {
	provider, ok := r.be.(expertRingRefillBackend)
	if !ok || r.shared != nil || !r.domainJoined() {
		return compute.Tensor{}, false
	}
	owner, ok := r.be.(compute.ForwardExecutionOwnerBackend)
	if !ok || owner.ForwardExecutionOwner() == nil || owner.ForwardExecutionOwner() != r.drainScope.owner ||
		!provider.OffloadRefillEnabled() {
		return compute.Tensor{}, false
	}
	id, dst, ok := r.refillVictim(m)
	if !ok || source.Dtype != dtype || dst.Dtype != dtype || !expertRingRefillSameBuffer(dst, dst) || expertRingRefillSameBuffer(dst, source) || !expertRingRefillGeometry(dst, source) {
		return compute.Tensor{}, false
	}
	expected := dst
	expected.Shape = slices.Clone(dst.Shape)
	if dst.Quant != nil {
		q := *dst.Quant
		expected.Quant = &q
	}
	epoch, policy, budget := r.policyGeneration, r.policy, r.pool.Budget()
	p := &expertRingRefill{id: m.ID, source: source, destination: dst, bytes: m.WeightBytes}
	r.refill = p // Retain before every provider callback and settlement callback.
	returned := false
	defer func() {
		if !returned && p.failure == nil {
			p.failure = compute.ErrOwnedTransferUnknown
		}
	}()
	fail := func(err error) {
		if p.failure == nil {
			p.failure = err
		}
		panic(&expertRingTransferError{id: m.ID, err: p.failure})
	}
	if backendExecutionOwner(dst.Backend()) != r.drainScope.owner {
		fail(compute.ErrOwnedTransferInvalid)
	}
	if !provider.CanRefillOffload(dst, source, m.WeightBytes) {
		if p.failure != nil {
			fail(p.failure)
		}
		r.refill = nil
		returned = true
		return compute.Tensor{}, false
	}
	if p.failure != nil {
		fail(p.failure)
	}
	if err := r.settleStaged(id, false); err != nil {
		fail(err)
	}
	// Settlement may call a provider. Revalidate the actual policy, hold state,
	// backing identity and source geometry instead of treating preview as a lease.
	again, live, valid := r.refillVictim(m)
	if p.failure != nil {
		fail(p.failure)
	}
	if !valid || again != id || !expertRingSameTensor(dst, live) ||
		epoch != r.policyGeneration || policy != r.policy || budget != r.pool.Budget() ||
		!r.domainJoined() || r.drainScope.owner != owner.ForwardExecutionOwner() ||
		!expertRingRefillGeometry(expected, source) || !expertRingRefillGeometry(expected, dst) {
		fail(compute.ErrOwnedTransferInvalid)
	}
	if !r.pool.Evict(id) {
		fail(compute.ErrOwnedTransferInvalid)
	}
	p.reserved = true
	delete(r.resident, id) // Old logical name is gone before ANY destructive write.
	if r.policy != ExpertRingEvictLRU {
		delete(r.lastUse, id)
	}
	r.evict++
	r.dropResidency(id)
	var result compute.OwnedTransferResult
	p.replacement, result = provider.RefillOffload(dst, source, m.WeightBytes)
	if p.failure != nil {
		fail(p.failure)
	}
	if result.State != compute.OwnedTransferComplete || result.Err != nil {
		cause := result.Err
		if cause == nil {
			cause = compute.ErrOwnedTransferUnknown
			if result.State == compute.OwnedTransferFailed {
				cause = compute.ErrOwnedTransferFailed
			}
		}
		fail(cause)
	}
	if !expertRingRefillGeometry(expected, p.replacement) ||
		!expertRingRefillSameBuffer(p.replacement, p.replacement) || expertRingRefillSameBuffer(dst, p.replacement) || backendExecutionOwner(p.replacement.Backend()) != r.drainScope.owner || !p.replacement.Ready() {
		fail(compute.ErrOwnedTransferInvalid)
	}
	if p.failure != nil {
		fail(p.failure)
	}
	// No callback may have changed admission while the slot was detached.
	if r.pool.Budget() != budget || r.policyGeneration != epoch || r.policy != policy ||
		!r.domainJoined() || r.drainScope.owner != owner.ForwardExecutionOwner() ||
		m.WeightBytes > r.pool.Budget()-r.pool.Used() {
		fail(compute.ErrOwnedTransferInvalid)
	}
	if _, exists := r.pool.Get(m.ID); exists {
		fail(compute.ErrOwnedTransferInvalid)
	}
	victims, err := r.pool.Admit(m)
	if err != nil {
		fail(err)
	}
	if len(victims) != 0 {
		fail(compute.ErrOwnedTransferInvalid)
	}
	r.resident[m.ID] = p.replacement
	p.reserved = false
	out = p.replacement
	r.refill = nil
	returned = true
	return out, true
}
