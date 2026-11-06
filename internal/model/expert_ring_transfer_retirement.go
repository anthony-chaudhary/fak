package model

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expertRingOwnedTransferFence opts an existing classed fence into checked
// retirement. Both descriptor owners must be the exact source/destination
// Tensors. The provider must independently retain or quarantine failed/unknown
// transfers, even after the execution scope releases its mutex. Ring reachability
// is not a native retain, and a failed scope drain does not keep that mutex held.
//
// RetireOwnedTransfer accepts the helper's one-time retention handoff. It may
// release transfer-specific retains/pins, never the destination being published.
// Two owner slots need not represent two native retain counts. An error or panic
// leaves handoff ownership uncertain; the ring retains the descriptor and never
// retries that callback. This does not qualify any legacy/native Fence provider.
type expertRingOwnedTransferFence interface {
	OwnedTransfer() *compute.OwnedTransfer
	RetireOwnedTransfer(compute.OwnedTransferDescriptor) error
}

type expertRingTransfer struct {
	id           polymodel.ModelID
	source       compute.Tensor
	tensor       compute.Tensor
	class        compute.MemoryClass
	fence        compute.Fence
	checked      expertRingOwnedTransferFence
	owned        *compute.OwnedTransfer
	retained     compute.OwnedTransferDescriptor
	handoffTried bool
	handoffDone  bool
	complete     bool
	overlapped   bool
	failure      error
}

// Keep the pending value's Fence observation contract. Done only queries
// visibility; it never waits or performs the provider's retirement handoff.
func (p *expertRingTransfer) Done() bool {
	if p.failure != nil {
		return false
	}
	if p.complete {
		return true
	}
	returned := false
	defer func() {
		if !returned {
			p.failure = compute.ErrOwnedTransferUnknown
		}
	}()
	var done bool
	if p.checked == nil {
		done = p.fence.Done()
	} else {
		result := p.owned.Query()
		done = result.State == compute.OwnedTransferComplete && result.Err == nil
	}
	returned = true
	return done
}

func (p *expertRingTransfer) Wait() {
	if _, err := p.settle(); err != nil {
		panic(err)
	}
}

type expertRingTransferError struct {
	id  polymodel.ModelID
	err error
}

func (e *expertRingTransferError) Error() string {
	return fmt.Sprintf("model: expert transfer %q: %v", e.id, e.err)
}
func (e *expertRingTransferError) Unwrap() error { return e.err }

func (p *expertRingTransfer) refused(err error) error {
	return &expertRingTransferError{id: p.id, err: err}
}

// Binding uses backing identity, not equal values or geometry. Conservatively
// refuse non-pointer buffers: this optional ownership extension needs a stable
// storage identity and must not guess identity from an opaque value.
func expertRingSameTensor(a, b compute.Tensor) bool {
	x, y := reflect.ValueOf(a.Buf()), reflect.ValueOf(b.Buf())
	return x.IsValid() && y.IsValid() && x.Kind() == reflect.Pointer &&
		y.Kind() == reflect.Pointer && !x.IsNil() && !y.IsNil() &&
		x.Type() == y.Type() && x.Pointer() == y.Pointer() &&
		a.Dtype == b.Dtype && a.Layout == b.Layout && a.Quant == b.Quant && slices.Equal(a.Shape, b.Shape)
}

func (r *pagedRing) retainStaged(id polymodel.ModelID, source, tensor compute.Tensor, fence compute.Fence) (err error) {
	if fence == nil {
		return nil
	}
	p := &expertRingTransfer{id: id, source: source, tensor: tensor, class: compute.MemoryOffload, fence: fence}
	if r.pending == nil {
		r.pending = make(map[polymodel.ModelID]*expertRingTransfer)
	}
	// Retain before invoking any provider method, including its binding accessor.
	r.pending[id] = p
	returned := false
	defer func() {
		if !returned {
			p.failure = compute.ErrOwnedTransferUnknown
		}
		if err != nil {
			p.failure = err
			err = p.refused(err)
		}
	}()
	if _, advertised := fence.(interface{ OwnedTransfer() *compute.OwnedTransfer }); !advertised {
		returned = true
		return nil // The incumbent legacy contract remains separate.
	}
	p.checked, _ = fence.(expertRingOwnedTransferFence)
	if p.checked == nil {
		returned = true
		return compute.ErrOwnedTransferInvalid
	}
	p.owned = p.checked.OwnedTransfer()
	owner, ownedBackend := r.be.(compute.ForwardExecutionOwnerBackend)
	_, classed := r.be.(compute.ClassedAsyncUploader)
	_, placement := r.be.(classedUploadBackend)
	if p.owned == nil || !ownedBackend || !classed || !placement || !r.domainJoined() {
		returned = true
		return compute.ErrOwnedTransferInvalid
	}
	d := p.owned.Descriptor()
	domain := owner.ForwardExecutionOwner()
	src, srcOK := d.SourceOwner.(compute.Tensor)
	dst, dstOK := d.DestinationOwner.(compute.Tensor)
	if domain == nil || domain != r.drainScope.owner || p.owned.Domain() != domain ||
		d.Domain == nil || d.Domain.ForwardExecutionOwner() != domain || d.Class != p.class ||
		!srcOK || !dstOK || !expertRingSameTensor(src, source) || !expertRingSameTensor(dst, tensor) {
		returned = true
		return compute.ErrOwnedTransferInvalid
	}
	returned = true
	return nil
}

func (p *expertRingTransfer) settle() (overlapped bool, err error) {
	if p.failure != nil {
		return false, p.refused(p.failure)
	}
	if p.complete {
		return p.overlapped, nil
	}
	returned := false
	defer func() {
		if !returned {
			p.failure = compute.ErrOwnedTransferUnknown
		}
	}()
	overlapped, err = p.observe()
	returned = true
	if err != nil {
		return false, p.refused(err)
	}
	p.complete, p.overlapped = true, overlapped
	return overlapped, nil
}

func (p *expertRingTransfer) observe() (bool, error) {
	if p.checked == nil {
		done := p.fence.Done()
		if !done {
			p.fence.Wait()
		}
		return done, nil
	}
	result := p.owned.Query()
	overlapped := result.State == compute.OwnedTransferComplete && result.Err == nil
	if result.State == compute.OwnedTransferPending && result.Err == nil {
		result = p.owned.Wait(context.Background())
	}
	if result.Err != nil {
		return false, result.Err
	}
	if result.State != compute.OwnedTransferComplete {
		return false, compute.ErrOwnedTransferPending
	}
	if p.handoffTried {
		return false, compute.ErrOwnedTransferUnknown
	}
	d, handed, err := p.owned.Retire()
	if err != nil || !handed {
		if err == nil {
			err = compute.ErrOwnedTransferInvalid
		}
		p.failure = err
		return false, err
	}
	p.retained = d
	p.handoffTried = true
	if err = p.checked.RetireOwnedTransfer(d); err != nil {
		p.failure = err
		return false, err
	}
	p.handoffDone = true
	return overlapped, nil
}

// Settle the actual policy's victims before either policy implementation mutates
// membership. Pool preview preserves LRU after repins; value-aware ranking has
// its own existing, non-mutating selector. No independent eviction clock is added.
func (r *pagedRing) settleAdmissionVictims(m polymodel.Model) error {
	if len(r.pending) == 0 || m.WeightBytes <= r.pool.Budget()-r.pool.Used() {
		return nil // No pending victim can be freed by this admission.
	}
	victims, err := r.pool.AdmissionVictims(m)
	if err != nil {
		return err
	}
	if r.policy != ExpertRingEvictLRU && m.WeightBytes > 0 && m.WeightBytes > r.pool.Budget()-r.pool.Used() {
		var ok bool
		victims, ok = r.victimsByHeat(r.pool.Budget() - m.WeightBytes)
		if !ok {
			return polymodel.ErrPinnedNoRoom
		}
	}
	for _, id := range victims {
		if err := r.settleStaged(id, false); err != nil {
			return err
		}
	}
	return nil
}

func (r *pagedRing) settleStaged(id polymodel.ModelID, demand bool) error {
	p := r.pending[id]
	if p == nil {
		return nil
	}
	overlapped, err := p.settle()
	if err != nil {
		return err
	}
	if demand {
		if overlapped {
			r.asyncOverlapped++
		} else {
			r.asyncWaited++
		}
	}
	delete(r.pending, id)
	return nil
}

// Failed admission can leave an unadmitted transfer. Its record stays reachable
// until completion and Free both return; an uncertain Free is never retried.
func (r *pagedRing) rollbackStaged(id polymodel.ModelID, tensor compute.Tensor) (err error) {
	p := r.pending[id]
	if p != nil {
		if _, err = p.settle(); err != nil {
			return err
		}
	}
	returned := false
	defer func() {
		if !returned && p != nil {
			p.failure = compute.ErrOwnedTransferUnknown
		}
	}()
	r.be.Free(tensor)
	returned = true
	delete(r.pending, id)
	return nil
}

// Attempt every pending retirement before propagating the first error/panic.
// Provider quarantine remains authoritative when the enclosing scope unlocks.
func (r *pagedRing) drainTransfers() {
	var first any
	failed := false
	for id, p := range r.pending {
		func() {
			returned := false
			defer func() {
				if !returned {
					value := recover()
					if !failed {
						first, failed = value, true
					}
				}
			}()
			var err error
			if _, resident := r.resident[id]; resident {
				err = r.settleStaged(id, false)
			} else {
				err = r.rollbackStaged(id, p.tensor)
			}
			if err != nil {
				panic(err)
			}
			returned = true
		}()
	}
	if failed {
		panic(first)
	}
}

func expertRingIsTransferError(err error) bool {
	var transfer *expertRingTransferError
	return errors.As(err, &transfer)
}
