package model

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expertRingTransfer is a consumer record, not a provider capability. Legacy
// fences keep their original contract; they are never promoted to checked
// completion. The owned arm consumes compute.OwnedTransfer without redefining it.
// Both uploadStagedAsync guards remain necessary: this record neither pins a
// source nor drains a recording domain before its execution owner unlocks.
type expertRingTransfer struct {
	id          polymodel.ModelID
	class       compute.MemoryClass
	source      any
	destination compute.Tensor
	fence       compute.Fence
	owned       *compute.OwnedTransfer
	retired     bool
	owners      compute.OwnedTransferDescriptor
}

// expertRingTransferError refuses publication or reclamation while retaining the
// original completion error. Void lifecycle callers panic with this error rather
// than continuing to Free; error-bearing demand callers return it.
type expertRingTransferError struct {
	ID     polymodel.ModelID
	Class  compute.MemoryClass
	Result compute.OwnedTransferResult
}

func (e *expertRingTransferError) Error() string {
	return fmt.Sprintf("model: ring transfer %q (%s) not retired: state=%d: %v", e.ID, e.Class, e.Result.State, e.Result.Err)
}

func (e *expertRingTransferError) Unwrap() error { return e.Result.Err }

// Done and Wait preserve the existing pending-map Fence view used by observers.
// The checked arm never treats an unsuccessful Wait as success, including when
// called through that compatibility view. Only settleStaged removes the record.
func (p *expertRingTransfer) Done() bool {
	if p.retired {
		return true
	}
	if p.owned != nil {
		result := p.owned.Query()
		return result.State == compute.OwnedTransferComplete && result.Err == nil && p.owned.Ready()
	}
	return p.fence == nil || p.fence.Done()
}

func (p *expertRingTransfer) Wait() {
	if _, err := p.settle(context.Background(), false); err != nil {
		panic(err)
	}
}

func (p *expertRingTransfer) settle(ctx context.Context, demand bool) (bool, error) {
	if p.retired {
		return true, nil
	}
	if p.owned == nil {
		landed := demand && (p.fence == nil || p.fence.Done())
		if p.fence != nil && !landed {
			p.fence.Wait()
		}
		return landed, nil
	}
	result := compute.OwnedTransferResult{State: compute.OwnedTransferPending}
	if demand {
		result = p.owned.Query()
	}
	landed := result.State == compute.OwnedTransferComplete && result.Err == nil
	if result.State == compute.OwnedTransferPending {
		result = p.owned.Wait(ctx)
	}
	if result.State != compute.OwnedTransferComplete || result.Err != nil {
		if result.Err == nil {
			result.Err = compute.ErrOwnedTransferPending
		}
		return false, &expertRingTransferError{ID: p.id, Class: p.class, Result: result}
	}
	owners, retired, err := p.owned.Retire()
	if err != nil || !retired {
		if err == nil {
			err = compute.ErrOwnedTransferInvalid
		}
		return false, &expertRingTransferError{ID: p.id, Class: p.class,
			Result: compute.OwnedTransferResult{State: compute.OwnedTransferUnknown, Err: err}}
	}
	// Keep the handed-back owners rooted through the caller's retirement decision.
	// Their allocator/pin contracts still belong to the enclosing domain owner.
	p.owners = owners
	p.retired = true
	return landed, nil
}

// rollbackStaged also handles an unadmitted allocation retained after a failed
// settlement. On refusal its pending record still owns the destination; retry
// or close must settle that same record rather than overwriting it.
func (r *pagedRing) rollbackStaged(id polymodel.ModelID, wt compute.Tensor) error {
	if err := r.settleStaged(context.Background(), id, false); err != nil {
		return err
	}
	r.be.Free(wt)
	return nil
}

func (r *pagedRing) settleStaged(ctx context.Context, id polymodel.ModelID, demand bool) error {
	f, ok := r.pending[id]
	if !ok {
		return nil
	}
	landed := false
	if p, checked := f.(*expertRingTransfer); checked {
		var err error
		landed, err = p.settle(ctx, demand)
		if err != nil {
			return err
		}
	} else if f != nil {
		landed = demand && f.Done()
		if !landed {
			f.Wait()
		}
	}
	// Delete only after settlement succeeds. A provider panic also leaves the
	// record (and its owners) reachable for the existing cleanup failure path.
	delete(r.pending, id)
	if demand && f != nil {
		if landed {
			r.asyncOverlapped++
		} else {
			r.asyncWaited++
		}
	}
	return nil
}

// Settle checked records before Admit or a victim policy can mutate the pool.
// Conservatively draining all checked records avoids duplicating Pool's victim
// selection. Legacy prefetch order/overlap is unchanged. This is NOT a domain
// retirement hook and earns no overlap credit for an undemanded transfer.
func (r *pagedRing) settleOwnedStaged() error {
	for id, f := range r.pending {
		p, ok := f.(*expertRingTransfer)
		if !ok {
			continue
		}
		if _, resident := r.resident[id]; !resident {
			if err := r.rollbackStaged(id, p.destination); err != nil {
				return err
			}
		} else if p.owned != nil {
			if err := r.settleStaged(context.Background(), id, false); err != nil {
				return err
			}
		}
	}
	return nil
}
