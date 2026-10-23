package compute

import (
	"context"
	"errors"
	"reflect"
	"sync"
)

// OwnedTransferState distinguishes successful visibility from an unsuccessful
// operation and from uncertainty about whether the device has stopped accessing
// its owners. Only OwnedTransferComplete permits publication or retirement.
type OwnedTransferState uint8

const (
	OwnedTransferUnknown OwnedTransferState = iota
	OwnedTransferPending
	OwnedTransferComplete
	OwnedTransferFailed // Checked quiescent completion, but the operation failed.
)

var (
	ErrOwnedTransferInvalid = errors.New("compute: invalid owned transfer")
	ErrOwnedTransferPending = errors.New("compute: owned transfer is pending")
	ErrOwnedTransferUnknown = errors.New("compute: owned transfer completion is unknown")
	ErrOwnedTransferFailed  = errors.New("compute: owned transfer failed")
)

// OwnedTransferResult is one checked observation. Complete proves both that the
// transfer is visible and that it no longer accesses either retained owner.
// Failed proves quiescence, not valid destination contents. Unknown proves
// neither. Err preserves the provider's original error without wrapping it.
type OwnedTransferResult struct {
	State OwnedTransferState
	Err   error
}

// OwnedTransferCompletion is an optional, error-bearing completion contract for
// one transfer. It deliberately does not adapt Fence: Wait returning or Done
// reporting true is not additional evidence of checked native completion.
// Query is nonblocking. Wait observes completion or returns when ctx is cancelled;
// cancellation alone must never be reported as Complete or quiescent Failed.
// Providers must retain the resources needed to query or drain this operation.
type OwnedTransferCompletion interface {
	Query() OwnedTransferResult
	Wait(context.Context) OwnedTransferResult
}

// OwnedTransferDescriptor transfers retention responsibility for one source and
// destination to the helper. Owners must be actual backing objects, including
// any borrowed Go slice, not just addresses or allocation IDs. The caller must
// already have acquired any native retain, pin, or mapping required for the copy;
// keeping a Go reference alive does not pin memory or prevent explicit Free.
// Neither owner may be freed, reused, or mutated incompatibly while retained.
// Domain supplies the existing complete recording-domain identity, not a new
// request lifetime. Its ForwardExecutionOwner mutex must be stable and non-nil.
type OwnedTransferDescriptor struct {
	Class            MemoryClass
	Domain           ForwardExecutionOwnerBackend
	SourceOwner      any
	DestinationOwner any
}

// OwnedTransfer retains a single classed transfer until checked completion.
// It must not be copied. It neither acquires the domain mutex nor submits work,
// frees allocations, releases pins, or replaces RequestLifetime.
//
// The enclosing domain owner must keep every undemanded transfer reachable,
// drain it before request/domain retirement, or quarantine the transfer AND its
// resources until an independent checked cleanup path proves reuse is safe.
// Cancellation and abandoned callers do not remove this obligation. Failed or
// unknown results are latched and never retired here; recovery belongs to that
// domain owner. No existing synchronous upload guard is relaxed by this helper.
type OwnedTransfer struct {
	mu         sync.Mutex
	descriptor OwnedTransferDescriptor
	domain     *sync.Mutex
	completion OwnedTransferCompletion
	result     OwnedTransferResult
	observing  bool
	retired    bool
}

// NewOwnedTransfer takes retention responsibility only on success. On an error
// the caller still owns both roots. An explicit MemoryUnknown class is allowed;
// an omitted class or missing owner/completion/domain is not.
func NewOwnedTransfer(descriptor OwnedTransferDescriptor, completion OwnedTransferCompletion) (*OwnedTransfer, error) {
	if descriptor.Class == "" || ownedTransferNil(descriptor.Domain) || ownedTransferNil(descriptor.SourceOwner) || ownedTransferNil(descriptor.DestinationOwner) || ownedTransferNil(completion) {
		return nil, ErrOwnedTransferInvalid
	}
	domain := descriptor.Domain.ForwardExecutionOwner()
	if domain == nil {
		return nil, ErrOwnedTransferInvalid
	}
	return &OwnedTransfer{
		descriptor: descriptor,
		domain:     domain,
		completion: completion,
		result:     OwnedTransferResult{State: OwnedTransferPending},
	}, nil
}

// Descriptor observes the retained roots for domain-owned draining/quarantine.
// It does not transfer ownership or authorize publication, mutation, or release.
// SourceOwner and DestinationOwner become nil after successful retirement.
func (t *OwnedTransfer) Descriptor() OwnedTransferDescriptor {
	if t == nil {
		return OwnedTransferDescriptor{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.descriptor
}

// Domain returns the captured ForwardExecutionOwner identity. The helper does
// not lock it; callers retain the existing domain's execution/retirement rules.
func (t *OwnedTransfer) Domain() *sync.Mutex {
	if t == nil {
		return nil
	}
	return t.domain
}

// Ready permits publication only after a checked successful observation and
// before ownership is handed back by Retire. It never polls the provider.
func (t *OwnedTransfer) Ready() bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.domain != nil && !t.retired && t.result.State == OwnedTransferComplete && t.result.Err == nil
}

// Query makes one nonblocking observation, or returns the cached terminal result.
func (t *OwnedTransfer) Query() OwnedTransferResult {
	return t.observe(nil, false)
}

// Wait asks the provider to wait using a non-nil context. If another observation
// is in progress (including a reentrant call), it returns the cached Pending
// result instead of invoking the provider concurrently or waiting on itself.
// A returned Pending result is not completion; the domain must observe again.
func (t *OwnedTransfer) Wait(ctx context.Context) OwnedTransferResult {
	return t.observe(ctx, true)
}

func (t *OwnedTransfer) observe(ctx context.Context, wait bool) (result OwnedTransferResult) {
	if t == nil {
		return OwnedTransferResult{State: OwnedTransferUnknown, Err: ErrOwnedTransferInvalid}
	}
	t.mu.Lock()
	if t.domain == nil {
		t.mu.Unlock()
		return OwnedTransferResult{State: OwnedTransferUnknown, Err: ErrOwnedTransferInvalid}
	}
	if t.result.State != OwnedTransferPending || t.observing {
		result = t.result
		t.mu.Unlock()
		return result
	}
	t.observing = true
	completion := t.completion
	t.mu.Unlock()

	// Provider callbacks run without our mutex. A panic leaves uncertainty
	// latched and roots retained, then continues unwinding with the same panic.
	result = OwnedTransferResult{State: OwnedTransferUnknown, Err: ErrOwnedTransferUnknown}
	defer func() {
		result = checkedOwnedTransferResult(result)
		t.mu.Lock()
		t.result = result
		t.observing = false
		t.mu.Unlock()
	}()
	if wait {
		return completion.Wait(ctx)
	}
	return completion.Query()
}

// Retire hands retention responsibility back exactly once, only after successful
// checked completion. The bool is true only for that handoff; subsequent calls
// return an empty descriptor, false, nil. The recipient must release or retain
// the returned roots according to their original allocator/pin contracts. Two
// owner slots do not imply two independent native retain counts.
//
// Retire never calls Free or user callbacks. Failed/unknown/pending transfers
// return no owners and keep their original completion error (or a state error).
func (t *OwnedTransfer) Retire() (OwnedTransferDescriptor, bool, error) {
	if t == nil {
		return OwnedTransferDescriptor{}, false, ErrOwnedTransferInvalid
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.domain == nil {
		return OwnedTransferDescriptor{}, false, ErrOwnedTransferInvalid
	}
	if t.retired {
		return OwnedTransferDescriptor{}, false, nil
	}
	if t.result.Err != nil {
		return OwnedTransferDescriptor{}, false, t.result.Err
	}
	if t.result.State != OwnedTransferComplete {
		return OwnedTransferDescriptor{}, false, ErrOwnedTransferPending
	}
	descriptor := t.descriptor
	t.descriptor.SourceOwner = nil
	t.descriptor.DestinationOwner = nil
	t.completion = nil
	t.retired = true
	return descriptor, true, nil
}

func checkedOwnedTransferResult(result OwnedTransferResult) OwnedTransferResult {
	switch result.State {
	case OwnedTransferComplete:
		if result.Err != nil {
			result.State = OwnedTransferFailed
		}
	case OwnedTransferFailed:
		if result.Err == nil {
			result.Err = ErrOwnedTransferFailed
		}
	case OwnedTransferPending:
		if result.Err != nil {
			result.State = OwnedTransferUnknown
		}
	default:
		result.State = OwnedTransferUnknown
		if result.Err == nil {
			result.Err = ErrOwnedTransferUnknown
		}
	}
	return result
}

func ownedTransferNil(value any) bool {
	if value == nil {
		return true
	}
	switch v := reflect.ValueOf(value); v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}
