package compute

import (
	"errors"
	"sync"
)

// TransferCompletionState describes one transfer, not the enclosing device queue.
// Unknown is deliberately the zero value and never grants publication or reuse.
type TransferCompletionState uint8

const (
	TransferUnknown TransferCompletionState = iota
	TransferPending
	TransferComplete
	TransferFailed
)

var ErrTransferNotComplete = errors.New("compute: transfer completion is not known successful")

// OwnedTransferDescriptor binds a transfer to already-acquired retention claims.
// Source and Destination must keep the actual backing allocations alive, not just
// their numeric addresses. In particular, Source must retain a borrowed Go slice.
// Their IDs and DomainID are immutable diagnostic identities supplied by the owner.
// Domain must be the same mutex returned by ForwardExecutionOwner for this domain.
// Holding a Go reference alone cannot prevent native Free: the provider must hold
// native retention claims before construction and release them only via Release.
// Release drops those two claims, not the caller's ownership of a published tensor;
// it must not fail, panic, or call back into this transfer.
type OwnedTransferDescriptor struct {
	Class                             MemoryClass
	DomainID, SourceID, DestinationID string
	Domain                            *sync.Mutex
	Source, Destination               any
	Release                           func()
}

// OwnedTransferWitness is a detached value observation. It contains neither
// owner handles nor callbacks; changing it cannot change the transfer. Failure
// is captured at observation time, while CompletionError preserves the cause.
type OwnedTransferWitness struct {
	Class                             MemoryClass
	DomainID, SourceID, DestinationID string
	State                             TransferCompletionState
	Failure                           string
	Retired                           bool
}

// OwnedTransfer is an optional, unconsumed lifetime helper, not an AsyncUploader
// implementation. Only a provider's verified completion observation may mark it
// complete. Query/wait errors (including cancellation) must be recorded, never
// converted into success. The first non-pending observation is immutable;
// failed/unknown transfers remain retained for domain-specific drain/quarantine.
// The enclosing owner MUST keep this helper reachable until retirement or transfer
// of its retained claims to quarantine, including undemanded prefetch and request
// cancellation. RequestLifetime alone does not satisfy that obligation. This
// helper neither drains a domain nor relaxes the model's synchronous guard.
// Methods are concurrency-safe; the helper must not be copied after first use.
type OwnedTransfer struct {
	mu         sync.Mutex
	descriptor OwnedTransferDescriptor
	witness    OwnedTransferWitness
	cause      error
	terminal   bool
}

// NewOwnedTransfer takes custody of retention claims only on success. A rejected
// descriptor leaves cleanup with its caller. Non-nil values must represent live
// owners (a typed nil or a bare native address is not a retention claim).
func NewOwnedTransfer(d OwnedTransferDescriptor) (*OwnedTransfer, error) {
	if d.Class == "" || d.DomainID == "" || d.SourceID == "" || d.DestinationID == "" ||
		d.Domain == nil || d.Source == nil || d.Destination == nil || d.Release == nil {
		return nil, errors.New("compute: incomplete owned transfer descriptor")
	}
	return &OwnedTransfer{descriptor: d, witness: OwnedTransferWitness{
		Class: d.Class, DomainID: d.DomainID, SourceID: d.SourceID,
		DestinationID: d.DestinationID, State: TransferPending,
	}}, nil
}

// Observe records a provider query/wait result. An error overrides even a claimed
// Complete result. Unsupported states become Unknown. No later optimistic result
// can erase a failure or uncertainty; recovery belongs to the enclosing domain.
func (t *OwnedTransfer) Observe(state TransferCompletionState, cause error) OwnedTransferWitness {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.descriptor.Domain == nil || t.terminal {
		return t.witness
	}
	if cause != nil {
		state = TransferFailed
	} else if state > TransferFailed {
		state = TransferUnknown
	}
	t.witness.State = state
	t.cause = cause
	if cause != nil {
		t.witness.Failure = cause.Error()
	}
	t.terminal = state != TransferPending
	return t.witness
}

// Witness returns a value snapshot without exposing retained backing storage.
func (t *OwnedTransfer) Witness() OwnedTransferWitness {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.witness
}

// CompletionError returns the original query/wait error, without wrapping it.
func (t *OwnedTransfer) CompletionError() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.cause
}

// Retire releases retention exactly once, and only after known success. A refusal
// preserves all owners and the original completion cause. The successful return
// occurs after Release finishes; duplicate calls wait for it and are harmless.
// This is not permission to destroy the domain or free the published destination.
func (t *OwnedTransfer) Retire() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.witness.State != TransferComplete {
		if t.cause != nil {
			return t.cause
		}
		return ErrTransferNotComplete
	}
	if !t.witness.Retired {
		t.descriptor.Release()
		t.descriptor.Source = nil
		t.descriptor.Destination = nil
		t.descriptor.Release = nil
		t.witness.Retired = true
	}
	return nil
}
