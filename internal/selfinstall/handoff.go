package selfinstall

import (
	"context"
	"errors"
	"sync"
)

// HandoffState is the externally visible phase of a graceful local handoff.
type HandoffState string

const (
	HandoffPending   HandoffState = "pending"
	HandoffDraining  HandoffState = "draining"
	HandoffHandedOff HandoffState = "handed_off"
	HandoffRefused   HandoffState = "refused"
)

var ErrHandoffDraining = errors.New("self-update handoff is draining; new admissions are refused")

// Handoff coordinates the admission boundary around a local process replacement.
// A zero value is ready for use.
type Handoff struct {
	mu       sync.Mutex
	state    HandoffState
	active   int
	drained  chan struct{}
	session  string
	revision string
	err      error
}

// HandoffSnapshot is a race-free view suitable for receipts and status output.
type HandoffSnapshot struct {
	State     HandoffState
	Active    int
	SessionID string
	Revision  string
	Err       error
}

// Admit registers one active call. Its release function must be called exactly once.
func (h *Handoff) Admit() (func(), error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.state == HandoffDraining || h.state == HandoffHandedOff || h.state == HandoffRefused {
		return nil, ErrHandoffDraining
	}
	if h.state == "" {
		h.state = HandoffPending
	}
	h.active++
	var once sync.Once
	return func() {
		once.Do(func() {
			h.mu.Lock()
			defer h.mu.Unlock()
			h.active--
			if h.active == 0 && h.drained != nil {
				close(h.drained)
				h.drained = nil
			}
		})
	}, nil
}

// Drain stops admissions, waits for active calls, then invokes the successor.
// launch must not return success until the successor has acknowledged its identity.
func (h *Handoff) Drain(ctx context.Context, sessionID, revision string, launch func(context.Context, string, string) error) HandoffSnapshot {
	h.mu.Lock()
	if h.state == HandoffDraining || h.state == HandoffHandedOff || h.state == HandoffRefused {
		h.mu.Unlock()
		return h.Snapshot()
	}
	h.state, h.session, h.revision = HandoffDraining, sessionID, revision
	if h.active == 0 {
		h.mu.Unlock()
	} else {
		h.drained = make(chan struct{})
		drained := h.drained
		h.mu.Unlock()
		select {
		case <-drained:
		case <-ctx.Done():
			h.refuse(ctx.Err())
			return h.Snapshot()
		}
	}
	if err := ctx.Err(); err != nil {
		h.refuse(err)
		return h.Snapshot()
	}
	if launch == nil {
		h.refuse(errors.New("self-update handoff has no successor launcher"))
		return h.Snapshot()
	}
	if err := launch(ctx, sessionID, revision); err != nil {
		h.refuse(err)
		return h.Snapshot()
	}
	h.mu.Lock()
	h.state = HandoffHandedOff
	h.mu.Unlock()
	return h.Snapshot()
}

func (h *Handoff) refuse(err error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.state, h.err = HandoffRefused, err
}

func (h *Handoff) Snapshot() HandoffSnapshot {
	h.mu.Lock()
	defer h.mu.Unlock()
	state := h.state
	if state == "" {
		state = HandoffPending
	}
	return HandoffSnapshot{State: state, Active: h.active, SessionID: h.session, Revision: h.revision, Err: h.err}
}
