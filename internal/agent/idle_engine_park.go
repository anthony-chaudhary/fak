// idle_engine_park.go — the idle-engine park lifecycle: an opt-in timer parks
// a loaded engine at zero active requests, and on-demand wakes reload it, with
// one lifetime mutex guarding every transition and engine read. Adapted from
// ashhart/TensorFold zig/src/server/idle_unload.zig (Manager) and
// idle_unload_policy.zig at 9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// (Apache-2.0): states resident/unloading/unloaded/loading/stopped in one
// mutex; parks only at zero active; concurrent wakes coalesce into one reload;
// failed opens stay parked and retryable; multi-rank GLM CUDA is refused
// because closing the primary rank ends the followers. Re-expressed in
// idiomatic Go; no upstream code is copied. The manager owns no clock: the
// caller's timer drives Tick, which keeps the lifecycle unit-testable.
package agent

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// IdleEngineState is the closed set of park-lifecycle states one engine moves
// through across unload/reload cycles.
type IdleEngineState int

const (
	// IdleEngineResident: the engine is loaded and serving.
	IdleEngineResident IdleEngineState = iota
	// IdleEngineUnloading: a park is in flight; the close has not finished.
	IdleEngineUnloading
	// IdleEngineUnloaded: parked; the next wake triggers a reload.
	IdleEngineUnloaded
	// IdleEngineLoading: a reload is in flight; concurrent wakes join it.
	IdleEngineLoading
	// IdleEngineStopped: terminal; the engine will not be reloaded.
	IdleEngineStopped
)

// String renders the closed state vocabulary.
func (s IdleEngineState) String() string {
	switch s {
	case IdleEngineResident:
		return "resident"
	case IdleEngineUnloading:
		return "unloading"
	case IdleEngineUnloaded:
		return "unloaded"
	case IdleEngineLoading:
		return "loading"
	case IdleEngineStopped:
		return "stopped"
	default:
		return "unknown"
	}
}

// Closed sentinel errors the park lifecycle returns; callers classify with
// errors.Is, never prose.
var (
	// ErrIdleEngineStopped reports a wake after Stop.
	ErrIdleEngineStopped = errors.New("idle engine stopped")
	// ErrIdleEngineOpenFailed wraps the open failure that left the engine
	// parked; the cause is also wrapped so the opener's own sentinel stays
	// reachable via errors.Is.
	ErrIdleEngineOpenFailed = errors.New("idle engine open failed")
	// ErrIdleUnloadRefused reports that the idle-unload policy refuses to
	// park this engine configuration.
	ErrIdleUnloadRefused = errors.New("idle unload refused")
)

// IdleEngineManager owns one engine's park/reload lifecycle. One mutex guards
// every state transition and every read of the active-request count; the open
// and shut callbacks run outside the lock, so a slow load never blocks the
// state machine.
type IdleEngineManager struct {
	mu         sync.Mutex
	state      IdleEngineState
	active     int
	lastActive time.Time
	openErr    error  // last failed open, surfaced to the wakers that joined it
	loadSeq    uint64 // bumped when a load attempt starts; identifies the attempt a waiter joined
	waiting    int    // wakers blocked behind an in-flight unload or load
	cond       *sync.Cond
	openFn     func() error
	shutFn     func() error
}

// NewIdleEngineManager builds a manager around an already-loaded engine
// (resident at now) whose lifecycle the open/shut callbacks drive.
func NewIdleEngineManager(now time.Time, open, shut func() error) *IdleEngineManager {
	m := &IdleEngineManager{
		state:      IdleEngineResident,
		lastActive: now,
		openFn:     open,
		shutFn:     shut,
	}
	m.cond = sync.NewCond(&m.mu)
	return m
}

// Wake acquires the engine for one request. Resident engines take one active
// slot; parked engines trigger exactly one shared reload, which every
// concurrent waker joins; a failed open leaves the engine parked and returns
// the failure to all wakers of that attempt. A successful wake pairs with one
// Release.
func (m *IdleEngineManager) Wake(now time.Time) error {
	m.mu.Lock()
	for {
		switch m.state {
		case IdleEngineResident:
			m.active++
			m.lastActive = now
			m.mu.Unlock()
			return nil
		case IdleEngineStopped:
			m.mu.Unlock()
			return ErrIdleEngineStopped
		case IdleEngineUnloading:
			// Park in flight: wait for the close to land, then re-decide.
			m.waiting++
			m.cond.Wait()
			m.waiting--
			continue
		case IdleEngineLoading:
			seq := m.loadSeq
			m.waiting++
			m.cond.Wait()
			m.waiting--
			if m.state == IdleEngineUnloaded && m.loadSeq == seq {
				// The reload this waker joined just failed: report it without
				// triggering another open; the next fresh wake retries.
				err := m.openErr
				m.mu.Unlock()
				return fmt.Errorf("%w: %w", ErrIdleEngineOpenFailed, err)
			}
			continue
		case IdleEngineUnloaded:
			m.state = IdleEngineLoading
			m.loadSeq++
			seq := m.loadSeq
			m.mu.Unlock()
			err := m.openFn()
			m.mu.Lock()
			if m.state != IdleEngineLoading || m.loadSeq != seq {
				// Stop raced ahead mid-load: honor it and drop the result.
				m.openErr = err
				m.cond.Broadcast()
				m.mu.Unlock()
				if err == nil {
					_ = m.shutFn() // best-effort: close what we just opened
				}
				if err != nil {
					return fmt.Errorf("%w: %w", ErrIdleEngineOpenFailed, err)
				}
				return ErrIdleEngineStopped
			}
			if err == nil {
				m.state = IdleEngineResident
				m.active++
				m.lastActive = now
				m.openErr = nil
			} else {
				// Failed open stays parked: retryable by the next wake.
				m.state = IdleEngineUnloaded
				m.openErr = err
			}
			m.cond.Broadcast()
			m.mu.Unlock()
			if err == nil {
				return nil
			}
			return fmt.Errorf("%w: %w", ErrIdleEngineOpenFailed, err)
		}
	}
}

// Release ends one active request started by a successful Wake. Releasing
// below zero active is ignored defensively; callers pair one Release per
// successful Wake.
func (m *IdleEngineManager) Release(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.active > 0 {
		m.active--
	}
	m.lastActive = now
}

// Tick is the idle timer's callback: it parks the engine only when it is
// resident, holds zero active requests, and has been quiet since its last
// activity for at least idle. Any other Tick is a no-op. A failed close keeps
// the engine resident and returns the close error.
func (m *IdleEngineManager) Tick(now time.Time, idle time.Duration) error {
	m.mu.Lock()
	if m.state != IdleEngineResident || m.active != 0 || now.Sub(m.lastActive) < idle {
		m.mu.Unlock()
		return nil
	}
	m.state = IdleEngineUnloading
	m.mu.Unlock()
	err := m.shutFn()
	m.mu.Lock()
	defer m.mu.Unlock()
	defer m.cond.Broadcast()
	if m.state == IdleEngineStopped {
		// Stop is terminal even when the close callback was already in flight.
		return err
	}
	if err != nil {
		// Close failed: the engine is still loaded; keep serving.
		m.state = IdleEngineResident
		return err
	}
	m.state = IdleEngineUnloaded
	m.lastActive = now
	return nil
}

// Stop moves the manager to the terminal state. An in-flight load is dropped:
// its freshly opened engine is shut best-effort so Stop always wins.
func (m *IdleEngineManager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.state == IdleEngineStopped {
		return
	}
	m.state = IdleEngineStopped
	m.cond.Broadcast()
}

// State snapshots the lifecycle state under the guard.
func (m *IdleEngineManager) State() IdleEngineState {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.state
}

// Active reports the current active-request count under the guard.
func (m *IdleEngineManager) Active() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.active
}

// Waiting reports how many wakers are blocked behind an in-flight unload or
// load; they all share that single transition instead of stacking engine
// opens.
func (m *IdleEngineManager) Waiting() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.waiting
}

// IdleUnloadRefused applies the idle-unload policy to one engine
// configuration. Multi-rank GLM on CUDA is refused: closing the primary rank
// terminates the follower ranks, so the park could never be retried. Every
// other configuration parks freely; the opt-in decision itself is the
// caller's, so no idle duration is consulted here.
func IdleUnloadRefused(backends []string, modelType string, worldSize int) error {
	if worldSize <= 1 {
		return nil
	}
	if !strings.EqualFold(strings.TrimSpace(modelType), "glm") {
		return nil
	}
	for _, b := range backends {
		if strings.EqualFold(strings.TrimSpace(b), "cuda") {
			return ErrIdleUnloadRefused
		}
	}
	return nil
}
