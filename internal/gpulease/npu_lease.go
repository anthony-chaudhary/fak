package gpulease

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrNPUBusy is the typed refusal returned when the XDNA 2 NPU is held exclusively
// by another session (#10686).
type ErrNPUBusy struct {
	HolderID       string        `json:"holder_id"`
	Remaining      time.Duration `json:"remaining"`
	ResidentModel  string        `json:"resident_model"`
	RequestedModel string        `json:"requested_model"`
}

func (e *ErrNPUBusy) Error() string {
	if e == nil {
		return "gpulease: NPU is busy"
	}
	return fmt.Sprintf("gpulease: NPU is busy (held by session %q with %v remaining; resident model %q)",
		e.HolderID, e.Remaining, e.ResidentModel)
}

// ErrNPUBusySentinel is the base sentinel error for errors.Is matching.
var ErrNPUBusySentinel = errors.New("gpulease: NPU is busy")

// Unwrap allows errors.Is(err, ErrNPUBusySentinel) to match *ErrNPUBusy.
func (e *ErrNPUBusy) Unwrap() error {
	return ErrNPUBusySentinel
}

// Is supports matching against ErrNPUBusySentinel or another *ErrNPUBusy.
func (e *ErrNPUBusy) Is(target error) bool {
	if target == nil {
		return false
	}
	if target == ErrNPUBusySentinel {
		return true
	}
	if e == nil {
		return false
	}
	t, ok := target.(*ErrNPUBusy)
	if !ok {
		return false
	}
	return t.HolderID == "" || t.HolderID == e.HolderID
}

// NPUBusyError is an alias for ErrNPUBusy to support alternate naming conventions.
type NPUBusyError = ErrNPUBusy

// NPULease represents an exclusive residency lease on the AMD XDNA 2 NPU accelerator.
type NPULease struct {
	SessionID  string        `json:"session_id"`
	ModelID    string        `json:"model_id"`
	AcquiredAt time.Time     `json:"acquired_at"`
	ExpiresAt  time.Time     `json:"expires_at"`
	SwapCost   time.Duration `json:"swap_cost"`
	Reloaded   bool          `json:"reloaded"`
	manager    *NPULeaseManager
	released   bool
}

// Release relinquishes the lease on the NPU accelerator.
// The model remains resident on the hardware until a subsequent lease swaps xclbin.
func (l *NPULease) Release() {
	if l == nil || l.manager == nil || l.released {
		return
	}
	l.manager.release(l)
}

// DefaultXCLBINSwapCost is the nominal xclbin reload and AIE-ML tile reconfiguration
// penalty when swapping the resident model on XDNA 2 hardware.
const DefaultXCLBINSwapCost = 350 * time.Millisecond

// NPULeaseManager arbitrates exclusive, single-model residency across concurrent sessions
// on the AMD XDNA 2 NPU accelerator (#10686).
type NPULeaseManager struct {
	mu             sync.Mutex
	residentModel  string
	activeLease    *NPULease
	xclbinSwapCost time.Duration
	nowFn          func() time.Time
}

// NPULeaseOption configures an NPULeaseManager instance.
type NPULeaseOption func(*NPULeaseManager)

// WithXCLBINSwapCost overrides the default xclbin swap cost.
func WithXCLBINSwapCost(d time.Duration) NPULeaseOption {
	return func(m *NPULeaseManager) {
		m.xclbinSwapCost = d
	}
}

// WithNowFunc overrides the clock function (primarily for deterministic unit testing).
func WithNowFunc(fn func() time.Time) NPULeaseOption {
	return func(m *NPULeaseManager) {
		m.nowFn = fn
	}
}

// NewNPULeaseManager instantiates an NPU lease arbiter.
func NewNPULeaseManager(opts ...NPULeaseOption) *NPULeaseManager {
	m := &NPULeaseManager{
		xclbinSwapCost: DefaultXCLBINSwapCost,
		nowFn:          time.Now,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Acquire requests exclusive residency for modelID under sessionID for the specified TTL.
// If the NPU is already held by another session, Acquire returns the typed refusal *ErrNPUBusy.
// If the requested model differs from the current resident model, an xclbin swap penalty is paid.
func (m *NPULeaseManager) Acquire(sessionID, modelID string, ttl time.Duration) (*NPULease, error) {
	if m == nil {
		return nil, errors.New("gpulease: nil NPULeaseManager")
	}
	if sessionID == "" {
		return nil, errors.New("gpulease: session ID is required")
	}
	if modelID == "" {
		return nil, errors.New("gpulease: model ID is required")
	}
	if ttl <= 0 {
		return nil, errors.New("gpulease: lease TTL must be positive")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	nowFn := m.nowFn
	if nowFn == nil {
		nowFn = time.Now
	}
	now := nowFn()
	if m.activeLease != nil {
		if !now.Before(m.activeLease.ExpiresAt) {
			// Active lease expired automatically
			m.activeLease = nil
		} else if m.activeLease.SessionID == sessionID {
			if m.activeLease.ModelID == modelID {
				// Same session, same model: renew TTL
				m.activeLease.ExpiresAt = now.Add(ttl)
				return m.activeLease, nil
			}
			// Same session, different model: swap resident model
			reloaded := true
			m.residentModel = modelID
			lease := &NPULease{
				SessionID:  sessionID,
				ModelID:    modelID,
				AcquiredAt: now,
				ExpiresAt:  now.Add(ttl),
				SwapCost:   m.xclbinSwapCost,
				Reloaded:   reloaded,
				manager:    m,
			}
			m.activeLease = lease
			return lease, nil
		} else {
			// Held by another session: typed refusal
			rem := m.activeLease.ExpiresAt.Sub(now)
			if rem < 0 {
				rem = 0
			}
			return nil, &ErrNPUBusy{
				HolderID:       m.activeLease.SessionID,
				Remaining:      rem,
				ResidentModel:  m.residentModel,
				RequestedModel: modelID,
			}
		}
	}

	reloaded := false
	swapCost := time.Duration(0)
	if m.residentModel != modelID {
		reloaded = true
		swapCost = m.xclbinSwapCost
		m.residentModel = modelID
	}

	lease := &NPULease{
		SessionID:  sessionID,
		ModelID:    modelID,
		AcquiredAt: now,
		ExpiresAt:  now.Add(ttl),
		SwapCost:   swapCost,
		Reloaded:   reloaded,
		manager:    m,
	}
	m.activeLease = lease
	return lease, nil
}

func (m *NPULeaseManager) release(l *NPULease) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.activeLease == l {
		m.activeLease = nil
		l.released = true
	}
}

// CurrentResident inspects the currently resident model and active lease holder, if any.
func (m *NPULeaseManager) CurrentResident() (modelID, holderID string, remaining time.Duration) {
	if m == nil {
		return "", "", 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	nowFn := m.nowFn
	if nowFn == nil {
		nowFn = time.Now
	}
	now := nowFn()
	modelID = m.residentModel
	if m.activeLease != nil {
		if now.Before(m.activeLease.ExpiresAt) {
			holderID = m.activeLease.SessionID
			remaining = m.activeLease.ExpiresAt.Sub(now)
		} else {
			m.activeLease = nil
		}
	}
	return modelID, holderID, remaining
}
