package servingsupervision

import (
	"context"
	"sync"
	"time"
)

// DrainManager supervises bounded request draining, traffic withdrawal, and inflight tracking.
type DrainManager struct {
	mu           sync.Mutex
	domainID     string
	memberID     string
	role         ServingRole
	observedGen  uint64
	phase        ServingPhase
	drainTimeout time.Duration
	inflight     int
	draining     bool
	drainCh      chan struct{}
	engine       string
}

// NewDrainManager constructs an active DrainManager initialized in PhaseReady.
func NewDrainManager(domainID, memberID string, role ServingRole, drainTimeout time.Duration, initialGen uint64) *DrainManager {
	if drainTimeout <= 0 {
		drainTimeout = 5 * time.Second
	}
	return &DrainManager{
		domainID:     domainID,
		memberID:     memberID,
		role:         role,
		observedGen:  initialGen,
		phase:        PhaseReady,
		drainTimeout: drainTimeout,
		drainCh:      make(chan struct{}, 1),
		engine:       EngineNative,
	}
}

// Acquire reserves capacity for an incoming request.
// Returns ErrTrafficWithdrawn if the domain is currently draining, recovering, quarantined, or stopped.
func (d *DrainManager) Acquire() (func(), error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.draining || d.phase != PhaseReady {
		return nil, ErrTrafficWithdrawn
	}

	d.inflight++
	var once sync.Once
	release := func() {
		once.Do(func() {
			d.mu.Lock()
			defer d.mu.Unlock()
			d.inflight--
			if d.inflight < 0 {
				d.inflight = 0
			}
			if d.draining && d.inflight == 0 {
				select {
				case d.drainCh <- struct{}{}:
				default:
				}
			}
		})
	}

	return release, nil
}

// Inflight returns the current number of in-flight requests.
func (d *DrainManager) Inflight() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.inflight
}

// Phase returns the current serving phase.
func (d *DrainManager) Phase() ServingPhase {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.phase
}

// SetPhase changes the active serving phase.
func (d *DrainManager) SetPhase(phase ServingPhase) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.phase = phase
}

// SetModelBackend updates the engine identity for receipts.
func (d *DrainManager) SetModelBackend(engine string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if engine != "" {
		d.engine = engine
	}
}

// ObservedGen returns the current generation counter.
func (d *DrainManager) ObservedGen() uint64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.observedGen
}

// Drain withdraws incoming traffic, awaits in-flight request completion until the deadline,
// and returns a verified ServingReceipt recording drained and lost work.
func (d *DrainManager) Drain(
	ctx context.Context,
	causeErr error,
	scope RestartScope,
	quarantined bool,
	nextGen uint64,
) (*ServingReceipt, error) {
	d.mu.Lock()
	start := time.Now()
	observed := d.observedGen
	if nextGen == 0 {
		if scope == ScopeNone {
			nextGen = observed
		} else {
			nextGen = observed + 1
		}
	}

	d.draining = true
	if quarantined {
		d.phase = PhaseQuarantined
	} else {
		d.phase = PhaseDraining
	}

	initialInflight := d.inflight
	d.mu.Unlock()

	var drained, lost int
	if initialInflight > 0 {
		drainCtx, cancel := context.WithTimeout(ctx, d.drainTimeout)
		defer cancel()

		select {
		case <-d.drainCh:
			// Drained cleanly
		case <-drainCtx.Done():
			// Timeout or cancellation reached
		}

		d.mu.Lock()
		remaining := d.inflight
		if remaining < 0 {
			remaining = 0
		}
		if remaining > initialInflight {
			remaining = initialInflight
		}
		drained = initialInflight - remaining
		lost = remaining
		d.mu.Unlock()
	}

	elapsed := time.Since(start)
	kind, _ := ClassifyError(causeErr)

	receipt := &ServingReceipt{
		Schema:          ServingReceiptSchema,
		Timestamp:       time.Now().UTC(),
		DomainID:        d.domainID,
		MemberID:        d.memberID,
		Role:            d.role,
		ObservedGen:     observed,
		NextGen:         nextGen,
		ErrorKind:       kind,
		RestartScope:    scope,
		InflightDrained: drained,
		InflightLost:    lost,
		DrainDuration:   elapsed,
		Quarantined:     quarantined,
		Engine:          d.engine,
		FallbackUsed:    false,
	}

	d.mu.Lock()
	d.observedGen = nextGen
	d.mu.Unlock()

	return receipt, nil
}

// Reset clears the draining flag, advances generation, and restores PhaseReady.
func (d *DrainManager) Reset(newGen uint64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.draining = false
	d.phase = PhaseReady
	if newGen > 0 {
		d.observedGen = newGen
	}
	// Drain any stale signal
	select {
	case <-d.drainCh:
	default:
	}
}
