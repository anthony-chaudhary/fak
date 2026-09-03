package servingsupervision

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ReadinessCheckFunc validates whether a replica is fully initialized and capable of inference.
type ReadinessCheckFunc func(ctx context.Context) error

// LivenessCheckFunc probes an active replica to ensure it has not deadlocked or silently hung.
type LivenessCheckFunc func(ctx context.Context) error

// ReplicaRestartHook executes low-level process or model reload during recovery.
type ReplicaRestartHook func(ctx context.Context, replicaID string) error

// ReplicaSupervisor enforces lowest-reasonable restart isolation for a single model worker.
type ReplicaSupervisor struct {
	mu             sync.Mutex
	domainID       string
	replicaID      string
	generation     uint64
	phase          ServingPhase
	spec           ServingDomainSpec
	drainMgr       *DrainManager
	restartCount   int
	quarantined    bool
	readinessCheck ReadinessCheckFunc
	livenessCheck  LivenessCheckFunc
	restartHook    ReplicaRestartHook
	lastReceipt    *ServingReceipt
	engine         string
}

// ReplicaOption configures optional behavior on a ReplicaSupervisor.
type ReplicaOption func(*ReplicaSupervisor)

// WithReadinessCheck attaches a custom readiness verification probe.
func WithReadinessCheck(fn ReadinessCheckFunc) ReplicaOption {
	return func(r *ReplicaSupervisor) {
		r.readinessCheck = fn
	}
}

// WithLivenessCheck attaches a periodic liveness health probe.
func WithLivenessCheck(fn LivenessCheckFunc) ReplicaOption {
	return func(r *ReplicaSupervisor) {
		r.livenessCheck = fn
	}
}

// WithReplicaRestartHook configures a callback invoked when the replica undergoes recovery.
func WithReplicaRestartHook(fn ReplicaRestartHook) ReplicaOption {
	return func(r *ReplicaSupervisor) {
		r.restartHook = fn
	}
}

// WithReplicaBackend sets the engine identity (default EngineNative).
func WithReplicaBackend(engine string) ReplicaOption {
	return func(r *ReplicaSupervisor) {
		r.engine = engine
	}
}

// NewReplicaSupervisor instantiates an isolated replica failure supervisor.
func NewReplicaSupervisor(spec ServingDomainSpec, replicaID string, opts ...ReplicaOption) *ReplicaSupervisor {
	if spec.DrainTimeout <= 0 {
		spec.DrainTimeout = 5 * time.Second
	}
	if spec.RestartBudget <= 0 {
		spec.RestartBudget = 3
	}
	spec.Role = RoleReplica

	r := &ReplicaSupervisor{
		domainID:   spec.DomainID,
		replicaID:  replicaID,
		generation: 1,
		phase:      PhaseStarting,
		spec:       spec,
		engine:     EngineNative,
	}

	for _, opt := range opts {
		opt(r)
	}

	r.drainMgr = NewDrainManager(spec.DomainID, replicaID, RoleReplica, spec.DrainTimeout, 1)
	r.drainMgr.SetModelBackend(r.engine)

	return r
}

// Start executes the initial readiness check and transitions the replica to PhaseReady.
func (r *ReplicaSupervisor) Start(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.phase = PhaseStarting
	r.drainMgr.SetPhase(PhaseStarting)

	if r.readinessCheck != nil {
		if err := r.readinessCheck(ctx); err != nil {
			r.phase = PhaseFailed
			r.drainMgr.SetPhase(PhaseFailed)
			return fmt.Errorf("replica %s readiness failed: %w", r.replicaID, err)
		}
	}

	r.phase = PhaseReady
	r.drainMgr.SetPhase(PhaseReady)
	return nil
}

// Execute wraps an inference request in bounded inflight tracking and error classification.
func (r *ReplicaSupervisor) Execute(ctx context.Context, fn func() error) error {
	release, err := r.drainMgr.Acquire()
	if err != nil {
		return err
	}
	defer release()

	reqErr := fn()
	if reqErr != nil {
		_, _ = r.HandleError(ctx, reqErr)
	}
	return reqErr
}

// HandleError processes a failure, ensuring leaf-only restart, isolation, and quarantine on budget exhaustion.
func (r *ReplicaSupervisor) HandleError(ctx context.Context, err error) (*ServingReceipt, error) {
	if err == nil {
		return nil, nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	kind, scope := ClassifyError(err)

	// ScopeNone: Request/application errors must NOT restart a healthy replica.
	if scope == ScopeNone || kind == ErrorKindRequestApplication {
		receipt := &ServingReceipt{
			Schema:          ServingReceiptSchema,
			Timestamp:       time.Now().UTC(),
			DomainID:        r.domainID,
			MemberID:        r.replicaID,
			Role:            RoleReplica,
			ObservedGen:     r.generation,
			NextGen:         r.generation,
			ErrorKind:       kind,
			RestartScope:    ScopeNone,
			InflightDrained: 0,
			InflightLost:    0,
			DrainDuration:   0,
			Quarantined:     r.quarantined,
			Engine:          r.engine,
			FallbackUsed:    false,
		}
		r.lastReceipt = receipt
		return receipt, nil
	}

	// Check if restart budget is already exhausted
	if r.restartCount >= r.spec.RestartBudget {
		r.quarantined = true
		r.phase = PhaseQuarantined
		receipt, drainErr := r.drainMgr.Drain(ctx, err, ScopeQuarantine, true, r.generation)
		if drainErr == nil && receipt != nil {
			receipt.Quarantined = true
			receipt.RestartScope = ScopeQuarantine
			receipt.ErrorKind = kind
			receipt.Engine = r.engine
			r.lastReceipt = receipt
		}
		return r.lastReceipt, ErrBudgetExhausted
	}

	// Initiate bounded drain of existing requests before restart
	nextGen := r.generation + 1
	receipt, _ := r.drainMgr.Drain(ctx, err, scope, false, nextGen)
	if receipt != nil {
		receipt.ErrorKind = kind
		receipt.RestartScope = scope
		receipt.Engine = r.engine
		r.lastReceipt = receipt
	}

	r.restartCount++
	r.phase = PhaseRecovering

	// Invoke restart hook if registered
	if r.restartHook != nil {
		if hookErr := r.restartHook(ctx, r.replicaID); hookErr != nil {
			r.phase = PhaseFailed
			r.drainMgr.SetPhase(PhaseFailed)
			return r.lastReceipt, fmt.Errorf("replica restart hook failed: %w", hookErr)
		}
	}

	// Verify readiness before restoring to service
	if r.readinessCheck != nil {
		if readyErr := r.readinessCheck(ctx); readyErr != nil {
			r.phase = PhaseFailed
			r.drainMgr.SetPhase(PhaseFailed)
			return r.lastReceipt, fmt.Errorf("replica failed readiness probe after restart: %w", readyErr)
		}
	}

	// Successfully recovered
	r.generation = nextGen
	r.phase = PhaseReady
	r.drainMgr.Reset(r.generation)

	return r.lastReceipt, nil
}

// CheckLiveness evaluates the replica's liveness probe, triggering recovery if it fails.
func (r *ReplicaSupervisor) CheckLiveness(ctx context.Context) error {
	r.mu.Lock()
	livenessFn := r.livenessCheck
	r.mu.Unlock()

	if livenessFn == nil {
		return nil
	}

	if err := livenessFn(ctx); err != nil {
		_, recErr := r.HandleError(ctx, WrapClassifiedError(ErrorKindFailedLiveness, ScopeLeafOnly, err))
		return recErr
	}
	return nil
}

// IsHealthy reports true if the replica is in PhaseReady and not quarantined.
func (r *ReplicaSupervisor) IsHealthy() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.phase == PhaseReady && !r.quarantined
}

// Phase returns current serving phase.
func (r *ReplicaSupervisor) Phase() ServingPhase {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.phase
}

// Generation returns the current generation counter.
func (r *ReplicaSupervisor) Generation() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.generation
}

// RestartCount returns how many times this replica has been recovered.
func (r *ReplicaSupervisor) RestartCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.restartCount
}

// Quarantined reports whether the replica is locked in quarantine.
func (r *ReplicaSupervisor) Quarantined() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.quarantined
}

// ResetQuarantine clears quarantine status and resets restart accounting.
func (r *ReplicaSupervisor) ResetQuarantine(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.quarantined = false
	r.restartCount = 0
	r.phase = PhaseStarting
	r.drainMgr.Reset(r.generation)

	if r.readinessCheck != nil {
		if err := r.readinessCheck(ctx); err != nil {
			r.phase = PhaseFailed
			r.drainMgr.SetPhase(PhaseFailed)
			return err
		}
	}

	r.phase = PhaseReady
	r.drainMgr.SetPhase(PhaseReady)
	return nil
}

// LastReceipt returns the most recently emitted supervision receipt.
func (r *ReplicaSupervisor) LastReceipt() *ServingReceipt {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.lastReceipt
}

// DomainID returns the failure domain ID.
func (r *ReplicaSupervisor) DomainID() string {
	return r.domainID
}

// ReplicaID returns the member replica ID.
func (r *ReplicaSupervisor) ReplicaID() string {
	return r.replicaID
}
