package control

import (
	"errors"
	"sync"
	"sync/atomic"
)

// ApplyResult contains the outcome of an apply or dry-run operation.
type ApplyResult struct {
	Status      string               `json:"status"`
	ConfigEpoch uint64               `json:"config_epoch"`
	Valid       bool                 `json:"valid"`
	Config      *ServingConfig       `json:"config,omitempty"`
	Diff        map[string]FieldDiff `json:"diff,omitempty"`
	Impact      ResourceImpact       `json:"impact"`
}

// QueueDepthSource provides live queue depth for drain estimations.
type QueueDepthSource func() int

// EpochObserver is notified whenever the active configuration epoch changes.
type EpochObserver func(vc VersionedConfig)

// Manager coordinates the hot-swappable control plane, shift-left validation,
// double-buffered LKG, monotonic epochs, and canary auto-rollback watchdog.
type Manager struct {
	mu           sync.Mutex
	active       atomic.Pointer[VersionedConfig]
	lkg          atomic.Pointer[VersionedConfig]
	eventStream  *EventStream
	watchdog     *Watchdog
	queueDepthFn QueueDepthSource
	observers    []EpochObserver
}

// NewManager initializes a Manager with the initial configuration at Epoch 1.
func NewManager(initial ServingConfig, wcfg WatchdogConfig, qfn QueueDepthSource) (*Manager, error) {
	if errs := Validate(initial); errs.HasErrors() {
		return nil, errs
	}

	stream := NewEventStream(1024)

	m := &Manager{
		eventStream:  stream,
		queueDepthFn: qfn,
	}

	initialVersioned := &VersionedConfig{
		Epoch:  1,
		Config: initial,
	}
	m.active.Store(initialVersioned)
	m.lkg.Store(initialVersioned)

	m.watchdog = NewWatchdog(wcfg, m.handleWatchdogRollback, m.handleWatchdogPromote)

	return m, nil
}

// Active returns the currently active VersionedConfig snapshot atomically.
// Lock-free via atomic.Pointer dereference (<1 ns CPU overhead).
func (m *Manager) Active() VersionedConfig {
	if m == nil {
		return VersionedConfig{Epoch: 0, Config: DefaultConfig()}
	}
	if v := m.active.Load(); v != nil {
		return *v
	}
	return VersionedConfig{Epoch: 0, Config: DefaultConfig()}
}

// LKG returns the Last-Known-Good VersionedConfig snapshot atomically.
func (m *Manager) LKG() VersionedConfig {
	if m == nil {
		return VersionedConfig{Epoch: 0, Config: DefaultConfig()}
	}
	if v := m.lkg.Load(); v != nil {
		return *v
	}
	return VersionedConfig{Epoch: 0, Config: DefaultConfig()}
}

// EventStream returns the manager's audit event journal.
func (m *Manager) EventStream() *EventStream {
	if m == nil {
		return nil
	}
	return m.eventStream
}

// Watchdog returns the canary watchdog instance.
func (m *Manager) Watchdog() *Watchdog {
	if m == nil {
		return nil
	}
	return m.watchdog
}

// RegisterObserver registers a callback invoked on active configuration changes.
func (m *Manager) RegisterObserver(obs EpochObserver) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.observers = append(m.observers, obs)
}

func (m *Manager) activeQueueDepth() int {
	if m.queueDepthFn != nil {
		return m.queueDepthFn()
	}
	return 0
}

// Apply executes a configuration update. If dryRun is true, it performs shift-left
// validation, computes diff and resource impact, and returns without altering state.
func (m *Manager) Apply(patch ConfigPatch, dryRun bool) (*ApplyResult, error) {
	if m == nil {
		return nil, errors.New("nil control manager")
	}

	cur := m.Active()
	candidate := cur.Config.Apply(patch)

	// Shift-left syntactic & relational invariant validation
	if errs := Validate(candidate); errs.HasErrors() {
		return nil, errs
	}

	diff := ComputeDiff(cur.Config, candidate)
	impact := ComputeImpact(cur.Config, candidate, m.activeQueueDepth())

	if dryRun {
		return &ApplyResult{
			Status:      "dry_run",
			ConfigEpoch: cur.Epoch,
			Valid:       true,
			Config:      &candidate,
			Diff:        diff,
			Impact:      impact,
		}, nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Re-fetch under lock to guarantee monotonic epoch ordering
	cur = *m.active.Load()
	candidate = cur.Config.Apply(patch)
	if errs := Validate(candidate); errs.HasErrors() {
		return nil, errs
	}

	newEpoch := cur.Epoch + 1
	nextVersioned := &VersionedConfig{
		Epoch:  newEpoch,
		Config: candidate,
	}

	// Atomic pointer swap to new active configuration
	m.active.Store(nextVersioned)

	// Arm watchdog for the new candidate configuration
	m.watchdog.StartEvaluation(newEpoch, candidate.DeclaredLatencySLAMS, candidate.SpeculativeAcceptanceThreshold)

	// Record audit event
	m.eventStream.Append(AuditEvent{
		Event:     EventSystemConfigApplied,
		FromEpoch: cur.Epoch,
		ToEpoch:   newEpoch,
		Detail:    "configuration hot-swapped and candidate canary evaluation started",
		Config:    candidate,
	})

	// Notify observers
	observers := make([]EpochObserver, len(m.observers))
	copy(observers, m.observers)
	for _, obs := range observers {
		obs(*nextVersioned)
	}

	return &ApplyResult{
		Status:      "applied",
		ConfigEpoch: newEpoch,
		Valid:       true,
		Config:      &candidate,
		Diff:        diff,
		Impact:      impact,
	}, nil
}

// DryRun evaluates a candidate configuration without mutating live state.
func (m *Manager) DryRun(patch ConfigPatch) (*ApplyResult, error) {
	return m.Apply(patch, true)
}

// Rollback restores the Last-Known-Good (LKG) configuration with a monotonic epoch bump.
func (m *Manager) Rollback(trigger, detail string) (*VersionedConfig, error) {
	if m == nil {
		return nil, errors.New("nil control manager")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	cur := *m.active.Load()
	lkg := *m.lkg.Load()

	newEpoch := cur.Epoch + 1
	restored := &VersionedConfig{
		Epoch:  newEpoch,
		Config: lkg.Config,
	}

	// Atomic rollback swap back to LKG
	m.active.Store(restored)

	// Record audit event
	eventName := EventSystemConfigAutomaticRollback
	if trigger == "manual" {
		eventName = EventSystemConfigRollbackManual
	}

	m.eventStream.Append(AuditEvent{
		Event:     eventName,
		FromEpoch: cur.Epoch,
		ToEpoch:   newEpoch,
		Trigger:   trigger,
		Detail:    detail,
		Config:    lkg.Config,
	})

	// Notify observers
	observers := make([]EpochObserver, len(m.observers))
	copy(observers, m.observers)
	for _, obs := range observers {
		obs(*restored)
	}

	return restored, nil
}

func (m *Manager) handleWatchdogRollback(trigger, detail string) error {
	_, err := m.Rollback(trigger, detail)
	return err
}

func (m *Manager) handleWatchdogPromote(epoch uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	cur := *m.active.Load()
	if cur.Epoch == epoch {
		m.lkg.Store(&cur)
		m.eventStream.Append(AuditEvent{
			Event:     EventSystemConfigCommittedLKG,
			FromEpoch: cur.Epoch,
			ToEpoch:   cur.Epoch,
			Detail:    "candidate stabilized throughout window; committed as LKG",
			Config:    cur.Config,
		})
	}
	return nil
}

// IngestTelemetry proxies telemetry to the watchdog and returns rollback outcome.
func (m *Manager) IngestTelemetry(sample TelemetrySample) (bool, string, string) {
	if m == nil || m.watchdog == nil {
		return false, "", ""
	}
	return m.watchdog.IngestTelemetry(sample)
}
