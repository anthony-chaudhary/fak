package gateway

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Live fleet membership + health/drain/failover — the loop the router reads.
//
// For many-node serving the router can only place load well if it knows,
// continuously, which workers exist and are healthy. Before this, the gateway had
// only non-fleet pieces: a one-shot reachability probe, an observability-only node
// watcher, a static flat endpoint list (no liveness), and the per-host gpulease
// flock (a single-machine floor, not a fleet view). FleetMembership supersedes the
// static endpoint list and the per-host lease *as the fleet view*: a continuous
// health loop marks each worker healthy/unhealthy/unknown with hysteresis, drain
// removes a worker without dropping its in-flight work, and Dispatch re-routes a
// request off a failed worker or returns a typed verdict — never a silent drop.
//
// It is engine-uniform: an external-adapter worker (vLLM/SGLang/Dynamo behind the
// EngineDriver seam, whose readiness this loop consumes where it exists) and a
// fak-native worker (which has no external lifecycle manager, so fak owns its
// membership directly) move through the same states, so the router routes against
// one surface for both. Routing POLICY (cache-aware / power-of-two / P/D-aware
// selection) belongs to the router skeleton and the residency index; this loop
// only marks workers admissible / non-admissible and supplies failover.

// WorkerRole is the placement hint a router uses in a disaggregated (prefill /
// decode) fleet. It is reported here, not acted on — routing policy is the
// router's; this loop only carries the hint alongside health.
type WorkerRole string

const (
	RolePrefill WorkerRole = "prefill"
	RoleDecode  WorkerRole = "decode"
	RoleUnified WorkerRole = "unified"
)

// EngineKind distinguishes a worker fronted by an external adapter (whose
// readiness this loop consumes) from a fak-native worker (whose membership fak
// owns directly). The health loop is identical for both.
type EngineKind string

const (
	EngineExternal EngineKind = "external"
	EngineNative   EngineKind = "native"
)

// WorkerHealth is a replica's liveness as the loop currently sees it. unknown is
// the state of a freshly-registered worker before its first probe resolves; an
// unknown worker is not admissible until it probes healthy.
type WorkerHealth string

const (
	HealthUnknown   WorkerHealth = "unknown"
	HealthHealthy   WorkerHealth = "healthy"
	HealthUnhealthy WorkerHealth = "unhealthy"
)

// WorkerSpec is one replica's stable identity in the registry — the fields a
// router reads to place (and, on failover, re-place) a request. It supersedes a
// flat endpoint-list entry by carrying role + engine kind; the registry adds live
// health on top.
type WorkerSpec struct {
	ID       string
	Role     WorkerRole
	Engine   EngineKind
	Endpoint string
}

var (
	// ErrNoHealthyWorker is the typed verdict a placement returns when no
	// admissible worker exists to route (or re-route) a request to. A caller
	// surfaces it rather than dropping the request silently.
	ErrNoHealthyWorker = errors.New("gateway: no admissible worker in fleet membership")
	// ErrWorkerExists guards a duplicate registration.
	ErrWorkerExists = errors.New("gateway: worker already registered")
	// ErrWorkerUnknown guards a mutation of an unregistered worker.
	ErrWorkerUnknown = errors.New("gateway: worker not registered")
)

// Probe reports whether a worker answered a liveness check. The registry calls it
// once per worker per health tick; it is engine-agnostic (an external-adapter
// probe reads K8s/Dynamo readiness or hits /health; a native worker pings its
// serve loop). A probe should respect ctx and return promptly.
type Probe func(ctx context.Context, spec WorkerSpec) bool

// MembershipKind tags a membership transition for the metrics surface.
type MembershipKind string

const (
	EventAdded         MembershipKind = "added"
	EventRemoved       MembershipKind = "removed"
	EventHealthChanged MembershipKind = "health_changed"
	EventDrainStarted  MembershipKind = "drain_started"
	EventFailover      MembershipKind = "failover"
)

// MembershipEvent is one transition the registry records. The serving-metrics
// surface drains the log (DrainEvents) and publishes a per-worker-labeled counter
// per Kind, so the operator and any autoscaler/planner can see membership move.
// WorkerID is the per-worker label every event carries.
type MembershipEvent struct {
	Kind     MembershipKind
	WorkerID string
	From     WorkerHealth // populated for health_changed
	To       WorkerHealth // populated for health_changed
	Draining bool         // populated for drain_started
}

type memberWorker struct {
	spec       WorkerSpec
	health     WorkerHealth
	draining   bool
	okStreak   int // consecutive successful probes (drives the healthy hysteresis)
	failStreak int // consecutive failed probes (drives the unhealthy hysteresis)
	inflight   int // in-flight requests acquired against this worker
}

// admissible reports whether a worker may receive NEW work: healthy and not
// draining. The caller must hold the registry lock.
func (w *memberWorker) admissible() bool {
	return w.health == HealthHealthy && !w.draining
}

// FleetMembership is the live membership + health/drain/failover registry the
// router reads. All methods are safe for concurrent use.
type FleetMembership struct {
	healthyAfter   int // consecutive OK probes to (re)admit a worker
	unhealthyAfter int // consecutive failed probes to evict a healthy worker (hysteresis)
	probe          Probe

	mu      sync.Mutex
	workers map[string]*memberWorker
	order   []string // stable registration order, for deterministic round-robin
	rr      uint64   // round-robin cursor over the admissible set
	events  []MembershipEvent
}

// MembershipConfig tunes the health-loop hysteresis. Zero values fall back to
// safe defaults (HealthyAfter=1, UnhealthyAfter=2) so a single missed beat does
// not flap a worker out of the admissible set.
type MembershipConfig struct {
	HealthyAfter   int
	UnhealthyAfter int
	Probe          Probe
}

// NewFleetMembership builds an empty registry. Register workers with Add and drive
// the health loop with RunHealthLoop (or call ProbeOnce directly in a test).
func NewFleetMembership(cfg MembershipConfig) *FleetMembership {
	ha := cfg.HealthyAfter
	if ha < 1 {
		ha = 1
	}
	ua := cfg.UnhealthyAfter
	if ua < 1 {
		ua = 2
	}
	return &FleetMembership{
		healthyAfter:   ha,
		unhealthyAfter: ua,
		probe:          cfg.Probe,
		workers:        make(map[string]*memberWorker),
	}
}

// Add registers a worker. Role defaults to unified and Engine to external when
// unset. A new worker starts unknown (not admissible) until its first probe.
func (m *FleetMembership) Add(spec WorkerSpec) error {
	if spec.ID == "" {
		return errors.New("gateway: worker spec has empty id")
	}
	if spec.Role == "" {
		spec.Role = RoleUnified
	}
	if spec.Engine == "" {
		spec.Engine = EngineExternal
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.workers[spec.ID]; ok {
		return fmt.Errorf("%w: %q", ErrWorkerExists, spec.ID)
	}
	m.workers[spec.ID] = &memberWorker{spec: spec, health: HealthUnknown}
	m.order = append(m.order, spec.ID)
	m.emit(MembershipEvent{Kind: EventAdded, WorkerID: spec.ID, To: HealthUnknown})
	return nil
}

// Remove drops a worker from the registry immediately, regardless of in-flight
// state. Prefer Drain for graceful removal.
func (m *FleetMembership) Remove(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.removeLocked(id)
}

func (m *FleetMembership) removeLocked(id string) error {
	if _, ok := m.workers[id]; !ok {
		return fmt.Errorf("%w: %q", ErrWorkerUnknown, id)
	}
	delete(m.workers, id)
	for i, wid := range m.order {
		if wid == id {
			m.order = append(m.order[:i], m.order[i+1:]...)
			break
		}
	}
	m.emit(MembershipEvent{Kind: EventRemoved, WorkerID: id})
	return nil
}

// emit appends a transition to the log under the registry lock. Keeping it a log
// (rather than a direct metrics call) frees the registry of a metrics dependency
// and lets a test assert the exact transitions; the metrics surface drains it.
func (m *FleetMembership) emit(ev MembershipEvent) {
	m.events = append(m.events, ev)
}

// DrainEvents returns and clears the accumulated transition log. The metrics
// surface calls it on its publish cadence; each event carries a per-worker label.
func (m *FleetMembership) DrainEvents() []MembershipEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.events) == 0 {
		return nil
	}
	out := m.events
	m.events = nil
	return out
}

// Admissible returns the specs the router may place NEW work on, in registration
// order — the live fleet view that replaces the static endpoint list.
func (m *FleetMembership) Admissible() []WorkerSpec {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []WorkerSpec
	for _, id := range m.order {
		if w := m.workers[id]; w != nil && w.admissible() {
			out = append(out, w.spec)
		}
	}
	return out
}

// WorkerStatus is the per-worker membership/health row the observability surface
// publishes (labeled by ID) — the live replacement for a static endpoint row.
type WorkerStatus struct {
	Spec     WorkerSpec
	Health   WorkerHealth
	Draining bool
	Inflight int
}

// Snapshot returns the current per-worker status in registration order.
func (m *FleetMembership) Snapshot() []WorkerStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]WorkerStatus, 0, len(m.order))
	for _, id := range m.order {
		if w := m.workers[id]; w != nil {
			out = append(out, WorkerStatus{Spec: w.spec, Health: w.health, Draining: w.draining, Inflight: w.inflight})
		}
	}
	return out
}

// ProbeOnce runs the configured probe against every registered worker once and
// updates health with hysteresis: unhealthyAfter consecutive failures evict a
// healthy worker from the admissible set; healthyAfter consecutive successes
// (re)admit it. It is the body the continuous loop runs each tick — tests call it
// directly to drive deterministic ticks. The probe runs OUTSIDE the lock (it may
// block on a network round-trip); results are applied under the lock.
func (m *FleetMembership) ProbeOnce(ctx context.Context) {
	if m.probe == nil {
		return
	}
	m.mu.Lock()
	specs := make([]WorkerSpec, 0, len(m.order))
	for _, id := range m.order {
		if w := m.workers[id]; w != nil {
			specs = append(specs, w.spec)
		}
	}
	m.mu.Unlock()

	results := make(map[string]bool, len(specs))
	for _, spec := range specs {
		results[spec.ID] = m.probe(ctx, spec)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, spec := range specs {
		if w := m.workers[spec.ID]; w != nil { // skip workers removed mid-probe
			m.applyProbeLocked(w, results[spec.ID])
		}
	}
}

// applyProbeLocked folds one probe result into a worker's streaks and crosses its
// health over the hysteresis thresholds. Caller holds the lock.
func (m *FleetMembership) applyProbeLocked(w *memberWorker, ok bool) {
	if ok {
		w.failStreak = 0
		w.okStreak++
		if w.health != HealthHealthy && w.okStreak >= m.healthyAfter {
			m.setHealthLocked(w, HealthHealthy)
		}
		return
	}
	w.okStreak = 0
	w.failStreak++
	if w.health != HealthUnhealthy && w.failStreak >= m.unhealthyAfter {
		m.setHealthLocked(w, HealthUnhealthy)
	}
}

func (m *FleetMembership) setHealthLocked(w *memberWorker, to WorkerHealth) {
	if w.health == to {
		return
	}
	from := w.health
	w.health = to
	m.emit(MembershipEvent{Kind: EventHealthChanged, WorkerID: w.spec.ID, From: from, To: to})
}

// RunHealthLoop probes the fleet every interval until ctx is cancelled — the
// continuous liveness loop. The testable body is ProbeOnce; this is the thin
// real-time driver. It probes immediately, then once per tick.
func (m *FleetMembership) RunHealthLoop(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = time.Second
	}
	m.ProbeOnce(ctx)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			m.ProbeOnce(ctx)
		}
	}
}

// Drain marks a worker non-admissible: the router immediately stops routing NEW
// work to it (Admissible / Pick drop it at once), while requests already Acquired
// against it are allowed to finish. The worker is removed automatically when its
// in-flight count reaches zero (see Release); a drained worker that is already
// idle is removed at once.
// withWorkerLock runs fn under mu against worker id, returning ErrWorkerUnknown
// when id is absent. Centralizes the lock + lookup + nil-guard the error-returning
// worker mutators share so a copy can't drop it.
func (m *FleetMembership) withWorkerLock(id string, fn func(w *memberWorker) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	w := m.workers[id]
	if w == nil {
		return fmt.Errorf("%w: %q", ErrWorkerUnknown, id)
	}
	return fn(w)
}

// withWorkerLockVoid is the no-return form: it runs fn under mu against worker id
// and is a no-op when id is absent.
func (m *FleetMembership) withWorkerLockVoid(id string, fn func(w *memberWorker)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	w := m.workers[id]
	if w == nil {
		return
	}
	fn(w)
}

func (m *FleetMembership) Drain(id string) error {
	return m.withWorkerLock(id, func(w *memberWorker) error {
		if !w.draining {
			w.draining = true
			m.emit(MembershipEvent{Kind: EventDrainStarted, WorkerID: id, Draining: true})
		}
		if w.inflight == 0 {
			return m.removeLocked(id)
		}
		return nil
	})
}

// Acquire marks the start of an in-flight request against a worker so a concurrent
// Drain waits for it before removing the worker. It refuses a non-admissible
// worker so new work never lands on a draining/unhealthy replica.
func (m *FleetMembership) Acquire(id string) error {
	return m.withWorkerLock(id, func(w *memberWorker) error {
		if !w.admissible() {
			return ErrNoHealthyWorker
		}
		w.inflight++
		return nil
	})
}

// Release marks an in-flight request done. If the worker was draining and this was
// its last in-flight request, the worker is removed (drain-before-remove complete).
func (m *FleetMembership) Release(id string) {
	m.withWorkerLockVoid(id, func(w *memberWorker) {
		if w.inflight > 0 {
			w.inflight--
		}
		if w.draining && w.inflight == 0 {
			_ = m.removeLocked(id)
		}
	})
}

// Pick returns the next admissible worker round-robin over the live admissible
// set — the placement read the router performs against membership. It reports
// false when no worker is admissible (the caller then returns a typed verdict
// rather than dropping the request). Pick does NOT acquire; use Dispatch for the
// acquire / failover / release lifecycle.
func (m *FleetMembership) Pick() (WorkerSpec, bool) {
	return m.pickExcept(nil)
}

// pickExcept returns the next admissible worker not in skip, advancing the
// round-robin cursor so repeated calls spread across the admissible set.
func (m *FleetMembership) pickExcept(skip map[string]struct{}) (WorkerSpec, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	adm := m.admissibleIDsLocked()
	n := uint64(len(adm))
	if n == 0 {
		return WorkerSpec{}, false
	}
	for i := uint64(0); i < n; i++ {
		id := adm[int((m.rr+i)%n)]
		if _, done := skip[id]; done {
			continue
		}
		m.rr += i + 1
		return m.workers[id].spec, true
	}
	return WorkerSpec{}, false
}

func (m *FleetMembership) admissibleIDsLocked() []string {
	var ids []string
	for _, id := range m.order {
		if w := m.workers[id]; w != nil && w.admissible() {
			ids = append(ids, id)
		}
	}
	return ids
}

// Dispatch routes a request to an admissible worker and retries on the NEXT
// admissible worker if the send fails mid-request — in-flight failover with no
// silent drop. It acquires the chosen worker (so a concurrent Drain waits), calls
// send, and releases it; on a send error it records a dispatch failure against
// that worker (crossing it toward unhealthy under the same hysteresis), emits a
// failover transition labeled with the worker the request moved OFF of, and tries
// the next distinct admissible worker. When every admissible worker has been
// tried and failed (or none was admissible to begin with), it returns
// ErrNoHealthyWorker — wrapping the last send error when there was one — a typed
// verdict the caller surfaces, never a dropped request. On success it returns the
// worker that served the request.
func (m *FleetMembership) Dispatch(ctx context.Context, send func(ctx context.Context, spec WorkerSpec) error) (WorkerSpec, error) {
	tried := make(map[string]struct{})
	var lastErr error
	for {
		spec, ok := m.pickExcept(tried)
		if !ok {
			if lastErr != nil {
				return WorkerSpec{}, fmt.Errorf("%w: every admissible worker failed: %w", ErrNoHealthyWorker, lastErr)
			}
			return WorkerSpec{}, ErrNoHealthyWorker
		}
		if err := m.Acquire(spec.ID); err != nil {
			// Worker went non-admissible between pick and acquire — skip it.
			tried[spec.ID] = struct{}{}
			continue
		}
		err := send(ctx, spec)
		m.Release(spec.ID)
		if err == nil {
			return spec, nil
		}
		lastErr = err
		tried[spec.ID] = struct{}{}
		m.markDispatchFailure(spec.ID)
	}
}

// markDispatchFailure records a mid-request send failure as a failed probe (so a
// worker that fails dispatches crosses to unhealthy under the same hysteresis) and
// emits a failover transition labeled with the worker the request moved off of.
func (m *FleetMembership) markDispatchFailure(id string) {
	m.withWorkerLockVoid(id, func(w *memberWorker) {
		m.emit(MembershipEvent{Kind: EventFailover, WorkerID: id})
		m.applyProbeLocked(w, false)
	})
}
