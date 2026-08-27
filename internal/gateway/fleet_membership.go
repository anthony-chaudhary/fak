package gateway

import (
	"context"
	"errors"
	"fmt"
	"strings"
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
//
// Models is what makes a HETEROGENEOUS worker set describable: the model ids this
// worker actually holds. An EMPTY Models means UNCONSTRAINED — the worker is a
// candidate for every model — which is exactly the pre-labeling behavior, so an
// existing configuration that never sets the field routes byte-identically. It is
// deliberately NOT read as "serves nothing": a fail-closed reading here would
// silently empty every deployed configuration the moment the field landed.
// The registry copies the slice on Add (a caller may reuse its own); a slice
// handed BACK by Admissible/Snapshot/Pick aliases the registry's copy and must be
// treated read-only.
type WorkerSpec struct {
	ID       string
	Role     WorkerRole
	Engine   EngineKind
	Endpoint string
	Models   []string
}

var (
	// ErrNoHealthyWorker is the typed verdict a placement returns when no
	// admissible worker exists to route (or re-route) a request to. A caller
	// surfaces it rather than dropping the request silently.
	ErrNoHealthyWorker = errors.New("gateway: no admissible worker in fleet membership")
	// ErrNoWorkerForModel is the typed verdict a placement returns when the
	// roster holds no worker for the requested model at all — a CONFIGURATION
	// fault, deliberately distinct from ErrNoHealthyWorker, which is an OUTAGE
	// (a worker holds the model but is unhealthy or draining). Folding the two
	// together would make a misconfigured model id look like a fleet failure and
	// send an operator hunting a health problem that does not exist. It is never
	// returned for an empty roster: with nothing registered there is no evidence
	// of a model mismatch, so an empty fleet stays ErrNoHealthyWorker exactly as
	// before.
	ErrNoWorkerForModel = errors.New("gateway: no worker in fleet membership holds the requested model")
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
	spec               WorkerSpec
	health             WorkerHealth
	draining           bool
	okStreak           int // consecutive successful probes (drives the healthy hysteresis)
	failStreak         int // consecutive failed probes (drives the unhealthy hysteresis)
	inflight           int // in-flight requests acquired against this worker
	bookedOutputBlocks int // anticipated decode blocks owned by those requests
}

// admissible reports whether a worker may receive NEW work: healthy and not
// draining. The caller must hold the registry lock.
func (w *memberWorker) admissible() bool {
	return w.health == HealthHealthy && !w.draining
}

// servesModel reports whether this worker holds model — the MODEL filter, which
// runs BEFORE the health filter so "nobody holds this model" is decided
// independently of liveness. Unconstrained on BOTH sides: a worker with no
// declared Models serves every model (today's behavior), and an empty model
// query means the caller did not constrain the request, so every worker matches
// (which is what keeps the un-modeled Pick/Dispatch path unchanged). The caller
// must hold the registry lock.
func (w *memberWorker) servesModel(model string) bool {
	if len(w.spec.Models) == 0 || model == "" {
		return true
	}
	for _, held := range w.spec.Models {
		if held == model {
			return true
		}
	}
	return false
}

// normalizeModels copies the caller's slice (so a caller that reuses its own
// backing array cannot mutate the registry afterwards) and drops blank entries.
// A spec whose entries are all blank normalizes to nil — i.e. UNCONSTRAINED, the
// fail-open reading; a malformed label must not silently strand a worker.
func normalizeModels(models []string) []string {
	if len(models) == 0 {
		return nil
	}
	out := make([]string, 0, len(models))
	for _, m := range models {
		if m = strings.TrimSpace(m); m != "" {
			out = append(out, m)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
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

// fleetReservation is one logical request's occupancy booking. A retry retargets
// the same reservation from the failed worker to an untried worker while holding
// the membership lock, so there is never a selected-but-unbooked fallback. Release
// is idempotent and never lets occupancy become negative.
type fleetReservation struct {
	mu                 sync.Mutex
	fleet              *FleetMembership
	workerID           string
	engine             EngineKind
	bookedOutputBlocks int
	released           bool
}

// fleetReservationPick runs while FleetMembership.mu is held. The statuses are
// the complete admissible, allowed, untried candidate set and carry the exact
// in-flight counts that the selected worker will be charged against.
type fleetReservationPickResult struct {
	workerID           string
	bookedOutputBlocks int
}

type fleetReservationPick func([]WorkerStatus) (fleetReservationPickResult, bool)

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
// unset. Models is copied and blank-stripped; leaving it empty registers an
// UNCONSTRAINED worker (a candidate for every model), which is the pre-labeling
// default every existing caller gets for free. A new worker starts unknown (not
// admissible) until its first probe.
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
	spec.Models = normalizeModels(spec.Models)
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
	Spec               WorkerSpec
	Health             WorkerHealth
	Draining           bool
	Inflight           int
	BookedOutputBlocks int
}

// Snapshot returns the current per-worker status in registration order.
func (m *FleetMembership) Snapshot() []WorkerStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]WorkerStatus, 0, len(m.order))
	for _, id := range m.order {
		if w := m.workers[id]; w != nil {
			out = append(out, WorkerStatus{Spec: w.spec, Health: w.health, Draining: w.draining, Inflight: w.inflight, BookedOutputBlocks: w.bookedOutputBlocks})
		}
	}
	return out
}

// reserveForModel atomically classifies the current roster, lets the router pick
// from the resulting live load snapshot, validates that decision, and charges the
// winner before releasing the membership lock. allowed restricts the registry to
// replicas the calling router can actually dial; skip excludes prior failed
// attempts. The picker must not call back into FleetMembership.
func (m *FleetMembership) reserveForModel(model string, engine EngineKind, allowed, skip map[string]struct{}, pick fleetReservationPick) (*fleetReservation, error) {
	if m == nil {
		return nil, ErrNoHealthyWorker
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	decision, err := m.reserveForModelLocked(model, engine, allowed, skip, pick)
	if err != nil {
		return nil, err
	}
	return &fleetReservation{
		fleet:              m,
		workerID:           decision.workerID,
		engine:             m.workers[decision.workerID].spec.Engine,
		bookedOutputBlocks: decision.bookedOutputBlocks,
	}, nil
}

// reserveForModelLocked is reserveForModel's lock-held core.
func (m *FleetMembership) reserveForModelLocked(model string, engine EngineKind, allowed, skip map[string]struct{}, pick fleetReservationPick) (fleetReservationPickResult, error) {
	ids, err := m.classifyForModelLocked(model)
	if err != nil {
		return fleetReservationPickResult{}, err
	}
	statuses := make([]WorkerStatus, 0, len(ids))
	candidates := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if allowed != nil {
			if _, ok := allowed[id]; !ok {
				continue
			}
		}
		if _, tried := skip[id]; tried {
			continue
		}
		w := m.workers[id]
		if w == nil || !w.admissible() {
			continue
		}
		if engine != "" && w.spec.Engine != engine {
			continue
		}
		statuses = append(statuses, WorkerStatus{
			Spec:               w.spec,
			Health:             w.health,
			Draining:           w.draining,
			Inflight:           w.inflight,
			BookedOutputBlocks: w.bookedOutputBlocks,
		})
		candidates[id] = struct{}{}
	}
	if len(statuses) == 0 {
		return fleetReservationPickResult{}, ErrNoHealthyWorker
	}
	if pick == nil {
		return fleetReservationPickResult{}, ErrNoHealthyWorker
	}
	decision, ok := pick(statuses)
	if !ok {
		return fleetReservationPickResult{}, ErrNoHealthyWorker
	}
	if _, ok := candidates[decision.workerID]; !ok {
		return fleetReservationPickResult{}, ErrNoHealthyWorker
	}
	w := m.workers[decision.workerID]
	if w == nil || !w.admissible() {
		return fleetReservationPickResult{}, ErrNoHealthyWorker
	}
	if decision.bookedOutputBlocks < 0 {
		decision.bookedOutputBlocks = 0
	}
	w.inflight++
	w.bookedOutputBlocks = saturatingNonnegativeAdd(w.bookedOutputBlocks, decision.bookedOutputBlocks)
	return decision, nil
}

// Engine returns the execution-engine class the reservation must retain across
// fallback. A native booking can therefore move workers without crossing into an
// external engine.
func (r *fleetReservation) Engine() EngineKind {
	if r == nil {
		return ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.engine
}

// WorkerID returns the worker currently carrying the reservation.
func (r *fleetReservation) WorkerID() string {
	if r == nil {
		return ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released {
		return ""
	}
	return r.workerID
}

// Retarget releases the failed physical attempt, records its membership failure,
// and books the next worker as one lock-held transition. If no fallback can be
// booked, the reservation is left released so a deferred Release stays a no-op.
func (r *fleetReservation) Retarget(model string, allowed, skip map[string]struct{}, pick fleetReservationPick) (string, error) {
	if r == nil || r.fleet == nil {
		return "", ErrNoHealthyWorker
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released {
		return "", ErrNoHealthyWorker
	}

	m := r.fleet
	m.mu.Lock()
	defer m.mu.Unlock()

	failed := r.workerID
	m.releaseReservationLocked(failed, r.bookedOutputBlocks)
	m.markDispatchFailureLocked(failed)

	decision, err := m.reserveForModelLocked(model, r.engine, allowed, skip, pick)
	if err != nil {
		r.workerID = ""
		r.released = true
		return "", err
	}
	r.workerID = decision.workerID
	r.bookedOutputBlocks = decision.bookedOutputBlocks
	return decision.workerID, nil
}

// Release frees the current occupancy booking exactly once.
func (r *fleetReservation) Release() {
	if r == nil || r.fleet == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released {
		return
	}
	r.fleet.mu.Lock()
	r.fleet.releaseReservationLocked(r.workerID, r.bookedOutputBlocks)
	r.fleet.mu.Unlock()
	r.workerID = ""
	r.released = true
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
		m.releaseWorkerLocked(id, w)
	})
}

func (m *FleetMembership) releaseLocked(id string) {
	w := m.workers[id]
	if w == nil {
		return
	}
	m.releaseWorkerLocked(id, w)
}

func (m *FleetMembership) releaseReservationLocked(id string, bookedOutputBlocks int) {
	w := m.workers[id]
	if w == nil {
		return
	}
	if bookedOutputBlocks > 0 {
		w.bookedOutputBlocks -= bookedOutputBlocks
		if w.bookedOutputBlocks < 0 {
			w.bookedOutputBlocks = 0
		}
	}
	m.releaseWorkerLocked(id, w)
}

func (m *FleetMembership) releaseWorkerLocked(id string, w *memberWorker) {
	if w.inflight > 0 {
		w.inflight--
	}
	if w.draining && w.inflight == 0 {
		_ = m.removeLocked(id)
	}
}

// Pick returns the next admissible worker round-robin over the live admissible
// set — the placement read the router performs against membership. It reports
// false when no worker is admissible (the caller then returns a typed verdict
// rather than dropping the request). Pick does NOT acquire; use Dispatch for the
// acquire / failover / release lifecycle.
func (m *FleetMembership) Pick() (WorkerSpec, bool) {
	spec, err := m.pickExceptForModel("", nil)
	return spec, err == nil
}

// PickForModel is Pick constrained to the workers that hold model: the model
// filter runs FIRST, so the two failure modes stay typed apart —
// ErrNoWorkerForModel when the roster holds no such worker (configuration) and
// ErrNoHealthyWorker when a holder exists but none is currently admissible
// (outage). An empty model is unconstrained and behaves exactly like Pick.
func (m *FleetMembership) PickForModel(model string) (WorkerSpec, error) {
	return m.pickExceptForModel(model, nil)
}

// CandidatesForModel returns the admissible workers that hold model, in
// registration order — the candidate set a router filters its replicas against.
// It applies the same model-before-health ordering as PickForModel and returns
// the same two typed verdicts instead of an empty set, so a caller never has to
// guess which of the two conditions emptied the set.
func (m *FleetMembership) CandidatesForModel(model string) ([]WorkerSpec, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids, err := m.classifyForModelLocked(model)
	if err != nil {
		return nil, err
	}
	out := make([]WorkerSpec, 0, len(ids))
	for _, id := range ids {
		out = append(out, m.workers[id].spec)
	}
	return out, nil
}

// classifyForModelLocked partitions the roster for model and is the single place
// the two typed verdicts are decided. It runs the MODEL filter over EVERY
// registered worker regardless of health, and only then the HEALTH filter over
// the holders — the ordering the whole design turns on:
//
//   - no registered worker holds model (roster non-empty) -> ErrNoWorkerForModel
//   - holders exist but none is admissible                -> ErrNoHealthyWorker
//   - an empty roster                                     -> ErrNoHealthyWorker
//
// The empty-roster arm is deliberate: with nothing registered there is no
// evidence distinguishing a bad model id from a fleet that has not come up, so
// it keeps the pre-existing verdict rather than inventing a configuration fault.
// The caller must hold the registry lock.
func (m *FleetMembership) classifyForModelLocked(model string) ([]string, error) {
	var ids []string
	held := false
	for _, id := range m.order {
		w := m.workers[id]
		if w == nil || !w.servesModel(model) {
			continue
		}
		held = true
		if w.admissible() {
			ids = append(ids, id)
		}
	}
	if !held && len(m.order) > 0 {
		return nil, fmt.Errorf("%w: %q", ErrNoWorkerForModel, model)
	}
	if len(ids) == 0 {
		return nil, ErrNoHealthyWorker
	}
	return ids, nil
}

// pickExceptForModel is the one placement primitive: it applies the
// model-before-health classification, then round-robins over the surviving
// candidates, skipping ids already tried. The cursor advances exactly as the
// un-modeled path always did, so an unconstrained call is unchanged.
func (m *FleetMembership) pickExceptForModel(model string, skip map[string]struct{}) (WorkerSpec, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	adm, err := m.classifyForModelLocked(model)
	if err != nil {
		return WorkerSpec{}, err
	}
	n := uint64(len(adm))
	for i := uint64(0); i < n; i++ {
		id := adm[int((m.rr+i)%n)]
		if _, done := skip[id]; done {
			continue
		}
		m.rr += i + 1
		return m.workers[id].spec, nil
	}
	return WorkerSpec{}, ErrNoHealthyWorker
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
	return m.DispatchForModel(ctx, "", send)
}

// DispatchForModel is Dispatch constrained to the workers that hold model. The
// model filter runs before placement, so failover re-places the request only
// onto OTHER holders of the same model — never onto a worker holding a different
// one. When the roster holds no worker for model it returns ErrNoWorkerForModel
// WITHOUT calling send even once: a configuration mistake must never become a
// dial to a wrong upstream. An empty model is unconstrained and is exactly
// Dispatch.
func (m *FleetMembership) DispatchForModel(ctx context.Context, model string, send func(ctx context.Context, spec WorkerSpec) error) (WorkerSpec, error) {
	tried := make(map[string]struct{})
	var lastErr error
	for {
		spec, pickErr := m.pickExceptForModel(model, tried)
		if pickErr != nil {
			// A model nobody holds is a configuration verdict, never an outage —
			// it is not wrapped in ErrNoHealthyWorker even mid-failover.
			if errors.Is(pickErr, ErrNoWorkerForModel) {
				return WorkerSpec{}, pickErr
			}
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
		m.markDispatchFailureWorkerLocked(id, w)
	})
}

func (m *FleetMembership) markDispatchFailureLocked(id string) {
	w := m.workers[id]
	if w == nil {
		return
	}
	m.markDispatchFailureWorkerLocked(id, w)
}

func (m *FleetMembership) markDispatchFailureWorkerLocked(id string, w *memberWorker) {
	m.emit(MembershipEvent{Kind: EventFailover, WorkerID: id})
	m.applyProbeLocked(w, false)
}
