package microagent

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// Gateway is the ONE in-process model/kernel seam every hosted microagent steps
// through — the same agent.Planner seam the fak serve gateway builds exactly once
// for all served sessions (internal/gateway.New). The host holds one Gateway and
// hands the SAME handle to every Step call; a hosted agent never dials its own.
// That is the structural inversion of production today: one guard process + one
// external CLI per agent (cmd/fak/dispatch_tick.go spawn path) becomes one
// goroutine per agent behind one shared gateway.
type Gateway = agent.Planner

// Microagent is one hosted agent loop. Step advances it by one unit of work
// (typically one model turn) through the host-shared gateway and reports
// done=true to retire. Step must honor ctx cancellation; the host also checks
// the context between steps, so a well-behaved agent stops at the next step
// boundary even if a single Step ignores ctx. The RunArm-backed implementation
// (per-turn stepping of the internal/agent loop) is the still-open #2001
// extraction; the host is deliberately loop-agnostic so it does not block on it.
type Microagent interface {
	Step(ctx context.Context, gw Gateway) (done bool, err error)
}

// EventKind classifies a host audit event.
type EventKind string

const (
	EventSpawn  EventKind = "spawn"  // agent accepted and enqueued
	EventReject EventKind = "reject" // spawn refused (bounded queue full)
	EventDone   EventKind = "done"   // Step reported done
	EventCancel EventKind = "cancel" // retired by Cancel/Close before done
	EventRetry  EventKind = "retry"  // Step failed and its evidence was fed back
	EventVerify EventKind = "verify" // Independent evidence accepted or refused completion
	EventError  EventKind = "error"  // Step returned a non-cancel error
)

// Event is one lifecycle record. The whole host shares ONE AuditSink — the
// in-process contrast to one hash-chained JSONL per guard process today.
type Event struct {
	Agent string // agent id (also its session.Table TraceID)
	Kind  EventKind
	Steps int    // completed Step calls when the event fired
	Err   string // non-empty on EventError
}

// AuditSink receives every host lifecycle event. Record is called under host
// locks for per-agent ordering (spawn strictly before retire), so it must be
// fast and must not call back into the Host.
type AuditSink interface{ Record(Event) }

type nopSink struct{}

func (nopSink) Record(Event) {}

// Result is one retired agent, collected via Reap.
type Result struct {
	ID    string
	Steps int   // completed Step calls
	Done  bool  // true: Step reported done; false: cancelled or errored
	Err   error // nil on done; ctx error on cancel; the Step error otherwise
}

// Config sizes a Host. The zero value of every field selects a usable default.
type Config struct {
	// Workers is K: how many Step drivers run concurrently. Default 8.
	Workers int
	// Queue bounds the pending-spawn queue. A Spawn beyond it refuses loudly
	// with ErrQueueFull instead of blocking or growing without bound. Default 256.
	Queue int
	// Sessions holds per-agent drive state: ONE bounded-LRU map entry per agent
	// keyed by its id as the TraceID (internal/session, limit 8192) — not one OS
	// process per agent. Default: session.NewTable().
	Sessions *session.Table
	// Audit is the host's ONE audit sink. Default: a no-op sink.
	Audit AuditSink
	// MaxTurns is the hard number of Step calls each agent may make. Values 1-3
	// enforce the bounded-microagent envelope; zero preserves the unbounded
	// compatibility posture. The host checks the limit before invoking Step, so
	// neither retries nor an untrusted agent can widen it.
	MaxTurns int
	// MaxRetries enables evidence-grounded retries after a failed Step. Zero
	// (the default) preserves the no-retry baseline. A retry is attempted only
	// when the Microagent implements RetryFeedback, so a failure can never be
	// retried blindly. The count is a hard per-agent ceiling.
	MaxRetries int
	// Verifier independently checks completion. Nil preserves baseline behavior.
	Verifier Verifier
	// SpawnBudget is the host-owned admission envelope for recursive children.
	// Nil refuses child requests while preserving ordinary top-level Spawn.
	SpawnBudget *SpawnBudget
	// Warm wires the two-watermark hibernation warm band (#5072, follow-on
	// #4035) into the Step loop: an agent that also implements Restorable is
	// enrolled on spawn (its fresh context frozen to disk, no goroutine held),
	// takes a resident slot around each Step, and hands it back at every step
	// boundary — warm into the reserve while residency is above low-water, cold
	// to the store otherwise — so N enrolled agents share R resident slots.
	// Nil (the default) is the byte-identical off posture: no enrollment, no
	// residency, no store, and the loop below is exactly the pre-band loop.
	Warm *WarmBand
}

// Defaults for Config zero values.
const (
	DefaultWorkers = 8
	DefaultQueue   = 256
)

// Structured refusals.
var (
	ErrNilGateway  = errors.New("microagent: NewHost requires the one shared gateway (nil Gateway)")
	ErrNilAgent    = errors.New("microagent: Spawn requires a non-nil Microagent")
	ErrQueueFull   = errors.New("microagent: pending-spawn queue is full")
	ErrDraining    = errors.New("microagent: host is draining; no new spawns")
	ErrClosed      = errors.New("microagent: host is closed")
	ErrDuplicateID = errors.New("microagent: agent id is already live or retired (an id is one agent lifetime)")
	ErrTurnBudget  = errors.New("microagent: hard turn budget exhausted")
)

// job is one accepted agent with its own cancelable context (derived from the
// host's, so Close cancels the whole fleet).
type job struct {
	id     string
	m      Microagent
	ctx    context.Context
	cancel context.CancelFunc
	// banded is set once, at enrollment, when the warm band is on AND this agent
	// implements Restorable. It is read only by the driving worker goroutine, so it
	// needs no lock. When false every band call below is a no-op and the loop is
	// byte-identically the pre-band loop.
	banded bool
}

// Host drives K concurrent Microagent.Step calls as goroutines, all sharing one
// in-process gateway (#2002, epic #2000 M2). Lifecycle: Spawn (bounded queue) →
// worker Step loop → retire (done / cancel / error) → Reap. Drain refuses new
// spawns and waits for the fleet to finish; Close cancels everything.
type Host struct {
	gw          Gateway
	sessions    *session.Table
	audit       AuditSink
	maxTurns    int
	maxRetries  int
	verifier    Verifier
	spawnBudget *SpawnBudget
	warm        *WarmBand

	queue chan *job

	ctx     context.Context
	cancel  context.CancelFunc
	workers sync.WaitGroup // the K Step drivers
	pending sync.WaitGroup // one count per accepted, not-yet-retired agent

	mu       sync.Mutex
	live     map[string]*job
	results  []Result
	draining bool
	closed   bool
}

// NewHost builds a running Host over the ONE shared gateway and starts its K
// workers. Callers own Close.
func NewHost(gw Gateway, cfg Config) (*Host, error) {
	if gw == nil {
		return nil, ErrNilGateway
	}
	workers := cfg.Workers
	if workers <= 0 {
		workers = DefaultWorkers
	}
	queue := cfg.Queue
	if queue <= 0 {
		queue = DefaultQueue
	}
	sessions := cfg.Sessions
	if sessions == nil {
		sessions = session.NewTable()
	}
	audit := cfg.Audit
	if audit == nil {
		audit = nopSink{}
	}
	h := &Host{
		gw:          gw,
		sessions:    sessions,
		audit:       audit,
		maxTurns:    max(0, cfg.MaxTurns),
		maxRetries:  max(0, cfg.MaxRetries),
		verifier:    cfg.Verifier,
		spawnBudget: cfg.SpawnBudget,
		warm:        cfg.Warm,
		queue:       make(chan *job, queue),
		live:        map[string]*job{},
	}
	h.ctx, h.cancel = context.WithCancel(context.Background())
	for i := 0; i < workers; i++ {
		h.workers.Add(1)
		go h.worker()
	}
	return h, nil
}

// Sessions exposes the host's per-agent state table (read via Get/Snapshot).
func (h *Host) Sessions() *session.Table { return h.sessions }

// Spawn accepts an agent under id and enqueues it for a worker. It refuses —
// never blocks — when the bounded queue is full (ErrQueueFull), when the host is
// draining or closed, and when the id already names a live or retired agent
// (ErrDuplicateID: an id is one agent lifetime, enforced by the session table's
// terminal-state machine). On acceptance the agent's session entry exists and
// is Running.
func (h *Host) Spawn(id string, m Microagent) error {
	if m == nil {
		return ErrNilAgent
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return ErrClosed
	}
	if _, dup := h.live[id]; dup {
		return ErrDuplicateID
	}
	// Per-agent state (#2002 scope 2): one session.Table map entry per agent,
	// keyed by id as the TraceID. A terminal (Stopped) entry refuses the
	// transition, which refuses a re-spawn of a retired id.
	if _, ok := h.sessions.Transition(id, session.Running, ""); !ok {
		return ErrDuplicateID
	}
	if h.draining {
		h.sessions.Reset(id)
		return ErrDraining
	}
	ctx, cancel := context.WithCancel(h.ctx)
	j := &job{id: id, m: m, ctx: ctx, cancel: cancel}
	select {
	case h.queue <- j:
	default:
		// Bounded queue (#2002 scope 3): refuse loudly. Reset the session entry
		// so the rejected id is spawnable again once there is room.
		h.sessions.Reset(id)
		cancel()
		h.audit.Record(Event{Agent: id, Kind: EventReject})
		return ErrQueueFull
	}
	h.live[id] = j
	h.pending.Add(1)
	h.audit.Record(Event{Agent: id, Kind: EventSpawn})
	return nil
}

// RequestChild routes recursive work through the host budget, queue, session,
// and audit path without exposing the Host itself to the requesting agent.
func (h *Host) RequestChild(request SpawnRequest, child Microagent) error {
	if h.spawnBudget == nil {
		return fmt.Errorf("%w: host has no recursive spawn budget", ErrSpawnBudget)
	}
	if err := h.spawnBudget.Admit(request); err != nil {
		return err
	}
	if err := h.Spawn(request.ChildID, child); err != nil {
		h.spawnBudget.release(request.ParentID)
		return err
	}
	return nil
}

// Cancel requests cancellation of one live (queued or running) agent. It
// reports whether the id named a live agent; retirement is asynchronous (the
// agent lands in Reap with Done=false and a context error).
func (h *Host) Cancel(id string) bool {
	h.mu.Lock()
	j := h.live[id]
	h.mu.Unlock()
	if j == nil {
		return false
	}
	j.cancel()
	return true
}

// Drain refuses new spawns and waits until every accepted agent has retired —
// queued agents included (they still run to completion; Drain is graceful, not
// a cancel). It returns ctx.Err() if the fleet does not finish in time.
func (h *Host) Drain(ctx context.Context) error {
	h.mu.Lock()
	h.draining = true
	h.mu.Unlock()
	done := make(chan struct{})
	go func() { h.pending.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close cancels every live agent, stops the K workers, and retires anything
// still queued (as cancelled). It waits for in-flight Step calls to return at
// their next boundary. Idempotent; Spawn after Close returns ErrClosed.
func (h *Host) Close() {
	h.mu.Lock()
	if h.closed {
		h.mu.Unlock()
		return
	}
	h.closed = true
	h.draining = true
	h.mu.Unlock()
	h.cancel()       // every job ctx derives from h.ctx
	h.workers.Wait() // workers retire their in-flight agent, then exit
	for {
		select {
		case j := <-h.queue: // never picked up by a worker
			h.retire(j, 0, false, context.Canceled)
		default:
			h.pending.Wait()
			return
		}
	}
}

// Reap returns and clears the retired-agent results accumulated so far.
func (h *Host) Reap() []Result {
	h.mu.Lock()
	out := h.results
	h.results = nil
	h.mu.Unlock()
	return out
}

// Live reports how many accepted agents have not yet retired.
func (h *Host) Live() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.live)
}

// worker is one of the K Step drivers: it pulls accepted agents off the bounded
// queue and drives each to retirement.
func (h *Host) worker() {
	defer h.workers.Done()
	for {
		select {
		case <-h.ctx.Done():
			return
		case j := <-h.queue:
			h.run(j)
		}
	}
}

// run drives ONE agent: Step through the shared gateway until done, error, or
// cancel, re-checking the agent's context at every step boundary.
func (h *Host) run(j *job) {
	steps, retries := 0, 0
	if err := h.enroll(j); err != nil {
		h.retire(j, steps, false, err)
		return
	}
	for {
		if j.ctx.Err() != nil {
			h.retire(j, steps, false, j.ctx.Err())
			return
		}
		if h.maxTurns > 0 && steps >= h.maxTurns {
			h.retire(j, steps, false, ErrTurnBudget)
			return
		}
		m, err := h.acquire(j)
		if err != nil {
			h.retire(j, steps, false, err)
			return
		}
		// Re-read per step rather than once before the loop: a banded agent's live
		// value is rebuilt by every cold Wake (into a fresh Blank vessel), so the
		// retry seam must be taken off the value actually being stepped. For an
		// unbanded agent m is always j.m, so this is the same value every time.
		feedback, canRetry := m.(RetryFeedback)
		done, err := m.Step(j.ctx, h.gw)
		steps++
		switch {
		case err != nil && j.ctx.Err() != nil:
			h.retire(j, steps, false, j.ctx.Err())
			return
		case err != nil && canRetry && retries < h.maxRetries:
			// The failed Step's actual error is the retry evidence. Feedback must
			// accept it before another Step is allowed; this makes blind retry
			// structurally impossible.
			if feedbackErr := feedback.RetryFeedback(j.ctx, err); feedbackErr != nil {
				h.retire(j, steps, false, errors.Join(err, feedbackErr))
				return
			}
			retries++
			h.audit.Record(Event{Agent: j.id, Kind: EventRetry, Steps: steps, Err: err.Error()})
			continue
		case err != nil:
			h.retire(j, steps, false, err)
			return
		case done:
			if h.verifier == nil {
				h.retire(j, steps, true, nil)
				return
			}
			verifyErr := h.verifier.Verify(j.ctx, VerificationInput{
				Agent: j.id,
				Steps: steps,
			})
			event := Event{Agent: j.id, Kind: EventVerify, Steps: steps}
			if verifyErr != nil {
				event.Err = verifyErr.Error()
			}
			h.mu.Lock()
			h.audit.Record(event)
			h.mu.Unlock()
			if verifyErr == nil {
				h.retire(j, steps, true, nil)
				return
			}
			verification := &VerificationError{Evidence: verifyErr}
			if canRetry && retries < h.maxRetries {
				if feedbackErr := feedback.RetryFeedback(j.ctx, verification); feedbackErr != nil {
					h.retire(j, steps, false, errors.Join(verification, feedbackErr))
					return
				}
				retries++
				h.mu.Lock()
				h.audit.Record(Event{Agent: j.id, Kind: EventRetry, Steps: steps, Err: verification.Error()})
				h.mu.Unlock()
				continue
			}
			h.retire(j, steps, false, verification)
			return
		}
		// Clean step boundary: Step returned, not done, no error. Hand the resident
		// slot back and let the #4035 WarmPark fold decide where this context goes —
		// warm into the reserve while residency is above low-water (its next Acquire
		// pops it back at ZERO Thaw), cold to the store otherwise. A RETRY does not
		// reach here on purpose: it keeps its slot, so the next Acquire returns the
		// same held value with no second admit and no round-trip.
		if err := h.yield(j); err != nil {
			h.retire(j, steps, false, err)
			return
		}
	}
}

// enroll registers j with the warm band, but only when the band is ON and the agent
// implements Restorable — the band must be able to rebuild the agent into a fresh
// vessel after it drops its last reference, which is the whole point of hibernating
// the context. An agent that is not Restorable (or any agent at all when Config.Warm
// is nil) stays unbanded and is stepped exactly the pre-band way.
//
// Enrollment freezes the agent's turn-0 context to disk, so it costs one file and no
// goroutine: N enrolled agents go on to share R resident slots.
func (h *Host) enroll(j *job) error {
	if h.warm == nil {
		return nil
	}
	r, ok := j.m.(Restorable)
	if !ok {
		return nil
	}
	if err := h.warm.Enroll(j.id, r); err != nil {
		return err
	}
	j.banded = true
	return nil
}

// acquire resolves the agent to step for this unit of work. Unbanded, that is just
// j.m. Banded, it takes a resident slot (blocking until one frees, or until the
// agent's context ends) and returns the LIVE agent the band resolved: the same value
// if this id is mid-retry, a warm reserve hit at 0 Thaw, else a cold Wake — the
// status-quo cold start a warm miss falls through to.
func (h *Host) acquire(j *job) (Microagent, error) {
	if !j.banded {
		return j.m, nil
	}
	return h.warm.Acquire(j.ctx, j.id)
}

// yield hands j's resident slot back at a clean step boundary. It is a no-op for an
// unbanded agent.
func (h *Host) yield(j *job) error {
	if !j.banded {
		return nil
	}
	return h.warm.Yield(j.id)
}

// retire records one agent's terminal outcome exactly once: session entry →
// Stopped with a reason, one audit event, one Reap result.
func (h *Host) retire(j *job, steps int, done bool, err error) {
	j.cancel()
	// Every terminal path funnels through here, so this one call covers done, error
	// and cancel alike: it releases any resident slot still held, drops the live
	// value, and Takes any warm residue back out so the reserve never carries a
	// finished agent. Retire is idempotent, so a double retire is harmless.
	if j.banded {
		h.warm.Retire(j.id)
	}
	reason, kind := "done", EventDone
	switch {
	case err != nil && j.ctx.Err() != nil && errors.Is(err, j.ctx.Err()):
		reason, kind = "cancelled", EventCancel
	case err != nil:
		reason, kind = "error: "+err.Error(), EventError
	}
	h.mu.Lock()
	delete(h.live, j.id)
	h.results = append(h.results, Result{ID: j.id, Steps: steps, Done: done, Err: err})
	h.sessions.Transition(j.id, session.Stopped, reason)
	ev := Event{Agent: j.id, Kind: kind, Steps: steps}
	if err != nil {
		ev.Err = err.Error()
	}
	h.audit.Record(ev)
	h.mu.Unlock()
	h.pending.Done()
}
