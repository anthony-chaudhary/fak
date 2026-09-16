package microagent

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Batch enrollment -- the fan-out seam the harness needs from the get-go.
//
// One Host already runs N agents behind K Step-driver goroutines (microagent.go),
// and the 100k witness (benchmark_100k_test.go) proves N=100000 contexts flow
// through ONE host. But the dispatch harness historically constructed a FRESH
// Host per routed tick with Config{Workers: 1, Queue: 1}, ran exactly one agent,
// and tore it down -- so a 1000-issue wave produced ~1000 host constructions at
// global peak concurrency 1, and the verified 100k capacity was unreachable from
// the real path.
//
// This file is the missing mechanism: a workload-budgeted driver that builds the
// ONE host a wave should share and admits its whole enrollment set into it. The
// design rule it encodes is that CONCURRENCY IS A PROPERTY OF THE HOST, NOT OF
// THE TICK -- a tick is one unit of routing, not one unit of host construction.

// DefaultWaveBudget bounds a shared harness host when a caller declares no
// budget. It is deliberately small: the budget bounds RESIDENT work, and a
// caller that wants more concurrency must say so explicitly rather than
// inheriting an unbounded default from the size of the backlog.
const DefaultWaveBudget = 16

// Enrollment is one agent's admission request against a shared host: the id it
// is known by (also its session.Table TraceID, so an id is one agent lifetime)
// and the Microagent to drive. It is the batch analogue of a single Spawn.
type Enrollment struct {
	ID    string
	Agent Microagent
}

// EnrollmentResult is the per-agent outcome of a batch admission. Admitted is
// false when the host refused THIS agent (bad id, duplicate, drained/closed, or
// a full queue); Err carries the typed refusal verbatim so a caller is never
// left guessing why one of a thousand enrollments did not land. Partial
// admission is therefore observable and the successful agents still run -- the
// caller decides whether a partial batch is acceptable.
type EnrollmentResult struct {
	ID       string
	Admitted bool
	Err      error
}

// WaveConfig sizes the ONE host a wave shares. Workers is the resident
// concurrency budget K (how many agents Step at once); Queue bounds how many
// additional accepted agents may wait behind them. Both are bounded by the
// caller's declared budget, never by the backlog length: a 1000-issue wave on a
// K=16 box runs 16 at a time with the rest queued, rather than opening 1000
// contexts at once. A WarmBand supplied here keeps total in-RAM contexts at its
// High watermark no matter how large N grows.
type WaveConfig struct {
	Workers int // resident Step drivers K (<=0 selects DefaultWaveBudget)
	Queue   int // pending-spawn queue (<=0 selects DefaultQueue)
	Warm    *WarmBand
}

// WaveHost is ONE shared in-process host sized from a workload budget, plus the
// batch admission verb the harness fans a wave through. It owns the Host it
// builds and must be Closed by the caller.
type WaveHost struct {
	host *Host
}

// NewWaveHost builds the shared host a wave enrolls into. It sizes the pool from
// cfg and wires the same optional mechanisms a single Host accepts (session
// table, audit sink, verifier, spawn budget, capabilities, warm band) via the
// base Config the caller already assembles -- a WaveHost is Host + batch admission
// + a budget-derived pool size, never a second execution path.
//
// The budget is clamped to at least 1 worker and at least Workers queue slots so
// a misconfigured zero still runs; it is never widened to the enrollment count.
func NewWaveHost(gw Gateway, base Config, cfg WaveConfig) (*WaveHost, error) {
	workers := cfg.Workers
	if workers <= 0 {
		workers = DefaultWaveBudget
	}
	queue := cfg.Queue
	if queue <= 0 {
		queue = DefaultQueue
	}
	base.Workers = workers
	base.Queue = queue
	if cfg.Warm != nil {
		base.Warm = cfg.Warm
	}
	h, err := NewHost(gw, base)
	if err != nil {
		return nil, err
	}
	return &WaveHost{host: h}, nil
}

// Host exposes the shared host so a caller can Reap/Drain/Live/Cancel/Sessions.
func (w *WaveHost) Host() *Host { return w.host }

// Enroll admits a batch of agents into the shared host. Every enrollment is
// attempted -- a refusal of one agent never aborts the batch -- and the results
// are returned in input order so the caller can map each back to its row. The
// host's bounded queue is the backpressure: enrollments beyond
// Workers+Queue are refused with ErrQueueFull rather than blocking or growing
// without bound, and the typed error is surfaced per agent.
//
// Enroll is the batch twin of Spawn: it adds no execution semantics of its own
// and preserves every per-agent guarantee Spawn already makes (id uniqueness,
// session entry, audit event, capability envelope).
func (w *WaveHost) Enroll(enrollments []Enrollment) []EnrollmentResult {
	results := make([]EnrollmentResult, len(enrollments))
	for i, e := range enrollments {
		err := w.host.Spawn(e.ID, e.Agent)
		results[i] = EnrollmentResult{ID: e.ID, Admitted: err == nil, Err: err}
	}
	return results
}

// Admitted counts the results that were accepted, for receipts.
func Admitted(results []EnrollmentResult) int {
	n := 0
	for _, r := range results {
		if r.Admitted {
			n++
		}
	}
	return n
}

// FirstRefusal returns the first typed refusal in the batch, or nil when every
// agent was admitted. It lets a caller fail loudly on a partial batch without
// discarding the ones that landed.
func FirstRefusal(results []EnrollmentResult) error {
	for _, r := range results {
		if !r.Admitted {
			return r.Err
		}
	}
	return nil
}

// DrainAll drains every accepted agent to retirement within budget and returns
// the reaped results. It is the wave-level counterpart of Host.Drain+Reap: one
// call drains the whole batch, so a caller fanning out N agents never loops.
//
// It returns ctx.Err() (wrapped with how many agents were still live) if the
// batch does not finish in time; the caller owns Close.
func (w *WaveHost) DrainAll(ctx context.Context) ([]Result, error) {
	err := w.host.Drain(ctx)
	results := w.host.Reap()
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			return results, fmt.Errorf("microagent: wave drain incomplete (%d still live): %w: %w", w.host.Live(), ErrWaveTimeout, err)
		}
		return results, fmt.Errorf("microagent: wave drain incomplete (%d still live): %w", w.host.Live(), err)
	}
	return results, nil
}

// RunWave is the one-call harness entry point: build the shared host, enroll the
// whole batch, drain it, and return the per-agent outcomes plus the admission
// results. It is deliberately the smallest end-to-end shape the dispatch wave
// needs -- construct ONCE, enroll N, drain ONCE -- so the harness never hand-rolls
// a host per agent again.
//
// A caller that needs to inspect admission before draining (e.g. to refuse a
// partial batch) should use NewWaveHost + Enroll + DrainAll directly.
func RunWave(ctx context.Context, gw Gateway, base Config, cfg WaveConfig, enrollments []Enrollment) ([]EnrollmentResult, []Result, error) {
	w, err := NewWaveHost(gw, base, cfg)
	if err != nil {
		return nil, nil, err
	}
	defer w.Close()
	admitted := w.Enroll(enrollments)
	if refusal := FirstRefusal(admitted); refusal != nil && Admitted(admitted) == 0 {
		// Nothing landed: there is no work to drain, and the refusal is the whole
		// story. A partial batch still drains its admitted agents below.
		return admitted, nil, refusal
	}
	results, drainErr := w.DrainAll(ctx)
	if drainErr != nil {
		return admitted, results, drainErr
	}
	return admitted, results, nil
}

// Close releases the shared host and every agent it still holds. Idempotent, via
// Host.Close.
func (w *WaveHost) Close() {
	if w.host != nil {
		w.host.Close()
	}
}

// BudgetForWave derives the resident worker budget for a wave from a declared
// ceiling and the batch size: the smaller of the caller's declared maximum and
// the number of agents actually enrolling, floored at 1. It is the pure rule the
// harness uses so the SAME input always yields the same pool size, and so the
// budget is a property of the declared capacity rather than of the backlog.
func BudgetForWave(declaredMax, batchSize int) int {
	if declaredMax <= 0 {
		declaredMax = DefaultWaveBudget
	}
	if batchSize <= 0 {
		return 1
	}
	return max(1, min(declaredMax, batchSize))
}

// ErrWaveTimeout is returned by DrainAll when the wave does not retire within
// its budget. It is a typed sentinel so a harness can distinguish a slow wave
// from a host construction or admission fault.
var ErrWaveTimeout = errors.New("microagent: wave did not drain within budget")

// DefaultWaveDrainTimeout is the liveness backstop a harness uses when no
// explicit wave budget is supplied. It is a ceiling, not a latency target.
const DefaultWaveDrainTimeout = 5 * time.Minute
