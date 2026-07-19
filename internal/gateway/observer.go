package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// observer.go — the stratified ASYNC observer rung on the result-admission chain (#2434).
//
// A blocking abi.ResultAdmitter (internal/abi/registry.go) sits ON the turn path: it
// returns a Verdict, so it can change (quarantine/transform) the admitted bytes, and it
// runs synchronously, so a slow one lags the turn. That is the right shape for a floor;
// it is the WRONG shape for observability (a PostToolUse-style notifier). This adds the
// missing stratum: a ResultObserver is handed a READ-ONLY COPY of the result AFTER the
// blocking chain has settled, delivered async off the turn path, under a per-rung latency
// budget and sample rate. The type enforces what a hook harness enforces only by
// convention — an observer that structurally cannot block (Observe runs on its own
// goroutine, its return is never awaited by the turn) and cannot mutate (it holds a copy
// by value, never a handle to the live message). A flaky observer that fails N times in a
// window is auto-disabled and journals HOOK_UNHEALTHY, so it degrades LOUDLY instead of
// silently lagging the fleet forever.

// ObservedResult is the read-only snapshot handed to a ResultObserver after the blocking
// result-admission chain settles. Every field is a value (string), and the whole struct
// is passed BY VALUE, so an observer receives its own copy: it structurally cannot reach
// back and change the admitted bytes the turn path has already forwarded. Strings are
// immutable in Go, so even the payload cannot be mutated through this handle.
type ObservedResult struct {
	// TraceID is the per-session trace the result was admitted under.
	TraceID string
	// ToolCallID pairs the result to the assistant tool_use that produced it (may be empty).
	ToolCallID string
	// Tool is the resolved tool name (or the "tool_result" placeholder for a nameless one).
	Tool string
	// ResultDigest is the content digest the admission ledger keyed on (screen-once identity).
	ResultDigest string
	// Verdict is the wire verdict KIND the blocking chain settled on (ALLOW/QUARANTINE/
	// TRANSFORM/DENY/...).
	Verdict string
	// Content is a copy of the admitted (post-chain, possibly paged-out) bytes.
	Content string
}

// ResultObserver is a NON-BLOCKING observability rung on the result-admission chain — the
// post-tool dual of abi.ResultAdmitter, minus the two powers an admitter holds: it returns
// no verdict (so it cannot change the admitted result) and it runs async off the turn path
// (so it cannot block it). A failing Observe counts only against the observer's own health
// budget (auto-disable), never against the result.
type ResultObserver interface {
	// ObserverID names this observer in /metrics and the HOOK_UNHEALTHY journal row.
	ObserverID() string
	// Observe is invoked async, off the turn path, with a read-only copy of one admitted
	// result. A returned error — or a run that overruns the rung's latency budget — counts
	// toward the failure window that auto-disables a flaky observer; it can NEVER change the
	// admitted bytes.
	Observe(ctx context.Context, r ObservedResult) error
}

const (
	// defaultObserverFailWindow is how many failures within the window auto-disable a rung.
	defaultObserverFailWindow = 5
	// defaultObserverWindow is the rolling window those failures are counted over.
	defaultObserverWindow = time.Minute
	// defaultObserverBudget is the per-observe latency ceiling; an overrun counts as a failure.
	defaultObserverBudget = 50 * time.Millisecond
)

// observerRung is one registered observer plus its per-rung budget, sample rate, and health
// state. It is never handed to callers; the stratum owns it.
type observerRung struct {
	obs        ResultObserver
	budget     time.Duration
	sampleRate float64

	mu        sync.Mutex
	failures  []time.Time // recent failure instants still inside the window
	sampleAcc float64     // deterministic sampler accumulator (no RNG — reproducible)
	disabled  bool
}

// shouldSample decides whether this candidate result is delivered, deterministically: the
// rate is accumulated and a delivery is emitted each time it crosses a whole unit. rate<=0
// never delivers; rate>=1 always delivers; rate=0.5 delivers every other candidate. No RNG,
// so a test can assert an exact delivery count.
func (r *observerRung) shouldSample() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sampleRate <= 0 {
		return false
	}
	if r.sampleRate >= 1 {
		return true
	}
	r.sampleAcc += r.sampleRate
	if r.sampleAcc >= 1 {
		r.sampleAcc -= 1
		return true
	}
	return false
}

func (r *observerRung) isDisabled() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.disabled
}

// observerStratum holds the registered observer rungs and their shared health policy plus a
// self-contained metric surface (a lag histogram + a per-observer auto-disable counter),
// rendered by writeMetrics the same way resumeProj renders its own family. It is nil-safe:
// every method no-ops on a nil receiver so a bare Server (built without New) is unaffected.
type observerStratum struct {
	mu    sync.Mutex
	rungs []*observerRung

	failWindow int
	window     time.Duration

	journalPath string
	now         func() time.Time
	logf        func(string, ...any)

	metricsMu sync.Mutex
	lag       *latencyCounter
	disabled  map[string]uint64 // observer id -> times auto-disabled

	wg sync.WaitGroup // lets a test wait for async delivery to settle
}

func newObserverStratum(journalPath string, logf func(string, ...any)) *observerStratum {
	return &observerStratum{
		failWindow:  defaultObserverFailWindow,
		window:      defaultObserverWindow,
		journalPath: journalPath,
		now:         time.Now,
		logf:        logf,
		lag:         newLatencyCounter(),
		disabled:    map[string]uint64{},
	}
}

// observerJournalPath is the HOOK_UNHEALTHY journal sink, mirroring reframeJournalPath: an
// empty path (the default) keeps the journal disabled; a write failure is telemetry-only.
func observerJournalPath() string { return os.Getenv("FAK_OBSERVER_JOURNAL") }

// register adds an observer rung. A non-positive budget falls back to the default; a
// non-positive sample rate means the observer never fires (registered but dormant).
func (st *observerStratum) register(obs ResultObserver, budget time.Duration, sampleRate float64) {
	if st == nil || obs == nil {
		return
	}
	if budget <= 0 {
		budget = defaultObserverBudget
	}
	st.mu.Lock()
	st.rungs = append(st.rungs, &observerRung{obs: obs, budget: budget, sampleRate: sampleRate})
	st.mu.Unlock()
}

// RegisterResultObserver adds a non-blocking async observer to the result-admission chain
// (#2434). budget is the per-observe latency ceiling (an overrun counts toward auto-disable);
// sampleRate in (0,1] is the deterministic fraction of admitted results delivered. Nil-safe;
// lazily builds the stratum so a directly-constructed Server (tests) can register too.
func (s *Server) RegisterResultObserver(obs ResultObserver, budget time.Duration, sampleRate float64) {
	if s == nil {
		return
	}
	if s.observers == nil {
		s.observers = newObserverStratum(observerJournalPath(), s.logf)
	}
	s.observers.register(obs, budget, sampleRate)
}

// dispatch hands each enabled, sampled observer a read-only copy of the settled results,
// async and off the turn path. It returns immediately; the observes run on their own
// goroutines. A nil stratum or an empty set is a no-op.
func (st *observerStratum) dispatch(ctx context.Context, results []ObservedResult) {
	if st == nil || len(results) == 0 {
		return
	}
	st.mu.Lock()
	rungs := append([]*observerRung(nil), st.rungs...)
	st.mu.Unlock()
	for _, rung := range rungs {
		if rung.isDisabled() {
			continue
		}
		for _, r := range results {
			if !rung.shouldSample() {
				continue
			}
			st.wg.Add(1)
			go st.deliver(ctx, rung, r)
		}
	}
}

// wait blocks until every dispatched observe has finished. Test-only synchronization: the
// production turn path never calls it (delivery is fire-and-forget off the turn).
func (st *observerStratum) wait() {
	if st == nil {
		return
	}
	st.wg.Wait()
}

func (st *observerStratum) deliver(ctx context.Context, rung *observerRung, r ObservedResult) {
	defer st.wg.Done()
	start := st.now()
	err := safeObserve(ctx, rung.obs, r)
	elapsed := st.now().Sub(start)
	st.recordLag(elapsed)
	overBudget := rung.budget > 0 && elapsed > rung.budget
	if err == nil && !overBudget {
		return
	}
	if st.noteFailure(rung) {
		st.disable(rung, r.TraceID, err, overBudget, elapsed)
	}
}

// safeObserve runs the observer and converts a panic into an error: a buggy observer must
// degrade against its own health budget, never crash the process from its goroutine.
func safeObserve(ctx context.Context, obs ResultObserver, r ObservedResult) (err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("observer panic: %v", rec)
		}
	}()
	return obs.Observe(ctx, r)
}

// noteFailure records one failure instant, prunes those older than the window, and reports
// whether this failure crossed the auto-disable threshold (and the rung was not already
// disabled) — so disable side effects fire exactly once.
func (st *observerStratum) noteFailure(rung *observerRung) (justDisabled bool) {
	now := st.now()
	rung.mu.Lock()
	defer rung.mu.Unlock()
	if rung.disabled {
		return false
	}
	cutoff := now.Add(-st.window)
	kept := rung.failures[:0]
	for _, t := range rung.failures {
		if t.After(cutoff) {
			kept = append(kept, t)
		}
	}
	kept = append(kept, now)
	rung.failures = kept
	if len(rung.failures) >= st.failWindow {
		rung.disabled = true
		return true
	}
	return false
}

// disable bumps the per-observer auto-disable counter, writes a HOOK_UNHEALTHY journal row
// naming the observer id, and logs the event. Called exactly once per rung (noteFailure
// gates it).
func (st *observerStratum) disable(rung *observerRung, traceID string, cause error, overBudget bool, elapsed time.Duration) {
	id := rung.obs.ObserverID()
	st.metricsMu.Lock()
	if st.disabled == nil {
		st.disabled = map[string]uint64{}
	}
	st.disabled[id]++
	st.metricsMu.Unlock()

	reason, detail := "OBSERVE_ERROR", ""
	if cause != nil {
		detail = cause.Error()
	}
	if overBudget {
		reason = "BUDGET_EXCEEDED"
		if detail == "" {
			detail = fmt.Sprintf("observe took %s over budget %s", elapsed, rung.budget)
		}
	}
	st.journal(observerHealthRow{
		Event:      "HOOK_UNHEALTHY",
		ObserverID: id,
		TraceID:    traceID,
		Reason:     reason,
		Failures:   st.failWindow,
		WindowSecs: st.window.Seconds(),
		Detail:     detail,
		TS:         st.now().UTC().Format(time.RFC3339Nano),
	})
	if st.logf != nil {
		st.logf("gateway: observer %q auto-disabled after %d failures in %s (HOOK_UNHEALTHY reason=%s): %s",
			id, st.failWindow, st.window, reason, detail)
	}
}

// observerHealthRow is one HOOK_UNHEALTHY journal record. The observer id is always named
// so an operator can tell WHICH notifier degraded, not just that one did.
type observerHealthRow struct {
	Event      string  `json:"event"`
	ObserverID string  `json:"observer_id"`
	TraceID    string  `json:"trace_id,omitempty"`
	Reason     string  `json:"reason"`
	Failures   int     `json:"failures"`
	WindowSecs float64 `json:"window_seconds"`
	Detail     string  `json:"detail,omitempty"`
	TS         string  `json:"ts"`
}

func (st *observerStratum) journal(row observerHealthRow) {
	if st.journalPath == "" {
		return
	}
	b, err := json.Marshal(row)
	if err != nil {
		return
	}
	f, err := os.OpenFile(st.journalPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(b, '\n'))
}

func (st *observerStratum) recordLag(d time.Duration) {
	secs := d.Seconds()
	if secs < 0 {
		secs = 0
	}
	st.metricsMu.Lock()
	st.lag.observe(secs)
	st.metricsMu.Unlock()
}

// writeMetrics renders the two families the issue asks for, always declaring HELP/TYPE so a
// panel exists from the first scrape (the dogfood-at-zero posture the deny-all family keeps):
//   - fak_gateway_observer_lag_seconds  — async observer delivery-latency distribution.
//   - fak_gateway_observer_disabled_total{observer} — times each observer auto-disabled.
func (st *observerStratum) writeMetrics(b *strings.Builder) {
	if st == nil {
		return
	}
	st.mu.Lock()
	ids := make([]string, 0, len(st.rungs))
	for _, r := range st.rungs {
		ids = append(ids, r.obs.ObserverID())
	}
	st.mu.Unlock()

	st.metricsMu.Lock()
	lag := st.lag.snapshot()
	disabled := make(map[string]uint64, len(st.disabled))
	for k, v := range st.disabled {
		disabled[k] = v
	}
	st.metricsMu.Unlock()

	writeHelpType(b, "fak_gateway_observer_lag_seconds",
		"Async result-observer delivery latency (#2434): wall-clock each non-blocking observer spent handling one admitted result, off the turn path. The observer_lag_seconds distribution an operator watches to see a stratum lagging behind admission — the hot-path cost this stratum removes from the blocking ResultAdmitter chain.",
		"histogram")
	writeHistogram(b, "fak_gateway_observer_lag_seconds", "", lag)

	writeHelpType(b, "fak_gateway_observer_disabled_total",
		"Async result observers auto-disabled after too many failures/budget-overruns in the health window (#2434), by observer id. Each increment corresponds to a HOOK_UNHEALTHY journal row naming the same observer — a flaky notifier degrading LOUDLY instead of silently lagging forever. 0 until an observer trips.",
		"counter")
	seen := map[string]bool{}
	sort.Strings(ids)
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		fmt.Fprintf(b, "fak_gateway_observer_disabled_total{observer=\"%s\"} %d\n", promQuote(id), disabled[id])
	}
	// An observer that tripped but is no longer registered still reports its count so a
	// disable is never silently dropped from the surface.
	extra := make([]string, 0)
	for id := range disabled {
		if !seen[id] {
			extra = append(extra, id)
		}
	}
	sort.Strings(extra)
	for _, id := range extra {
		fmt.Fprintf(b, "fak_gateway_observer_disabled_total{observer=\"%s\"} %d\n", promQuote(id), disabled[id])
	}
}
