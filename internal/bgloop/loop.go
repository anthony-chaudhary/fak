package bgloop

import (
	"context"
	"sync"
	"time"
)

// Loop is one unit of recurring background work the kernel supervises while it is up.
// It is pure configuration; the live runtime state lives in the supervisor's private
// loopState and is read out as a Status.
type Loop struct {
	// Name uniquely identifies the loop. It is the label in /metrics and the JSON
	// snapshot, so keep it stable and metric-safe (lowercase, no spaces). Required.
	Name string

	// Interval is the wait between the END of one tick and the START of the next.
	// A value <= 0 means CONTINUOUS: the next tick begins as soon as the last
	// returns, so such a Tick must pace itself (block until there is work) or it
	// will spin. Most kernel loops want a positive interval.
	Interval time.Duration

	// Tick is the work performed each iteration. It MUST honor ctx cancellation
	// (return promptly when ctx is Done) so the kernel can shut down cleanly. A
	// returned error is recorded and triggers backoff; a panic is recovered,
	// recorded, and also triggers backoff — neither ever crashes the kernel.
	// Required.
	Tick func(ctx context.Context) error
}

// State is the observable lifecycle phase of a supervised loop.
type State string

const (
	// StateIdle: the loop is waiting out its interval before the next tick.
	StateIdle State = "idle"
	// StateRunning: the loop is inside Tick right now.
	StateRunning State = "running"
	// StateBackoff: the last tick failed (error or panic); the loop is waiting out
	// the exponential backoff before retrying.
	StateBackoff State = "backoff"
	// StatePaused: the admit gate refused this fire (operator backpressure); the
	// loop is holding and will re-check on the next interval.
	StatePaused State = "paused"
	// StateStopped: the supervisor shut down or the lifecycle context was cancelled;
	// the loop's goroutine has exited.
	StateStopped State = "stopped"
)

// Status is a point-in-time snapshot of one loop's progress — the observability
// surface a /metrics scrape, a /v1/fak/loops reader, or `fak bgloop status` renders.
// The JSON tags are the stable export schema; new fields are additive (omitempty).
type Status struct {
	Name       string    `json:"name"`
	State      State     `json:"state"`
	Interval   string    `json:"interval"`             // human-readable (e.g. "30s"); "continuous" when <= 0
	StartedAt  time.Time `json:"started_at,omitempty"` // when the loop's goroutine began
	Ticks      uint64    `json:"ticks"`                // ticks that returned nil
	Errors     uint64    `json:"errors"`               // ticks that returned a non-nil error
	Panics     uint64    `json:"panics"`               // ticks that panicked (recovered)
	Restarts   uint64    `json:"restarts"`             // backoff cycles entered (errors + panics)
	Pauses     uint64    `json:"pauses"`               // fires the admit gate refused
	LastTickAt time.Time `json:"last_tick_at,omitempty"`
	LastErrAt  time.Time `json:"last_err_at,omitempty"`
	LastErr    string    `json:"last_err,omitempty"`
	LastDurMS  float64   `json:"last_dur_ms,omitempty"` // wall time of the last completed tick
	NextTickAt time.Time `json:"next_tick_at,omitempty"`
}

// loopState is the supervisor's private, mutex-guarded live state for one loop.
type loopState struct {
	name     string
	interval time.Duration
	tick     func(context.Context) error

	mu         sync.Mutex
	state      State
	startedAt  time.Time
	ticks      uint64
	errors     uint64
	panics     uint64
	restarts   uint64
	pauses     uint64
	lastTickAt time.Time
	lastErrAt  time.Time
	lastErr    string
	lastDur    time.Duration
	nextTickAt time.Time
}

func (ls *loopState) begin() {
	ls.mu.Lock()
	ls.startedAt = time.Now()
	ls.state = StateIdle
	ls.mu.Unlock()
}

func (ls *loopState) setState(s State) {
	ls.mu.Lock()
	ls.state = s
	ls.mu.Unlock()
}

func (ls *loopState) markRunning() {
	ls.mu.Lock()
	ls.state = StateRunning
	ls.mu.Unlock()
}

// recordTick folds one completed tick's outcome into the live counters.
func (ls *loopState) recordTick(err error, panicked bool, dur time.Duration) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.lastDur = dur
	now := time.Now()
	ls.lastTickAt = now
	switch {
	case panicked:
		ls.panics++
		ls.lastErrAt = now
		if err != nil {
			ls.lastErr = err.Error()
		}
	case err != nil:
		ls.errors++
		ls.lastErrAt = now
		ls.lastErr = err.Error()
	default:
		ls.ticks++
		ls.lastErr = ""
	}
}

// markBackoff records that the loop is waiting out a failure backoff and bumps the
// restart counter; next is when the retry is scheduled.
func (ls *loopState) markBackoff(next time.Time) {
	ls.mu.Lock()
	ls.state = StateBackoff
	ls.restarts++
	ls.nextTickAt = next
	ls.mu.Unlock()
}

func (ls *loopState) markPaused() {
	ls.mu.Lock()
	ls.state = StatePaused
	ls.pauses++
	ls.mu.Unlock()
}

func (ls *loopState) markIdle(next time.Time) {
	ls.mu.Lock()
	ls.state = StateIdle
	ls.nextTickAt = next
	ls.mu.Unlock()
}

func (ls *loopState) snapshot() Status {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	st := Status{
		Name:       ls.name,
		State:      ls.state,
		Interval:   intervalString(ls.interval),
		StartedAt:  ls.startedAt,
		Ticks:      ls.ticks,
		Errors:     ls.errors,
		Panics:     ls.panics,
		Restarts:   ls.restarts,
		Pauses:     ls.pauses,
		LastTickAt: ls.lastTickAt,
		LastErrAt:  ls.lastErrAt,
		LastErr:    ls.lastErr,
		NextTickAt: ls.nextTickAt,
	}
	if ls.lastDur > 0 {
		st.LastDurMS = float64(ls.lastDur) / float64(time.Millisecond)
	}
	return st
}

func intervalString(d time.Duration) string {
	if d <= 0 {
		return "continuous"
	}
	return d.String()
}
