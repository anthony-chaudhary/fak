package taskmgr

import (
	"sync"
	"time"
)

// stallsignal.go — the REPEAT-STALL witness (#768). The time-based LivenessStalled
// in taskmgr.go answers "has anything happened lately?"; this answers the orthogonal
// question "is something happening, but the SAME thing over and over?". An agent loop
// can heartbeat steadily (so it reads time-LIVE) while being wedged — re-issuing the
// identical tool call (a Read of the same path, a duplicate search) turn after turn,
// making no progress. The kernel already content-addresses a tool call's arguments;
// this folds that hash into a per-task signal: when one input hash recurs more than a
// small threshold inside a single turn, the task is RepeatStalled.
//
// It is a SEPARATE, composable tracker — not a field on taskState — so it is purely
// additive: a host that wants the signal constructs a RepeatStallMonitor alongside its
// Manager and feeds it the same tool-input hashes the kernel already computes. The
// monitor never blocks or fails a call; it only OBSERVES and reports, the way every
// other taskmgr rung reports rather than enforces.

// DefaultRepeatStallThreshold is the repeat count at which a within-turn tool-input
// hash flips a task to RepeatStalled. The issue's "> 2x in one turn" maps to: the
// FIRST call is work, the SECOND is a retry, the THIRD identical call inside the same
// turn is the signal that the loop is wedged. So the third occurrence (count == 3)
// trips it.
const DefaultRepeatStallThreshold = 3

// RepeatSignal is the verdict ObserveToolCall returns for one observed tool call. It
// is a pure value: the caller decides what to do (surface it, refuse the beat, route
// the task to a replan). Stalled is the one-bit gate; the rest is evidence.
type RepeatSignal struct {
	// Stalled is true once Count has reached the monitor's threshold for Hash within
	// the current turn — the task is repeating the same tool input and not progressing.
	Stalled bool `json:"stalled"`
	// Hash is the tool-input hash this observation carried (echoed for the caller's log).
	Hash string `json:"hash,omitempty"`
	// Count is how many times Hash has been seen in the current turn, including this
	// observation. A fresh hash is 1; the threshold (default 3) is the trip point.
	Count int `json:"count"`
	// Turn is the turn index this observation landed in (0-based; advanced by NextTurn).
	Turn int `json:"turn"`
	// FirstSeenUnixNano is when Hash first appeared in the current turn — so a host can
	// report how long the loop has been wedged on it.
	FirstSeenUnixNano int64 `json:"first_seen_unix_nano,omitempty"`
}

// RepeatStallMonitor tracks within-turn tool-input repetition per task. It is
// concurrency-safe and clock-injectable (mirroring Manager), so a test proves the
// trip math without sleeping. Construct it once per process alongside the Manager.
type RepeatStallMonitor struct {
	mu        sync.Mutex
	clock     func() time.Time
	threshold int
	tasks     map[string]*repeatState
}

type repeatState struct {
	turn   int
	counts map[string]int
	first  map[string]time.Time
}

// RepeatStallOption configures a RepeatStallMonitor at construction.
type RepeatStallOption func(*RepeatStallMonitor)

// WithRepeatClock injects the monitor's clock (defaults to time.Now). A nil clock is
// ignored, mirroring WithClock on the Manager.
func WithRepeatClock(clock func() time.Time) RepeatStallOption {
	return func(r *RepeatStallMonitor) {
		if clock != nil {
			r.clock = clock
		}
	}
}

// WithRepeatThreshold overrides DefaultRepeatStallThreshold. A value < 2 is ignored
// (a threshold of 1 would flag every single call as a stall, which is meaningless).
func WithRepeatThreshold(threshold int) RepeatStallOption {
	return func(r *RepeatStallMonitor) {
		if threshold >= 2 {
			r.threshold = threshold
		}
	}
}

// NewRepeatStallMonitor builds a monitor with the default threshold and clock, plus
// any overriding options.
func NewRepeatStallMonitor(opts ...RepeatStallOption) *RepeatStallMonitor {
	r := &RepeatStallMonitor{
		clock:     time.Now,
		threshold: DefaultRepeatStallThreshold,
		tasks:     map[string]*repeatState{},
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Threshold reports the configured repeat count at which a hash trips the stall.
func (r *RepeatStallMonitor) Threshold() int { return r.threshold }

// ObserveToolCall records one tool-input hash for taskID in the current turn and
// returns the resulting signal. An empty hash is a no-op observation (Count 0, never
// stalled): a caller that cannot content-address a call should not have it counted as
// a repeat of "the empty call". The same hash seen `threshold` times within one turn
// returns Stalled=true; advancing the turn (NextTurn) clears the counts so a legitimate
// re-issue in a new turn starts fresh.
func (r *RepeatStallMonitor) ObserveToolCall(taskID, hash string) RepeatSignal {
	if hash == "" {
		return RepeatSignal{Hash: "", Count: 0, Turn: r.currentTurn(taskID)}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	st := r.stateLocked(taskID)
	st.counts[hash]++
	count := st.counts[hash]
	if count == 1 {
		st.first[hash] = r.clock()
	}
	first := st.first[hash]
	return RepeatSignal{
		Stalled:           count >= r.threshold,
		Hash:              hash,
		Count:             count,
		Turn:              st.turn,
		FirstSeenUnixNano: unixNanoOrZero(first),
	}
}

// NextTurn advances taskID to the next turn and clears its within-turn repeat counts,
// returning the new turn index. A host calls it at each model-turn boundary so the
// "within ONE turn" scope is honored: the same tool call across DIFFERENT turns is
// normal agent work, only a within-turn repeat is the wedge this catches.
func (r *RepeatStallMonitor) NextTurn(taskID string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	st := r.stateLocked(taskID)
	st.turn++
	st.counts = map[string]int{}
	st.first = map[string]time.Time{}
	return st.turn
}

// Forget drops all repeat state for taskID — a host calls it when the task ends so the
// monitor does not retain per-task maps for the life of the process.
func (r *RepeatStallMonitor) Forget(taskID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.tasks, taskID)
}

// currentTurn reads the turn index for taskID without recording an observation (used
// for the empty-hash no-op path). An unseen task is turn 0.
func (r *RepeatStallMonitor) currentTurn(taskID string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if st, ok := r.tasks[taskID]; ok {
		return st.turn
	}
	return 0
}

func (r *RepeatStallMonitor) stateLocked(taskID string) *repeatState {
	st, ok := r.tasks[taskID]
	if !ok {
		st = &repeatState{counts: map[string]int{}, first: map[string]time.Time{}}
		r.tasks[taskID] = st
	}
	return st
}
