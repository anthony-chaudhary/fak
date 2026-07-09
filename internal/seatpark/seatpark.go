// Package seatpark is a pure, bounded park-and-retry fold for the no-seat
// transient (REFUSE_NO_ACCOUNT). When dispatch preflight refuses a launch
// because every Claude seat is busy or throttled, re-attempting on the very
// next tick ("bursting") just re-checks a wall that only a PEER FINISHING can
// move -- burning ticks, log noise, and preflight probes against a shared
// capacity window that reopens on its own. This kernel folds a task's
// consecutive no-seat parks plus a caller-supplied clock into a bounded
// exponential backoff and returns a closed verdict: SEAT_READY (attempt now),
// SEAT_PARKED (still inside the current backoff window -- skip this tick
// cheaply), or SEAT_EXHAUSTED (the bounded retry budget is spent -- stop
// re-offering this cycle and surface for the next one) -- instead of an
// unbounded busy-retry (#3523).
//
// Pure: same Input in, same Decision out; zero I/O, zero clock reads -- the
// caller supplies "now" as data (Input.NowUnix), the same discipline
// internal/attemptbudget and internal/dispatchorder already use for
// clock-dependent folds. It never decides WHY a seat was unavailable; it only
// counts parks, applies the backoff schedule, and thresholds facts the caller
// already gathered.
//
// Distinct from internal/attemptbudget, which folds an issue's POST-attempt
// failure history (a worker ran and failed): seatpark folds the PRE-attempt
// seat-contention refusal, where no worker ever launched and so no Attempt is
// ever recorded. The two compose -- seatpark decides whether a launch is even
// allowed to be tried; attemptbudget decides what to do once one has run.
package seatpark

import "fmt"

// Status is the closed park-and-retry verdict for one seat-blocked task. It is
// an additive set -- a new value is an added constant plus a knownStatus arm,
// never a free-text field -- so the dispatcher's skip-reason surface stays a
// typed, verifiable vocabulary.
type Status string

const (
	// StatusReady means there is no active park window (the task has never
	// parked, or its backoff window has elapsed): attempt a launch now. A first
	// no-seat encounter is READY, not a burst -- bursting is only the UNBOUNDED
	// immediate re-attempt this kernel replaces.
	StatusReady Status = "SEAT_READY"
	// StatusParked means the task is seat-blocked and still inside its current
	// backoff window: skip this tick cheaply rather than re-probe a wall only a
	// peer finishing can move.
	StatusParked Status = "SEAT_PARKED"
	// StatusExhausted means the bounded retry budget (Policy.MaxParks) is spent:
	// stop re-offering the task this cycle and surface it for the next one, so a
	// persistently seat-starved task neither bursts nor parks forever.
	StatusExhausted Status = "SEAT_EXHAUSTED"
)

// knownStatus reports whether s is one of the closed Status set.
func knownStatus(s Status) bool {
	switch s {
	case StatusReady, StatusParked, StatusExhausted:
		return true
	}
	return false
}

// Valid reports whether s is a member of the closed Status vocabulary.
func (s Status) Valid() bool { return knownStatus(s) }

// Sourced default policy constants. The cap is anchored to a real in-repo
// precedent rather than invented: a no-seat refusal is the same shape of wall as
// internal/attemptbudget's FailureClassRateLimit -- "a shared capacity window
// reopening on its own" (attemptbudget.go:121) -- so the longest a task waits
// between no-seat retries matches that 5-minute window.
const (
	// DefaultMaxParks bounds the retries: after this many consecutive no-seat
	// parks a task is EXHAUSTED for the cycle instead of parked forever.
	DefaultMaxParks = 5
	// DefaultBaseSeconds is the first park window -- a peer finishing frees a seat
	// on the order of tens of seconds, so the first retry waits that long rather
	// than re-bursting immediately.
	DefaultBaseSeconds int64 = 30
	// DefaultFactor is the geometric growth per park (30s -> 60s -> 120s -> ...).
	DefaultFactor = 2
	// DefaultCapSeconds caps the backoff window at attemptbudget's rate-limit
	// precedent (5m): a shared capacity window that reopens on its own.
	DefaultCapSeconds int64 = 5 * 60
)

// Policy is the bounded backoff schedule. Every field zero-valued => the
// documented default is used, so a caller may pass a zero Policy for sane
// behavior and override only the fields it tunes.
type Policy struct {
	// MaxParks is the bounded retry budget before SEAT_EXHAUSTED. <=0 =>
	// DefaultMaxParks.
	MaxParks int
	// BaseSeconds is the first park window in seconds. <=0 => DefaultBaseSeconds.
	BaseSeconds int64
	// Factor is the geometric growth per park. <2 => DefaultFactor (a factor
	// below 2 would not actually back off).
	Factor int
	// CapSeconds is the maximum park window in seconds. <=0 => DefaultCapSeconds.
	CapSeconds int64
}

// withDefaults returns the policy with every unset/invalid field filled from the
// documented defaults, so Decide never divides an empty schedule.
func (p Policy) withDefaults() Policy {
	if p.MaxParks <= 0 {
		p.MaxParks = DefaultMaxParks
	}
	if p.BaseSeconds <= 0 {
		p.BaseSeconds = DefaultBaseSeconds
	}
	if p.Factor < 2 {
		p.Factor = DefaultFactor
	}
	if p.CapSeconds <= 0 {
		p.CapSeconds = DefaultCapSeconds
	}
	return p
}

// Input is one seat-blocked task's park-and-retry facts.
type Input struct {
	TaskID string `json:"task_id,omitempty"`
	// Parks is how many times this task has ALREADY been parked on a no-seat
	// refusal this cycle. 0 => the task has not parked yet (its next launch is a
	// first attempt, not a burst).
	Parks int `json:"parks"`
	// LastParkUnix is when the task was most recently parked (unix seconds).
	// 0 => never parked; the backoff window then has no anchor and the task is
	// READY.
	LastParkUnix int64 `json:"last_park_unix,omitempty"`
	// NowUnix is the caller-supplied clock reading used to test whether the
	// current backoff window has elapsed. 0 => the caller does not supply a
	// clock; the Decision still reports the window that WOULD apply, but a task
	// is never reported PARKED without a "now" to compare against (fail toward
	// READY, never silently stall). This package never reads a clock itself.
	NowUnix int64 `json:"now_unix,omitempty"`
	// Policy is the backoff schedule; a zero Policy uses the documented defaults.
	Policy Policy `json:"policy,omitempty"`
}

// Decision is the verdict for one seat-blocked task.
type Decision struct {
	TaskID string `json:"task_id,omitempty"`
	Status Status `json:"status"`
	Parks  int    `json:"parks"`
	// MaxParks is the effective bounded retry budget (after defaults).
	MaxParks int `json:"max_parks"`
	// BackoffSeconds is the window applying to the CURRENT park count (0 when the
	// task has never parked).
	BackoffSeconds int64 `json:"backoff_seconds,omitempty"`
	// NextRetryUnix is LastParkUnix + BackoffSeconds -- the earliest time the
	// task should be re-attempted. 0 when the task has never parked.
	NextRetryUnix int64 `json:"next_retry_unix,omitempty"`
	// Detail is a free-text note for the trace.
	Detail string `json:"detail,omitempty"`
}

// ShouldAttempt reports whether the dispatcher should try to launch the task now.
func (d Decision) ShouldAttempt() bool { return d.Status == StatusReady }

// Retryable reports whether the task will be re-offered later (PARKED) rather
// than dropped for the cycle (EXHAUSTED).
func (d Decision) Retryable() bool { return d.Status != StatusExhausted }

// Decide folds one task's Input into a Decision, in this order: EXHAUSTED once
// Parks reaches the bounded budget (a hard stop that overrides an open window);
// otherwise PARKED when the task has parked, a clock is supplied, and now is
// still before NextRetryUnix; otherwise READY. The Decision always carries the
// backoff window and NextRetryUnix that apply to the current park count, so a
// report can show the schedule even for an EXHAUSTED task.
func Decide(in Input) Decision {
	p := in.Policy.withDefaults()
	d := Decision{
		TaskID:         in.TaskID,
		Parks:          in.Parks,
		MaxParks:       p.MaxParks,
		BackoffSeconds: backoffSeconds(p, in.Parks),
	}
	if in.LastParkUnix > 0 && in.Parks > 0 {
		d.NextRetryUnix = in.LastParkUnix + d.BackoffSeconds
	}

	switch {
	case in.Parks >= p.MaxParks:
		d.Status = StatusExhausted
		d.Detail = fmt.Sprintf("%d no-seat parks reached the bounded budget of %d; stop re-offering this cycle", in.Parks, p.MaxParks)
	case d.NextRetryUnix > 0 && in.NowUnix > 0 && in.NowUnix < d.NextRetryUnix:
		d.Status = StatusParked
		d.Detail = fmt.Sprintf("seat-blocked; parked %ds until %d (park %d of %d)", d.BackoffSeconds, d.NextRetryUnix, in.Parks, p.MaxParks)
	default:
		d.Status = StatusReady
		if in.Parks == 0 {
			d.Detail = "no prior no-seat park; attempt now"
		} else {
			d.Detail = fmt.Sprintf("backoff window elapsed after park %d of %d; attempt now", in.Parks, p.MaxParks)
		}
	}
	return d
}

// backoffSeconds is the geometric window that applies after `parks` consecutive
// parks: min(cap, base * factor^(parks-1)). parks<=0 (never parked) yields 0 --
// there is no active window to wait out. The cap is applied inside the loop so a
// large park count can never overflow the running product.
func backoffSeconds(p Policy, parks int) int64 {
	if parks <= 0 {
		return 0
	}
	w := p.BaseSeconds
	for i := 1; i < parks; i++ {
		w *= int64(p.Factor)
		if w >= p.CapSeconds {
			return p.CapSeconds
		}
	}
	if w > p.CapSeconds {
		return p.CapSeconds
	}
	return w
}
