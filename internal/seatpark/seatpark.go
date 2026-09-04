// Package seatpark provides a pure, bounded park-and-retry fold for the
// no-seat transient refusal (REFUSE_NO_ACCOUNT). It folds consecutive parks and
// a clock into a bounded exponential backoff schedule, returning SEAT_READY,
// SEAT_PARKED, or SEAT_EXHAUSTED.
package seatpark

import "fmt"

// Status is the closed park-and-retry verdict for one seat-blocked task.
type Status string

const (
	// StatusReady indicates the backoff window has elapsed or never parked: attempt launch now.
	StatusReady Status = "SEAT_READY"
	// StatusParked indicates the task is seat-blocked inside its backoff window: skip this tick.
	StatusParked Status = "SEAT_PARKED"
	// StatusExhausted indicates the bounded retry budget is spent: stop re-offering this cycle.
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

// Default policy constants for bounded backoff.
const (
	// DefaultMaxParks is the default bounded retry budget before exhaustion.
	DefaultMaxParks = 5
	// DefaultBaseSeconds is the initial park window in seconds.
	DefaultBaseSeconds int64 = 30
	// DefaultFactor is the geometric growth multiplier per consecutive park.
	DefaultFactor = 2
	// DefaultCapSeconds caps backoff at 5 minutes, matching rate-limit precedents.
	DefaultCapSeconds int64 = 5 * 60
)

// Policy configures the bounded backoff schedule. Zero values default to documented constants.
type Policy struct {
	// MaxParks is the retry budget before SEAT_EXHAUSTED (default DefaultMaxParks).
	MaxParks int
	// BaseSeconds is the initial park window in seconds (default DefaultBaseSeconds).
	BaseSeconds int64
	// Factor is the geometric growth multiplier (minimum 2, default DefaultFactor).
	Factor int
	// CapSeconds is the maximum park window in seconds (default DefaultCapSeconds).
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

// Input captures one seat-blocked task's park-and-retry state.
type Input struct {
	TaskID string `json:"task_id,omitempty"`
	// Parks is the count of consecutive no-seat parks this cycle.
	Parks int `json:"parks"`
	// LastParkUnix is the unix timestamp of the most recent park.
	LastParkUnix int64 `json:"last_park_unix,omitempty"`
	// NowUnix is the caller-supplied clock timestamp (0 fails toward READY).
	NowUnix int64 `json:"now_unix,omitempty"`
	// Policy is the backoff schedule (zero values use defaults).
	Policy Policy `json:"policy,omitempty"`
}

// Decision is the adjudicated verdict and retry schedule for a seat-blocked task.
type Decision struct {
	TaskID string `json:"task_id,omitempty"`
	Status Status `json:"status"`
	Parks  int    `json:"parks"`
	// MaxParks is the effective bounded retry budget.
	MaxParks int `json:"max_parks"`
	// BackoffSeconds is the computed backoff window applying to the current park count.
	BackoffSeconds int64 `json:"backoff_seconds,omitempty"`
	// NextRetryUnix is the earliest unix timestamp to re-attempt the task.
	NextRetryUnix int64 `json:"next_retry_unix,omitempty"`
	// Detail is a diagnostic summary for tracing.
	Detail string `json:"detail,omitempty"`
}

// ShouldAttempt reports whether the dispatcher should try to launch the task now.
func (d Decision) ShouldAttempt() bool { return d.Status == StatusReady }

// Retryable reports whether the task will be re-offered later rather than dropped.
func (d Decision) Retryable() bool { return d.Status != StatusExhausted }

// Decide folds task Input into a Decision: EXHAUSTED at budget, PARKED in window, or READY.
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

// backoffSeconds computes min(cap, base * factor^(parks-1)) without integer overflow.
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
