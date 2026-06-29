package loopmgr

import (
	"fmt"
	"time"
)

// RestartPolicy is the pure crashloop fence for a supervised loop restart
// sequence. It does not spawn, sleep, read a clock, or sample randomness: the
// caller supplies now and, when jitter is enabled, a deterministic jitter source.
type RestartPolicy struct {
	// MaxAttempts is the number of restart attempts allowed in one consecutive
	// failure streak. A caller passes the attempts already consumed; once that
	// count reaches MaxAttempts the policy gives up with a typed reason.
	MaxAttempts uint64 `json:"max_attempts,omitempty"`

	// BaseDelay is the first retry delay. Each later attempt doubles from this
	// base until MaxDelay caps the total delay.
	BaseDelay time.Duration `json:"base_delay,omitempty"`

	// MaxDelay caps the total delay, including jitter.
	MaxDelay time.Duration `json:"max_delay,omitempty"`

	// Jitter is the maximum deterministic jitter window. The core never samples
	// random state; it passes this bound to the injected RestartJitter.
	Jitter time.Duration `json:"jitter,omitempty"`

	// ResetAfter is the healthy run window required to clear the attempt counter
	// after a success. Zero means any successful run clears the streak.
	ResetAfter time.Duration `json:"reset_after,omitempty"`
}

// RestartJitter returns a deterministic jitter offset for the one-based attempt
// and the configured window. Implementations should return a duration in
// [0, window]; out-of-range values are clamped by the policy.
type RestartJitter func(attempt uint64, window time.Duration) time.Duration

// Closed restart-decision vocabulary. WATCHDOG_RESTART_EXHAUSTED is the typed
// give-up reason downstream watchdogs route on instead of silently crashlooping.
const (
	ReasonRestartScheduled     = "WATCHDOG_RESTART_SCHEDULED"
	ReasonRestartDebounced     = "WATCHDOG_RESTART_DEBOUNCED"
	ReasonRestartExhausted     = "WATCHDOG_RESTART_EXHAUSTED"
	ReasonRestartPolicyInvalid = "WATCHDOG_RESTART_POLICY_INVALID"
)

// ValidRestartReason reports whether reason is a named restart-policy verdict.
func ValidRestartReason(reason string) bool {
	switch reason {
	case ReasonRestartScheduled, ReasonRestartDebounced, ReasonRestartExhausted, ReasonRestartPolicyInvalid:
		return true
	default:
		return false
	}
}

// RestartDecision is the structured verdict returned by RestartPolicy.Decide.
// Restart means a restart remains allowed and is scheduled for RestartAtUnixNano;
// GiveUp means the caller must stop restarting and surface Reason.
type RestartDecision struct {
	Restart           bool          `json:"restart"`
	GiveUp            bool          `json:"give_up,omitempty"`
	Reason            string        `json:"reason"`
	Summary           string        `json:"summary"`
	Attempt           uint64        `json:"attempt,omitempty"`
	After             time.Duration `json:"-"`
	AfterNanos        int64         `json:"after_nanos,omitempty"`
	RestartAtUnixNano int64         `json:"restart_at_unix_nano,omitempty"`
}

// Validate checks that the restart policy is well formed. The zero value is
// invalid because an uncapped or zero-delay restart policy would be a storm, not
// a bounded retry contract.
func (p RestartPolicy) Validate() error {
	if p.MaxAttempts == 0 {
		return fmt.Errorf("restart policy max_attempts must be > 0")
	}
	if p.BaseDelay <= 0 {
		return fmt.Errorf("restart policy base_delay must be > 0")
	}
	if p.MaxDelay <= 0 {
		return fmt.Errorf("restart policy max_delay must be > 0")
	}
	if p.MaxDelay < p.BaseDelay {
		return fmt.Errorf("restart policy max_delay %s is below base_delay %s", p.MaxDelay, p.BaseDelay)
	}
	if p.Jitter < 0 {
		return fmt.Errorf("restart policy jitter must be >= 0")
	}
	if p.ResetAfter < 0 {
		return fmt.Errorf("restart policy reset_after must be >= 0")
	}
	return nil
}

// Decide returns the restart decision for a failure streak. attempts is the
// number of restart attempts already consumed after consecutive failures; the
// returned Attempt is the next one-based attempt to schedule. Re-evaluating the
// same attempts + lastFailure while now is inside the backoff window returns the
// same RestartAtUnixNano and the remaining After duration, so callers can
// debounce duplicate watchdog ticks without spawning a second process.
func (p RestartPolicy) Decide(attempts uint64, lastFailure, now time.Time, jitter RestartJitter) RestartDecision {
	if err := p.Validate(); err != nil {
		return restartGiveUp(ReasonRestartPolicyInvalid, attempts, fmt.Sprintf("invalid restart policy: %v", err))
	}
	if attempts >= p.MaxAttempts {
		return restartGiveUp(ReasonRestartExhausted, attempts,
			fmt.Sprintf("restart attempts exhausted (%d/%d)", attempts, p.MaxAttempts))
	}

	lastFailure = lastFailure.UTC()
	now = now.UTC()
	if lastFailure.IsZero() {
		lastFailure = now
	}

	delay := p.BackoffDelay(attempts, jitter)
	restartAt := lastFailure.Add(delay)
	after := restartAt.Sub(now)
	if after < 0 {
		after = 0
	}

	reason := ReasonRestartScheduled
	if now.After(lastFailure) && now.Before(restartAt) {
		reason = ReasonRestartDebounced
	}
	attempt := attempts + 1
	return RestartDecision{
		Restart:           true,
		Reason:            reason,
		Summary:           fmt.Sprintf("restart attempt %d/%d scheduled after %s", attempt, p.MaxAttempts, after),
		Attempt:           attempt,
		After:             after,
		AfterNanos:        int64(after),
		RestartAtUnixNano: restartAt.UnixNano(),
	}
}

// BackoffDelay returns the capped exponential delay for the next restart
// attempt, including deterministic injected jitter when configured.
func (p RestartPolicy) BackoffDelay(attempts uint64, jitter RestartJitter) time.Duration {
	delay := cappedExponentialDelay(p.BaseDelay, p.MaxDelay, attempts)
	if p.Jitter > 0 && jitter != nil {
		offset := jitter(attempts+1, p.Jitter)
		if offset < 0 {
			offset = 0
		}
		if offset > p.Jitter {
			offset = p.Jitter
		}
		delay = capDelay(delay, offset, p.MaxDelay)
	}
	return delay
}

// ShouldResetAfterSuccess reports whether a successful run from started to ended
// is long enough to clear the consecutive restart-attempt counter.
func (p RestartPolicy) ShouldResetAfterSuccess(started, ended time.Time) bool {
	if ended.Before(started) {
		return false
	}
	if p.ResetAfter <= 0 {
		return true
	}
	return ended.Sub(started) >= p.ResetAfter
}

// AttemptsAfterSuccess returns the next attempt count after a successful run.
// A success that lasts through the clear window resets the count to zero;
// shorter successes preserve the streak so flapping loops do not get free retries.
func (p RestartPolicy) AttemptsAfterSuccess(attempts uint64, started, ended time.Time) uint64 {
	if attempts == 0 {
		return 0
	}
	if p.ShouldResetAfterSuccess(started, ended) {
		return 0
	}
	return attempts
}

func restartGiveUp(reason string, attempts uint64, summary string) RestartDecision {
	return RestartDecision{
		GiveUp:  true,
		Reason:  reason,
		Summary: summary,
		Attempt: attempts,
	}
}

func cappedExponentialDelay(base, max time.Duration, attempts uint64) time.Duration {
	delay := base
	for i := uint64(0); i < attempts; i++ {
		if delay >= max {
			return max
		}
		if delay > max/2 {
			return max
		}
		delay *= 2
	}
	if delay > max {
		return max
	}
	return delay
}

func capDelay(delay, offset, max time.Duration) time.Duration {
	if delay >= max {
		return max
	}
	if offset > max-delay {
		return max
	}
	return delay + offset
}
