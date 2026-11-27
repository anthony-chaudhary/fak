package ops

import (
	"fmt"
	"time"
)

const (
	// DefaultBaseProbeInterval matches Engine.Tick's 60s quick-check cadence so
	// the first re-probe waits at least one normal tick instead of polling faster.
	DefaultBaseProbeInterval = 1 * time.Minute

	// DefaultMaxProbeInterval caps the exponential probe backoff at 15m,
	// matching Engine.Tick's storage-reclaim cadence and <= 50% of
	// DefaultMaxPause (30m) so worst-case probe duty stays <= 1/15 (~6.7%) of
	// per-minute ticks while remaining logarithmic over the default pause window.
	DefaultMaxProbeInterval = 15 * time.Minute
)

// ReactivationPolicy configures the bounded probe schedule and forced-resume
// ceiling for an ops pause (#14479).
type ReactivationPolicy struct {
	BaseProbeInterval time.Duration
	MaxProbeInterval  time.Duration
	MaxPause          time.Duration
	Now               time.Time
}

// DefaultReactivationPolicy returns the default bounded reactivation policy.
func DefaultReactivationPolicy() ReactivationPolicy {
	return ReactivationPolicy{
		BaseProbeInterval: DefaultBaseProbeInterval,
		MaxProbeInterval:  DefaultMaxProbeInterval,
		MaxPause:          defaultMaxPauseDuration,
	}
}

// ProbeInterval returns the exponential-with-cap delay before probe attempt
// number `attempts` (0-indexed).
func ProbeInterval(attempts uint64, policy ReactivationPolicy) time.Duration {
	base := policy.BaseProbeInterval
	if base <= 0 {
		base = DefaultBaseProbeInterval
	}
	maxInt := policy.MaxProbeInterval
	if maxInt <= 0 {
		maxInt = DefaultMaxProbeInterval
	}
	if maxInt < base {
		maxInt = base
	}
	interval := base
	for i := uint64(0); i < attempts; i++ {
		if interval >= maxInt || interval > maxInt/2 {
			return maxInt
		}
		interval *= 2
	}
	if interval > maxInt {
		return maxInt
	}
	return interval
}

// ShouldProbe reports whether now falls on or after the stretched exponential
// probe schedule for pauseState.
func ShouldProbe(now time.Time, pauseState PauseState, policy ReactivationPolicy) bool {
	if !pauseState.Paused {
		return false
	}
	last := pauseState.LastProbeAt
	if last.IsZero() {
		last = pauseState.PausedAt
	}
	if last.IsZero() || now.IsZero() {
		return false
	}
	return now.Sub(last) >= ProbeInterval(pauseState.ProbeAttempts, policy)
}

// ShouldResumeAt evaluates whether a pause should end at now given a fresh
// hostSample, returning true and a closed-vocabulary reason_class when either
// the hard MaxPause ceiling or the declared reactivate_when condition is met.
func ShouldResumeAt(now time.Time, hostSample *HostLoad, pauseState PauseState, policy ReactivationPolicy) (bool, PauseReasonClass) {
	policy.Now = now
	return ShouldResume(hostSample, pauseState, policy)
}

// ShouldResume evaluates the declared un-pause condition against a fresh
// hostSample and enforces the hard MaxPause ceiling at paused_at + MaxPause.
// An UNKNOWN/ERROR/nil hostSample never resumes on its own and never extends
// the pause.
func ShouldResume(hostSample *HostLoad, pauseState PauseState, policy ReactivationPolicy) (bool, PauseReasonClass) {
	if !pauseState.Paused {
		return false, ""
	}
	maxPause := effectiveMaxPause(pauseState, policy)
	if !policy.Now.IsZero() && !pauseState.PausedAt.IsZero() && maxPause > 0 {
		if !policy.Now.Before(pauseState.PausedAt.Add(maxPause)) {
			return true, PauseReasonMaxPauseReached
		}
	}

	ratio, measured := hostPressureRatio(hostSample)
	if !measured {
		return false, ""
	}
	if pauseState.ReactivateWhen.PressureRatioBelow > 0 {
		required := pauseState.ReactivateWhen.ConsecutiveSamples
		if required <= 0 {
			required = 1
		}
		if ratio < pauseState.ReactivateWhen.PressureRatioBelow && pauseState.ConsecutiveClearSamples+1 >= required {
			return true, PauseReasonConditionCleared
		}
	}
	return false, ""
}

// ApplyProbeSample records one fresh probe sample against pauseState without I/O,
// updating ProbeAttempts, LastProbeAt, ConsecutiveClearSamples, and
// LastProbeResult, and returns the ShouldResumeAt verdict.
func ApplyProbeSample(pauseState PauseState, now time.Time, hostSample *HostLoad, policy ReactivationPolicy) (PauseState, bool, PauseReasonClass) {
	next := pauseState
	if !next.Paused {
		return next, false, ""
	}
	resume, reason := ShouldResumeAt(now, hostSample, pauseState, policy)
	next.ProbeAttempts++
	if !now.IsZero() {
		next.LastProbeAt = now.UTC()
	}
	if resume {
		next.LastProbeResult = string(reason)
		return next, true, reason
	}

	ratio, measured := hostPressureRatio(hostSample)
	if !measured {
		next.ConsecutiveClearSamples = 0
		if hostSample != nil && hostSample.State != "" {
			next.LastProbeResult = string(hostSample.State)
		} else {
			next.LastProbeResult = string(HostSampleUnknown)
		}
		return next, false, ""
	}

	if next.ReactivateWhen.PressureRatioBelow > 0 && ratio < next.ReactivateWhen.PressureRatioBelow {
		next.ConsecutiveClearSamples++
		required := next.ReactivateWhen.ConsecutiveSamples
		if required <= 0 {
			required = 1
		}
		next.LastProbeResult = fmt.Sprintf("clear:%d/%d", next.ConsecutiveClearSamples, required)
	} else {
		next.ConsecutiveClearSamples = 0
		next.LastProbeResult = fmt.Sprintf("busy:%.2f", ratio)
	}
	return next, false, ""
}

func effectiveMaxPause(pauseState PauseState, policy ReactivationPolicy) time.Duration {
	if pauseState.ReactivateWhen.MaxPause != "" {
		if parsed, err := time.ParseDuration(pauseState.ReactivateWhen.MaxPause); err == nil && parsed > 0 {
			return parsed
		}
	}
	if policy.MaxPause > 0 {
		return policy.MaxPause
	}
	return defaultMaxPauseDuration
}
