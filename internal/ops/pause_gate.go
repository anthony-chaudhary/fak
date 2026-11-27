package ops

import (
	"fmt"
	"strings"
	"time"
)

const (
	defaultPressurePauseRatio  = 0.85
	defaultPressureResumeRatio = 0.75
	defaultMaxPauseDuration    = 30 * time.Minute
	defaultMaxConsecutiveSkips = 10
)

// PhaseAction is the closed vocabulary for a pause-gate outcome (#14477).
type PhaseAction string

const (
	ActionRun                   PhaseAction = "RUN"
	ActionSkip                  PhaseAction = "SKIP"
	ActionRunPressureReliefOnly PhaseAction = "RUN_PRESSURE_RELIEF_ONLY"
)

// ValidPhaseAction reports whether a is a member of the closed action vocabulary.
func ValidPhaseAction(a PhaseAction) bool {
	switch a {
	case ActionRun, ActionSkip, ActionRunPressureReliefOnly:
		return true
	}
	return false
}

// PhaseClass declares at the call site whether a maintenance phase is
// pressure relief (never gated) or deferrable junkwork (gatable).
type PhaseClass string

const (
	PhasePressureRelief PhaseClass = "PRESSURE_RELIEF"
	PhaseDeferrable     PhaseClass = "DEFERRABLE"
)

// PauseDirective is the pure output of PauseEvaluator.Decide for one ops tick.
type PauseDirective struct {
	JunkworkAction PhaseAction      `json:"junkwork_action"`
	ReliefAction   PhaseAction      `json:"relief_action"`
	ReasonClass    PauseReasonClass `json:"reason_class,omitempty"`
	TriggerValue   string           `json:"trigger_value,omitempty"`
	ShouldPause    bool             `json:"should_pause,omitempty"`
	ShouldResume   bool             `json:"should_resume,omitempty"`
	DeadlineHit    bool             `json:"deadline_hit,omitempty"`
}

// ForPhase returns the action for a phase based on its call-site class.
// Pressure relief is never gated; only explicitly deferrable phases can skip.
func (d PauseDirective) ForPhase(class PhaseClass) PhaseAction {
	if class == PhaseDeferrable {
		return d.JunkworkAction
	}
	return d.ReliefAction
}

// PauseEvaluator is the injectable pause-gate seam consulted by Engine.Tick (#14477).
type PauseEvaluator interface {
	Decide(cycle LoadCycle, pauseState PauseState, cfg Config) PauseDirective
}

// DefaultPauseEvaluator implements the pressure-reactive pause rules:
//
//   - Opt-in and default-off (`cfg.ReactivePauseEnabled` must be true).
//   - Fail-open on UNKNOWN/ERROR host samples: an unreadable sampler is
//     indistinguishable from a failed health check, and pausing maintenance on a
//     failed health check is an invisible outage. (Contrast with
//     internal/harnessres/governor.go, which fails closed to MinWorkers for
//     worker-wave sizing.)
//   - Pressure relief is never gated; only explicitly deferrable junkwork skips.
//   - Every pause has an exit condition: trigger clearing or a hard deadline
//     handoff (#14479) so a SKIP streak cannot run forever.
type DefaultPauseEvaluator struct{}

// Decide evaluates the current tick's host sample, durable pause state, and
// config into a deterministic PauseDirective with no side effects.
func (DefaultPauseEvaluator) Decide(cycle LoadCycle, pauseState PauseState, cfg Config) PauseDirective {
	pauseRatio, resumeRatio, maxPause, maxSkips := effectivePauseThresholds(cfg)
	ratio, measured := hostPressureRatio(cycle.Host)

	if pauseState.Paused {
		if deadlineReached(cycle.Started, pauseState, maxPause, maxSkips) {
			reason := pauseState.ReasonClass
			if maxPauseWallReached(cycle.Started, pauseState, maxPause) {
				reason = PauseReasonMaxPauseReached
			}
			return PauseDirective{
				JunkworkAction: ActionRun,
				ReliefAction:   ActionRun,
				ReasonClass:    reason,
				TriggerValue:   "deadline_reached",
				ShouldResume:   true,
				DeadlineHit:    true,
			}
		}

		if pauseState.ReasonClass == PauseReasonHostPressure && cfg.ReactivePauseEnabled {
			if cycle.Host == nil || cycle.Host.State != HostSampleMeasured || !measured {
				state := "unknown"
				if cycle.Host != nil && cycle.Host.State != "" {
					state = strings.ToLower(string(cycle.Host.State))
				}
				return PauseDirective{
					JunkworkAction: ActionRun,
					ReliefAction:   ActionRun,
					ReasonClass:    PauseReasonUnknownHostState,
					TriggerValue:   "host_sample_" + state,
					ShouldResume:   true,
				}
			}
			if ratio < resumeRatio {
				return PauseDirective{
					JunkworkAction: ActionRun,
					ReliefAction:   ActionRun,
					TriggerValue:   fmt.Sprintf("ram_pressure=%.2f", ratio),
					ShouldResume:   true,
				}
			}
		}

		trigger := pauseState.ReasonDetail
		if measured {
			trigger = fmt.Sprintf("ram_pressure=%.2f", ratio)
		}
		return PauseDirective{
			JunkworkAction: ActionSkip,
			ReliefAction:   ActionRunPressureReliefOnly,
			ReasonClass:    pauseState.ReasonClass,
			TriggerValue:   trigger,
		}
	}

	if !cfg.ReactivePauseEnabled {
		return PauseDirective{
			JunkworkAction: ActionRun,
			ReliefAction:   ActionRun,
		}
	}

	if cycle.Host == nil || cycle.Host.State != HostSampleMeasured || !measured {
		return PauseDirective{
			JunkworkAction: ActionRun,
			ReliefAction:   ActionRun,
			ReasonClass:    PauseReasonUnknownHostState,
		}
	}

	if ratio >= pauseRatio {
		return PauseDirective{
			JunkworkAction: ActionSkip,
			ReliefAction:   ActionRunPressureReliefOnly,
			ReasonClass:    PauseReasonHostPressure,
			TriggerValue:   fmt.Sprintf("ram_pressure=%.2f", ratio),
			ShouldPause:    true,
		}
	}

	return PauseDirective{
		JunkworkAction: ActionRun,
		ReliefAction:   ActionRun,
	}
}

func effectivePauseThresholds(cfg Config) (pauseRatio, resumeRatio float64, maxPause time.Duration, maxSkips int) {
	pauseRatio = cfg.PressurePauseRatio
	if pauseRatio <= 0 || pauseRatio > 1 {
		pauseRatio = defaultPressurePauseRatio
	}
	resumeRatio = cfg.PressureResumeRatio
	if resumeRatio <= 0 || resumeRatio >= pauseRatio {
		resumeRatio = pauseRatio - 0.10
		if resumeRatio <= 0 {
			resumeRatio = defaultPressureResumeRatio
		}
	}
	maxPause = cfg.MaxPauseDuration
	if maxPause <= 0 {
		maxPause = defaultMaxPauseDuration
	}
	maxSkips = cfg.MaxConsecutiveSkips
	if maxSkips <= 0 {
		maxSkips = defaultMaxConsecutiveSkips
	}
	return pauseRatio, resumeRatio, maxPause, maxSkips
}

func hostPressureRatio(h *HostLoad) (float64, bool) {
	if h == nil || h.State != HostSampleMeasured || !h.HaveRAM || h.TotalRAMBytes == 0 {
		return 0, false
	}
	if h.AvailRAMBytes >= h.TotalRAMBytes {
		return 0, true
	}
	used := h.TotalRAMBytes - h.AvailRAMBytes
	return float64(used) / float64(h.TotalRAMBytes), true
}

func maxPauseWallReached(now time.Time, pauseState PauseState, cfgMaxPause time.Duration) bool {
	maxPause := effectiveMaxPause(pauseState, ReactivationPolicy{MaxPause: cfgMaxPause})
	return maxPause > 0 && !pauseState.PausedAt.IsZero() && !now.IsZero() && !now.Before(pauseState.PausedAt.Add(maxPause))
}

func deadlineReached(now time.Time, pauseState PauseState, cfgMaxPause time.Duration, maxSkips int) bool {
	if maxSkips > 0 && pauseState.PausedTicks >= uint64(maxSkips) {
		return true
	}
	return maxPauseWallReached(now, pauseState, cfgMaxPause)
}
