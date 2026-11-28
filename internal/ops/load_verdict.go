package ops

// DefaultIdleStreakForStretch is the minimum consecutive measured IDLE ticks
// required before cadence stretching is licensed (#14478).
const DefaultIdleStreakForStretch = 3

// StretchPolicy configures how consecutive IDLE verdicts license cadence stretching.
type StretchPolicy struct {
	RequiredIdleStreak int `json:"required_idle_streak"`
}

// StretchDecision is the pure output of evaluating a tick verdict against the
// current idle-streak state.
type StretchDecision struct {
	Verdict        LoadVerdict `json:"verdict"`
	IdleStreak     int         `json:"idle_streak"`
	StretchAllowed bool        `json:"stretch_allowed"`
}

// ClassifyLoadCycle classifies a completed fak-ops-load/1 tick and pause state
// into the closed LoadVerdict vocabulary with no I/O (#14478).
func ClassifyLoadCycle(cycle LoadCycle, pauseState PauseState) LoadVerdict {
	if cycle.Verdict == LoadError {
		return LoadError
	}
	if cycle.Mutations > 0 || anyPhaseMutated(cycle.Phases) {
		return LoadProductive
	}

	deferred := cycle.PauseAction == string(ActionSkip) ||
		cycle.ReasonClass != "" ||
		(pauseState.Paused && pauseState.ReasonClass != "") ||
		cycle.Verdict == LoadSkipped ||
		cycle.Verdict == LoadPressureRelief ||
		cycle.Verdict == LoadPressureReliefOnly

	if deferred {
		if anyReliefRan(cycle.Phases) {
			return LoadPressureRelief
		}
		if !anyPhaseRan(cycle.Phases) {
			return LoadSkipped
		}
	}

	if cycle.Host == nil || cycle.Host.State != HostSampleMeasured {
		return LoadUnknown
	}
	return LoadIdle
}

// EvaluateStretch updates the consecutive IDLE streak for a classified tick
// verdict and reports whether cadence stretching is licensed (#14478).
//
// Stretching is licensed only by a sustained IDLE run under a measured host
// state. Any other verdict (or an unmeasured host sample) resets the streak to
// zero and licenses no stretch.
func EvaluateStretch(prevStreak int, verdict LoadVerdict, host *HostLoad, policy StretchPolicy) StretchDecision {
	required := policy.RequiredIdleStreak
	if required <= 0 {
		required = DefaultIdleStreakForStretch
	}
	if prevStreak < 0 {
		prevStreak = 0
	}

	if verdict == LoadIdle && host != nil && host.State == HostSampleMeasured {
		next := prevStreak + 1
		return StretchDecision{
			Verdict:        verdict,
			IdleStreak:     next,
			StretchAllowed: next >= required,
		}
	}

	return StretchDecision{
		Verdict:        verdict,
		IdleStreak:     0,
		StretchAllowed: false,
	}
}

// ClassifyAndEvaluateStretch classifies a completed LoadCycle and updates the
// idle streak under policy with no I/O (#14478).
func ClassifyAndEvaluateStretch(prevStreak int, cycle LoadCycle, pauseState PauseState, policy StretchPolicy) StretchDecision {
	verdict := ClassifyLoadCycle(cycle, pauseState)
	return EvaluateStretch(prevStreak, verdict, cycle.Host, policy)
}

func anyPhaseMutated(phases []PhaseLoad) bool {
	for _, p := range phases {
		if p.Mutated {
			return true
		}
	}
	return false
}

func anyPhaseRan(phases []PhaseLoad) bool {
	for _, p := range phases {
		if p.Ran {
			return true
		}
	}
	return false
}
