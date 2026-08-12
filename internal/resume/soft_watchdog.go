package resume

// soft_watchdog.go — issue #5287, a leaf of the supervision epic (#4748):
// the soft watchdog's diagnostic state-dump core.
//
// The trajectory watchdog (trajectory_watchdog.go) decides NUDGE/REVIVE for an
// alive-but-stalled session but records only an action plus a reason string —
// the evidence of WHY the session wedged is discarded before the intervention
// runs. The soft watchdog is the observe-only twin (sglang's soft/hard
// watchdog split, clean-room): when a session is alive with a stalled
// witnessed curve past a soft grace window, it captures a structured
// diagnostic dump — the last progress marker, elapsed-since-progress, the
// pending action, and the liveness-vs-progress split — BEFORE the nudge/revive
// decision fires, and exactly once per stall episode. Dead sessions are left
// to the hard revive path untouched, and the intervention core itself is
// byte-identical to running without the soft watchdog: soft observes, hard
// decides.

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// DefaultSoftStallGrace is the soft timeout: how long a living session's
// witnessed curve must have been stalled (no forward-progress marker) before
// the diagnostic dump fires. It is deliberately shorter than any revive
// escalation so the dump always precedes the intervention it explains.
const DefaultSoftStallGrace = 2 * time.Minute

// SoftSplitAliveWithoutProgress is the pinned liveness-vs-progress verdict a
// soft dump carries: the heartbeat says alive, the witnessed curve says no
// forward progress — exactly the regime the soft watchdog exists to explain.
const SoftSplitAliveWithoutProgress = "ALIVE_WITHOUT_PROGRESS"

// SoftWatchdogObservation is one clock-injected look at a session. The core
// never reads the wall clock or touches IO — Now is supplied by the caller so
// every capture decision is deterministic and replayable.
type SoftWatchdogObservation struct {
	SessionID          string
	Alive              bool           // heartbeat/process liveness (the liveness half of the split)
	Signal             trajctl.Signal // witnessed curve signal (the progress half of the split)
	LastProgressMarker string         // last witnessed forward-progress marker (curve detail, commit, turn)
	LastProgressAt     time.Time      // when that marker was witnessed
	PendingAction      string         // what the session claims to be doing right now
	Now                time.Time      // injected clock
}

// SoftStateDump is the structured diagnostic captured from an alive-but-stalled
// session before the nudge/revive decision runs, durable-record ready.
type SoftStateDump struct {
	SessionID                  string         `json:"session_id"`
	CapturedAtUnixMillis       int64          `json:"captured_at_unix_millis"`
	Signal                     trajctl.Signal `json:"signal"`
	LastProgressMarker         string         `json:"last_progress_marker"`
	ElapsedSinceProgressMillis int64          `json:"elapsed_since_progress_millis"`
	PendingAction              string         `json:"pending_action"`
	Alive                      bool           `json:"alive"`
	ProgressStalled            bool           `json:"progress_stalled"`
	LivenessVsProgress         string         `json:"liveness_vs_progress"`
	Reason                     string         `json:"reason"`
}

// DecideSoftStateDump is the clock-free capture core. It returns a dump only
// when the observation shows a session that is alive, whose witnessed curve is
// stalled, and whose stall has outlived the grace window (grace <= 0 selects
// DefaultSoftStallGrace). Healthy, slow, or drifting sessions capture nothing;
// a dead session captures nothing because the hard revive path owns it and a
// dump would race the relaunch instead of explaining a live wedge. A stall
// carrying no witnessed progress timestamp (zero LastProgressAt) captures
// nothing either: the soft timeout is the gate, and an unknown stall clock
// cannot be proven to have elapsed — treating it as infinitely old would dump
// on every never-scored session and make the diagnostic worthless.
func DecideSoftStateDump(in SoftWatchdogObservation, grace time.Duration) (SoftStateDump, bool) {
	if grace <= 0 {
		grace = DefaultSoftStallGrace
	}
	if !in.Alive {
		return SoftStateDump{}, false
	}
	if in.Signal != trajctl.SignalStall {
		return SoftStateDump{}, false
	}
	if in.LastProgressAt.IsZero() {
		return SoftStateDump{}, false
	}
	elapsed := in.Now.Sub(in.LastProgressAt)
	if elapsed < grace {
		return SoftStateDump{}, false
	}
	return SoftStateDump{
		SessionID:                  in.SessionID,
		CapturedAtUnixMillis:       in.Now.UnixMilli(),
		Signal:                     in.Signal,
		LastProgressMarker:         in.LastProgressMarker,
		ElapsedSinceProgressMillis: elapsed.Milliseconds(),
		PendingAction:              in.PendingAction,
		Alive:                      true,
		ProgressStalled:            true,
		LivenessVsProgress:         SoftSplitAliveWithoutProgress,
		Reason:                     "session is alive but its witnessed curve stalled past the soft grace window; diagnostic state captured before nudge/revive",
	}, true
}

// SoftWatchdog folds a stream of observations and guarantees exactly one dump
// per stall episode per session: the first qualifying observation captures,
// later looks at the same unbroken stall stay silent, and any non-stall
// observation (progress resumed, or the session died into the hard path)
// closes the episode and re-arms the session for a future one.
type SoftWatchdog struct {
	grace    time.Duration
	captured map[string]bool
}

// NewSoftWatchdog builds an episode tracker; grace <= 0 selects
// DefaultSoftStallGrace.
func NewSoftWatchdog(grace time.Duration) *SoftWatchdog {
	if grace <= 0 {
		grace = DefaultSoftStallGrace
	}
	return &SoftWatchdog{grace: grace, captured: make(map[string]bool)}
}

// Observe folds one observation, returning (dump, true) at most once per
// stall episode.
func (w *SoftWatchdog) Observe(in SoftWatchdogObservation) (SoftStateDump, bool) {
	if !in.Alive || in.Signal != trajctl.SignalStall {
		delete(w.captured, in.SessionID) // episode over: re-arm
		return SoftStateDump{}, false
	}
	if w.captured[in.SessionID] {
		return SoftStateDump{}, false
	}
	dump, ok := DecideSoftStateDump(in, w.grace)
	if ok {
		w.captured[in.SessionID] = true
	}
	return dump, ok
}

// ObserveThenDecide runs the soft diagnostic capture and THEN the unchanged
// intervention core, in that order — the returned dump (nil when none fired)
// is the forensic record the ensuing nudge/revive carries. The decision is
// exactly DecideTrajectoryWatchdog's: the soft watchdog never alters the
// intervention, it only makes the stalled verdict carry evidence.
func (w *SoftWatchdog) ObserveThenDecide(in SoftWatchdogObservation, nudgeAttempted bool) (*SoftStateDump, TrajectoryWatchdogDecision) {
	dump, ok := w.Observe(in)
	decision := DecideTrajectoryWatchdog(TrajectoryWatchdogInput{Alive: in.Alive, Signal: in.Signal, NudgeAttempted: nudgeAttempted})
	if !ok {
		return nil, decision
	}
	return &dump, decision
}
