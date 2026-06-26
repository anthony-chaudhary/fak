// Package lifebridge is the explicit converter between the two altitudes of the
// ONE agent lifecycle machine (epic #912): the served-session drive state
// (internal/session.RunState) and the loop supervisor state
// (internal/loopmgr.LoopState).
//
// It owns NEITHER type — it sits above both and asserts they are one vocabulary by
// pivoting every conversion through the shared internal/lifecycle.Phase. Keeping
// the converter here (rather than making one package import the other) is what lets
// session and loopmgr stay decoupled from each other while still being provably the
// same skeleton: each depends only on the shared leaf, and this package is the
// single seam where the two meet.
//
// The conversion is TOTAL and HONEST about the layer-specific extras. A state with
// no peer at the other altitude — session's Throttled (a pace modifier the
// supervisor does not model) and the supervisor's Armed/Disabled (schedule states a
// live sequence has no peer for) — converts to (zero, false), never to a silent
// default. Fail at the boundary, do not guess.
package lifebridge

import (
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// RunToLoop maps a served-session RunState to the supervisor's LoopState through
// the shared Phase. ok is false for Throttled (no supervisor peer) and any
// out-of-range value.
func RunToLoop(rs session.RunState) (loopmgr.LoopState, bool) {
	p, ok := rs.Phase()
	if !ok {
		return "", false
	}
	return loopmgr.LoopStateFromPhase(p)
}

// LoopToRun maps a supervisor LoopState to the served-session RunState through the
// shared Phase. ok is false for Armed/Disabled (no served-session peer) and any
// unknown string.
func LoopToRun(ls loopmgr.LoopState) (session.RunState, bool) {
	p, ok := ls.Phase()
	if !ok {
		return 0, false
	}
	return session.RunStateFromPhase(p)
}
