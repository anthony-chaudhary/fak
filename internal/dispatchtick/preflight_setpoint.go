package dispatchtick

import (
	"strconv"
	"strings"
)

// SetpointConcurrencyEnv is the operator-written live concurrency setpoint (#4036): a
// small externally-writable value dispatch polls each tick. A caller reads it (env,
// then a setpoint file, per its own resolution order) and folds it through
// ReconcileSetpoint. Empty/unset means "no operator setpoint" -- a total no-op.
const SetpointConcurrencyEnv = "FAK_DISPATCH_SETPOINT"

// SetpointPlan is the pure result of folding an operator concurrency setpoint against
// the live fleet, mirroring the serving autoscaler's level-triggered reconcile onto the
// dispatch cap (#4036): grow immediately toward a higher setpoint, contract only as
// in-use workers drain toward a lower one -- never a mid-issue kill.
type SetpointPlan struct {
	// Active is false when no operator setpoint is set (setpoint <= 0). An inactive plan
	// is a total no-op: DesiredCap/ContractionTarget/Draining are all zero, so a caller
	// that applies it is byte-identical to before the setpoint existed.
	Active bool `json:"active"`
	// DesiredCap is the cap to converge toward now: == setpoint whenever active. On a
	// grow the caller raises the configured cap toward it (still min()'d by the hard
	// host/seat/config ceilings, so a setpoint can never overbook the box). On a drain it
	// is below live and is realized via ContractionTarget, not a raise.
	DesiredCap int `json:"desired_cap"`
	// ContractionTarget is the post-drain worker count when the setpoint is BELOW live,
	// else 0. It feeds PreflightInput.ContractionTarget (#4038), which caps admits at the
	// target so no new worker lands on capacity that is about to be reclaimed -- the
	// shrink-on-drain half. Zero means no contraction is pending.
	ContractionTarget int `json:"contraction_target"`
	// Draining is the surplus worker count to mark draining (live - setpoint) when
	// shrinking, else 0. These workers are removed only as they finish -- never killed.
	Draining int `json:"draining"`
	// Mode names the level-triggered branch: "inactive" | "grow" | "steady" | "drain".
	Mode string `json:"mode"`
}

// ReconcileSetpoint folds an operator-written concurrency setpoint against the live
// worker count into a level-triggered plan (#4036). It is a pure function -- the whole
// grow-now / shrink-on-drain decision with no I/O -- so the caller owns polling the
// setpoint source and applying the plan.
//
//	setpoint <= 0     inactive  -- no operator setpoint; a total no-op (byte-identical)
//	setpoint >  live  grow      -- raise the cap toward the setpoint immediately
//	setpoint == live  steady    -- already at the setpoint; hold
//	setpoint <  live  drain     -- mark the surplus draining toward the setpoint; no kill
//
// Growth is realized by the caller raising the configured cap toward DesiredCap (still
// bounded by the hard physical/seat/config ceilings downstream); contraction is realized
// by feeding ContractionTarget into the #4038 pending-contraction cap term.
func ReconcileSetpoint(live, setpoint int) SetpointPlan {
	if setpoint <= 0 {
		return SetpointPlan{Mode: "inactive"}
	}
	switch {
	case setpoint > live:
		return SetpointPlan{Active: true, DesiredCap: setpoint, Mode: "grow"}
	case setpoint == live:
		return SetpointPlan{Active: true, DesiredCap: setpoint, Mode: "steady"}
	default: // setpoint < live: shrink-on-drain
		return SetpointPlan{
			Active:            true,
			DesiredCap:        setpoint,
			ContractionTarget: setpoint,
			Draining:          live - setpoint,
			Mode:              "drain",
		}
	}
}

// ParseConcurrencySetpoint parses an operator setpoint string into a setpoint value.
// A blank, non-integer, or non-positive string yields 0 -- the inactive sentinel that
// ReconcileSetpoint treats as a no-op -- so a malformed or cleared setpoint file can
// never accidentally drain the fleet to zero. A caller resolving env then file passes
// each candidate through this; the first positive result wins.
func ParseConcurrencySetpoint(s string) int {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || n < 0 {
		return 0
	}
	return n
}
