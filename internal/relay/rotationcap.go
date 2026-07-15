// Rung G5 (issue #1892): per-hour rotation cap — over a caller-supplied
// rotations-per-hour ceiling a proposed leg is held (RELAY_ROTATION_CAPPED) instead of
// rotating, bounding a runaway trigger before it burns an unbounded number of windows.
//
// The spine (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md) makes the ceiling
// policy DATA, not a magic number in code: the caller supplies the cap alongside the
// wall-clock seconds. Pure fold like its G4/G6 siblings (hysteresis.go, noprogress.go):
// no clock, no I/O, no config read — only the accepted-rotation timestamps still inside
// the trailing hour, folded one proposed rotation at a time.
package relay

// ReasonRotationCapped is the closed relay reason token emitted when a proposed
// rotation would exceed the caller-supplied rotations-per-hour ceiling: hold the leg
// instead of rotating, and let the window slide before proposing again. It joins the
// Reason* discipline (ReasonNoProgress, ReasonGoalDone, …) so a supervisor reads a
// checkable cause, never free text.
const ReasonRotationCapped = "RELAY_ROTATION_CAPPED"

// RotationCap holds proposed rotations once the trailing wall-clock hour already
// carries MaxPerHour accepted rotations. The zero value is an unconfigured,
// non-tripping cap (MaxPerHour 0 never holds — an unset policy never refuses); set
// MaxPerHour to the ceiling the operator states.
type RotationCap struct {
	// MaxPerHour is the rotations-per-hour ceiling. <= 0 disables the cap — an unset
	// policy never holds.
	MaxPerHour int

	// accepted is the unix-seconds timestamps of accepted rotations still inside the
	// trailing hour; entries strictly older than one hour are pruned at each fold.
	accepted []int64
}

// Admit folds one proposed rotation at nowUnixSec (caller-supplied wall-clock, unix
// seconds, non-decreasing across calls) and reports whether the rotation may proceed.
// It first prunes accepted entries at or beyond one hour old (an entry exactly one
// hour old is outside the trailing window). With MaxPerHour <= 0 the cap is disabled:
// the rotation passes through without being recorded. If the trailing hour already
// holds MaxPerHour accepted rotations the proposal is held — allow is false, reason is
// RELAY_ROTATION_CAPPED, and the held rotation is NOT recorded (a rejected rotation
// consumes no slot). Otherwise the rotation is accepted and recorded.
func (c *RotationCap) Admit(nowUnixSec int64) (allow bool, reason string) {
	kept := c.accepted[:0]
	for _, ts := range c.accepted {
		if ts > nowUnixSec-3600 {
			kept = append(kept, ts)
		}
	}
	c.accepted = kept

	if c.MaxPerHour <= 0 {
		return true, ""
	}
	if len(c.accepted) >= c.MaxPerHour {
		return false, ReasonRotationCapped
	}
	c.accepted = append(c.accepted, nowUnixSec)
	return true, ""
}

// CountInWindow returns the number of accepted rotations still inside the trailing
// hour as of the last fold — the operator/debug read of how close the relay is to the
// ceiling.
func (c *RotationCap) CountInWindow() int { return len(c.accepted) }
