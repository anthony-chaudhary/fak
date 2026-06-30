package loopmgr

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"strings"
	"time"
)

// Schedule decides WHEN a recurring job fires, with correctness as a primitive.
// It is the schedule half the field re-debugs per vendor, written once over the
// shipped loopmgr ledger: overlap-lock (no duplicate fire on wake-from-sleep),
// a named missed-run policy (catch-up vs skip, never defaulted), and
// deterministic per-job jitter (the thundering-herd guard).
//
// Schedule owns schedule SEMANTICS only. It reads no clock and performs no I/O:
// the OS scheduler still owns wall-clock liveness, and the caller folds the
// ledger (Summarize) and supplies the resulting snapshot. The fire decision is
// returned as a structured FireDecision the caller journals like every other
// event, so the "did it fire?" question is answered from the ledger, not from
// the schedule's word.

// MissedRunPolicy names what a schedule does when one or more fire windows
// elapsed unobserved (a sleep, a gateway restart, a long-running prior run).
// It is a CLOSED set with NO zero-value default: an empty policy is a
// configuration error, never a silent fall-through. This is the part the issue
// calls out — "explicit, never defaulted".
type MissedRunPolicy string

const (
	// MissedCatchUp fires once for the missed window: the schedule still owes a
	// run, so it runs now (a single catch-up, not one-per-missed-window — the
	// overlap-lock collapses a backlog to a single live run).
	MissedCatchUp MissedRunPolicy = "catch-up"
	// MissedSkip drops the missed window: the schedule re-aligns to the next
	// boundary and does not run for the windows that elapsed unobserved.
	MissedSkip MissedRunPolicy = "skip"
)

// ValidMissedRunPolicy reports whether p is a named member of the closed set.
// The empty string is INVALID by design: the policy must be chosen explicitly.
func ValidMissedRunPolicy(p MissedRunPolicy) bool {
	switch p {
	case MissedCatchUp, MissedSkip:
		return true
	default:
		return false
	}
}

// Schedule is the recurring-fire definition for one loop. It carries the
// cadence (IntervalSeconds), the explicit missed-run policy, and the jitter
// window — all pure data, validated by Validate. The cron-expression projection
// (P4.3, the OS scheduler) consumes Schedule; loopmgr itself decides the fire.
type Schedule struct {
	// JobID is the schedule identity. Jitter is derived from it, so two jobs
	// with distinct ids get distinct (but each deterministic) offsets.
	JobID string `json:"job_id"`

	// IntervalSeconds is the nominal cadence: a fire is due every this-many
	// seconds from the schedule's anchor. Must be > 0.
	IntervalSeconds int64 `json:"interval_seconds"`

	// MissedRun is the explicit, named policy for elapsed-unobserved windows.
	// There is no default: Validate rejects an empty policy.
	MissedRun MissedRunPolicy `json:"missed_run"`

	// JitterSeconds is the width of the deterministic per-job jitter window in
	// seconds. 0 disables jitter (the fire lands exactly on the boundary). The
	// offset is in [0, JitterSeconds), stable for a given JobID, so a fleet of
	// jobs that share a boundary spread out instead of storming together.
	JitterSeconds int64 `json:"jitter_seconds,omitempty"`
}

// FireReason is the closed vocabulary FireDecision carries. Like the governor's
// refusal reasons, each is emittable and verifiable — a downstream routes on the
// reason, never on free text.
const (
	// ReasonScheduleFire: a fire window is due and no prior run is in flight.
	ReasonScheduleFire = "SCHEDULE_FIRE"
	// ReasonScheduleNotDue: now is before the next due boundary.
	ReasonScheduleNotDue = "SCHEDULE_NOT_DUE"
	// ReasonOverlapLock: the prior run's start has no matching end — refuse so a
	// wake-from-sleep cannot double-fire the same job.
	ReasonOverlapLock = "OVERLAP_LOCK"
	// ReasonMissedSkipped: a window was missed and the policy is skip — re-align,
	// do not run.
	ReasonMissedSkipped = "MISSED_SKIPPED"
	// ReasonMissedCaughtUp: a window was missed and the policy is catch-up — run
	// now to settle the owed fire.
	ReasonMissedCaughtUp = "MISSED_CAUGHT_UP"
)

// FireDecision is the structured verdict Next returns. Fire is true to proceed.
// FireAtUnixNano is the (jittered) boundary the decision is anchored to, so the
// caller journals a deterministic fire time, not "whenever Next happened to run".
type FireDecision struct {
	JobID          string `json:"job_id"`
	Fire           bool   `json:"fire"`
	Reason         string `json:"reason"`
	Summary        string `json:"summary"`
	FireAtUnixNano int64  `json:"fire_at_unix_nano,omitempty"`
}

// Validate checks a Schedule is well-formed: a job id, a positive interval, and
// a NAMED missed-run policy. The policy check is the load-bearing one — a zero
// MissedRun is rejected so the policy can never be silently defaulted.
func (s Schedule) Validate() error {
	if strings.TrimSpace(s.JobID) == "" {
		return fmt.Errorf("schedule job_id is required")
	}
	if s.IntervalSeconds <= 0 {
		return fmt.Errorf("schedule %q interval_seconds = %d, want > 0", s.JobID, s.IntervalSeconds)
	}
	if !ValidMissedRunPolicy(s.MissedRun) {
		return fmt.Errorf("schedule %q missed_run = %q, want an explicit %q or %q (never defaulted)",
			s.JobID, s.MissedRun, MissedCatchUp, MissedSkip)
	}
	if s.JitterSeconds < 0 {
		return fmt.Errorf("schedule %q jitter_seconds = %d, want >= 0", s.JobID, s.JitterSeconds)
	}
	return nil
}

// JitterOffsetNanos returns the deterministic per-job jitter offset in
// nanoseconds, in [0, JitterSeconds). It is a pure function of JobID, so the
// SAME job always lands at the SAME offset within its window and two distinct
// jobs sharing a boundary spread apart. 0 jitter yields 0. The offset is derived
// from a SHA-256 of the job id (uniformly distributed, stable across processes
// and platforms — not Go's map-seeded hash).
func (s Schedule) JitterOffsetNanos() int64 {
	if s.JitterSeconds <= 0 {
		return 0
	}
	sum := sha256.Sum256([]byte(s.JobID))
	v := binary.BigEndian.Uint64(sum[:8])
	windowNanos := uint64(s.JitterSeconds) * uint64(time.Second)
	return int64(v % windowNanos)
}

// runInFlight reports whether the loop has an unmatched start — a run that
// started and has no end yet. This is the overlap-lock predicate, read straight
// off the fold: a loop snapshot with a non-zero in-flight count has a live run.
// It shares LoopSnapshot.Concurrent's Started-Ended definition so the binary
// overlap-lock and the governor's tunable concurrency budget never drift apart.
// (Summarize counts start/end events; a witnessed/claimed end both increment
// Ended via the EventEnd fold.)
func runInFlight(loop LoopSnapshot) bool {
	return loop.Concurrent() > 0
}

// anchorUnixNano is the schedule's alignment anchor: the last observed event for
// this loop, falling back to startUnixNano when the loop has never fired. The
// schedule's boundaries are multiples of IntervalSeconds from this anchor.
func (s Schedule) anchorUnixNano(loop LoopSnapshot, startUnixNano int64) int64 {
	if loop.LastEventUnixNano > 0 {
		return loop.LastEventUnixNano
	}
	return startUnixNano
}

// Next decides whether the job should fire at time now, given the folded loop
// snapshot (the ledger's word on the prior run) and the schedule's anchor
// startUnixNano (used only when the loop has never fired). It is PURE: no clock
// read, no I/O. The decision order is fixed so the reason is deterministic:
//
//  1. overlap-lock first — a live run always wins, even past a due boundary, so
//     a wake-from-sleep cannot double-fire.
//  2. not-due — now is before the next boundary; nothing owed.
//  3. due, with at most the current window owed — a normal fire.
//  4. due, with one or more EARLIER windows missed — honor MissedRun: skip
//     re-aligns (no run), catch-up fires once now.
//
// The returned FireAtUnixNano is the jittered boundary, so the caller journals a
// stable fire time independent of when Next was invoked within the window.
func (s Schedule) Next(loop LoopSnapshot, now time.Time, startUnixNano int64) FireDecision {
	d := FireDecision{JobID: s.JobID}

	// (1) Overlap-lock: a prior run is still in flight. Refuse unconditionally.
	if runInFlight(loop) {
		d.Reason = ReasonOverlapLock
		d.Summary = fmt.Sprintf("prior run in flight (started=%d ended=%d) — refusing duplicate fire", loop.Started, loop.Ended)
		return d
	}

	nowNanos := now.UTC().UnixNano()
	anchor := s.anchorUnixNano(loop, startUnixNano)
	intervalNanos := s.IntervalSeconds * int64(time.Second)
	jitter := s.JitterOffsetNanos()

	// Boundaries recur every intervalNanos starting one interval AFTER the
	// anchor (the anchor is when the job last ran, not a due boundary). We count
	// how many whole intervals have completed since the anchor, offset by jitter:
	// 0 => not yet due, 1 => the current window is owed, >1 => earlier windows
	// were missed.
	elapsed := nowNanos - (anchor + jitter)
	windows := int64(0)
	if elapsed >= 0 {
		windows = elapsed / intervalNanos
	}

	if windows < 1 {
		d.Reason = ReasonScheduleNotDue
		d.FireAtUnixNano = anchor + intervalNanos + jitter
		d.Summary = "not due: next jittered boundary not reached"
		return d
	}

	// The first owed boundary (the on-time fire). windows==1 is the normal
	// on-time fire; windows>1 means (windows-1) earlier windows elapsed
	// unobserved.
	firedBoundary := anchor + intervalNanos + jitter
	// The next boundary at/after now — where skip re-aligns to.
	dueBoundary := anchor + (windows+1)*intervalNanos + jitter

	if windows == 1 {
		d.Fire = true
		d.Reason = ReasonScheduleFire
		d.FireAtUnixNano = firedBoundary
		d.Summary = "due: firing on the current window"
		return d
	}

	// windows > 1: at least one earlier window was missed. Honor the policy.
	missed := windows - 1
	switch s.MissedRun {
	case MissedSkip:
		d.Reason = ReasonMissedSkipped
		// Re-align to the NEXT boundary after now; do not run for the gap.
		d.FireAtUnixNano = dueBoundary
		d.Summary = fmt.Sprintf("skip policy: %d missed window(s) dropped, re-aligned to next boundary", missed)
		return d
	case MissedCatchUp:
		d.Fire = true
		d.Reason = ReasonMissedCaughtUp
		// A single catch-up run, anchored to the FIRST missed boundary, settles
		// the owed fire; the overlap-lock collapses the backlog to this one run.
		d.FireAtUnixNano = firedBoundary
		d.Summary = fmt.Sprintf("catch-up policy: %d missed window(s) settled by one run now", missed)
		return d
	default:
		// Unreachable for a Validate'd schedule; fail closed rather than fire.
		d.Reason = ReasonScheduleNotDue
		d.Summary = fmt.Sprintf("invalid missed_run policy %q — refusing to fire", s.MissedRun)
		return d
	}
}
