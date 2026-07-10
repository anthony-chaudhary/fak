// Issue #4143: the watch-goal idle terminator — a benign park for a relay whose goal is to
// WATCH an invariant rather than drive it to done.
//
// The gap it closes. The G-track escapes assume every leg is trying to MAKE progress: G4
// hysteresis (hysteresis.go) withholds a re-arm until verified progress moves, and G6 the
// no-progress escape (noprogress.go) halts with RELAY_NO_PROGRESS after K legs that advance the
// verified cursor by nothing. That is exactly right for a build-to-done relay, but a WATCH goal
// ("keep the tree green", "hold this invariant") is CORRECTLY idle for long stretches: no
// verified progress, yet nothing is wrong. Fed straight into G6 such a relay looks identical to
// a stuck spin and would trip RELAY_NO_PROGRESS — a false alarm — or, if K were raised to hide
// it, would blunt the escape for genuinely stuck relays too.
//
// The terminator distinguishes the two using DURABLE STATE ONLY, never a self-report. A leg is
// correctly idle when BOTH:
//   - the watched invariant HOLDS when re-checked against ground truth (a dos_verify / CI
//     verdict the caller supplies through the WatchWitness port), and
//   - the durable ledger shows ZERO admitted pending work.
//
// When both hold the leg parks with RELAY_IDLE_PARKED — a benign terminal-for-now, not an
// OPERATOR_GATE alarm and not a non-refusing advisory: the relay stops spinning windows but
// re-arms on the next witnessed external event (a git ref change, a CI verdict flip) through
// the ExternalEvent re-arm trigger below.
//
// Fail-closed by construction, so it can only make the no-progress escape MORE conservative,
// never suppress a genuinely stuck relay. An idle read that cannot be verified — an unknown
// invariant, or a pending count the ledger could not return — is NOT idle: it falls through to
// the ordinary empty-leg path and still feeds RELAY_NO_PROGRESS. So the ONLY way to skip the
// no-progress counter is to positively prove, against durable witnesses, that the invariant
// holds with nothing pending. A stuck relay (invariant violated, or work pending) is never
// mistaken for idle.
//
// Pure fold, like its G-track siblings: no clock, no I/O. The witness verdict and the pending
// count are read by the caller from durable state and handed in as plain values, exactly as
// noprogress.go takes a VerifiedProgress and armtriggers.go takes plain AxisUsage numbers.
package relay

// ReasonIdleParked is the closed relay reason token (TRUE_DRAIN category,
// docs/notes/RELAY-REASON-VOCABULARY-2026-07-01.md) emitted when a watch-goal relay verifies —
// against durable witnesses — that its invariant holds with zero admitted pending work, and so
// parks idle instead of spinning fresh windows. Unlike RELAY_NO_PROGRESS it is NOT an alarm and
// does not escalate to a human: it is a clean, re-armable pause. The relay re-arms on the next
// witnessed external event (ExternalEvent.Rearmed). It joins the Reason* discipline so a
// supervisor reads a checkable, closed-vocabulary cause rather than free text.
const ReasonIdleParked = "RELAY_IDLE_PARKED"

// WatchVerdict is the closed durable read of whether a watch relay's invariant holds when
// re-checked against ground truth. It is never a self-report — the caller obtains it from a
// dos_verify / CI witness — and it is tri-state so an unreadable witness (WatchUnknown) is kept
// distinct from a witnessed violation (WatchViolated), the same fail-closed distinction the
// resolver's verified/dangling/unknown draws.
type WatchVerdict string

const (
	// WatchHolds means ground truth confirms the watched invariant is currently satisfied —
	// the necessary condition (with zero pending work) for a benign idle park.
	WatchHolds WatchVerdict = "holds"
	// WatchViolated means ground truth shows the invariant broken: there IS work to do, so the
	// relay is active-but-unadvanced, never idle.
	WatchViolated WatchVerdict = "violated"
	// WatchUnknown means the witness could not be read (an unreachable checker, a timeout). It
	// is NOT idle: fail closed to the ordinary empty-leg path so the no-progress escape still
	// sees the leg.
	WatchUnknown WatchVerdict = "unknown"
)

// WatchWitness re-checks a watch relay's invariant against ground truth and returns the closed
// WatchVerdict. It is injected like Resolver so the idle fold is unit-testable without a live
// checker; a production witness shells dos_verify / reads a CI verdict. A conforming witness
// MUST fail closed — an error or an unreachable checker is WatchUnknown, never a false
// WatchHolds — since a false "holds" is the one reading that could park a genuinely stuck relay.
type WatchWitness interface {
	CheckInvariant() WatchVerdict
}

// IdleObservation is the durable-state-only evidence one completed watch-relay leg presents to
// the idle predicate: the invariant verdict re-checked against ground truth, and the count of
// admitted-but-unfinished work items read from the durable ledger. Neither comes from the leg's
// own narration — both are witnessed, exactly like the ledger-verified progress the no-progress
// escape consumes.
type IdleObservation struct {
	// Invariant is the WatchWitness verdict for this leg's watched invariant.
	Invariant WatchVerdict
	// PendingAdmitted is the number of admitted-but-unfinished work items the durable ledger
	// records for this relay. Zero means nothing is queued to do.
	PendingAdmitted int
	// PendingKnown is false when the ledger could not be read. An unread pending count is never
	// treated as zero: fail closed, the leg is not idle.
	PendingKnown bool
}

// idle reports whether this leg is CORRECTLY idle: the invariant holds against ground truth AND
// the ledger positively confirms zero admitted pending work. Any other combination — an unknown
// or violated invariant, an unread pending count, or any positive pending work — is not idle, so
// the caller feeds the leg to the ordinary no-progress path (fail closed).
func (o IdleObservation) idle() bool {
	return o.Invariant == WatchHolds && o.PendingKnown && o.PendingAdmitted == 0
}

// IdleAwareEscape wraps the G6 no-progress escape with the watch-goal idle terminator. It holds
// a NoProgressEscape and adds one rule ahead of it: a leg that is positively proven idle parks
// with RELAY_IDLE_PARKED and is NOT counted as an empty leg, so a correctly-idle watch relay
// never drifts toward RELAY_NO_PROGRESS. Every other leg is folded through the embedded escape
// UNCHANGED, so a stuck relay trips exactly as before. The zero value embeds an unconfigured
// (non-tripping) escape; set Escape.MaxEmptyLegs to arm the no-progress backstop.
type IdleAwareEscape struct {
	// Escape is the underlying G6 no-progress counter. Idle-proven legs bypass it; every other
	// leg is folded through it with identical semantics.
	Escape NoProgressEscape

	// parked records whether the most recent ObserveLeg parked the relay idle — the operator/
	// supervisor read of "this watch relay is idle, awaiting a witnessed event", distinct from
	// the empty-run count the underlying escape tracks.
	parked bool
}

// IdleLegOutcome is the closed result of folding one watch-relay leg: whether the relay must
// halt (the G6 backstop), whether it parked idle this leg, and the closed reason token
// (RELAY_NO_PROGRESS on halt, RELAY_IDLE_PARKED on an idle park, empty otherwise).
type IdleLegOutcome struct {
	// Halt is the G6 verdict: the no-progress escape tripped and automation must stop.
	Halt bool
	// Parked is true when this leg was proven idle and parked with RELAY_IDLE_PARKED — a benign
	// terminal-for-now, re-armable via ExternalEvent, NOT a halt.
	Parked bool
	// Reason is the closed reason token: ReasonNoProgress when Halt, ReasonIdleParked when
	// Parked, "" otherwise. Halt and Parked are mutually exclusive.
	Reason string
}

// ObserveLeg folds one completed watch-relay leg into the escape, idle-aware. Order and
// fail-closed rules:
//
//  1. If the leg made verified forward progress, it is delegated to the embedded escape (which
//     resets the empty run and raises the high-water mark) — progress is progress, never a park.
//  2. Else if the leg is POSITIVELY proven idle (invariant holds against ground truth AND the
//     ledger confirms zero admitted pending work), the relay parks: Parked, ReasonIdleParked,
//     no halt, and the empty-run counter is left UNTOUCHED (neither incremented nor reset) — a
//     correctly-idle leg simply is not counted toward RELAY_NO_PROGRESS.
//  3. Otherwise (an unknown/violated invariant, an unread pending count, or pending work) the
//     leg is delegated to the embedded escape UNCHANGED — the ordinary empty-leg path that can
//     trip RELAY_NO_PROGRESS. This is the fail-closed edge: an unverifiable idle-vs-stuck read
//     defaults to counting as no-progress.
//
// The idle branch can therefore only make the escape MORE tolerant of a PROVEN-idle relay; it
// can never suppress the counter for a relay that cannot prove it is idle.
func (e *IdleAwareEscape) ObserveLeg(now VerifiedProgress, obs IdleObservation) IdleLegOutcome {
	// A leg that made real verified forward progress is never a park: delegate so the escape
	// resets its empty run and raises the high-water mark. Only a NON-advancing, positively
	// proven-idle leg parks (and is left uncounted).
	if !e.Escape.Advances(now) && obs.idle() {
		e.parked = true
		return IdleLegOutcome{Parked: true, Reason: ReasonIdleParked}
	}
	e.parked = false
	halt, reason := e.Escape.ObserveLeg(now)
	return IdleLegOutcome{Halt: halt, Reason: reason}
}

// Parked reports whether the most recent ObserveLeg parked the relay idle.
func (e *IdleAwareEscape) Parked() bool { return e.parked }

// AxisExternalEvent is the re-arm trigger kind a PARKED watch relay arms on. It sits alongside
// the budget axes (armtriggers.go: context/turns/wall/spend) but is a different KIND of trigger:
// the budget axes arm a RUNNING leg on window pressure, whereas this axis re-arms a PARKED leg
// on a witnessed change in external ground truth. It leaves the budget-axis arming math wholly
// untouched — it is evaluated by ExternalEvent.Rearmed, not by ArmTriggers.Cross.
const AxisExternalEvent Axis = "external_event"

// ExternalEvent is the witnessed re-arm trigger for a watch relay parked with RELAY_IDLE_PARKED.
// It pins a ground-truth token (a git ref SHA, a CI verdict string) captured WHEN THE RELAY
// PARKED, and re-arms when the currently-observed ground-truth token DIFFERS from it — a
// witnessed flip, never a model assertion that "something changed". This is what turns an idle
// park back into an active leg without a human, while keeping the "verified, never claimed"
// discipline: both tokens are observations of ground truth the caller reads, not narration.
type ExternalEvent struct {
	// Kind names what is watched — "git_ref", "ci_verdict" — for the re-arm detail. Display-only.
	Kind string
	// ParkedToken is the witnessed ground-truth value captured when the relay parked (the
	// baseline the re-arm compares against).
	ParkedToken string
	// ObservedToken is the current witnessed ground-truth value read now.
	ObservedToken string
}

// Rearmed reports whether the witnessed external event flipped since the relay parked. It is a
// pure comparison of two ground-truth tokens and fails closed on a missing witness: BOTH tokens
// must be non-empty (an empty observed token is "could not witness now"; an empty parked token
// is "no baseline was pinned") — either alone yields no re-arm. Given two real observations, the
// relay re-arms exactly when they differ.
func (e ExternalEvent) Rearmed() bool {
	if e.ObservedToken == "" || e.ParkedToken == "" {
		return false
	}
	return e.ObservedToken != e.ParkedToken
}
