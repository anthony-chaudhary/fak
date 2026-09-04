// Package loopunblock is the GENERIC head-of-line unblocker for any
// worklist-draining loop — a normal loop, a super loop, or a meta-loop over
// super loops. It is the always-on "watchdog" rung the fleet was missing: a
// small, critical, fast fold whose only job is to keep a loop that has ALREADY
// selected its next move from stalling on a move it cannot make.
//
// # The gap it fills
//
// The loop ladder already has a liveness rung (internal/loopmgr FoldHealth:
// live/stale/dark), a backpressure rung (loopmgr Governor: back a refusal-storming
// loop off), a crashloop fence (loopmgr RestartPolicy), and a select rung
// (internal/superloop Walk/Drive: fold a worst-first worklist and name the head to
// enter). None of them solves HEAD-OF-LINE BLOCKING: superloop.Drive ALWAYS
// returns the worst-first head (rank 1), and the impure shell then passes that head
// through an admission gate (region admission over the live lease fabric). When the
// gate REFUSES the head — its lease overlaps a live peer, its account is capped, or
// a dead worker left a stale lease — the shell re-folds and Drive names the SAME
// blocked head again. The queue behind the head never drains even though other
// members are perfectly enterable. One stuck head stalls the whole loop.
//
// That is textbook head-of-line blocking, and clearing it is a small volume of
// work that must be done fast and continuously — exactly a watchdog's shape. This
// leaf is that watchdog's decision core.
//
// # What it decides
//
// Given the worst-first candidate list a loop already produced, each candidate
// tagged with whether admission would ADMIT it (and, if not, WHY, in a closed
// [BlockCause] vocabulary), [Decide] returns one [Decision] from a closed [Action]
// vocabulary:
//
//   - Enter          — the head is admittable; nothing was blocked. Enter it.
//   - ClearThenEnter — the head is blocked on an AUTO-RELEASABLE cause (a stale,
//     dead-worker lease). Clear the blocker in place, then enter the head. This is
//     the true fix: the worst-first order is preserved.
//   - Bypass         — the head is blocked on a cause that will not clear in place,
//     but a lower-ranked candidate IS admittable. Route around the head to that
//     candidate (out-of-order completion) so the queue keeps draining. The skipped
//     heads are recorded so the bypass is auditable, never silent.
//   - Wait          — the head is blocked on a TRANSIENT cause (a live peer lease,
//     a capped seat) and nothing is admittable to bypass to. Back off and retry;
//     the cause self-resolves when the peer finishes or the cap window resets.
//   - Escalate      — the head is blocked on a cause that will NOT self-resolve
//     (an unreadable member, an unknown block) with nothing to bypass to. Only an
//     operator can move it; surface it rather than spin.
//   - StandDown     — the candidate list is empty. The loop's worklist is clean;
//     there is nothing to unblock. A no-op, stated explicitly rather than inferred.
//
// # Generic by construction
//
// The fold takes only an abstract [Candidate] list — an id, a worst-first rank, and
// an [AdmitStatus] — never a superloop.WorkItem or any concrete loop type. Any loop
// that can produce a worst-first head list and report per-head admission can drive
// this watchdog: the superloop drive shell maps its admission-gated worklist heads
// into []Candidate; a dispatch loop maps its lane-pressure picks; a future loop maps
// whatever it queues. The seam stays in the impure shell (cmd/fak), which reads the
// live lease fabric / account caps to fill each AdmitStatus and APPLIES the returned
// Action (releasing a stale lease, entering the bypass target). This core reads no
// clock, no files, and takes no admission action — same candidates in, same decision
// out — so it is table-testable in isolation and cannot be reddened by churn in the
// large, contended packages it serves.
package loopunblock

import (
	"fmt"
	"strings"
)

// Schema is the decision payload's self-describing version tag.
const Schema = "fak.loop-unblock.v1"

// BlockCause is the CLOSED vocabulary of why a worklist head could not be admitted.
// A caller must map its admission refusal onto one of these — the whole point of the
// fold is that every reason a head stalls has a defined disposition (clear in place,
// bypass, wait, or escalate), so a novel free-text refusal can never silently strand
// a loop. CauseNone is the admittable case; every other value is a real block.
type BlockCause string

const (
	// CauseNone: admission would ADMIT this candidate — it is not blocked.
	CauseNone BlockCause = ""
	// CauseLeaseStale: the head's lease is held by a DEAD worker (a pid that is gone,
	// a lease past its liveness horizon). The one AUTO-RELEASABLE block: releasing a
	// dead-worker lease is the safe, fast, in-place fix that preserves worst-first
	// order. This is the watchdog's canonical hand-holding move.
	CauseLeaseStale BlockCause = "lease_stale"
	// CauseLeaseLive: the head's lease overlaps a LIVE peer's lease (a real
	// COLLISION_RISK). Transient: it clears when the peer finishes. Not clearable in
	// place — bypass if something else is enterable, else wait.
	CauseLeaseLive BlockCause = "lease_live"
	// CauseCapped: entering the head needs a resource that is currently exhausted (a
	// usage-limited account seat, a rate-capped gateway). Transient: the cap window
	// resets. Not clearable in place — bypass if something else is enterable, else wait.
	CauseCapped BlockCause = "capped"
	// CauseBudgetHeld: the head's declared budget dimension is UNBUDGETED/held (the
	// superloop generation-budget "hold" state), so the loop is not authorized to spend
	// on it now. Will NOT self-resolve on a timer — bypass to a budgeted member if one
	// is enterable, else escalate for an operator budget decision.
	CauseBudgetHeld BlockCause = "budget_held"
	// CauseUnmeasured: the head's status could not be read, so it cannot be entered as
	// a work item (it needs a MEASURE, not an enter). Will not self-resolve — bypass to
	// a measured member if one is enterable, else escalate to repair the reader.
	CauseUnmeasured BlockCause = "unmeasured"
	// CauseUnknown: admission refused for a reason outside this vocabulary. Fail
	// conservative — never auto-act on a block we cannot classify: bypass if possible,
	// otherwise escalate to a human.
	CauseUnknown BlockCause = "unknown"
)

// Transient reports whether a cause self-resolves on a timer — the peer lease frees,
// the cap window resets — so WAIT is a sound disposition when nothing is admittable
// to bypass to. A non-transient block (budget held, unmeasured, unknown) will still
// be there on the next tick, so waiting on it only spins; it escalates instead.
func (c BlockCause) Transient() bool {
	switch c {
	case CauseLeaseLive, CauseCapped:
		return true
	default:
		return false
	}
}

// Clearable reports whether a cause can be safely resolved IN PLACE by the watchdog,
// preserving the worst-first head. Only a stale (dead-worker) lease qualifies:
// releasing it is safe and idempotent. Every other block is either transient (wait),
// routable (bypass), or an operator's call (escalate) — never auto-cleared.
func (c BlockCause) Clearable() bool { return c == CauseLeaseStale }

// Action is the CLOSED vocabulary of what the watchdog does about the head. It is the
// one thing a shell routes on; each value's meaning is fixed in the package doc.
type Action string

const (
	ActionEnter          Action = "enter"            // head admittable — enter it
	ActionClearThenEnter Action = "clear_then_enter" // clear a stale lease, then enter the head
	ActionBypass         Action = "bypass"           // route around a blocked head to the next admittable candidate
	ActionWait           Action = "wait"             // transient block, nothing to bypass to — back off
	ActionEscalate       Action = "escalate"         // unresolvable block, nothing to bypass to — surface to operator
	ActionStandDown      Action = "stand_down"       // empty worklist — nothing to unblock
)

// Reason tokens: the closed, grep-able prefix on each Decision.Reason so a downstream
// consumer (a log fold, a `--json` gate) can route on the disposition without string
// matching the prose.
const (
	ReasonEnter     = "UNBLOCK_ENTER"
	ReasonCleared   = "UNBLOCK_CLEARED"
	ReasonBypass    = "UNBLOCK_BYPASS"
	ReasonWait      = "UNBLOCK_WAIT"
	ReasonEscalate  = "UNBLOCK_ESCALATE"
	ReasonStandDown = "UNBLOCK_STAND_DOWN"
)

// AdmitStatus is the per-candidate admission verdict the shell fills from the live
// world (the lease fabric, account caps). Admittable is the one load-bearing bit;
// when false, Cause explains the block (never CauseNone) and Detail carries the
// shell's one-line evidence for the audit trail.
type AdmitStatus struct {
	Admittable bool       `json:"admittable"`
	Cause      BlockCause `json:"cause,omitempty"`
	Detail     string     `json:"detail,omitempty"`
}

// Admittable builds the "admission would admit this" status.
func Admittable() AdmitStatus { return AdmitStatus{Admittable: true, Cause: CauseNone} }

// Blocked builds a refused status with a classified cause. An empty or CauseNone
// cause on a blocked candidate is coerced to CauseUnknown so a blocked head can never
// masquerade as admittable — the fold fails conservative.
func Blocked(cause BlockCause, detail string) AdmitStatus {
	if cause == CauseNone {
		cause = CauseUnknown
	}
	return AdmitStatus{Admittable: false, Cause: cause, Detail: detail}
}

// Candidate is one worst-first entry in the loop's worklist as this watchdog sees it:
// an opaque id, its 1-based worst-first Rank (the loop's own ordering), and the
// admission verdict for entering it now. It is deliberately NOT a superloop.WorkItem —
// keeping the fold generic over any worklist-draining loop.
type Candidate struct {
	ID    string      `json:"id"`
	Rank  int         `json:"rank"`
	Admit AdmitStatus `json:"admit"`
}

// Policy tunes the watchdog. The ZERO VALUE is the intended always-on autonomous
// unblocker: it clears stale leases and bypasses blocked heads without an operator,
// because the whole point is a fast, continuous, hands-off unblock. The two opt-OUT
// knobs let an operator dial that back without changing the classification.
type Policy struct {
	// Manual, when true, makes the watchdog ADVISORY only: it still classifies the
	// block and names the exact Action, but marks every decision Auto=false so a human
	// applies it. Use when an operator wants the diagnosis without the hands.
	Manual bool `json:"manual,omitempty"`
	// NoBypass, when true, forbids routing around a blocked head: strict worst-first is
	// preserved even at the cost of stalling. A head that would have been bypassed then
	// falls through to Wait (transient cause) or Escalate (not), same as if nothing were
	// admittable behind it. Clearing a stale lease in place is still allowed — it does
	// not change which member runs.
	NoBypass bool `json:"no_bypass,omitempty"`
}

// Decision is the folded verdict: the single next move for the stalled loop, why, and
// whether the watchdog may apply it autonomously.
type Decision struct {
	Schema string `json:"schema"`
	Action Action `json:"action"`
	// Auto is true when the watchdog may APPLY this action without an operator. It is
	// true for the autonomous moves (enter/clear/bypass/wait/stand-down) under the
	// default policy, and always false for Escalate (nothing to auto-do) and for every
	// action under a Manual policy.
	Auto bool `json:"auto"`
	// Head is the worst-first head id — the candidate the loop WANTED to enter. Empty
	// only on StandDown (no candidates).
	Head string `json:"head,omitempty"`
	// HeadRank is the head's 1-based worst-first rank, echoed for the audit line.
	HeadRank int `json:"head_rank,omitempty"`
	// HeadCause is why the head was blocked (CauseNone when the head was admittable).
	HeadCause BlockCause `json:"head_cause,omitempty"`
	// Enter is the candidate id to actually enter now: the head (Enter/ClearThenEnter)
	// or the bypass target (Bypass). Empty on Wait/Escalate/StandDown — nothing to enter.
	Enter string `json:"enter,omitempty"`
	// Bypassed lists the blocked head ids skipped to reach Enter, in worst-first order —
	// the head-of-line victims, recorded so a bypass is always auditable, never silent.
	Bypassed []string `json:"bypassed,omitempty"`
	// Reason is a one-line account prefixed with a closed Reason token for routing.
	Reason string `json:"reason"`
}

// Decide folds a worst-first candidate list plus a policy into the single next move
// for a stalled loop. It is PURE: no clock, no I/O, no admission action — the shell
// fills each AdmitStatus and applies the returned Action.
//
// Invariant: loop unblock decisions are fail-closed and deterministic. An empty candidate
// list stands down, and unknown or unclassifiable blocks fail closed by escalating to an operator
// rather than speculating or taking unsafe autonomous actions.
//
// Guard condition: when the head is blocked on non-transient or unclassifiable causes
// and no bypass targets are admittable, the decision must escalate rather than loop or wait.
//
// The decision order is fixed so the verdict is deterministic:
//
//  1. no candidates            -> StandDown (nothing queued to unblock)
//  2. head admittable          -> Enter (it was never blocked)
//  3. head lease is stale      -> ClearThenEnter (release it in place, keep worst-first)
//  4. a later candidate admits  -> Bypass (route around the head), unless NoBypass
//  5. head cause is transient  -> Wait (it self-resolves; retry)
//  6. otherwise                -> Escalate (unresolvable, nothing to bypass to)
func Decide(cands []Candidate, pol Policy) Decision {
	if len(cands) == 0 {
		return Decision{
			Schema: Schema, Action: ActionStandDown, Auto: !pol.Manual,
			Reason: ReasonStandDown + ": no candidates queued — the worklist is clean, nothing to unblock",
		}
	}

	head := cands[0]
	d := Decision{
		Schema:    Schema,
		Head:      head.ID,
		HeadRank:  head.Rank,
		HeadCause: head.Admit.Cause,
	}

	// (2) The head was enterable all along — no block.
	if head.Admit.Admittable {
		d.Action, d.Auto, d.Enter = ActionEnter, !pol.Manual, head.ID
		d.HeadCause = CauseNone
		d.Reason = fmt.Sprintf("%s: head %q (rank %d) is admittable — enter it", ReasonEnter, head.ID, head.Rank)
		return d
	}

	// (3) A stale, dead-worker lease is the one block the watchdog clears IN PLACE,
	// preserving the worst-first head. Allowed even under NoBypass (it does not change
	// which member runs) but suppressed to advisory under Manual.
	if head.Admit.Cause.Clearable() {
		d.Action, d.Auto, d.Enter = ActionClearThenEnter, !pol.Manual, head.ID
		d.Reason = fmt.Sprintf("%s: head %q (rank %d) blocked by a stale (dead-worker) lease — release it in place, then enter the head",
			ReasonCleared, head.ID, head.Rank)
		return d
	}

	// (4) Route around the blocked head to the first admittable candidate behind it,
	// unless the operator forbade bypass. This is the head-of-line drain: one stuck
	// head no longer stalls the members queued behind it.
	if !pol.NoBypass {
		if target, skipped, ok := firstAdmittable(cands); ok {
			d.Action, d.Auto = ActionBypass, !pol.Manual
			d.Enter, d.Bypassed = target.ID, skipped
			d.Reason = fmt.Sprintf("%s: head %q (rank %d) blocked (%s); routing around %d stalled head(s) to admittable %q (rank %d)",
				ReasonBypass, head.ID, head.Rank, causeWord(head.Admit.Cause), len(skipped), target.ID, target.Rank)
			return d
		}
	}

	// (5) Nothing admittable to enter. A transient block self-resolves, so back off and
	// retry rather than escalate a race that is about to clear on its own.
	if head.Admit.Cause.Transient() {
		d.Action, d.Auto = ActionWait, !pol.Manual
		d.Reason = fmt.Sprintf("%s: head %q (rank %d) blocked by a transient %s and nothing else is admittable — back off and retry",
			ReasonWait, head.ID, head.Rank, causeWord(head.Admit.Cause))
		return d
	}

	// (6) An unresolvable block with nothing to bypass to. Only an operator moves it.
	d.Action, d.Auto = ActionEscalate, false
	d.Reason = fmt.Sprintf("%s: head %q (rank %d) blocked by %s that will not self-resolve, and nothing else is admittable — surface to an operator",
		ReasonEscalate, head.ID, head.Rank, causeWord(head.Admit.Cause))
	return d
}

// firstAdmittable returns the first admittable candidate in worst-first order and the
// ids of the blocked candidates skipped to reach it (always including the head, which
// by the time we call this is known blocked). ok is false when NONE are admittable.
func firstAdmittable(cands []Candidate) (target Candidate, skipped []string, ok bool) {
	for _, c := range cands {
		if c.Admit.Admittable {
			return c, skipped, true
		}
		skipped = append(skipped, c.ID)
	}
	return Candidate{}, nil, false
}

// causeWord renders a cause for a human-readable reason line, falling back to the raw
// token so an unmapped cause is still legible rather than blank.
func causeWord(c BlockCause) string {
	switch c {
	case CauseLeaseStale:
		return "a stale lease"
	case CauseLeaseLive:
		return "a live peer lease"
	case CauseCapped:
		return "an exhausted resource cap"
	case CauseBudgetHeld:
		return "a held budget"
	case CauseUnmeasured:
		return "an unreadable status"
	case CauseUnknown:
		return "an unclassified block"
	case CauseNone:
		return "no block"
	default:
		if s := strings.TrimSpace(string(c)); s != "" {
			return s
		}
		return "an unclassified block"
	}
}
