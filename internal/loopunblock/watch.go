package loopunblock

// watch.go — the MONITOR half of the generic watchdog. [Decide] answers "given a stuck
// loop's worst-first candidates, what do I do THIS tick?"; [Assess] answers the
// always-on question around it: "is this loop actually STALLED on a head-of-line block,
// how long has it been stuck, must it escalate to a human, and how soon should the
// watchdog look again?"
//
// Together they are the whole watchdog the goal describes — small, critical, always-on,
// fast to respond: each tick the shell calls Decide, records the resulting [Tick], and
// calls Assess over the recent history to learn whether the loop is draining (come back
// FAST), stuck on a transient block (back OFF so the watchdog does not spin), or stalled
// past the escalation horizon (surface to an operator). The pacing is the "fast response
// timing" made explicit: a watchdog that re-checks a draining queue quickly and a
// blocked one slowly spends its small budget where it matters.
//
// Pure, like the rest of the package: Assess reads no clock (now is supplied) and no
// files. Same history + policy + now in, same verdict out.

import "fmt"

// WatchSchema tags the stall-watch verdict payload, distinct from the decision Schema.
const WatchSchema = "fak.loop-watch.v1"

// Progressed reports whether an action ENTERED a member this tick — the watchdog made
// forward progress. Enter, ClearThenEnter, and Bypass all put a member into flight;
// Wait and Escalate enter nothing; StandDown had nothing to enter. Exposed because a
// shell driving the watchdog often wants the same progress/no-progress split Assess uses.
func (a Action) Progressed() bool {
	switch a {
	case ActionEnter, ActionClearThenEnter, ActionBypass:
		return true
	default:
		return false
	}
}

// blocks reports whether an action is a BLOCKED tick that counts toward a stall streak —
// the head could not be entered and nothing was routed around it. Wait and Escalate
// block; Enter/ClearThenEnter/Bypass are progress; StandDown is an empty queue (nothing
// to stall on). This is the internal predicate the streak fold keys on.
func (a Action) blocks() bool { return a == ActionWait || a == ActionEscalate }

// Tick is one watchdog observation: the [Decide] action this tick produced, the
// worst-first head it concerned, and when it happened (unix nanoseconds). A shell appends
// one per watchdog tick and hands the recent tail to [Assess].
type Tick struct {
	Action   Action `json:"action"`
	Head     string `json:"head,omitempty"`
	UnixNano int64  `json:"unix_nano"`
}

// StallPolicy tunes the stall / escalation / pacing thresholds. The ZERO VALUE is usable:
// [Assess] fills every unset field with a documented default, so a caller can pass
// StallPolicy{} and get sane always-on behavior (the same zero-value-friendly contract
// [Policy] and loopmgr.HealthThresholds keep).
type StallPolicy struct {
	// StallAfter is the number of consecutive BLOCKED ticks on the SAME head before the
	// loop is declared stalled. <=0 -> DefaultStallAfter. 1 means "stalled on the first
	// blocked tick"; higher tolerates a brief block before raising the stall flag.
	StallAfter int `json:"stall_after,omitempty"`
	// EscalateAfterSeconds is how long a stall may persist — measured from the FIRST
	// blocked tick in the current streak — before the watchdog escalates to an operator
	// even when the block is nominally transient. <=0 -> DefaultEscalateAfterSeconds.
	EscalateAfterSeconds int64 `json:"escalate_after_seconds,omitempty"`
	// FastDelaySeconds is the re-check window when the watchdog is actively DRAINING (it
	// entered a member this tick) — respond fast, keep the queue moving. <=0 -> default.
	FastDelaySeconds int64 `json:"fast_delay_seconds,omitempty"`
	// SlowDelaySeconds is the backed-off re-check window when the head is in a transient
	// wait, already escalated, or there is nothing to do — so the always-on watchdog does
	// not spin its small budget on a block it cannot move. <=0 -> default.
	SlowDelaySeconds int64 `json:"slow_delay_seconds,omitempty"`
}

// Pacing / stall defaults. Conservative: a few blocked ticks before "stalled", a 15-min
// escalation horizon, a brisk 15s drain re-check, and a 2-min back-off when blocked.
const (
	DefaultStallAfter           = 3
	DefaultEscalateAfterSeconds = 900
	DefaultFastDelaySeconds     = 15
	DefaultSlowDelaySeconds     = 120
)

func (p StallPolicy) stallAfter() int {
	if p.StallAfter > 0 {
		return p.StallAfter
	}
	return DefaultStallAfter
}

func (p StallPolicy) escalateAfterSeconds() int64 {
	if p.EscalateAfterSeconds > 0 {
		return p.EscalateAfterSeconds
	}
	return DefaultEscalateAfterSeconds
}

func (p StallPolicy) fastDelaySeconds() int64 {
	if p.FastDelaySeconds > 0 {
		return p.FastDelaySeconds
	}
	return DefaultFastDelaySeconds
}

func (p StallPolicy) slowDelaySeconds() int64 {
	if p.SlowDelaySeconds > 0 {
		return p.SlowDelaySeconds
	}
	return DefaultSlowDelaySeconds
}

// StallVerdict is the monitor-half output: whether the loop is stalled on one head, how
// long, whether the watchdog must escalate, and how soon it should tick again.
type StallVerdict struct {
	Schema string `json:"schema"`
	// Stalled is true when the same head has been blocked for at least StallAfter
	// consecutive newest ticks — a head-of-line stall, not just a one-tick block.
	Stalled bool `json:"stalled"`
	// Head is the head the streak is stuck on ("" when not stalled).
	Head string `json:"head,omitempty"`
	// BlockedStreak is the number of consecutive newest ticks blocked on that same head
	// (0 when the newest tick made progress or the queue was empty).
	BlockedStreak int `json:"blocked_streak"`
	// StallAgeSeconds is now minus the first blocked tick in the streak (0 when not
	// stalled or when the streak's first tick has no usable timestamp).
	StallAgeSeconds int64 `json:"stall_age_seconds,omitempty"`
	// Escalate is true when a stall has aged past EscalateAfterSeconds, or the newest tick
	// already decided Escalate — the watchdog can no longer self-resolve and a human is owed.
	Escalate bool `json:"escalate"`
	// NextDelaySeconds is when the watchdog should tick again: the fast window while
	// draining, the slow window while blocked / escalated / idle.
	NextDelaySeconds int64 `json:"next_delay_seconds"`
	// Reason is a one-line account of the verdict.
	Reason string `json:"reason"`
}

// Assess folds a recent watchdog tick history (oldest -> newest) plus a policy and the
// current time into the monitor verdict. It is PURE: no clock, no I/O.
//
// The logic keys on the NEWEST tick:
//   - it made progress (entered/cleared/bypassed) -> not stalled; come back FAST (draining).
//   - it stood down (empty queue)                 -> not stalled; back off SLOW (idle).
//   - it blocked (wait/escalate)                  -> count the streak of consecutive newest
//     ticks that blocked on the SAME head; that streak is the head-of-line stall. Stalled
//     once it reaches StallAfter; Escalate once it ages past EscalateAfterSeconds (or the
//     newest tick already escalated). A blocked watchdog backs off SLOW so it does not spin.
//
// The same-head requirement is what makes this a HEAD-OF-LINE detector: if the blocking
// head keeps CHANGING tick to tick, the queue is draining (different members block in
// turn), not stuck behind one — so the streak resets and the loop is not called stalled.
func Assess(history []Tick, pol StallPolicy, nowUnixNano int64) StallVerdict {
	v := StallVerdict{Schema: WatchSchema}
	if len(history) == 0 {
		v.NextDelaySeconds = pol.fastDelaySeconds()
		v.Reason = "no ticks observed yet — check again on the fast cadence"
		return v
	}

	newest := history[len(history)-1]
	if newest.Action.Progressed() {
		v.NextDelaySeconds = pol.fastDelaySeconds()
		v.Reason = fmt.Sprintf("the watchdog is draining the queue (%s) — keep the fast cadence", newest.Action)
		return v
	}
	if !newest.Action.blocks() { // StandDown or any non-progress, non-block action.
		v.NextDelaySeconds = pol.slowDelaySeconds()
		v.Reason = fmt.Sprintf("nothing to enter (%s) — back off to the slow cadence", newest.Action)
		return v
	}

	// Blocked: count the consecutive newest ticks blocked on the SAME head.
	streak := 0
	firstUnixNano := newest.UnixNano
	for i := len(history) - 1; i >= 0; i-- {
		t := history[i]
		if !t.Action.blocks() || t.Head != newest.Head {
			break
		}
		streak++
		firstUnixNano = t.UnixNano
	}

	v.Head = newest.Head
	v.BlockedStreak = streak
	v.Stalled = streak >= pol.stallAfter()
	if firstUnixNano > 0 && nowUnixNano > firstUnixNano {
		v.StallAgeSeconds = (nowUnixNano - firstUnixNano) / 1_000_000_000
	}
	agedOut := v.Stalled && v.StallAgeSeconds >= pol.escalateAfterSeconds()
	v.Escalate = newest.Action == ActionEscalate || agedOut
	// A blocked watchdog always backs off: it cannot move the block by re-checking fast.
	v.NextDelaySeconds = pol.slowDelaySeconds()

	switch {
	case v.Escalate:
		v.Reason = fmt.Sprintf("stalled on head %q and must escalate to an operator (streak %d, age %ds)",
			newest.Head, streak, v.StallAgeSeconds)
	case v.Stalled:
		v.Reason = fmt.Sprintf("head-of-line stalled on %q (streak %d >= %d) — still transient; backing off before escalation",
			newest.Head, streak, pol.stallAfter())
	default:
		v.Reason = fmt.Sprintf("head %q blocked (streak %d < %d) — not yet a stall; back off to the slow cadence",
			newest.Head, streak, pol.stallAfter())
	}
	return v
}
