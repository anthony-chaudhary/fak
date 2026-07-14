package rsiloop

// cadence.go — CRASH-SAFE self-improvement cadence counters (#2877, part of the
// Hermes-inspiration epic #2871). The nudge state that schedules a periodic
// self-improvement review — turns-since-review, skill-iters — is DERIVED from the
// durable session ledger (internal/sessionledger, the #1352/#2382 family) on every
// read. There is no live counter object at all.
//
// THE PROBLEM (the Hermes mechanism this improves on). Hermes' nudge counters live
// on the in-memory AIAgent, so a gateway rebuild (cache miss / 1h idle eviction /
// restart) resets them to 0 and the review loop never fires. They patched it by
// re-hydrating the counters from conversation_history (their #22357) — a hydration
// hack bolted onto a fragile design.
//
// WHAT THIS DOES. fak's session ledger already durably appends every session event
// (the gateway writes a turn row per served request, and Append persists BEFORE
// returning), so the cadence counters can be a PURE FOLD over the trace's chain:
// turns-since-review is the count of completed-turn rows after the most recent
// review-fired row; skill-iters likewise for skill-loop iteration rows. A process
// death loses nothing because nothing lives outside the ledger — crash-safe by
// construction, not by rehydration.
//
// THE CONFUSION-RISK FENCE (#2877's own): the counters are derived from the ledger
// on EVERY read (SessionCadence → Chain → DeriveCadence), never persisted-and-
// rehydrated once at startup. A hydrate-once copy would be Hermes' patch with extra
// steps; a fold has no copy to go stale.
//
// RELATION TO THE SIBLING GATES. This file owns only the SCHEDULE state — "is a
// review due by cadence?". Whether a due review is WORTH firing is the value gate's
// question (reviewgate.go, #2837), and the stateless novelty/friction trigger
// (internal/nightrun/learningnudge.go, #2910) is the measured alternative to any
// fixed cadence. All three coordinate on the same Track A review-fire path; this
// leaf changes none of their decisions, it only makes the cadence input durable.
//
// Pure and deterministic: the same chain folds to the same counters every time.
// The only I/O is the ledger's own (already-durable) append/read.

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// Ledger entry kinds the cadence fold reads. CadenceKindTurnComplete is the kind
// the gateway already appends per served request (internal/gateway/session_admit.go
// writes the same literal — it cannot import this tier-4 package, so the string is
// the contract). The review/skill kinds are appended via the Mark helpers below.
const (
	// CadenceKindTurnComplete marks one completed session turn (the gateway's row).
	CadenceKindTurnComplete = "turn_complete"
	// CadenceKindSkillIter marks one skill-loop iteration (Hermes' "tool iteration"
	// axis), appended by the RSI skill loop via MarkSkillIter.
	CadenceKindSkillIter = "skill_iter"
	// CadenceKindReviewFired marks a fired self-improvement review; it is the row
	// that resets both since-review counters, appended via MarkReviewFired.
	CadenceKindReviewFired = "review_fired"
)

// Cadence is the derived (never stored) self-improvement schedule state for one
// session trace. Every field is a fold over durable rows; a rebuilt process derives
// the identical value from the identical chain.
type Cadence struct {
	Turns            int // completed turns in the whole session
	TurnsSinceReview int // completed turns after the most recent fired review
	SkillIters       int // skill-loop iterations after the most recent fired review
	ReviewsFired     int // self-improvement reviews fired in the whole session
}

// CadenceConfig is the review schedule: fire when either since-review counter
// reaches its cadence. A non-positive cadence disables that axis.
type CadenceConfig struct {
	ReviewEveryTurns      int // fire a review every N completed turns
	ReviewEverySkillIters int // fire a review every N skill-loop iterations
}

// DefaultCadenceConfig mirrors Hermes' shipped defaults (a nudge every 10 user
// turns / 10 tool iterations) so like-for-like comparisons hold; the value gate
// (#2837), not this constant, decides whether a due review is worth its cost.
func DefaultCadenceConfig() CadenceConfig {
	return CadenceConfig{ReviewEveryTurns: 10, ReviewEverySkillIters: 10}
}

// ReviewDue reports whether the schedule calls for a review NOW: either
// since-review counter has reached its (enabled) cadence.
func (c Cadence) ReviewDue(cfg CadenceConfig) bool {
	if cfg.ReviewEveryTurns > 0 && c.TurnsSinceReview >= cfg.ReviewEveryTurns {
		return true
	}
	if cfg.ReviewEverySkillIters > 0 && c.SkillIters >= cfg.ReviewEverySkillIters {
		return true
	}
	return false
}

// DeriveCadence is the pure fold: oldest-to-newest ledger entries in, counters out.
// A review-fired row resets both since-review counters; rows of any other kind are
// ignored, so the cadence fold coexists with every other kind a trace accumulates.
func DeriveCadence(entries []sessionledger.Entry) Cadence {
	var c Cadence
	for _, e := range entries {
		switch e.Kind {
		case CadenceKindTurnComplete:
			c.Turns++
			c.TurnsSinceReview++
		case CadenceKindSkillIter:
			c.SkillIters++
		case CadenceKindReviewFired:
			c.ReviewsFired++
			c.TurnsSinceReview = 0
			c.SkillIters = 0
		}
	}
	return c
}

// SessionCadence derives the cadence for trace from the durable ledger — the
// derive-on-every-read entry point a live loop calls at each turn boundary. A trace
// the ledger has never seen is a fresh session: zero cadence, no error.
func SessionCadence(l *sessionledger.Ledger, trace string) (Cadence, error) {
	entries, err := l.Chain(trace)
	if err != nil {
		// Chain errors only for an unknown trace head or a broken chain; an unknown
		// trace is simply a session with no history yet.
		if l.Head(trace) == "" {
			return Cadence{}, nil
		}
		return Cadence{}, err
	}
	return DeriveCadence(entries), nil
}

// MarkSkillIter durably records one skill-loop iteration for trace. The append
// persists before returning, so a process killed on the very next instruction
// still counts this iteration after rebuild.
func MarkSkillIter(l *sessionledger.Ledger, trace string) error {
	return appendCadenceRow(l, trace, CadenceKindSkillIter)
}

// MarkReviewFired durably records a fired self-improvement review for trace,
// resetting both since-review counters for every subsequent derivation.
func MarkReviewFired(l *sessionledger.Ledger, trace string) error {
	return appendCadenceRow(l, trace, CadenceKindReviewFired)
}

func appendCadenceRow(l *sessionledger.Ledger, trace, kind string) error {
	if l == nil {
		return errors.New("rsiloop: cadence ledger is required")
	}
	_, err := l.Append(trace, kind, nil)
	return err
}
