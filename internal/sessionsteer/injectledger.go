package sessionsteer

import "sort"

// This file is the repeat-injection bound (#5922, epic #5917): a per-rule ledger that
// decides whether a named rule may inject AGAIN, counted in completed turns. It keeps the
// package's contract — no I/O, no clock, no randomness: the ledger is a pure state machine
// over two caller-reported events (TurnEnd, RecordInjection), so every verdict is a
// deterministic function of the event history and the whole thing is exercised by tables.
//
// Restored-age semantics (a decision, not an accident — pinned by test): persistence
// carries rule NAMES, not ages. Restore marks each name as injected at the ledger's
// CURRENT completed-turn count (zero on a fresh resume), so a restored `after-gap` rule
// becomes eligible only after a FULL fresh gap of completed turns, and a restored `once`
// rule stays suppressed for the session's lifetime. Storing ages instead would buy
// precision at the cost of a cross-session turn-clock; the full-fresh-gap restart is the
// conservative direction (a resume never makes a rule fire SOONER than it would have).
//
// Abort/error rollback (a decision, not an accident — pinned by test): RecordInjection is
// in-memory only; nothing is persisted until the caller snapshots InjectedRules at a turn
// boundary. When a turn dies before the injection is actually delivered, the caller MUST
// call Rollback(rule) so live state matches what a resume would rebuild (the record was
// never persisted, so a reload would make the rule eligible again). A caller that crashes
// before Rollback converges to the same answer through the resume path — the divergence
// window is the live session only, and Rollback exists precisely to close it.

// RepeatMode is the closed vocabulary for how often a named rule may re-inject.
type RepeatMode string

const (
	// RepeatOnce fires a rule at most once per session lifetime: after it holds an
	// injection record it is suppressed forever (including across a resume — the record
	// persists by name).
	RepeatOnce RepeatMode = "once"
	// RepeatAfterGap re-fires a rule only once at least the configured gap of COMPLETED
	// turns has elapsed since its last injection: turns-since-injection >= gap. The gap is
	// measured at turn end, never in stream chunks — a chunk count would make the budget
	// depend on response length, which is meaningless as a nag bound.
	RepeatAfterGap RepeatMode = "after-gap"
)

// DefaultRepeatGap is the default after-gap re-fire distance in completed turns.
const DefaultRepeatGap = 10

// NormalizeRepeatMode maps an arbitrary mode string onto the closed vocabulary, defaulting
// unrecognized input to RepeatOnce (fail-closed: an unknown mode never grants unbounded
// repetition).
func NormalizeRepeatMode(s string) RepeatMode {
	if RepeatMode(s) == RepeatAfterGap {
		return RepeatAfterGap
	}
	return RepeatOnce
}

// InjectionLedger is the per-rule repeat bound: which rules have injected, at which
// completed-turn count, and how many turns have completed. The zero value is ready to use.
// It is deliberately event-driven — the caller reports turn ends and injections; the
// ledger never reads a clock or the filesystem.
type InjectionLedger struct {
	turns          int            // completed turns (advances on TurnEnd ONLY)
	chunks         int            // stream chunks observed within the current turn (never advances turns)
	lastInjectedAt map[string]int // rule name -> completed-turn count at its last injection
}

// NewInjectionLedger returns an empty ledger at turn zero.
func NewInjectionLedger() *InjectionLedger { return &InjectionLedger{} }

// CompletedTurns reports how many turns have ended since the ledger was created/restored.
func (l *InjectionLedger) CompletedTurns() int { return l.turns }

// TurnEnd records the completion of one turn. It is the ONLY event that advances the
// completed-turn counter, and it resets the within-turn chunk count.
func (l *InjectionLedger) TurnEnd() {
	l.turns++
	l.chunks = 0
}

// StreamChunk records one streaming delta within the current turn. It deliberately does
// NOT advance the completed-turn counter — the gap is a nag budget in turns, and counting
// chunks would tie it to response length. Exposed as an explicit event so the transport
// contract is testable, not implicit.
func (l *InjectionLedger) StreamChunk() { l.chunks++ }

// ChunksThisTurn reports the stream chunks observed since the last TurnEnd (observability
// only; no verdict depends on it).
func (l *InjectionLedger) ChunksThisTurn() int { return l.chunks }

// Allow reports whether the named rule may inject now under the given mode. A rule with no
// injection record is always allowed. gap <= 0 selects DefaultRepeatGap. The boundary is
// inclusive: an after-gap rule re-fires at exactly gap completed turns since its last
// injection, not gap+1.
func (l *InjectionLedger) Allow(rule string, mode RepeatMode, gap int) bool {
	last, injected := l.lastInjectedAt[rule]
	if !injected {
		return true
	}
	if NormalizeRepeatMode(string(mode)) == RepeatOnce {
		return false
	}
	if gap <= 0 {
		gap = DefaultRepeatGap
	}
	return l.turns-last >= gap
}

// RecordInjection marks the rule as injected at the current completed-turn count. Call it
// when the injection is emitted; if the turn then dies before delivery, undo it with
// Rollback (see the file header for why).
func (l *InjectionLedger) RecordInjection(rule string) {
	if l.lastInjectedAt == nil {
		l.lastInjectedAt = make(map[string]int)
	}
	l.lastInjectedAt[rule] = l.turns
}

// Rollback erases the rule's in-memory injection record — the abort/error path. A turn
// that dies before the injection is delivered never persisted anything, so a resume would
// treat the rule as eligible; Rollback makes the LIVE session agree with that instead of
// silently suppressing a rule the model never saw. A rollback of an unrecorded rule is a
// no-op.
func (l *InjectionLedger) Rollback(rule string) { delete(l.lastInjectedAt, rule) }

// InjectedRules is the persistence snapshot: the sorted names of every rule holding an
// injection record. Names only — no ages — by decision (see the file header).
func (l *InjectionLedger) InjectedRules() []string {
	names := make([]string, 0, len(l.lastInjectedAt))
	for name := range l.lastInjectedAt {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Restore rehydrates persisted rule names on session resume, marking each as injected at
// the ledger's CURRENT completed-turn count. On a fresh ledger that is turn zero, so a
// restored after-gap rule waits a full fresh gap and a restored once rule stays suppressed
// — the documented restored-age decision.
func (l *InjectionLedger) Restore(names []string) {
	for _, name := range names {
		l.RecordInjection(name)
	}
}
