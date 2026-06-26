// Package lifecycle is the ONE canonical vocabulary for an agent's run-state —
// the shared skeleton that both the served session (internal/session.RunState)
// and the loop supervisor (internal/loopmgr.LoopState) spell.
//
// THE DEDUPE IT GROUNDS.
// Before this package the lifecycle vocabulary was written twice as unrelated Go
// types: session.RunState (a uint8 enum: running/throttled/paused/draining/stopped)
// and loopmgr.LoopState (a string folded from a hash-chained ledger:
// armed/running/paused/draining/stopped/disabled). Both carried the same four
// tokens — running/paused/draining/stopped — with no shared definition and no
// converter, so the two machines only LOOKED like one. This package is the literal
// dedupe that makes the "one machine seen at two altitudes" claim load-bearing
// (epic #912): the four common tokens are defined here exactly once, both layers
// source them, and the cross-layer converter (internal/lifebridge) pivots through
// the Phase below instead of two hand-kept string tables drifting apart.
//
// THE SHARED CORE, NOT THE WHOLE VERB SET.
// The shared skeleton is the four states both layers carry. The layer-specific
// extras — session's Throttled (a pace modifier, not a position) and the
// supervisor's Armed/Disabled (schedule states a live sequence has no peer for) —
// stay in their own packages and map to/from this core EXPLICITLY: a state with no
// shared peer converts to (zero, false), never to a silent default. Fail at the
// boundary, do not guess.
//
// This package is a foundation leaf: stdlib-only, depends on nothing, registers
// nothing. Parse fails closed so an unrecognized wire token is rejected at the
// edge rather than coerced to Running.
package lifecycle

// Phase is a position in the shared lifecycle skeleton: the four states both the
// served-session machine and the supervisor machine carry. It is ordered as the
// lifecycle progresses (Running -> Paused -> Draining -> Stopped), but conversion
// to/from the layer types is always by explicit switch, never by ordinal — the two
// enums number their states differently and must never be cast numerically.
type Phase uint8

const (
	// Running: the sequence advances at its budget/pace.
	Running Phase = iota
	// Paused: held at a turn boundary without ending — a resume is a state flip,
	// not a cold re-attach.
	Paused
	// Draining: a stop was requested; taken at the NEXT turn boundary, never
	// mid-decode.
	Draining
	// Stopped: terminal.
	Stopped
)

// The canonical wire tokens — the single definition of the lowercase strings both
// layers emit and accept. They are UNTYPED string constants on purpose: a caller
// in another package uses them in a typed-constant expression (e.g.
// `loopmgr.StateRunning LoopState = lifecycle.TokenRunning`) so the supervisor's
// constants are SOURCED here, not re-spelled.
const (
	TokenRunning  = "running"
	TokenPaused   = "paused"
	TokenDraining = "draining"
	TokenStopped  = "stopped"
)

// String renders a Phase as its canonical lowercase wire token. An out-of-range
// value renders "unknown" rather than panicking — a wire-derived value is never
// trusted to be in range.
func (p Phase) String() string {
	switch p {
	case Running:
		return TokenRunning
	case Paused:
		return TokenPaused
	case Draining:
		return TokenDraining
	case Stopped:
		return TokenStopped
	}
	return "unknown"
}

// Parse maps a wire token back to a Phase. The bool is false for any token outside
// the shared core — including a layer-specific extra like "throttled"/"armed",
// which is a real token but not a shared-lifecycle Phase — so a caller fails closed
// rather than defaulting an unknown verb to Running.
func Parse(s string) (Phase, bool) {
	switch s {
	case TokenRunning:
		return Running, true
	case TokenPaused:
		return Paused, true
	case TokenDraining:
		return Draining, true
	case TokenStopped:
		return Stopped, true
	}
	return 0, false
}
