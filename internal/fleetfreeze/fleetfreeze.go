// Package fleetfreeze is the operator freeze gate for the parallel-agent
// dispatch fleet: a documented switch that HOLDS new worker spawns while
// STILL ALLOWING the progress-harvesting paths (witness-close and
// status-refresh) to keep running.
//
// # The freeze contract
//
// An operator needs to stop the fleet from starting NEW work — to drain,
// to cool down after a rate-limit scare, or to freeze the world before a
// risky change — WITHOUT going dark on the work already in flight. A blunt
// "disable the dispatcher" switch throws away both: it stops spawns AND
// stops the fleet from noticing that in-flight workers finished, so issues
// that are actually done never get closed and the status view goes stale.
//
// This package draws the line by OPERATION CLASS rather than by process.
// A freeze is a predicate over what an operation would DO, evaluated by
// [Allowed]:
//
//   - OpSpawn        — starts NEW work. HELD when frozen.
//   - OpWitnessClose — closes an issue whose worker PROVED it done.
//     ALWAYS allowed: harvesting finished progress is never unsafe, and
//     blocking it is the failure mode a freeze must avoid.
//   - OpStatusRefresh — re-reads live fleet/issue state for the operator.
//     ALWAYS allowed: read-only, and the operator needs it MOST while frozen.
//   - OpComment       — posts a progress/status comment. ALWAYS allowed:
//     annotation, not new work.
//
// The rule, stated once: a freeze holds exactly the spawn class and nothing
// else. When not frozen every class is allowed. This is the whole safety
// property, and [Allowed] is its single enforcement point — callers on the
// spawn path consult it before launching a worker; callers on the
// close/status/comment paths consult it too and are told to proceed.
//
// # Determinism
//
// Nothing here reads the clock. [Freeze] takes the freeze instant as a Unix
// timestamp argument so the same inputs always yield the same [State] and the
// same [Decision] — the whole package is a pure function of its inputs, which
// is what makes the witness test (spawn held while close/status pass) a
// deterministic proof rather than a timing-dependent flake.
//
// # Scope
//
// Stdlib-only, off the hot path. This is the reusable core; a later
// `fak fleet freeze` CLI wrapper drives [Freeze]/[Unfreeze] and prints
// [State.String] for its dry-run status output.
package fleetfreeze

import (
	"fmt"
	"time"
)

// OpClass names a class of fleet operation by what it would DO, so a freeze
// can hold new work while letting progress harvesting continue.
type OpClass int

const (
	// OpSpawn starts a NEW worker / new unit of work. This is the only class
	// a freeze holds.
	OpSpawn OpClass = iota
	// OpWitnessClose closes an issue whose worker has PROVED it done. Always
	// allowed — harvesting finished progress is the point a freeze protects.
	OpWitnessClose
	// OpStatusRefresh re-reads live fleet/issue state for the operator.
	// Always allowed — read-only, and needed most during a freeze.
	OpStatusRefresh
	// OpComment posts a progress or status comment. Always allowed —
	// annotation of existing work, not new work.
	OpComment
)

// String renders an OpClass for operator-facing output and reasons.
func (o OpClass) String() string {
	switch o {
	case OpSpawn:
		return "spawn"
	case OpWitnessClose:
		return "witness-close"
	case OpStatusRefresh:
		return "status-refresh"
	case OpComment:
		return "comment"
	default:
		return fmt.Sprintf("opclass(%d)", int(o))
	}
}

// holdsSpawn reports whether an operation class is the spawn class that a
// freeze holds. Kept as one predicate so the contract lives in a single place:
// only the spawn class is ever held, everything else is progress harvesting.
func (o OpClass) holdsSpawn() bool { return o == OpSpawn }

// State is the operator freeze state. The zero value is the running (not
// frozen) fleet, so an un-set gate allows everything. Build a frozen state
// with [Freeze] and clear it with [Unfreeze].
type State struct {
	// Frozen is true when new spawns are held.
	Frozen bool
	// Reason is the operator-supplied justification, surfaced in held-spawn
	// decisions and dry-run status. Empty when not frozen.
	Reason string
	// SinceUnix is the Unix second at which the freeze took effect, taken as
	// input (not read from the clock) to keep the package deterministic. Zero
	// when not frozen.
	SinceUnix int64
}

// Freeze returns a frozen State recorded at nowUnix (a caller-supplied Unix
// second, never the clock) with the given operator reason. A blank reason is
// backfilled so held-spawn decisions always carry an actionable explanation.
func Freeze(reason string, nowUnix int64) State {
	if reason == "" {
		reason = "operator freeze (no reason given)"
	}
	return State{Frozen: true, Reason: reason, SinceUnix: nowUnix}
}

// Unfreeze returns the running State — the zero value — in which every
// operation class is allowed.
func Unfreeze() State { return State{} }

// Decision is the verdict for one operation class against a State: whether it
// may proceed and, when held, an actionable reason naming the freeze.
type Decision struct {
	// Allow reports whether the operation may proceed.
	Allow bool
	// Reason explains a held decision (Allow == false) in operator terms. It
	// is empty when Allow is true.
	Reason string
}

// Allowed applies the freeze contract to one operation class.
//
// When state is not frozen, every class is allowed. When frozen, OpSpawn is
// HELD with an actionable reason naming the freeze and its since-time, while
// OpWitnessClose, OpStatusRefresh, and OpComment remain ALLOWED so progress
// harvesting continues uninterrupted.
func Allowed(state State, op OpClass) Decision {
	if !state.Frozen || !op.holdsSpawn() {
		return Decision{Allow: true}
	}
	return Decision{
		Allow: false,
		Reason: fmt.Sprintf(
			"%s held: fleet frozen since %s (%s); witness-close and status-refresh still allowed",
			op, formatSince(state.SinceUnix), state.Reason,
		),
	}
}

// formatSince renders a freeze instant for operator output. A zero timestamp
// (a freeze recorded without a clock) is reported as "unknown time" rather
// than the Unix epoch, which would mislead an operator.
func formatSince(unix int64) string {
	if unix == 0 {
		return "unknown time"
	}
	return time.Unix(unix, 0).UTC().Format(time.RFC3339)
}

// String renders the freeze state for an operator dry-run status line: whether
// the fleet is frozen, the reason, the since-time, and — the reassurance a
// freeze must give — that progress harvesting continues.
func (s State) String() string {
	if !s.Frozen {
		return "fleet-freeze: RUNNING (spawns allowed; all operation classes allowed)"
	}
	return fmt.Sprintf(
		"fleet-freeze: FROZEN since %s — %s; new spawns HELD, witness-close + status-refresh still allowed",
		formatSince(s.SinceUnix), s.Reason,
	)
}
