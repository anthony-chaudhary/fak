// guard_core_lock.go supplies the admission decision for the `--core-lock-all`
// launch mode (#5175, epic #5170, Track C): the strongest amendment posture, in
// which EVERY knob behaves as a RATCHET — no channel (operator overlay, live
// reload, or self-authored proposal) may widen the floor for the life of the
// session, and only tighten-only amendments are admitted. By design it is the
// launch-wide generalization of the per-overlay self-tighten gate (#5181,
// cmd/fak/guard_self_tighten.go): where that gate governs one self-authored
// overlay, this posture would clamp a whole session, so an operator could start
// a run in a mode where the floor only ever gets stricter.
//
// THE POSTURE IS NOW ARMED (#5423). It shipped as the DECISION half only — the
// guard entrypoint declared its flags on a flag.NewFlagSet (cmd/fak/guard.go)
// that never saw this file, nothing set the session posture, and nothing outside
// this file's own tests called either helper below, so no live session was ever
// clamped. Both ends of the wiring now exist:
//
//   - LAUNCH. cmdGuard calls guardLaunchCoreLockAll on the raw launch argv,
//     BEFORE the FlagSet parse (the FlagSet is flag.ExitOnError, so an
//     unregistered flag reaching it would abort the launch), and records the
//     posture in guardCoreLockAllState.
//   - AMENDMENT. Every live floor-amendment site in this binary consults
//     guardCoreLockAllAdmitAmendment before installing a proposed floor:
//     applyPolicyRuntimeLocked (cmd/fak/policy_reload_widen.go, the --policy FILE
//     reload) and guardReloadDefaultFloor (cmd/fak/guard_startup.go, the
//     built-in-floor reload the allow watcher and POST /v1/fak/policy/reload
//     drive on an ordinary `fak guard -- claude`). A refusal is NAMED on stderr
//     and recorded as a rejected CONFIG_SWAP journal row, and the proposed floor
//     is never installed.
//
// The LAUNCH-boundary floor install (loadGuardCapabilityFloor) is deliberately
// NOT gated: it is the assembly of the floor this posture then clamps, not an
// amendment to it. Refusing there would refuse the session's own first floor.
//
// Note what this posture adds ON TOP of the existing gates rather than
// duplicating: the --policy reload path already refuses a widening, but that
// refusal is escapable with FAK_POLICY_RELOAD_ALLOW_WIDEN=1 and is skipped
// entirely on the non-enforcing call, and the built-in-floor reload had no
// widening gate at all. Under core-lock-all neither escape exists — the session
// is ratchet-tighten-only for its whole life, which is the property an operator
// buys by passing the flag.
package main

import (
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guardCoreLockAllState is the session-wide posture, set once at guard launch
// and read from every amendment site. It is atomic rather than a plain bool
// because the reload paths that read it run on gateway HTTP handler goroutines
// and on the allow-watcher goroutine, while the write happens on the launch
// goroutine — a plain bool would be a data race the detector reports even though
// the write provably precedes every read in wall-clock terms.
var guardCoreLockAllState atomic.Bool

// setGuardCoreLockAll records the launch posture. Called exactly once, from
// cmdGuard, before the gateway binds or any reload path can run.
func setGuardCoreLockAll(active bool) { guardCoreLockAllState.Store(active) }

// guardCoreLockAllActive reports whether THIS process was launched with
// --core-lock-all. False for every non-guard command (`fak serve`, `fak policy
// load`), which is why those paths keep their pre-#5423 behaviour byte for byte.
func guardCoreLockAllActive() bool { return guardCoreLockAllState.Load() }

// guardLaunchCoreLockAll peels --core-lock-all off the GUARD side of a launch
// argv and reports the posture. It splits at the first bare `--` and strips only
// the flags before it, so a wrapped child that happens to take a flag of the same
// name still receives its own argv byte for byte — `fak guard --core-lock-all --
// claude --core-lock-all` clamps the session and passes the second occurrence
// through untouched. With no `--` the whole argv is guard-side, which is the
// help/`--dump-policy` shape.
func guardLaunchCoreLockAll(argv []string) (active bool, rest []string) {
	for i, a := range argv {
		if a == "--" {
			active, head := stripCoreLockAllFlag(argv[:i])
			return active, append(head, argv[i:]...)
		}
	}
	return stripCoreLockAllFlag(argv)
}

// guardCoreLockAllAdmitAmendment is the seam the live amendment sites call: it
// reads THIS session's posture and routes the proposed floor through the verdict
// below. Split out from coreLockAllReloadVerdict so the verdict stays a pure
// function of its arguments (directly unit-testable with no process state) while
// the callers do not each have to remember to read the posture var.
func guardCoreLockAllAdmitAmendment(current, proposed adjudicator.Policy) (admit bool, reason string) {
	return coreLockAllReloadVerdict(guardCoreLockAllActive(), current, proposed)
}

// stripCoreLockAllFlag extracts the --core-lock-all boolean flag from a launch
// argv, returning whether it was present and the remaining args with the flag
// removed. It peels the flag out of argv rather than declaring it on a FlagSet,
// the shape the non-guard precedents in this tree use (stripNoReuseFlag in
// cmd/fak/benchloop.go, stripFlags in cmd/fak/debug.go). The guard entrypoint
// uses a FlagSet, so the peel has to happen before fs.Parse sees the argv;
// guardLaunchCoreLockAll above is the `--`-aware wrapper cmdGuard actually calls.
// Pure and testable: it takes an argv and returns one, touching no state.
func stripCoreLockAllFlag(args []string) (coreLock bool, rest []string) {
	rest = make([]string, 0, len(args))
	for _, a := range args {
		if a == "--core-lock-all" {
			coreLock = true
			continue
		}
		rest = append(rest, a)
	}
	return coreLock, rest
}

// coreLockAllReloadVerdict decides whether one reload/overlay amendment is
// admissible under core-lock-all, reading the posture from its `active` argument
// rather than from any process-wide state. It admits only a tighten-only or
// no-op delta from the current effective floor to the proposed one; any
// widening, or any movement of a FROZEN floor element, is refused regardless of
// which channel proposed it. When core-lock-all is NOT active it admits
// unconditionally (normal per-channel gating applies elsewhere), which is what
// lets the reload paths call it on EVERY amendment: an ordinary launch takes the
// inactive branch and behaves exactly as it did before #5423. The classes come from the canonical
// policy.DiffAmendment engine, so the verdict means exactly what the Class()
// fold in internal/policy/amendment_delta.go says it means. As in the #5181
// sibling, the AmendmentFrozenViolation branch is unreachable by construction
// today (DiffAmendment emits only registered, non-FROZEN field names; the FROZEN
// knobs carry Field:"" and are compiled-in only), so it is fail-closed insurance
// against a future edit rather than exercised logic.
func coreLockAllReloadVerdict(active bool, current, proposed adjudicator.Policy) (admit bool, reason string) {
	if !active {
		return true, "core-lock-all inactive: normal amendment gating applies"
	}
	switch policy.DiffAmendment(current, proposed).Class() {
	case policy.AmendmentNone:
		return true, "core-lock-all: no-op amendment admitted"
	case policy.AmendmentTighten:
		return true, "core-lock-all: tighten-only amendment admitted"
	case policy.AmendmentWiden:
		return false, "core-lock-all: WIDENING refused — session is ratchet-tighten-only"
	case policy.AmendmentFrozenViolation:
		return false, "core-lock-all: FROZEN-floor movement refused"
	default:
		return false, "core-lock-all: unclassified amendment refused (fail-closed)"
	}
}
