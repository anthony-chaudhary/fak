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
// AND TODAY THERE IS NO LAUNCH MODE AND NO CALLER. This ships as the DECISION
// half only, exactly as its #5181 sibling does: no other Go file in the module
// names the flag, the guard entrypoint declares its flags on a flag.NewFlagSet
// (cmd/fak/guard.go) that never sees this file, and nothing outside this file's
// own tests calls either helper below. So no live session is clamped by this
// posture yet, and the paragraph above states a DESIGN relationship, not a
// wiring claim — naming a call site that does not exist is the failure mode this
// repo treats as an unwitnessed claim. Promotion evidence is a guard launch that
// actually routes an amendment through the verdict below and journals its class;
// the launch wiring is filed rather than half-done here.
package main

import (
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// stripCoreLockAllFlag extracts the --core-lock-all boolean flag from a launch
// argv, returning whether it was present and the remaining args with the flag
// removed. It peels the flag out of argv rather than declaring it on a FlagSet,
// the shape the non-guard precedents in this tree use (stripNoReuseFlag in
// cmd/fak/benchloop.go, stripFlags in cmd/fak/debug.go); the guard entrypoint
// itself uses a FlagSet today, so adopting this helper there is a wiring
// decision nobody has made yet. Pure and testable, and called only by the tests.
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
// unconditionally (normal per-channel gating applies elsewhere), so a reload
// path could afford to call it on every amendment — but no reload path calls it
// today; its only callers are the tests. The classes come from the canonical
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
