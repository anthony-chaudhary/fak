package architest

import (
	"path/filepath"
	"testing"
)

// synthToolGuardCallee is the decide-path function that binds a LIVE self-synthesized
// tool's exec to the agent's envelope (#543). decide.go calls it as
// `synthToolSelfModify(args, &a.authored, p.SelfModifyGlobs)` and denies SELF_MODIFY
// on a non-empty result. The exact name is the wiring this gate witnesses: a self-edit
// that deletes the call re-opens the synth->exec hole, and renaming the function is a
// visible, reviewable edit that must update this constant too.
const synthToolGuardCallee = "synthToolSelfModify"

// TestSynthToolGuardWiredInDecide is the cross-package half of the #543 floor — the
// self-synthesized-tool twin of TestShellSelfModifyGuardWiredInDecide. The hole it
// guards: fak adjudicates every NAMED tool call, but an agent can synthesize its own
// helper mid-run — write `helper.py` to the unguarded task workdir, then run
// `python helper.py internal/abi/x.go`. That exec carries no write verb, no inline-eval
// flag, and no `>` redirect, so commandWrites (and thus commandSelfModify) treats it as
// read-shaped and waves it through even when its command line reaches a guarded tree —
// laundering a self-edit past the floor through one opaque exec.
//
// decide.go closes that hole by ALSO calling synthToolSelfModify(args, &a.authored,
// p.SelfModifyGlobs) on the decision path: a command that execs a script the SAME agent
// authored earlier this run (the per-run `authored` ledger) is treated as write-shaped,
// so a synth-tool that reaches a guarded glob is denied SELF_MODIFY exactly as the
// equivalent direct write is.
//
// WHY A GATE (and why it BITES): nothing in the type system keeps that call on the
// decide path. A refactor — or a self-editing loop that does so deliberately — could
// delete the synthToolSelfModify call from decide.go and the kernel would compile and
// pass every other test while the synth->exec path silently regained the ability to
// launder a self-authored script's guarded-tree reach past the floor. The in-package
// behavioral floor (TestSynthToolEnvelopeBindsGuardedReach) lives INSIDE a witness tree
// (internal/adjudicator), so the same self-edit that drops the guard could drop the test
// that guards it — the exact circularity the witness-tree gates exist to break. This gate
// is the belt to that test's suspenders: it lives OUTSIDE adjudicator, re-derives the
// wiring from adjudicator's source, and fails if decide.go stops invoking the synth-tool
// guard.
//
// It is NOT a duplicate of TestShellSelfModifyGuardWiredInDecide / the inline-eval gate:
// those witness commandSelfModify and the interpreterEvalFlags floor (a WRITE-shaped
// command into a guarded tree); THIS one witnesses the synth->exec floor (an EXEC of an
// agent-authored script reaching a guarded tree), which fires where the write-verb floors
// do not. All three rungs are needed for the floor to bite the full laundering surface.
//
// Seeded GREEN: decide.go's Adjudicate method calls synthToolSelfModify. Deleting that
// call — re-opening the #543 hole — turns this gate RED from outside every guarded tree.
func TestSynthToolGuardWiredInDecide(t *testing.T) {
	internal := internalDir(t)
	adjDir := filepath.Join(internal, "adjudicator")

	const decideFn = "Adjudicate"
	if !bodyCallsFunc(t, adjDir, decideFn, synthToolGuardCallee) {
		t.Errorf("adjudicator.%s does not call %s — the #543 self-synthesized-tool guard is NOT "+
			"on the decision path. An agent that writes `helper.py` to the unguarded workdir then runs "+
			"`python helper.py internal/abi/x.go` carries no write verb / no inline-eval flag / no '>' "+
			"redirect, so commandSelfModify never fires for it; decide MUST also call "+
			"%s(args, &a.authored, p.SelfModifyGlobs) so a synth-tool exec reaching a guarded tree is "+
			"denied SELF_MODIFY. Without that call a self-improving loop can launder a self-edit past the "+
			"floor through one opaque exec. Re-add the call in decide.go (the in-package "+
			"TestSynthToolEnvelopeBindsGuardedReach is its behavioral twin), or, if %s was renamed, update "+
			"decideFn/synthToolGuardCallee in this gate.",
			decideFn, synthToolGuardCallee, synthToolGuardCallee, synthToolGuardCallee)
	}
}
