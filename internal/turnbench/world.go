package turnbench

// world.go adds the WORLD-PLUGGABLE replay entry point. RunWithCalls (turnbench.go)
// installs the frozen AIRLINE tool world (agent.Configure) — the world the
// turntax/guard demos replay against. A demo that wants a DIFFERENT tool world (a
// coding-agent file world, say, where read_file is allow-listed and cacheable while
// write_file / delete_path / run_shell fall to the structural DEFAULT_DENY floor)
// passes its OWN installer here instead. Everything downstream — the real-kernel
// replay, the live-verdict classification, the safety floor, the consistency check —
// is world-AGNOSTIC, so a new world reuses the exact same grounded machinery (no
// second classifier to drift from classify()).

import (
	"context"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// RunWithWorld is RunWithCalls generalized over the tool world: `configure` installs
// the policy / grammar / engine the trace's tools are adjudicated against, replacing
// the built-in agent.Configure() airline world. A nil configure replays against
// whatever world is already installed in the process-global drivers.
//
// The two real replays (vDSO on, then the on/off ablation) and the per-call
// dispositions are derived EXACTLY as RunWithCalls derives them — RunWithCalls is now
// literally RunWithWorld(ctx, t, cm, agent.Configure) — so a custom-world report is
// the same artifact shape, with the same single-source-of-truth classification, just
// adjudicated against a different floor. The replay isolates the per-arm cross-call
// state (vDSO tier-2 cache + IFC taint ledger), so back-to-back runs against
// different worlds in one process stay reproducible.
func RunWithWorld(ctx context.Context, t *Trace, cm CostModel, configure func()) (*Report, []CallDisposition, error) {
	cm = withCostModelVersion(cm)
	// Install the tool world (idempotent) so the trace's tools trigger the REAL
	// rungs: read-only+idempotent reads -> vDSO tier-2 dedup, write-shaped /
	// unsanctioned tools -> capability-floor deny, etc.
	if configure != nil {
		configure()
	}

	// The on-arm calibrates (its p50 is the reported 1-shot serve cost) and collects
	// the per-call dispositions; the off-arm never needs either, so it skips both.
	on, onClass, onSafety, localNs, disp, err := replay(ctx, t, true, true, true)
	if err != nil {
		return nil, nil, err
	}

	_, offClass, _, _, _, err := replay(ctx, t, false, false, false)
	if err != nil {
		return nil, nil, err
	}

	rep := &Report{
		Provenance: Provenance{
			AppVersion:   appversion.Current(),
			Command:      "fak turntax --suite " + t.SliceID,
			SliceID:      t.SliceID,
			WorkloadHash: t.WorkloadHash(),
			GoVersion:    runtime.Version(),
			OS:           runtime.GOOS,
			GeneratedBy:  "fak/internal/turnbench",
		},
		Calls:        len(t.Calls),
		Cost:         cm,
		Class:        onClass,
		Counters:     on,
		LocalServeNs: localNs,
		Net:          netFor(onClass.turnsSaved(), cm),
		TurnKinds:    TurnKinds{Forced: onClass.forcedTurns(), Elision: onClass.elisionTurns()},
		VDSOOffNet:   netFor(offClass.turnsSaved(), cm),
		Safety: SafetyFloor{
			InjectionsAdmittedBaseline:  onSafety.poison,
			InjectionsAdmittedFak:       0,
			DestructiveExecutedBaseline: onSafety.destructiveExecuted,
			DestructiveExecutedFak:      0,
		},
		Levers:      levers(onClass),
		Sensitivity: sensitivity(onClass.turnsSaved(), cm),
	}
	rep.ConsistencyCheck = consistencyCheck(on, onClass)
	return rep, disp, nil
}
