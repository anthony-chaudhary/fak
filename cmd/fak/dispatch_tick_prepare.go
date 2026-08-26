package main

import (
	"io"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

type dispatchTickPreparation struct {
	root, runsDir       string
	started, spawnStart time.Time
	timings             map[string]int64
	registry            map[string]any
	witnessedSlots      map[string]any
	witnessRecords      []dispatchtick.WitnessRecord
	freshWitnessRecords []dispatchtick.WitnessRecord
	heldNoCommit        map[int]bool
	recoverableNoCommit map[int]bool
	preflight           map[string]any
	account             dispatchtick.Account
	terminal            map[string]any
}

func prepareDispatchTickEvaluation(opts *dispatchTickOptions, stderr io.Writer) (dispatchTickPreparation, error) {
	root, err := filepath.Abs(opts.Workspace)
	if err != nil {
		return dispatchTickPreparation{}, err
	}
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if opts.WorkerTimeoutS <= 0 {
		opts.WorkerTimeoutS = dispatchtick.DefaultWorkerTimeoutS
	}
	// #1411: scope this tick's native issue routing to the named view. The CLI
	// flag defaults View to `current`; a programmatic tick (wave/sweep/garden)
	// that leaves View empty keeps today's full-backlog behavior.
	dispatchTickView = opts.View
	// #5416 tracks D/E/F: publish this tick's placement settings to the package seams the
	// placement, escalation, sidecar, and journal helpers read. They are DECLARED on the tick's
	// own config surface (--placement-evidence / --rung-placement / --accounts-roster) rather
	// than read out of the process environment, which is the CONFIG_NOT_ENV rule
	// internal/envconfiglint ratchets: an env var is for a secret, a behavioral setting belongs
	// where `--help` can name it and a caller can set it per invocation. Written here, beside
	// dispatchTickView and for the same reason — the readers hang off several call chains whose
	// many stubs keep their signature. A programmatic tick (wave / sweep / garden) that leaves
	// them zero gets the pre-seam posture: both switches off, no roster override.
	dispatchPlacementEvidence = opts.PlacementEvidence
	dispatchRungPlacement = opts.RungPlacement
	dispatchAccountsRoster = opts.AccountsRoster

	// #3405: publish this tick's host-probe shell-reuse setting into the probe seam BEFORE
	// preflight, which is what actually runs the probes. Without this call the whole reuse
	// spine is inert code -- landed, tested, and never reached -- so every Windows tick keeps
	// paying one `powershell` process, and one ConPTY/conhost, per probe (#3153). The arm
	// hands back the teardown, deferred here rather than folded into `finish` so that the
	// early error returns above the funnel cannot leak a warm console either: no console
	// outlives the tick that opened it.
	// Per-phase wall-clock attribution (observability): a slow tick is otherwise a
	// black box -- the dominant cost (the ~40s fleet_sessions.py registry scan) and
	// the per-tick subprocess fan-out (preflight PowerShell probes, router gh/dos
	// spawns) were only ever asserted in comments, never witnessed. `timings` holds
	// int64-millisecond durations, keyed per phase; a phase that did not run this
	// tick simply has no key (mirroring the omitempty provenance sub-maps). It is
	// attached as payload["timings_ms"] and stamped with `total` inside `finish`
	// (the single funnel every verdict return passes through), and folded into the
	// loop-ledger metrics under *_ms names by recordDispatchTickLoop so cross-tick
	// per-phase percentiles become a later fold. Purely additive: no decision reads it.
	t0 := time.Now()
	timings := map[string]int64{}
	var spawnStart time.Time

	reg := map[string]any{"skipped": true}
	if opts.Refresh {
		tReg := time.Now()
		reg = dispatchRefreshRegistry(root, stderr)
		dispatchStampMs(timings, "registry_refresh", tReg)
	}

	// Commit-time diff-witness binding (#1324 proposal #2), ported from the Python
	// dispatcher: grade each finished (dead-pid) worker's slot through `dos
	// commit-audit` and record the verdict in a .witness sidecar, so a bare `exit 0`
	// never silently counts as productive. Live ticks only (the audit + the sidecar
	// write are the side effects, and a dry run must stay byte-identical); fail-open.
	// The re-blockable guard refusals it surfaces (self_modify / policy_block) feed
	// the pick's hold set below (#1396).
	durableWitnessRecords := readDurableDispatchWitnesses(runsDir)
	witnessedSlots := map[string]any{"skipped": true, "durable_records": len(durableWitnessRecords), "decision_records": len(durableWitnessRecords)}
	witnessRecords := append([]dispatchtick.WitnessRecord(nil), durableWitnessRecords...)
	var freshWitnessRecords []dispatchtick.WitnessRecord
	if opts.Live {
		tWitness := time.Now()
		witnessedSlots, freshWitnessRecords = witnessExitedWorkers(root, runsDir, true)
		witnessRecords = mergeDispatchWitnessRecords(durableWitnessRecords, freshWitnessRecords)
		witnessedSlots["durable_records"] = len(durableWitnessRecords)
		witnessedSlots["decision_records"] = len(witnessRecords)
		// Durable records participate in decisions but never duplicate evidence rows.
		if dispatchPlacementEvidenceEnabled() {
			if ev := appendDispatchTurnOutcomes(runsDir, freshWitnessRecords); len(ev) > 0 {
				witnessedSlots["turn_evidence"] = ev
			}
		}
		dispatchStampMs(timings, "witness", tWitness)
	}
	heldNoCommit := dispatchtick.HeldNoCommitIssues(witnessRecords)
	recoverableNoCommit := map[int]bool{}
	for issue := range dispatchtick.ModelDowngradeReDispatch(witnessRecords, workerDowngradeChain(opts.Backend)) {
		recoverableNoCommit[issue] = true
	}

	tPreflight := time.Now()
	pre, preflightTimings, err := dispatchPreflightTimed(root, stderr, opts.MaxWorkers, opts.WorkKind, dispatchtick.ProductForBackend(opts.Backend))
	if err != nil {
		return dispatchTickPreparation{}, err
	}
	dispatchStampMs(timings, "preflight", tPreflight)
	for name, ms := range preflightTimings {
		timings["preflight_"+name] = ms
	}
	// #6508: the gate must REFUSE, not merely warn, when the binary that adjudicated this
	// preflight is unreviewable or disagrees with the one that will front the worker it
	// admits. The decision is internal/selfinstall.ApplyGateSkew (tested there); this seam
	// only folds it into the payload, so it exits through the same !preOK path below.
	dispatchApplyBinSkew(pre, dispatchtick.PreflightOKVerdict)
	preOK := dispatchMapString(pre, "verdict") == dispatchtick.PreflightOKVerdict
	account := accountFromMap(mapAt(pre, "account"))
	if opts.Account != nil {
		account = *opts.Account
	}

	// A refused preflight cannot launch, so do not pay the issue router's GitHub/DOS
	// subprocess fan-out merely to decorate a terminal capacity verdict (#6762).
	// Keep this before resolveDispatchTickPick: lane/issue selection is actionable only
	// after capacity admission. The empty pick intentionally omits lane-pick evidence.
	if !preOK {
		payload := seedDispatchTickPayload(root, *opts, reg, pre, account, dispatchTickPick{})
		payload["ok"] = false
		payload["action"] = "refused"
		payload["verdict"] = firstString(dispatchMapString(pre, "verdict"), "REFUSE")
		payload["reason"] = "preflight refused: " + dispatchMapString(pre, "reason")
		if worklist, ok := pre["janitor_worklist"].([]int); ok && len(worklist) > 0 {
			payload["janitor_worklist"] = worklist
			if opts.Live {
				payload["janitor_reaped"] = dispatchReapWorklist(worklist)
			}
		}
		payload["timings_ms"] = timings
		timings["total"] = time.Since(t0).Milliseconds()
		if opts.RecordLoop {
			payload["loop_ledger"] = recordDispatchTickLoop(root, opts.LoopLedger, payload)
		}
		// Readiness is an observation ledger, not a launch ledger. Persist dry
		// terminal outcomes so a later readiness read does not report an erased
		// or stale blocker; recordDispatchPayload creates a launch sidecar only
		// when a real spawned payload carries a repo-pulse receipt.
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return dispatchTickPreparation{terminal: payload}, nil
	}

	return dispatchTickPreparation{
		root: root, runsDir: runsDir, started: t0, spawnStart: spawnStart, timings: timings,
		registry: reg, witnessedSlots: witnessedSlots, witnessRecords: witnessRecords,
		freshWitnessRecords: freshWitnessRecords, heldNoCommit: heldNoCommit,
		recoverableNoCommit: recoverableNoCommit, preflight: pre, account: account,
	}, nil
}
