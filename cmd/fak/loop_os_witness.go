package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// The OS-scheduler rung for `fak loop health` (#4989).
//
// The ledger plane derives a loop's liveness from ONE input: the loop's
// last-appended ledger row vs its cadence. A loop whose OS scheduled task fires
// successfully on cadence but appends no ledger row on that tick therefore reads
// DARK — a false-DARK, since the loop is demonstrably alive at the OS layer.
// `fak schedscan` already decodes exactly the missing signal (LastTaskResult /
// LastRunTime per fleet task), but the two planes were never joined: neither
// loopmgr nor loopfleet imported the OS-task reader.
//
// This file is that join, and it lives HERE (cmd/fak) rather than in loopmgr on
// purpose. loopmgr's fold is pure — no clock, no I/O, cross-platform — so dragging a
// Windows PowerShell probe into it would cost exactly the property that makes it
// testable. Instead the platform read stays at the command layer, which decodes the
// witness with schedscan's own table (reused, never forked, so the 0x0 decode cannot
// drift between the two surfaces) and hands loopmgr pure data.

// gardenTickTaskLabel is the Windows scheduled task that drives the garden loop.
// Per the documented cron emit (`fak cron emit --target taskscheduler --label
// FleetStaleWorkGarden --command 'fak garden --check' --interval 1h`, cron.go) it
// runs `fak garden --check` — the READ-ONLY CI-gate fold. The ledger row for
// gardenTickLoopID is written only by witnessGardenTick, which is reached only from
// the acting `fak garden tick` path, never from `--check`. So this task fires 0x0
// hourly while its loop's ledger goes hours stale: the named false-DARK this rung
// exists to stop mislabeling.
const gardenTickTaskLabel = "FleetStaleWorkGarden"

// loopOSTaskMap is the EXPLICIT loop-id <-> OS-task-label map the reconciliation is
// keyed by. Explicit is the contract, not a convenience: a task with no mapped loop
// is out of scope, and a loop with no mapped task is never touched by the OS rung.
// Nothing is inferred from name similarity — an accidental match between an
// unrelated Fleet* task and a loop id would fabricate liveness, which is the one
// failure this rung must never have.
func loopOSTaskMap() map[string]string {
	return map[string]string{
		gardenTickLoopID: gardenTickTaskLabel,
	}
}

// loopOSWitnesses joins decoded schedscan rows onto loop ids through taskFor,
// producing the pure witness map loopmgr.FoldHealthWithOS consumes.
//
// Fail-closed at every step, since a wrong witness here silently promotes a genuinely
// dead loop out of DARK:
//   - a mapped task absent from the scan yields NO entry (the loop stays as the
//     ledger derived it);
//   - Fired is set only for a result of exactly 0x0. It deliberately does NOT use
//     schedscan's "ok" severity, which also covers 0x41300 ("the task is ready to run
//     at its next scheduled time") — a *status*, not evidence that a run happened;
//   - an unparseable/absent LastRunTime leaves LastRunUnixNano 0, which loopmgr reads
//     as "cannot place on the freshness timeline" and refuses to corroborate.
func loopOSWitnesses(rows []schedScanTaskInfo, taskFor map[string]string) map[string]loopmgr.OSTaskInfo {
	byName := make(map[string]schedScanTaskInfo, len(rows))
	for _, r := range rows {
		byName[strings.TrimSpace(r.TaskName)] = r
	}
	out := make(map[string]loopmgr.OSTaskInfo, len(taskFor))
	for loopID, label := range taskFor {
		row, ok := byName[label]
		if !ok {
			continue
		}
		lastRun, _ := parseSchedRunTime(row.LastRunTime)
		out[loopID] = loopmgr.OSTaskInfo{
			TaskLabel:       label,
			Fired:           decodeSchedTaskResult(row.LastTaskResult).Code == 0,
			LastRunUnixNano: lastRun,
		}
	}
	return out
}

// parseSchedRunTime parses the ISO-8601 LastRunTime the schedscan probe emits (it
// stringifies the DateTime with .ToString('o') precisely so this stays RFC3339 and
// never the ambiguous \/Date(...)\/ form). Unreadable -> 0/false, which the caller
// carries through as "no usable last-run time" rather than a zero timestamp that
// would read as 1970.
func parseSchedRunTime(s string) (int64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return 0, false
	}
	return t.UTC().UnixNano(), true
}

// loopOSTaskRows acquires the raw scheduled-task rows the witness map is built from,
// from a captured snapshot (--sched-from, works on any OS and is what the regression
// test drives) or a live Windows query (--os-tasks).
//
// The OS rung is OPT-IN. The live probe enumerates every task on the box through
// PowerShell — seconds of latency — and `fak loop health` is a read-only fold run in
// loops and gates, so making it unconditional would trade a rendering fix for a
// GATE_LATENCY_REGRESSION. Default-off also means the rung cannot change any existing
// caller's verdict.
//
// An acquisition failure is surfaced and degraded, never fatal: the health fold still
// renders from the ledger plane, and every loop simply keeps the verdict it already
// had. Losing the OS witness can only cost the re-description, never invent one.
func loopOSTaskRows(stderr io.Writer, schedFrom string, live bool) []schedScanTaskInfo {
	var raw string
	switch {
	case schedFrom != "":
		b, err := os.ReadFile(schedFrom)
		if err != nil {
			fmt.Fprintf(stderr, "fak loop health: --sched-from: %v (OS-task rung skipped; loops keep their ledger verdict)\n", err)
			return nil
		}
		raw = string(b)
	case live:
		if runtime.GOOS != "windows" {
			fmt.Fprintln(stderr, "fak loop health: --os-tasks queries the Windows Task Scheduler; run on Windows or pass --sched-from <json> (OS-task rung skipped)")
			return nil
		}
		ctx, cancel := context.WithTimeout(ctx(), 60*time.Second)
		defer cancel()
		out, err := schedScanQueryLive(ctx, watchdogRunCommand)
		if err != nil && strings.TrimSpace(out) == "" {
			fmt.Fprintf(stderr, "fak loop health: --os-tasks live query failed: %v (OS-task rung skipped)\n", err)
			return nil
		}
		raw = out
	default:
		return nil
	}
	rows, err := parseSchedTaskJSON(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop health: parse scheduled-task JSON: %v (OS-task rung skipped)\n", err)
		return nil
	}
	return rows
}
