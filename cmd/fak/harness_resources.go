package main

import (
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

// appendHarnessResources writes one durable row of the guard session's harness
// hardware-resource use (CPU/mem/IO for the kernel + agent halves) to
// .fak/nightrun/harness-resources.jsonl — the sibling of the cache-savings ledger
// (appendObservedCacheSavings). Best-effort: a write failure never fails the session,
// and an empty snapshot (resource-stats off) is never reached because the caller gates
// on the sampler being present. Epic #2044 / #2046.
func appendHarnessResources(mode, provider, agent string, snap harnessres.Snapshot) {
	_ = appendHarnessResourcesTo(nightrunLedgerPath(harnessres.DefaultLedgerRel), mode, provider, agent, snap, time.Now())
}

// appendHarnessResourcesTo is the testable core: it renders the row and appends it to
// path, creating the parent directory if needed.
func appendHarnessResourcesTo(path, mode, provider, agent string, snap harnessres.Snapshot, now time.Time) error {
	line, err := snap.MarshalLedgerRow(mode, provider, agent, now)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(line, '\n'))
	return err
}

// foldGuardStopCounts folds the session's guard-stops ledger into the operator-question
// counts that ride on the harness-resources row (operator-directed asks + fail-open
// stops). It reads the SAME ledger `fak guard-stops` folds, so the harness tick and the
// guard-stops tally agree. The counts are the ledger-to-date totals observed at this
// tick (a gauge, not a per-session delta); a fleet consumer diffs consecutive ticks.
//
// Zero-safe and fail-open (#4348): no ledger wired, an unreadable file, or an empty
// ledger all yield an explicit zero — never a gap — so every harness row states how
// often the harness stopped to ask a human, even when that answer is "never".
func foldGuardStopCounts() harnessres.GuardStopCounts {
	return guardStopCountsFrom(guardStopsLedgerResolved())
}

// guardStopCountsFrom is the testable core: fold the guard-stops ledger at path into the
// harness-row counts. An empty path or a read error is a zero reading, not a failure.
func guardStopCountsFrom(ledger string) harnessres.GuardStopCounts {
	if strings.TrimSpace(ledger) == "" {
		return harnessres.GuardStopCounts{}
	}
	content, err := readGuardStopsLedger(ledger)
	if err != nil {
		return harnessres.GuardStopCounts{}
	}
	sum := summarizeGuardStops(content, 0)
	return harnessres.GuardStopCounts{
		OperatorDirected: sum.OperatorDirected,
		FailOpen:         sum.FailOpen,
	}
}
