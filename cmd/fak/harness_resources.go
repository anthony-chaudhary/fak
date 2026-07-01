package main

import (
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

// appendHarnessResources writes one durable row of the guard session's harness
// hardware-resource use (CPU/mem/IO for the kernel + agent halves) to
// docs/nightrun/harness-resources.jsonl — the sibling of the cache-savings ledger
// (appendObservedCacheSavings). Best-effort: a write failure never fails the session,
// and an empty snapshot (resource-stats off) is never reached because the caller gates
// on the sampler being present. Epic #2044 / #2046.
func appendHarnessResources(mode, provider, agent string, snap harnessres.Snapshot) {
	_ = appendHarnessResourcesTo(harnessres.DefaultLedgerRel, mode, provider, agent, snap, time.Now())
}

// appendHarnessResourcesTo is the testable core: it renders the row and appends it to
// path, creating the parent directory if needed (a guard run from a fresh cwd may not
// have docs/nightrun yet).
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
