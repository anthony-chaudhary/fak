package main

// leaseref_sync_ambient.go is the #2302 wiring: make the `fak leaseref sync`
// convergence AMBIENT at the loop-drive tick boundary, so cross-node lease
// visibility no longer depends on an operator remembering a refspec. The
// boundary MODEL lives in internal/loopdrive (LeaseRefSyncPlanForSurface —
// fetch-only before the decide read, push-only after the tick wrote a lease
// ref, both advisory); this file is the shell side that runs that plan against
// a real leaseref.Store.
//
// The dispatch tick's lane-lease acquire (acquireDispatchLaneLease) is the
// other #2302 surface; it shares this helper by passing the
// LeaseRefSyncSurfaceDispatchPreflight surface. It is wired in its own commit
// — the boundary is identical to the loop-drive one wired here.
//
// Sync is NONFATAL by design (internal/loopdrive.ReportLeaseRefSync): a node
// that cannot reach origin — single-box dev, air-gapped CI, a clone with no
// remote configured — continues on local lease evidence, the same fail-open
// posture as the dispatch tick's existing acquire. Transport failures fold
// into a LeaseRefSyncReport a caller may surface but never act on as a verdict.

import (
	"context"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
)

// ambientLeaseRefSync is the test seam: tests swap it to a recorder to witness
// that a tick ran the sync plan at the right boundary without driving real git.
// The default runs the real plan against the store.
var ambientLeaseRefSync = ambientLeaseRefSyncImpl

// ambientLeaseRefSyncRemote resolves the remote sync converges with. Defaults
// to "origin" (the protocol in docs/notes/MULTINODE-DEV-GROUNDWORK-2026-07-02.md);
// FAK_LEASEREF_REMOTE overrides it for setups whose canonical remote is named
// otherwise.
func ambientLeaseRefSyncRemote() string {
	if v := strings.TrimSpace(os.Getenv("FAK_LEASEREF_REMOTE")); v != "" {
		return v
	}
	return "origin"
}

// ambientLeaseRefSyncEnabled reports whether ambient sync is on. It is ON by
// default — #2302 makes convergence ambient, not operator muscle memory;
// FAK_LEASEREF_SYNC=off (also 0/false/no/disable) disables it for air-gapped
// runs or tests that want zero git transport.
func ambientLeaseRefSyncEnabled() bool {
	raw, ok := os.LookupEnv("FAK_LEASEREF_SYNC")
	if !ok {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "1", "on", "true", "yes", "enable", "enabled":
		return true
	}
	return false
}

// ambientLeaseRefSyncImpl runs the surface's sync plan against store and folds
// the per-step transport results into one advisory report. It never turns a
// transport failure into a fatal verdict — the caller proceeds on local lease
// evidence regardless.
func ambientLeaseRefSyncImpl(surface loopdrive.LeaseRefSyncSurface, store *leaseref.Store, remote string, written bool) loopdrive.LeaseRefSyncReport {
	if !ambientLeaseRefSyncEnabled() {
		return loopdrive.LeaseRefSyncReport{Outcome: loopdrive.LeaseRefSyncOK, Summary: "lease-ref sync disabled (FAK_LEASEREF_SYNC=off)"}
	}
	if strings.TrimSpace(remote) == "" {
		remote = ambientLeaseRefSyncRemote()
	}
	steps := loopdrive.LeaseRefSyncPlanForSurface(surface, loopdrive.LeaseRefSyncPlanInput{Remote: remote, LeaseRefsWritten: written})
	attempts := make([]loopdrive.LeaseRefSyncAttempt, 0, len(steps))
	for _, step := range steps {
		push, fetch, err := loopdrive.LeaseRefSyncDirections(step)
		if err != nil {
			attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step, Err: err.Error()})
			continue
		}
		if _, err := store.Sync(context.Background(), step.Remote, push, fetch); err != nil {
			attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step, Err: err.Error()})
			continue
		}
		attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step})
	}
	return loopdrive.ReportLeaseRefSync(attempts)
}

// syncLoopDriveTickLeaseRefs runs the loop-drive-tick sync boundary against the
// hold's store. written is false before the decide read (fetch-only: converge
// peer lease refs so admission sees them) and true after the hold acquired or
// renewed its lease ref (push-only: publish the local write).
func syncLoopDriveTickLeaseRefs(store *leaseref.Store, written bool) loopdrive.LeaseRefSyncReport {
	return ambientLeaseRefSync(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, ambientLeaseRefSyncRemote(), written)
}
