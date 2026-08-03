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
//
// Nonfatal was only HALF true until #5564, because "the node cannot reach
// origin" and "the node cannot FINISH reaching origin" are different failures
// and only the first one used to end. This file is where the second one is
// given an end: it is the single funnel every ambient sync goes through, so the
// convergence budget below is the one place that decides how much wall-clock a
// best-effort sync may take from a tick (see ambientLeaseRefSyncBudget).

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

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

// ambientLeaseRefSyncBudget bounds ONE ambient convergence call — the WHOLE
// plan, not one step (#5564). Before it existed this file handed store.Sync a
// context.Background(); Sync shells out to a real `git push` / `git fetch`, and
// internal/leaseref carries no deadline of its own, so an auth-stalled or
// black-holed remote pinned the loop-drive tick inside exec.Cmd.Run with
// nothing anywhere below it able to end the wait.
//
// WHY THE WHOLE PLAN, NOT ONE STEP PER BUDGET. LeaseRefSyncPlanForSurface
// returns up to two steps (fetch-before-decide, push-after-write) and
// loopDriveRegionHold.ensure runs this boundary twice a turn, so a per-step
// bound is not a bound on the tick: it multiplies by the plan's length, and it
// would grow silently the day a surface gains a third step. One deadline shared
// by every step of one call means the boundary costs at most this much whatever
// the plan's shape. A step that inherits an already-spent context fails at once
// and is reported — the honest answer after the convergence budget is gone.
//
// WHY THIS NUMBER. The repo already bounds this exact operation once: a single
// store.Sync push of refs/fak/locks/* to origin is bounded by leasePublishTimeout
// (30s, cmd/fak/leasewrite_endpoint.go), whose own doc records why it may be
// that wide — the publish runs on its own goroutine and holds no caller waiting.
// This call does hold a caller waiting; the tick blocks on it before it may
// decide and again after it writes. So the ambient budget is deliberately
// tighter than its asynchronous sibling's, while staying several times over the
// couple of seconds a healthy push of a handful of tiny blob refs costs. It is
// a CONVERGENCE bound, not a correctness one: the surface it guards is
// explicitly nonfatal, so spending the budget costs cross-node visibility until
// the next tick and never a wrong verdict — which is what makes erring on the
// short side the safe direction here.
//
// A var rather than a const ONLY so a test can shrink it and witness the
// give-up path without sleeping. Nothing in production assigns it.
var ambientLeaseRefSyncBudget = 10 * time.Second

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
	// ONE deadline for the whole plan, so the tick's exposure is this budget and
	// not this budget times however many steps the surface plans.
	ctx, cancel := context.WithTimeout(context.Background(), ambientLeaseRefSyncBudget)
	defer cancel()
	attempts := make([]loopdrive.LeaseRefSyncAttempt, 0, len(steps))
	for _, step := range steps {
		push, fetch, err := loopdrive.LeaseRefSyncDirections(step)
		if err != nil {
			attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step, Err: err.Error()})
			continue
		}
		if _, err := store.Sync(ctx, step.Remote, push, fetch); err != nil {
			attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step, Err: ambientLeaseRefSyncStepErr(ctx, err)})
			continue
		}
		attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step})
	}
	return loopdrive.ReportLeaseRefSync(attempts)
}

// ambientLeaseRefSyncStepErr names the one failure the raw leaseref error cannot
// name for itself: a spent convergence budget. leaseref reports transport
// outcomes in git's own terms, and git does not know it was killed by a
// deadline — a cancelled push surfaces as a plain non-zero exit, and a step that
// starts after the budget is already gone surfaces as "git not executable". Both
// would blame the remote for a deadline THIS file set, so the ledger reader
// would chase a network that was fine. Every other error crosses unchanged; the
// underlying text is kept either way, since it still says which direction and
// which remote failed.
func ambientLeaseRefSyncStepErr(ctx context.Context, err error) string {
	if ctx.Err() != nil {
		return fmt.Sprintf("lease-ref sync budget %s spent: %v", ambientLeaseRefSyncBudget, err)
	}
	return err.Error()
}

// syncLoopDriveTickLeaseRefs runs the loop-drive-tick sync boundary against the
// hold's store. written is false before the decide read (fetch-only: converge
// peer lease refs so admission sees them) and true after the hold acquired or
// renewed its lease ref (push-only: publish the local write).
func syncLoopDriveTickLeaseRefs(store *leaseref.Store, written bool) loopdrive.LeaseRefSyncReport {
	return ambientLeaseRefSync(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, ambientLeaseRefSyncRemote(), written)
}
