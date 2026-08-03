package main

// Witnesses for the #2302 ambient leaseref-sync wiring (leaseref_sync_ambient.go):
// the boundary runs the convergence plan, stays NONFATAL when origin is
// unreachable (the fail-open posture the dispatch tick already uses), honors the
// FAK_LEASEREF_SYNC disable and FAK_LEASEREF_REMOTE override, and the loop-drive
// region hold actually drives it at the before-decide / after-write boundaries.

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
)

func TestAmbientLeaseRefSyncDisabledByEnv(t *testing.T) {
	t.Setenv("FAK_LEASEREF_SYNC", "off")
	store := leaseref.NewInDir("")
	report := ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
	if report.Outcome != loopdrive.LeaseRefSyncOK {
		t.Fatalf("disabled sync outcome = %q, want ok", report.Outcome)
	}
	if report.Fatal {
		t.Fatalf("disabled sync must never be fatal: %+v", report)
	}
	if len(report.Failures) != 0 {
		t.Fatalf("disabled sync ran transport anyway: %+v", report.Failures)
	}
}

func TestAmbientLeaseRefSyncRemoteOverride(t *testing.T) {
	t.Setenv("FAK_LEASEREF_REMOTE", "upstream")
	if got := ambientLeaseRefSyncRemote(); got != "upstream" {
		t.Fatalf("remote = %q, want upstream", got)
	}
}

func TestAmbientLeaseRefSyncDefaultsOrigin(t *testing.T) {
	t.Setenv("FAK_LEASEREF_REMOTE", "")
	if got := ambientLeaseRefSyncRemote(); got != "origin" {
		t.Fatalf("default remote = %q, want origin", got)
	}
}

// TestAmbientLeaseRefSyncDegradesNonFatalWithoutOrigin: in a clone with no
// configured remote (single-box dev, CI) the convergence transport fails, but
// the boundary folds it to a NONFATAL degraded report — the caller must keep
// running on local lease evidence, never honest-stop on a dead link.
func TestAmbientLeaseRefSyncDegradesNonFatalWithoutOrigin(t *testing.T) {
	dir := initRegionTestRepo(t)
	store := leaseref.NewInDir(dir)

	report := ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
	if report.Outcome != loopdrive.LeaseRefSyncDegraded {
		t.Fatalf("outcome = %q, want degraded (no origin configured)", report.Outcome)
	}
	if report.Fatal {
		t.Fatalf("transport failure must be surfaced, not fatal: %+v", report)
	}
	if report.Reason != loopdrive.ReasonLeaseRefSyncTransport {
		t.Fatalf("reason = %q, want %q", report.Reason, loopdrive.ReasonLeaseRefSyncTransport)
	}
	if len(report.Failures) == 0 {
		t.Fatalf("expected surfaced failures, got none: %+v", report)
	}

	// written=false runs only the before-decide fetch step: one failure, still
	// nonfatal — admission proceeds on the local lease set.
	report = ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", false)
	if report.Outcome != loopdrive.LeaseRefSyncDegraded || report.Fatal {
		t.Fatalf("fetch-only degrade = %+v, want nonfatal degraded", report)
	}
}

// TestAmbientLeaseRefSyncGivesUpOnAStalledGitRunner is the #5564 witness: with a
// git that never returns — an auth-stalled or black-holed remote, simulated
// deterministically through internal/leaseref's injected-Runner seam rather than
// against a real network — the ambient boundary must GIVE UP and report, not
// block the tick that called it.
//
// It fails in two distinguishable ways on a regression, both of them loud. If
// the call site goes back to an unexpirable context the runner's <-ctx.Done()
// never fires and the select below fails the test in bounded time with the
// diagnosis, instead of wedging the package suite until the go-test timeout —
// which is exactly how this bug spent its career being misread as a flake. If
// the deadline survives but moves to a per-step budget, the second transport
// call arrives with a LIVE context and that assertion names it.
func TestAmbientLeaseRefSyncGivesUpOnAStalledGitRunner(t *testing.T) {
	restore := ambientLeaseRefSyncBudget
	ambientLeaseRefSyncBudget = 200 * time.Millisecond
	t.Cleanup(func() { ambientLeaseRefSyncBudget = restore })

	type transportCall struct {
		verb           string
		hasDeadline    bool
		expiredOnEntry bool
	}
	var calls []transportCall

	// The stall. Only the two transport verbs hang; the local preamble git calls
	// answer normally, so what is being witnessed is a wedged NETWORK git and not
	// a store that cannot read its own ref database.
	stalled := func(ctx context.Context, _ string, args ...string) (string, int, error) {
		verb := ""
		if len(args) != 0 {
			verb = args[0]
		}
		switch verb {
		case "push", "fetch":
			_, hasDeadline := ctx.Deadline()
			calls = append(calls, transportCall{verb: verb, hasDeadline: hasDeadline, expiredOnEntry: ctx.Err() != nil})
			<-ctx.Done() // returns ONLY when the context does — never, if it cannot expire
			return "", -1, ctx.Err()
		case "for-each-ref":
			// A non-empty local namespace, so the push direction is really attempted
			// rather than short-circuited as PushSkippedEmpty (leaseref/sync.go).
			return "refs/fak/locks/loop-region-loop\n", 0, nil
		default:
			return "", 0, nil
		}
	}
	store := leaseref.NewWithRunner(stalled, t.TempDir())

	// written=true plans BOTH steps (fetch-before-decide, push-after-write), which
	// is what makes the one-budget-per-call property observable.
	done := make(chan loopdrive.LeaseRefSyncReport, 1)
	go func() {
		done <- ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
	}()
	var report loopdrive.LeaseRefSyncReport
	select {
	case report = <-done:
	case <-time.After(30 * ambientLeaseRefSyncBudget):
		t.Fatal("ambient lease-ref sync never returned against a stalled git: the sync context cannot expire, so a slow remote hangs the tick (#5564)")
	}

	if report.Outcome != loopdrive.LeaseRefSyncDegraded {
		t.Fatalf("outcome = %q, want degraded — a give-up must be SURFACED, not swallowed: %+v", report.Outcome, report)
	}
	if report.Fatal {
		t.Fatalf("a spent convergence budget must stay advisory, not fatal: %+v", report)
	}
	if report.Reason != loopdrive.ReasonLeaseRefSyncTransport {
		t.Fatalf("reason = %q, want %q", report.Reason, loopdrive.ReasonLeaseRefSyncTransport)
	}
	if len(report.Failures) != 2 {
		t.Fatalf("failures = %d, want both planned steps reported: %+v", len(report.Failures), report.Failures)
	}
	for _, f := range report.Failures {
		if !strings.Contains(f, "budget") {
			t.Errorf("failure %q does not name the deadline as the cause — the ledger reader would blame the remote", f)
		}
	}

	if len(calls) != 2 {
		t.Fatalf("transport calls = %+v, want one fetch then one push", calls)
	}
	if calls[0].verb != "fetch" || calls[1].verb != "push" {
		t.Fatalf("transport order = %+v, want fetch-before-decide then push-after-write", calls)
	}
	if !calls[0].hasDeadline {
		t.Error("the first transport call carried NO deadline: the ambient sync is back on an unexpirable context (#5564)")
	}
	if calls[0].expiredOnEntry {
		t.Error("the first transport call started already expired: the budget is not being spent on real work")
	}
	if !calls[1].expiredOnEntry {
		t.Error("the second transport call got a fresh live context: the budget is per-STEP, so the tick's exposure still multiplies by the plan's length")
	}
}

// TestLoopDriveEnsureRunsAmbientSyncAtBoundaries: a fresh hold's ensure() must
// drive the ambient sync BOTH before the decide read (fetch-only, written=false)
// AND after the fenced acquire succeeds (push, written=true), at the loop-drive
// tick surface — the actual #2302 wiring, witnessed through the test seam.
func TestLoopDriveEnsureRunsAmbientSyncAtBoundaries(t *testing.T) {
	initRegionTestRepo(t)

	type rec struct {
		surface loopdrive.LeaseRefSyncSurface
		remote  string
		written bool
	}
	var got []rec
	orig := ambientLeaseRefSync
	ambientLeaseRefSync = func(surface loopdrive.LeaseRefSyncSurface, store *leaseref.Store, remote string, written bool) loopdrive.LeaseRefSyncReport {
		got = append(got, rec{surface, remote, written})
		return loopdrive.LeaseRefSyncReport{Outcome: loopdrive.LeaseRefSyncOK}
	}
	t.Cleanup(func() { ambientLeaseRefSync = orig })

	spec := loopdrive.Spec{Loop: "region-loop", Lane: "gateway"}
	hold := newLoopDriveRegionHold(loopDriveOptions{}, spec)
	defer hold.release()
	if refuse, err := hold.ensure(time.Now()); err != nil || refuse != nil {
		t.Fatalf("fresh acquire must hold: refuse=%+v err=%v", refuse, err)
	}
	if !hold.held {
		t.Fatal("hold must be held after a clean acquire")
	}

	var sawFetchBeforeDecide, sawPushAfterWrite bool
	for _, c := range got {
		if c.surface != loopdrive.LeaseRefSyncSurfaceLoopDriveTick {
			t.Fatalf("sync surface = %q, want loop_drive_tick", c.surface)
		}
		if c.remote != "origin" {
			t.Fatalf("remote = %q, want origin", c.remote)
		}
		if !c.written {
			sawFetchBeforeDecide = true
		} else {
			sawPushAfterWrite = true
		}
	}
	if !sawFetchBeforeDecide {
		t.Fatalf("ensure never ran the before-decide fetch sync: %+v", got)
	}
	if !sawPushAfterWrite {
		t.Fatalf("ensure never ran the after-write push sync: %+v", got)
	}

	// A renew (second ensure on the held lease) must also publish the write.
	got = nil
	if refuse, err := hold.ensure(time.Now().Add(time.Minute)); err != nil || refuse != nil {
		t.Fatalf("renew must hold: refuse=%+v err=%v", refuse, err)
	}
	sawPushAfterWrite = false
	for _, c := range got {
		if c.written {
			sawPushAfterWrite = true
		}
	}
	if !sawPushAfterWrite {
		t.Fatalf("renew never ran the after-write push sync: %+v", got)
	}
}
