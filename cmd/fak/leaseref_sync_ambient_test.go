package main

// Witnesses for the #2302 ambient leaseref-sync wiring (leaseref_sync_ambient.go):
// the boundary runs the convergence plan, stays NONFATAL when origin is
// unreachable (the fail-open posture the dispatch tick already uses), honors the
// FAK_LEASEREF_SYNC disable and FAK_LEASEREF_REMOTE override, and the loop-drive
// region hold actually drives it at the before-decide / after-write boundaries.

import (
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
