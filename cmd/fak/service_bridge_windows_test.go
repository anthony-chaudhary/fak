//go:build windows

package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scmbridge"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// TestWindowsSCMReadBackIsAuthoritativeAndQueryOnly witnesses the #4756
// read-back half against the REAL local Service Control Manager with
// query-only rights: no service is created, started, stopped, or changed.
// The EventLog inbox service exists on every Windows machine, always runs,
// and runs as the same least-privilege LocalService principal the machine
// projection demands — so the principal fold, the status→phase map, and the
// binary-provenance digest are all exercised against live SCM state.
func TestWindowsSCMReadBackIsAuthoritativeAndQueryOnly(t *testing.T) {
	got, err := windowsSCMReadBack("EventLog")
	if err != nil {
		t.Skipf("SCM query-only access unavailable in this environment: %v", err)
	}
	if !got.Present || got.Manager != scmbridge.ManagerSCM || got.UnitName != "EventLog" {
		t.Fatalf("read-back = %+v", got)
	}
	if got.Status != "running" || got.PID <= 0 {
		t.Fatalf("EventLog status = %q pid=%d", got.Status, got.PID)
	}
	if scmbridge.PhaseFromSCMState(got.Status, got.PID) != servicespec.PhaseReady {
		t.Fatalf("phase map broke on live state %q", got.Status)
	}
	if !strings.Contains(strings.ToLower(got.Principal), "localservice") {
		t.Fatalf("EventLog principal = %q", got.Principal)
	}
	if len(got.Command) == 0 || got.Command[0] == "" {
		t.Fatalf("command read-back = %+v", got.Command)
	}
	if got.BinarySHA256 == "" {
		t.Fatalf("installed-binary provenance digest missing (command %q)", got.Command[0])
	}
	if !got.StartOnBoot || got.StartDisabled {
		t.Fatalf("EventLog trigger read-back = boot:%v disabled:%v", got.StartOnBoot, got.StartDisabled)
	}
}

// A service that does not exist reads back Present=false without error —
// the reconcile absent axis, witnessed against the live SCM.
func TestWindowsSCMReadBackAbsentService(t *testing.T) {
	got, err := windowsSCMReadBack("FakDefinitelyNotInstalled8472")
	if err != nil {
		t.Skipf("SCM query-only access unavailable in this environment: %v", err)
	}
	if got.Present {
		t.Fatalf("phantom service: %+v", got)
	}
}
