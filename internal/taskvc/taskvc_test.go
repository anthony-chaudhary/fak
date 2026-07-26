package taskvc

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDeclaredTaskNames pins the literal-only parse (verify the verifier). The
// single-quoted default is the fleet's static, bindable form; the interpolated and
// empty forms are NOT knowable without running the script and must never count as
// coverage — that distinction is the whole name-drift check.
func TestDeclaredTaskNames(t *testing.T) {
	src := `
[CmdletBinding()]
param(
  [string]$TaskName = 'FleetScoutLoop',
  [string]$CaptureTaskName = 'FakLogvaultCapture',
  [string]$VerifyTaskName  = "FakLogvaultVerify",
  [string]$EmptyTaskName = '',
  [string]$TemplatedTaskName = "FleetWorktreeDoctor-$repoSlug"
)
schtasks /Delete /TN $TaskName /F
`
	got := DeclaredTaskNames(src)
	want := []string{"FakLogvaultCapture", "FakLogvaultVerify", "FleetScoutLoop"}
	if len(got) != len(want) {
		t.Fatalf("DeclaredTaskNames = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("DeclaredTaskNames[%d] = %q, want %q", i, got[i], want[i])
		}
	}

	// A script that registers nothing statically yields nothing — never a panic
	// and never a bogus empty-string name.
	if n := DeclaredTaskNames("# no task declared here\n"); len(n) != 0 {
		t.Errorf("no-declaration script: got %v, want none", n)
	}
}

// TestVerifyAcceptsBoundInstaller is the happy path: a coverage claim whose
// installer really does declare that task name holds up.
func TestVerifyAcceptsBoundInstaller(t *testing.T) {
	declared := map[string][]string{"tools/register_x.ps1": {"FleetX"}}
	inv := []Coverage{{Task: "FleetX", Status: StatusInstaller, Installer: "tools/register_x.ps1"}}
	if off := Verify(inv, declared); len(off) != 0 {
		t.Fatalf("bound installer: want 0 offenses, got %v", off)
	}
}

// TestVerifyCatchesNameDrift is the #3323 regression: the installer still exists,
// but its -TaskName default has drifted to another name, so a reinstall would
// register a SECOND loop instead of updating the live one. Verify must refuse.
func TestVerifyCatchesNameDrift(t *testing.T) {
	declared := map[string][]string{"tools/register_x.ps1": {"FleetXRenamed"}}
	inv := []Coverage{{Task: "FleetX", Status: StatusInstaller, Installer: "tools/register_x.ps1"}}

	off := Verify(inv, declared)
	if len(off) != 1 {
		t.Fatalf("drifted installer: want 1 offense, got %v", off)
	}
	if off[0].Reason != ReasonInstallerNameDrift {
		t.Errorf("reason = %q, want %q", off[0].Reason, ReasonInstallerNameDrift)
	}
	if off[0].Task != "FleetX" {
		t.Errorf("task = %q, want FleetX", off[0].Task)
	}

	// A claim naming an installer that is gone entirely is the same class of lie.
	off = Verify(inv, map[string][]string{})
	if len(off) != 1 || off[0].Reason != ReasonInstallerNameDrift {
		t.Fatalf("deleted installer: want 1 INSTALLER_NAME_DRIFT offense, got %v", off)
	}
}

// TestVerifyRequiresReasonForGaps: a drift/orphan row must say WHY. An unexplained
// gap is indistinguishable from an oversight, and the residual is the whole point
// of the inventory.
func TestVerifyRequiresReasonForGaps(t *testing.T) {
	off := Verify([]Coverage{
		{Task: "FleetOrphan", Status: StatusOrphan},
		{Task: "FleetDrifted", Status: StatusDrift},
	}, map[string][]string{})
	if len(off) != 2 {
		t.Fatalf("reasonless gaps: want 2 offenses, got %v", off)
	}
	for _, o := range off {
		if o.Reason != ReasonOrphanTask {
			t.Errorf("%s reason = %q, want %q", o.Task, o.Reason, ReasonOrphanTask)
		}
	}

	// With reasons recorded, the same gaps are accepted: Verify refuses
	// OVERCLAIMING, never an honestly-recorded gap.
	ok := Verify([]Coverage{
		{Task: "FleetOrphan", Status: StatusOrphan, Reason: "script lives outside the repo"},
		{Task: "FleetDrifted", Status: StatusDrift, Reason: "installer name is interpolated"},
	}, map[string][]string{})
	if len(ok) != 0 {
		t.Fatalf("explained gaps: want 0 offenses, got %v", ok)
	}
}

// TestVerifyRejectsUnknownStatus keeps the status vocabulary closed — a typo'd
// status must not silently pass as coverage.
func TestVerifyRejectsUnknownStatus(t *testing.T) {
	off := Verify([]Coverage{{Task: "FleetX", Status: Status("probably-fine")}}, map[string][]string{})
	if len(off) != 1 || off[0].Reason != ReasonOrphanTask {
		t.Fatalf("unknown status: want 1 ORPHAN_FLEET_TASK offense, got %v", off)
	}
}

// TestFleetTaskInventoryBindsToVersionedInstallers is the live trunk guard: every
// StatusInstaller row in the real inventory must resolve against the real tracked
// installers in this repo. This is what makes the FleetStaleWorkGarden bug class
// non-recurring — rename or delete an installer's -TaskName default without
// updating inventory.go and this reds.
func TestFleetTaskInventoryBindsToVersionedInstallers(t *testing.T) {
	root := repoRoot(t)
	offenses, err := ScanTree(root)
	if err != nil {
		t.Fatalf("ScanTree(%s): %v", root, err)
	}
	for _, o := range offenses {
		t.Errorf("fleet task coverage: %s", o)
	}
	if len(offenses) > 0 {
		t.Fatalf("%d fleet task(s) claim coverage that the tree does not back", len(offenses))
	}
}

// TestInventoryIsCopied guards the shared record: a caller mutating the returned
// slice must not corrupt the inventory for the next caller.
func TestInventoryIsCopied(t *testing.T) {
	first := Inventory()
	if len(first) == 0 {
		t.Fatal("Inventory() is empty; the fleet posture record should not be blank")
	}
	original := first[0]
	first[0] = Coverage{Task: "clobbered", Status: StatusOrphan, Reason: "mutation probe"}
	if again := Inventory(); again[0] != original {
		t.Errorf("Inventory() leaked its backing array: got %+v, want %+v", again[0], original)
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod above %s", dir)
		}
		dir = parent
	}
}
