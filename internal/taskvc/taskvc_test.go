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
  [string]$TemplatedTaskName = "FleetSomething-$repoSlug"
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
	if off := Verify(inv, declared, nil); len(off) != 0 {
		t.Fatalf("bound installer: want 0 offenses, got %v", off)
	}
}

// TestVerifyAcceptsTrackedCapture is the XML tier's happy path: a row whose export
// really is tracked, and which says what the export does not restore, holds up.
func TestVerifyAcceptsTrackedCapture(t *testing.T) {
	captured := map[string]bool{"tools/scheduled-tasks/FleetX.xml": true}
	inv := []Coverage{{
		Task:    "FleetX",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FleetX.xml",
		Reason:  "launches an out-of-tree script; the XML restores the schedule, not the script",
	}}
	if off := Verify(inv, nil, captured); len(off) != 0 {
		t.Fatalf("tracked capture: want 0 offenses, got %v", off)
	}
}

// TestVerifyCatchesUntrackedCapture: a capture claim is a promise that a reimage can
// read the file back. An export that was written but never committed cannot keep it,
// and that is precisely the state a half-finished capture leaves behind.
func TestVerifyCatchesUntrackedCapture(t *testing.T) {
	inv := []Coverage{{
		Task:    "FleetX",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FleetX.xml",
		Reason:  "out-of-tree script",
	}}
	off := Verify(inv, nil, map[string]bool{})
	if len(off) != 1 {
		t.Fatalf("untracked capture: want 1 offense, got %v", off)
	}
	if off[0].Reason != ReasonMissingCapture {
		t.Errorf("reason = %q, want %q", off[0].Reason, ReasonMissingCapture)
	}

	// A StatusXML row naming no capture at all is the same failure, caught before
	// the tree is even consulted.
	off = Verify([]Coverage{{Task: "FleetY", Status: StatusXML, Reason: "why"}}, nil, map[string]bool{})
	if len(off) != 1 || off[0].Reason != ReasonMissingCapture {
		t.Fatalf("captureless xml row: want 1 MISSING_TASK_CAPTURE offense, got %v", off)
	}

	// A capture named by a NON-xml row is held to the same standard: a drift row
	// leaning on an export is making the identical reimage promise.
	off = Verify([]Coverage{{
		Task: "FleetZ", Status: StatusDrift, Capture: "tools/scheduled-tasks/FleetZ.xml", Reason: "interpolated name",
	}}, nil, map[string]bool{})
	if len(off) != 1 || off[0].Reason != ReasonMissingCapture {
		t.Fatalf("drift row with untracked capture: want 1 MISSING_TASK_CAPTURE offense, got %v", off)
	}
}

// TestVerifyRequiresResidualOnCapture: an XML capture versions the task, never the
// script the task launches. A row that claims the XML tier without naming what a
// restore would still be missing is overclaiming, and that is the whole distinction
// between this tier and a real installer.
func TestVerifyRequiresResidualOnCapture(t *testing.T) {
	captured := map[string]bool{"tools/scheduled-tasks/FleetX.xml": true}
	off := Verify([]Coverage{{
		Task: "FleetX", Status: StatusXML, Capture: "tools/scheduled-tasks/FleetX.xml",
	}}, nil, captured)
	if len(off) != 1 || off[0].Reason != ReasonOrphanTask {
		t.Fatalf("reasonless capture: want 1 ORPHAN_FLEET_TASK offense, got %v", off)
	}
}

// TestUncoveredIsTheDoneCondition pins #3323's done condition as a predicate: a row
// is uncovered only when it has NEITHER an installer NOR an export. Note the third
// row — an honestly-reasoned orphan still fails, because Verify asks "is the claim
// true?" while Uncovered asks "is it good enough?".
func TestUncoveredIsTheDoneCondition(t *testing.T) {
	got := Uncovered([]Coverage{
		{Task: "ByInstaller", Status: StatusInstaller, Installer: "tools/register_x.ps1"},
		{Task: "ByCapture", Status: StatusXML, Capture: "tools/scheduled-tasks/ByCapture.xml", Reason: "r"},
		{Task: "Lost", Status: StatusOrphan, Reason: "honestly recorded, still lost on a reimage"},
	})
	if len(got) != 1 {
		t.Fatalf("Uncovered: want 1 row, got %v", got)
	}
	if got[0].Task != "Lost" {
		t.Errorf("Uncovered[0] = %q, want Lost", got[0].Task)
	}
}

// TestVerifyCatchesNameDrift is the #3323 regression: the installer still exists,
// but its -TaskName default has drifted to another name, so a reinstall would
// register a SECOND loop instead of updating the live one. Verify must refuse.
func TestVerifyCatchesNameDrift(t *testing.T) {
	declared := map[string][]string{"tools/register_x.ps1": {"FleetXRenamed"}}
	inv := []Coverage{{Task: "FleetX", Status: StatusInstaller, Installer: "tools/register_x.ps1"}}

	off := Verify(inv, declared, nil)
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
	off = Verify(inv, map[string][]string{}, nil)
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
	}, map[string][]string{}, nil)
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
	}, map[string][]string{}, nil)
	if len(ok) != 0 {
		t.Fatalf("explained gaps: want 0 offenses, got %v", ok)
	}
}

// TestVerifyRejectsUnknownStatus keeps the status vocabulary closed — a typo'd
// status must not silently pass as coverage.
func TestVerifyRejectsUnknownStatus(t *testing.T) {
	off := Verify([]Coverage{{Task: "FleetX", Status: Status("probably-fine")}}, map[string][]string{}, nil)
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

// TestFleetInventoryHasNoOrphans is #3323's done condition held against the real
// record: every enabled fak-fleet task must have an installer or an exported task
// XML, i.e. zero rows survive Uncovered. TestFleetTaskInventoryBindsToVersioned-
// Installers cannot catch this — Verify only refuses claims that are UNTRUE, so an
// honestly-reasoned orphan passes it while still being lost on a reimage. This is
// the assertion that makes "zero orphans" a gate rather than a comment.
func TestFleetInventoryHasNoOrphans(t *testing.T) {
	for _, c := range Uncovered(Inventory()) {
		t.Errorf("%s is rebuildable from neither an installer nor an exported task XML: %s (%s)",
			c.Task, c.Reason, ReasonOrphanTask)
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

func BenchmarkDeclaredTaskNames(b *testing.B) {
	src := `
[CmdletBinding()]
param(
  [string]$TaskName = 'FleetScoutLoop',
  [string]$CaptureTaskName = 'FakLogvaultCapture',
  [string]$VerifyTaskName  = "FakLogvaultVerify",
  [string]$EmptyTaskName = '',
  [string]$TemplatedTaskName = "FleetSomething-$repoSlug"
)
schtasks /Delete /TN $TaskName /F
`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DeclaredTaskNames(src)
	}
}

func BenchmarkVerify(b *testing.B) {
	inv := Inventory()
	declared := map[string][]string{
		"tools/register_logvault_backup.ps1":       {"FakLogvaultCapture", "FakLogvaultVerify"},
		"tools/install_self_update_schedule.ps1":   {"FakSelfUpdate"},
		"tools/register_bench_plan_doc.ps1":        {"FleetBenchPlanDoc"},
		"tools/register_dos_dispatch_watchdog.ps1": {"FleetDOSDispatchWatchdog"},
		"tools/register_dispatch_status_doc.ps1":   {"FleetDispatchStatusDoc"},
		"tools/register_idea_scout.ps1":            {"FleetIdeaScout"},
		"tools/register_issue_dispatch.ps1":        {"FleetIssueDispatch"},
		"tools/register_proc_resource_guard.ps1":   {"FleetProcResourceGuard"},
		"tools/register_push_lag_pusher.ps1":       {"FleetPushLagPusher"},
		"tools/register_resolve_progress.ps1":      {"FleetResolveProgress"},
		"tools/register_resume_watchdog.ps1":       {"FleetResumeWatchdog"},
		"tools/register_runaway_reaper.ps1":        {"FleetRunawayReaper"},
		"tools/register_scout_loop.ps1":            {"FleetScoutLoop"},
		"tools/register_stale_work_watchdog.ps1":   {"FleetStaleWorkGarden"},
		"tools/register_supervisor_watchdog.ps1":   {"FleetSupervisorWatchdog"},
		"tools/register_worktree_doctor.ps1":       {"FleetWorktreeDoctor"},
	}
	captured := map[string]bool{
		"tools/scheduled-tasks/ClaudeAccountBackup.xml":          true,
		"tools/scheduled-tasks/July4CacheValueAutospawn.xml":     true,
		"tools/scheduled-tasks/FakFleetJanitorHeadless.xml":      true,
		"tools/scheduled-tasks/FakBenchmarkFleetLoop.xml":        true,
		"tools/scheduled-tasks/FakMetaSuperloopNight100.xml":     true,
		"tools/scheduled-tasks/FakOvernightMixedProfiles100.xml": true,
		"tools/scheduled-tasks/FleetGLM52CampaignStop.xml":       true,
		"tools/scheduled-tasks/FakReapOrphanTails.xml":           true,
		"tools/scheduled-tasks/FleetOwnerSeatResume.xml":         true,
		"tools/scheduled-tasks/FleetStrandedRecovery.xml":        true,
		"tools/scheduled-tasks/FleetWatchdogWatchdogAudit.xml":   true,
		"tools/scheduled-tasks/UserSeatDrain-1010.xml":           true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Verify(inv, declared, captured)
	}
}

func BenchmarkUncovered(b *testing.B) {
	inv := Inventory()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Uncovered(inv)
	}
}

func BenchmarkInventory(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Inventory()
	}
}

func BenchmarkOffenseString(b *testing.B) {
	off := Offense{
		Task:   "FleetStaleWorkGarden",
		Reason: ReasonInstallerNameDrift,
		Detail: "installer tools/register_stale_work_watchdog.ps1 drifted",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = off.String()
	}
}
