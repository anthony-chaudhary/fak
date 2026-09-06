package taskvc

import (
	"fmt"
	"strings"
	"testing"
)

func BenchmarkDeclaredTaskNames(b *testing.B) {
	minimalSrc := "param([string]$TaskName = 'FleetMinimalLoop')\n"

	mixedSrc := `
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
	largeSrc := sampleLargeInstallerScript()

	b.Run("Minimal", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = DeclaredTaskNames(minimalSrc)
		}
	})

	b.Run("ProductionMixed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = DeclaredTaskNames(mixedSrc)
		}
	})

	b.Run("LargeScript", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(largeSrc)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = DeclaredTaskNames(largeSrc)
		}
	})
}

func BenchmarkVerify(b *testing.B) {
	inv := Inventory()
	declared := productionDeclared()
	captured := productionCaptured()
	dirtyInv := sampleDirtyInventory()

	b.Run("ProductionClean", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Verify(inv, declared, captured)
		}
	})

	b.Run("WithOffenses", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Verify(dirtyInv, declared, captured)
		}
	})

	for _, count := range []int{100, 500, 1000} {
		scaledInv, scaledDecl, scaledCap := generateSyntheticFleet(count)
		b.Run(fmt.Sprintf("Scaled_%d", count), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = Verify(scaledInv, scaledDecl, scaledCap)
			}
		})
	}
}

func BenchmarkUncovered(b *testing.B) {
	inv := Inventory()
	scaledInv, _, _ := generateSyntheticFleet(1000)

	b.Run("Production", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Uncovered(inv)
		}
	})

	b.Run("ScaledDegraded_1000", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Uncovered(scaledInv)
		}
	})
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

func BenchmarkVerifyPipeline(b *testing.B) {
	dirtyInv := sampleDirtyInventory()
	declared := productionDeclared()
	captured := productionCaptured()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offenses := Verify(dirtyInv, declared, captured)
		for j := range offenses {
			_ = offenses[j].String()
		}
	}
}

func productionDeclared() map[string][]string {
	return map[string][]string{
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
}

func productionCaptured() map[string]bool {
	return map[string]bool{
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
}

func sampleDirtyInventory() []Coverage {
	return []Coverage{
		{Task: "ValidInstallerTask", Status: StatusInstaller, Installer: "tools/register_logvault_backup.ps1"},
		{Task: "DriftedTask", Status: StatusInstaller, Installer: "tools/register_bench_plan_doc.ps1"},
		{Task: "MissingInstallerTask", Status: StatusInstaller, Installer: "tools/untracked_installer.ps1"},
		{Task: "MissingXMLTask", Status: StatusXML, Capture: "tools/scheduled-tasks/Untracked.xml", Reason: "out of tree"},
		{Task: "EmptyCaptureXML", Status: StatusXML, Reason: "no capture path"},
		{Task: "ReasonlessXML", Status: StatusXML, Capture: "tools/scheduled-tasks/ClaudeAccountBackup.xml"},
		{Task: "ReasonlessDrift", Status: StatusDrift},
		{Task: "ReasonlessOrphan", Status: StatusOrphan},
		{Task: "UnknownStatusTask", Status: Status("unknown_tier")},
	}
}

func sampleLargeInstallerScript() string {
	var sb strings.Builder
	sb.WriteString("# Fleet maintenance and supervision task installer\n")
	sb.WriteString("[CmdletBinding()]\nparam(\n")
	for i := 0; i < 20; i++ {
		fmt.Fprintf(&sb, "  [string]$TaskName%d = 'FleetMaintenanceLoop%02d',\n", i, i)
		fmt.Fprintf(&sb, "  [string]$DynTaskName%d = \"FleetDynamic-$env:COMPUTERNAME-%d\",\n", i, i)
	}
	sb.WriteString("  [string]$PrimaryTaskName = 'FleetMainWatchdog'\n)\n\n")
	sb.WriteString("function Register-FleetTask {\n")
	sb.WriteString("  param([string]$Name, [string]$Action)\n")
	sb.WriteString("  Write-Verbose \"Registering $Name with action $Action\"\n")
	sb.WriteString("  schtasks /Create /TN $Name /TR $Action /SC DAILY /F\n")
	sb.WriteString("}\n\n")
	for i := 0; i < 50; i++ {
		fmt.Fprintf(&sb, "Register-FleetTask -Name \"Task_%d\" -Action \"powershell.exe -NoProfile -File script_%d.ps1\"\n", i, i)
	}
	return sb.String()
}

func generateSyntheticFleet(count int) ([]Coverage, map[string][]string, map[string]bool) {
	inv := make([]Coverage, count)
	declared := make(map[string][]string, count/2)
	captured := make(map[string]bool, count/2)

	for i := 0; i < count; i++ {
		task := fmt.Sprintf("FleetTask_%04d", i)
		switch i % 5 {
		case 0, 1:
			inst := fmt.Sprintf("tools/register_task_%04d.ps1", i)
			declared[inst] = []string{task}
			inv[i] = Coverage{
				Task:      task,
				Status:    StatusInstaller,
				Installer: inst,
			}
		case 2:
			capPath := fmt.Sprintf("tools/scheduled-tasks/%s.xml", task)
			captured[capPath] = true
			inv[i] = Coverage{
				Task:    task,
				Status:  StatusXML,
				Capture: capPath,
				Reason:  "out-of-tree script execution",
			}
		case 3:
			inv[i] = Coverage{
				Task:   task,
				Status: StatusDrift,
				Reason: "runtime parameter interpolation",
			}
		case 4:
			if i%10 == 4 {
				inv[i] = Coverage{
					Task:    task,
					Status:  StatusXML,
					Capture: fmt.Sprintf("tools/scheduled-tasks/Missing_%04d.xml", i),
					Reason:  "untracked export",
				}
			} else {
				inv[i] = Coverage{
					Task:   task,
					Status: StatusOrphan,
					Reason: "untracked loop on fleet machine",
				}
			}
		}
	}
	return inv, declared, captured
}
