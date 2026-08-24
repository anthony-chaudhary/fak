package hostresurrect

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

func TestPlanJoinsCrashToLiveRowsIdempotentlyAndCapsWave(t *testing.T) {
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt-1", Class: "wt-render-av"}
	rows := []guardsessions.Row{
		{Schema: guardsessions.Schema, Handle: "g2", Interactive: true, CWD: `C:\b`, Command: []string{"claude"}, ResumeHandle: "g2", StartedAt: "2026-07-14T20:00:02Z"},
		{Schema: guardsessions.Schema, Handle: "g1", Interactive: true, CWD: `C:\a`, Command: []string{"claude", "--resume", "stale"}, ResumeHandle: "g1", StartedAt: "2026-07-14T20:00:01Z"},
		{Schema: guardsessions.Schema, Handle: "ended", Interactive: true, CWD: `C:\c`, Command: []string{"claude"}, ResumeHandle: "ended", EndedAt: "2026-07-14T20:01:00Z"},
	}
	cohort := Cohort{Sessions: []CohortEntry{{Handle: "g1", PID: 1, StartedAt: rows[1].StartedAt}, {Handle: "g2", PID: 2, StartedAt: rows[0].StartedAt}}}
	rows[0].PID, rows[1].PID = 2, 1
	rows[0].HostRecovery, rows[1].HostRecovery = true, true
	got, _ := Plan(sig, rows, cohort, map[string]bool{Key("evt-1", "g1"): true}, 1)
	if len(got) != 1 || got[0].Session != "g2" || got[0].CWD != `C:\b` {
		t.Fatalf("Plan = %+v", got)
	}
	if got[0].Schema != Schema || got[0].EventID != "evt-1" {
		t.Fatalf("incomplete request: %+v", got[0])
	}
	if len(got[0].Command) != 3 || got[0].Command[1] != "--resume" || got[0].Command[2] != "g2" {
		t.Fatalf("cold relaunch command: %v", got[0].Command)
	}
}

func TestPlanReplacesStaleResumeHandle(t *testing.T) {
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt"}
	rows := []guardsessions.Row{optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "g", Interactive: true, CWD: `C:\a`, Command: []string{"claude", "--resume", "old"}, ResumeHandle: "new"})}
	rows[0].PID = 1
	got, _ := Plan(sig, rows, Cohort{Sessions: []CohortEntry{{Handle: "g", PID: 1, StartedAt: rows[0].StartedAt}}}, nil, 1)
	if len(got) != 1 || got[0].Command[2] != "new" {
		t.Fatalf("Plan=%+v", got)
	}
}

func TestPlanReplacesContinueWithExplicitResume(t *testing.T) {
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt"}
	rows := []guardsessions.Row{optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "g", Interactive: true, CWD: `C:\a`, Command: []string{"claude", "--continue"}, ResumeHandle: "g"})}
	rows[0].PID = 1
	got, _ := Plan(sig, rows, Cohort{Sessions: []CohortEntry{{Handle: "g", PID: 1, StartedAt: rows[0].StartedAt}}}, nil, 1)
	if len(got) != 1 || len(got[0].Command) != 3 || got[0].Command[1] != "--resume" || got[0].Command[2] != "g" {
		t.Fatalf("Plan=%+v", got)
	}
}
func TestRecentCountUsesRollingWindow(t *testing.T) {
	now := time.Date(2026, 7, 14, 20, 0, 0, 0, time.UTC)
	if got := RecentCount([]time.Time{now.Add(-299 * time.Second), now.Add(-301 * time.Second), now.Add(time.Second)}, now, 300*time.Second); got != 1 {
		t.Fatalf("RecentCount=%d", got)
	}
}

func TestPlanExcludesMoreThanWaveOfStaleRows(t *testing.T) {
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt-mixed"}
	rows := make([]guardsessions.Row, 0, MaxLaunchesPerWindow+2)
	for i := 0; i < MaxLaunchesPerWindow+1; i++ {
		rows = append(rows, optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: fmt.Sprintf("stale-%02d", i), PID: 100 + i, Interactive: true, CWD: `C:\stale`, Command: []string{"claude"}, ResumeHandle: fmt.Sprintf("stale-%02d", i), StartedAt: fmt.Sprintf("2026-07-14T19:%02d:00Z", i)}))
	}
	current := optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "current", PID: 4242, Interactive: true, CWD: `C:\current`, Command: []string{"claude"}, ResumeHandle: "current", StartedAt: "2026-07-14T20:00:00Z"})
	rows = append(rows, current)
	cohort := Cohort{CapturedAt: "2026-07-14T20:00:01Z", Sessions: []CohortEntry{{Handle: current.Handle, PID: current.PID, StartedAt: current.StartedAt}}}

	got, counts := Plan(sig, rows, cohort, nil, MaxLaunchesPerWindow)
	if len(got) != 1 || got[0].Session != current.Handle {
		t.Fatalf("Plan=%+v", got)
	}
	if counts.Inventory != MaxLaunchesPerWindow+2 || counts.Candidates != 1 || counts.ExcludedNotInCohort != MaxLaunchesPerWindow+1 || counts.Selected != 1 {
		t.Fatalf("counts=%+v", counts)
	}
}

func TestPlanRejectsCohortPIDMismatch(t *testing.T) {
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt-reuse"}
	row := optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "g", PID: 42, Interactive: true, CWD: `C:\repo`, Command: []string{"claude"}, ResumeHandle: "g"})
	got, counts := Plan(sig, []guardsessions.Row{row}, Cohort{Sessions: []CohortEntry{{Handle: "g", PID: 41, StartedAt: row.StartedAt}}}, nil, 1)
	if len(got) != 0 || counts.ExcludedPIDMismatch != 1 {
		t.Fatalf("Plan=%+v counts=%+v", got, counts)
	}
}

func TestCaptureKeepsNewestRowPerLivePID(t *testing.T) {
	now := time.Date(2026, 8, 24, 2, 0, 0, 0, time.UTC)
	rows := []guardsessions.Row{
		{Schema: guardsessions.Schema, Handle: "old-alias", PID: 42, Interactive: true, CWD: `C:\repo`, Command: []string{"codex"}, ResumeHandle: "alias", StartedAt: now.Add(-time.Minute).Format(time.RFC3339Nano)},
		{Schema: guardsessions.Schema, Handle: "current", PID: 42, Interactive: true, CWD: `C:\repo`, Command: []string{"codex"}, ResumeHandle: "alias", StartedAt: now.Format(time.RFC3339Nano)},
	}
	cohort := Capture(rows, now, func(pid int) bool { return pid == 42 }, func(pid int) (time.Time, bool) { return now.Add(-2 * time.Hour), pid == 42 })
	if len(cohort.Sessions) != 1 || cohort.Sessions[0].Handle != "current" {
		t.Fatalf("cohort=%+v", cohort)
	}
}

func TestPlanScopesTerminalCrashToOwningHost(t *testing.T) {
	now := time.Date(2026, 8, 24, 3, 0, 0, 0, time.UTC)
	one := optIn(guardsessions.NewInteractiveRow("one", "claude", 41, `C:\one`, "", "", now, []string{"claude"}))
	two := optIn(guardsessions.NewInteractiveRow("two", "claude", 42, `C:\two`, "", "", now.Add(-time.Second), []string{"claude"}))
	cohort := Cohort{Sessions: []CohortEntry{
		{Handle: one.Handle, PID: one.PID, StartedAt: one.StartedAt, HostPID: 100},
		{Handle: two.Handle, PID: two.PID, StartedAt: two.StartedAt, HostPID: 200},
	}}
	signal := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "wt", HostPID: 100, FaultingApp: "WindowsTerminal.exe"}
	got, counts := Plan(signal, []guardsessions.Row{one, two}, cohort, nil, 10)
	if len(got) != 1 || got[0].Session != one.Handle || counts.Candidates != 1 || counts.ExcludedHostMismatch != 1 {
		t.Fatalf("got=%+v counts=%+v", got, counts)
	}
}

func TestPlanTerminalCrashFailsClosedWithoutHostBinding(t *testing.T) {
	now := time.Date(2026, 8, 24, 3, 0, 0, 0, time.UTC)
	row := optIn(guardsessions.NewInteractiveRow("one", "claude", 41, `C:\one`, "", "", now, []string{"claude"}))
	signal := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "wt", HostPID: 100, FaultingApp: "WindowsTerminal.exe"}
	got, counts := Plan(signal, []guardsessions.Row{row}, Cohort{Sessions: []CohortEntry{{Handle: row.Handle, PID: row.PID, StartedAt: row.StartedAt}}}, nil, 10)
	if len(got) != 0 || counts.ExcludedHostMismatch != 1 {
		t.Fatalf("got=%+v counts=%+v", got, counts)
	}
}

func optIn(row guardsessions.Row) guardsessions.Row {
	row.HostRecovery = true
	return row
}

func TestPlanRejectsCodexWithoutExactThreadBinding(t *testing.T) {
	row := optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "guard-handle", PID: 42, Interactive: true, CWD: `C:\repo`, Command: []string{"codex"}, ResumeHandle: "guard-handle"})
	got, counts := Plan(hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt"}, []guardsessions.Row{row}, Cohort{Sessions: []CohortEntry{{Handle: row.Handle, PID: row.PID}}}, nil, 10)
	if len(got) != 0 || counts.Candidates != 1 || counts.ExcludedInvalidResume != 1 {
		t.Fatalf("got=%+v counts=%+v", got, counts)
	}
}

func TestPlanPreservesExactCodexThreadBinding(t *testing.T) {
	const thread = "01a03148-a78e-7af1-b4b3-dda99323b55a"
	command := []string{`C:\tools\codex.exe`, "resume", thread}
	row := optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "guard-handle", PID: 42, Interactive: true, CWD: `C:\repo`, Command: command, ResumeHandle: "guard-handle"})
	got, counts := Plan(hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt"}, []guardsessions.Row{row}, Cohort{Sessions: []CohortEntry{{Handle: row.Handle, PID: row.PID}}}, nil, 10)
	if len(got) != 1 || counts.Selected != 1 || !reflect.DeepEqual(got[0].Command, command) {
		t.Fatalf("got=%+v counts=%+v", got, counts)
	}
}

func TestPlanPreservesCodexThreadBindingAfterLauncherConfig(t *testing.T) {
	const thread = "01a02fe5-37cd-7252-ba6e-5bd8b19ee8d8"
	command := []string{"codex", `developer_instructions="guard steering"`, "resume", thread}
	row := optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "guard-handle", PID: 42, Interactive: true, CWD: `C:\repo`, Command: command, ResumeHandle: "guard-handle"})
	got, counts := Plan(hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt"}, []guardsessions.Row{row}, Cohort{Sessions: []CohortEntry{{Handle: row.Handle, PID: row.PID}}}, nil, 10)
	if len(got) != 1 || counts.Selected != 1 || !reflect.DeepEqual(got[0].Command, command) {
		t.Fatalf("got=%+v counts=%+v", got, counts)
	}
}

func TestPlanRepairsLegacyCodexGuardResumeSuffix(t *testing.T) {
	const thread = "01a02fe5-37cd-7252-ba6e-5bd8b19ee8d8"
	command := []string{"codex", `developer_instructions="guard steering"`, "resume", thread, "--resume", "guard-handle"}
	want := command[:4]
	row := optIn(guardsessions.Row{Schema: guardsessions.Schema, Handle: "guard-handle", PID: 42, Interactive: true, CWD: `C:\repo`, Command: command, ResumeHandle: "guard-handle"})
	got, counts := Plan(hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt"}, []guardsessions.Row{row}, Cohort{Sessions: []CohortEntry{{Handle: row.Handle, PID: row.PID}}}, nil, 10)
	if len(got) != 1 || counts.Selected != 1 || !reflect.DeepEqual(got[0].Command, want) {
		t.Fatalf("got=%+v counts=%+v", got, counts)
	}
}
