package hostresurrect

import (
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
	got := Plan(sig, rows, map[string]bool{Key("evt-1", "g1"): true}, 1)
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
	rows := []guardsessions.Row{{Schema: guardsessions.Schema, Handle: "g", Interactive: true, CWD: `C:\a`, Command: []string{"claude", "--resume", "old"}, ResumeHandle: "new"}}
	got := Plan(sig, rows, nil, 1)
	if len(got) != 1 || got[0].Command[2] != "new" {
		t.Fatalf("Plan=%+v", got)
	}
}

func TestPlanReplacesContinueWithExplicitResume(t *testing.T) {
	sig := hostfault.HostCrashSignal{Schema: hostfault.HostCrashSignalSchema, EventID: "evt"}
	rows := []guardsessions.Row{{Schema: guardsessions.Schema, Handle: "g", Interactive: true, CWD: `C:\a`, Command: []string{"claude", "--continue"}, ResumeHandle: "g"}}
	got := Plan(sig, rows, nil, 1)
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
