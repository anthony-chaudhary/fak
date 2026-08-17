package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func TestSuperloopNoProgressStreakResetsOnWitnessedProgress(t *testing.T) {
	events := []loopmgr.Event{
		{LoopID: "night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusFailed},
		{LoopID: "night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusWitnessedDone},
		{LoopID: "night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusRefused},
		{LoopID: "other", Kind: loopmgr.EventEnd, Status: loopmgr.StatusFailed},
	}
	if got := superloopNoProgressStreak(events, "night"); got != 1 {
		t.Fatalf("streak = %d, want 1 after reset", got)
	}
}

func TestSuperloopNoProgressStreakTreatsMeasuredClosureAsProgress(t *testing.T) {
	events := []loopmgr.Event{
		{LoopID: "night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusFailed},
		{LoopID: "night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusFailed, Metrics: map[string]int64{"closed_now": 1}},
		{LoopID: "night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusFailed},
	}
	if got := superloopNoProgressStreak(events, "night"); got != 1 {
		t.Fatalf("streak = %d, want measured progress reset", got)
	}
}

func TestApplySuperloopNoProgressEscalationClimbsAndCaps(t *testing.T) {
	old := superloopEscalationEvents
	t.Cleanup(func() { superloopEscalationEvents = old })
	base := superloop.DriveDecision{Intent: "night", Enter: true, Member: superloop.Member{Kind: superloop.KindSurface, Ref: "actionable", Enter: "dispatch"}, Action: "dispatch"}
	for streak, want := range map[int]string{1: "actionable", 2: "no-progress-replan", 3: "no-progress-unblock", 4: "no-progress-unstick", 8: "no-progress-operator-decision"} {
		events := make([]loopmgr.Event, streak)
		for i := range events {
			events[i] = loopmgr.Event{LoopID: "night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusFailed}
		}
		superloopEscalationEvents = func(string) ([]loopmgr.Event, error) { return events, nil }
		got, gotStreak := applySuperloopNoProgressEscalation(t.TempDir(), base)
		if gotStreak != streak || got.Member.Ref != want {
			t.Fatalf("streak %d = (%d,%s), want %s", streak, gotStreak, got.Member.Ref, want)
		}
	}
}

func TestEscalationCommandsUseExecutableFrontDoors(t *testing.T) {
	want := map[int]string{
		0: "go run ./cmd/fak dispatch sweep --live",
		1: "go run ./cmd/fak dispatch sweep --live",
		2: "dos plan --workspace . --once --json",
		3: "go run ./cmd/fak-dev issue repair --live --json",
		4: "go run ./cmd/fak stale-work loop --live-issues --live-launch --json",
		5: "dos decisions --workspace . --json",
	}
	for streak, command := range want {
		if got := superloop.EscalateNoProgress(streak).Command; got != command {
			t.Errorf("streak %d command = %q, want %q", streak, got, command)
		}
	}
}

func TestNoProgressStreakMatchesDriveLedgerLoopID(t *testing.T) {
	events := []loopmgr.Event{{LoopID: "superloop-night", Kind: loopmgr.EventEnd, Status: loopmgr.StatusFailed}}
	if got := superloopNoProgressStreak(events, "night"); got != 1 {
		t.Fatalf("streak = %d, want drive ledger event to count", got)
	}
}
