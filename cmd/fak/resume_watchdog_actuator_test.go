package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resumeactuator"
)

func TestResumeWatchdogBrokerUsesFakManagedEnvelopeForEveryHarness(t *testing.T) {
	tests := []struct {
		name string
		plan resume.WatchdogPlanRow
		want string
	}{
		{name: "claude", plan: resume.WatchdogPlanRow{Session: "claude-session"}, want: "fak-bin m -- claude-bin --resume claude-session"},
		{name: "codex", plan: resume.WatchdogPlanRow{Harness: "codex", Session: "codex-session", Rollout: "rollout.jsonl", GoalFile: "goal.json", ResultFile: "result.json"}, want: "fak-bin m -- fak-bin codex-resume"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := rwResumeBrokerAttempt("fak-bin", "claude-bin", tt.plan, ".claude-test", nil)
			if joined := strings.Join(got.Argv, " "); !strings.Contains(joined, tt.want) {
				t.Fatalf("argv = %q, want managed envelope containing %q", joined, tt.want)
			}
		})
	}
}

func TestResumeWatchdogManagedArgvRejectsUnsupportedHarness(t *testing.T) {
	_, err := rwManagedResumeArgv("fak", "claude", resume.WatchdogPlanRow{Harness: "opencode", Session: "s"}, nil)
	if !errors.Is(err, resumeactuator.ErrUnknownAdapter) {
		t.Fatalf("error = %v, want ErrUnknownAdapter", err)
	}
}
