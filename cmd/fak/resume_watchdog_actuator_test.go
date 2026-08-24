package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

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
		{name: "codex", plan: resume.WatchdogPlanRow{Harness: "codex", Session: "codex-session", Rollout: "rollout.jsonl", GoalFile: "goal.json", ResultFile: "result.json"}, want: "fak-bin m --codex-loop-gate off -- fak-bin codex-resume"},
		{name: "opencode", plan: resume.WatchdogPlanRow{Harness: "opencode", Session: "opencode-session"}, want: "fak-bin m -- opencode run --session opencode-session"},
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
	_, err := rwManagedResumeArgv("fak", "claude", resume.WatchdogPlanRow{Harness: "gemini", Session: "s"}, nil)
	if !errors.Is(err, resumeactuator.ErrUnknownAdapter) {
		t.Fatalf("error = %v, want ErrUnknownAdapter", err)
	}
}

func TestResumeWatchdogOpenCodeSpawnDoesNotRequireClaude(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "launched")
	var argv []string
	if runtime.GOOS == "windows" {
		script := filepath.Join(dir, "opencode.cmd")
		if err := os.WriteFile(script, []byte("@echo launched>\"%~1\"\r\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		argv = []string{"cmd.exe", "/c", script, marker}
	} else {
		script := filepath.Join(dir, "opencode")
		if err := os.WriteFile(script, []byte("#!/bin/sh\nprintf launched > \"$1\"\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		argv = []string{script, marker}
	}
	p := resume.WatchdogPlanRow{Harness: resumeactuator.HarnessOpenCode, Session: "opencode-session", CWD: dir}
	grant := launchBrokerGrant{Argv: argv, Env: envMap(os.Environ()), CWD: dir}
	if _, err := rwSpawnResume("", p, "", dir, grant); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(marker); err == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("OpenCode process was not launched")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
