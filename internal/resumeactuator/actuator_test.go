package resumeactuator

import (
	"errors"
	"reflect"
	"testing"
)

func TestManagedArgvKeepsFAKOuterAndHarnessAdjustmentSmall(t *testing.T) {
	tests := []struct {
		name string
		req  Request
		want []string
	}{
		{
			name: "legacy claude",
			req:  Request{Session: "claude-session", Prompt: "continue", ClaudeExe: "claude-bin"},
			want: []string{"fak-bin", "m", "--provider", "anthropic", "--budget-envelope", "{}", "--", "claude-bin", "--resume", "claude-session", "-p", "continue", "--dangerously-skip-permissions"},
		},
		{
			name: "codex",
			req:  Request{Harness: "CoDeX", Session: "codex-session", Rollout: "rollout.jsonl", GoalFile: "goal.json", ResultFile: "result.json", CWD: "repo"},
			want: []string{"fak-bin", "m", "--provider", "anthropic", "--budget-envelope", "{}", "--", "fak-bin", "codex-resume", "--json", "--rollout", "rollout.jsonl", "--cwd", "repo", "--prompt-file", "goal.json", "--result-file", "result.json", "codex-session"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.req.ManagedArgv("fak-bin", []string{"--provider", "anthropic"}, []string{"--budget-envelope", "{}"})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("argv mismatch\n got: %#v\nwant: %#v", got, tt.want)
			}
		})
	}
}

func TestManagedArgvFailsClosedForUnknownHarness(t *testing.T) {
	_, err := (Request{Harness: "opencode", Session: "s"}).ManagedArgv("fak", nil, nil)
	if !errors.Is(err, ErrUnknownAdapter) {
		t.Fatalf("error = %v, want ErrUnknownAdapter", err)
	}
}

func TestCodexNativeResumeNeedsOnlySession(t *testing.T) {
	got, err := (Request{Harness: "codex", Session: "s", Prompt: "continue", CodexExe: "codex-bin"}).ManagedArgv("fak", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"fak", "m", "--", "codex-bin", "exec", "resume", "--json", "--dangerously-bypass-approvals-and-sandbox", "s", "continue"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got=%q want=%q", got, want)
	}
}

func TestCodexPartialDurableCoordinatesFailClosed(t *testing.T) {
	_, err := (Request{Harness: "codex", Session: "s", Rollout: "r"}).ManagedArgv("fak", nil, nil)
	if !errors.Is(err, ErrMissingCoordinate) {
		t.Fatalf("error = %v, want ErrMissingCoordinate", err)
	}
}
