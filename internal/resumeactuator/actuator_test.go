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
			want: []string{"fak-bin", "m", "--codex-loop-gate", "off", "--provider", "anthropic", "--budget-envelope", "{}", "--", "fak-bin", "codex-resume", "--json", "--rollout", "rollout.jsonl", "--cwd", "repo", "--prompt-file", "goal.json", "--result-file", "result.json", "codex-session"},
		},
		{
			name: "opencode",
			req:  Request{Harness: "OpenCode", Session: "opencode-session", Prompt: "continue", OpenCodeExe: "opencode-bin"},
			want: []string{"fak-bin", "m", "--provider", "anthropic", "--budget-envelope", "{}", "--", "opencode-bin", "run", "--session", "opencode-session", "continue"},
		},
		{
			name: "fak",
			req:  Request{Harness: HarnessFak, Session: "fak-session", Prompt: "continue", FakExe: "fak-custom"},
			want: []string{"fak-bin", "m", "--provider", "anthropic", "--budget-envelope", "{}", "--", "fak-custom", "agent", "--native", "--resume", "fak-session", "--task", "continue"},
		},
		{
			name: "fak-native",
			req:  Request{Harness: HarnessFakNative, Session: "native-session"},
			want: []string{"fak-bin", "m", "--provider", "anthropic", "--budget-envelope", "{}", "--", "fak-bin", "agent", "--native", "--resume", "native-session"},
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
	_, err := (Request{Harness: "gemini", Session: "s"}).ManagedArgv("fak", nil, nil)
	if !errors.Is(err, ErrUnknownAdapter) {
		t.Fatalf("error = %v, want ErrUnknownAdapter", err)
	}
}

func TestCodexNativeResumeNeedsOnlySession(t *testing.T) {
	got, err := (Request{Harness: "codex", Session: "s", Prompt: "continue", CodexExe: "codex-bin"}).ManagedArgv("fak", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"fak", "m", "--codex-loop-gate", "off", "--", "codex-bin", "exec", "resume", "--json", "--dangerously-bypass-approvals-and-sandbox", "s", "continue"}
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

func TestOpenCodeResumeRequiresSession(t *testing.T) {
	_, err := (Request{Harness: HarnessOpenCode}).ManagedArgv("fak", nil, nil)
	if !errors.Is(err, ErrMissingCoordinate) {
		t.Fatalf("error = %v, want ErrMissingCoordinate", err)
	}
}

func TestFakResumeRequiresSession(t *testing.T) {
	_, err := (Request{Harness: HarnessFak}).ManagedArgv("fak", nil, nil)
	if !errors.Is(err, ErrMissingCoordinate) {
		t.Fatalf("error = %v, want ErrMissingCoordinate", err)
	}
	_, err = (Request{Harness: HarnessFakNative}).ManagedArgv("fak", nil, nil)
	if !errors.Is(err, ErrMissingCoordinate) {
		t.Fatalf("error = %v, want ErrMissingCoordinate", err)
	}
}

func TestFakHarnessNormalization(t *testing.T) {
	variants := []string{"fak", "FAK", "fak-native", "FaK-NaTiVe", "  fak  ", "  fak-native  "}
	for _, v := range variants {
		name, err := (Request{Harness: v}).HarnessName()
		if err != nil {
			t.Fatalf("variant %q failed: %v", v, err)
		}
		if name != HarnessFak {
			t.Errorf("variant %q got %q, want %q", v, name, HarnessFak)
		}
	}
}

func TestFakContinuationArgv(t *testing.T) {
	// 1. Without prompt, default fakExe
	got, err := (Request{Harness: HarnessFak, Session: "s1"}).ContinuationArgv("fak")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"fak", "agent", "--native", "--resume", "s1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}

	// 2. With prompt
	got, err = (Request{Harness: HarnessFakNative, Session: "s2", Prompt: "do work"}).ContinuationArgv("fak-bin")
	if err != nil {
		t.Fatal(err)
	}
	want = []string{"fak-bin", "agent", "--native", "--resume", "s2", "--task", "do work"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}

	// 3. With explicit FakExe override in Request
	got, err = (Request{Harness: HarnessFak, Session: "s3", FakExe: "/usr/local/bin/fak"}).ContinuationArgv("fak-bin")
	if err != nil {
		t.Fatal(err)
	}
	want = []string{"/usr/local/bin/fak", "agent", "--native", "--resume", "s3"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
