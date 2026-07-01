package dispatchtick

import (
	"reflect"
	"strings"
	"testing"
)

func TestBuildWorkerCommandMatchesBackends(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		model   string
		want    []string
	}{
		{
			name:    "claude prompt",
			backend: "claude",
			want:    []string{"claude", "-p", "--permission-mode", "bypassPermissions", "resolve it"},
		},
		{
			name:    "opencode pins model",
			backend: "opencode",
			model:   "glm-5.2",
			want:    []string{"opencode", "run", "--print-logs", "--dangerously-skip-permissions", "-m", "glm-5.2", "resolve it"},
		},
		{
			name:    "codex exec",
			backend: "codex",
			model:   "gpt-5-codex",
			want: []string{
				"codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
				"--skip-git-repo-check", "-m", "gpt-5-codex", "resolve it",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildWorkerCommand(tt.backend, "resolve it", tt.model)
			if err != nil {
				t.Fatalf("BuildWorkerCommand: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("command = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestPickTargetIssueSkipsLiveAndCooling(t *testing.T) {
	got, ok := PickTargetIssue([]int{10, 11, 12}, map[int]bool{10: true, 11: true})
	if !ok || got != 12 {
		t.Fatalf("PickTargetIssue = %d/%v, want 12/true", got, ok)
	}
	if got, ok := PickTargetIssue([]int{10, 11}, map[int]bool{10: true, 11: true}); ok || got != 0 {
		t.Fatalf("PickTargetIssue all skipped = %d/%v, want 0/false", got, ok)
	}
}

func TestWaveMembershipEnvAndAccountSidecar(t *testing.T) {
	env := WaveMembershipEnv(Membership{Rank: 2, WaveID: "wave-abc", Size: 5, Shortfall: 1})
	wantEnv := map[string]string{
		"FLEET_WAVE_ID":        "wave-abc",
		"FLEET_WAVE_RANK":      "2",
		"FLEET_WAVE_SIZE":      "5",
		"FLEET_WAVE_SHORTFALL": "1",
	}
	if !reflect.DeepEqual(env, wantEnv) {
		t.Fatalf("membership env = %#v, want %#v", env, wantEnv)
	}

	side := AccountSidecar(Account{Tag: "acct-a", Tier: float64(2), Model: "glm", Dir: "C:/acct"})
	if side["tag"] != "acct-a" || side["tier"] != float64(2) || side["model"] != "glm" || side["dir"] != "C:/acct" {
		t.Fatalf("account sidecar = %#v", side)
	}
}

func TestGuardedLaunchCommand(t *testing.T) {
	raw := []string{"claude", "-p", "prompt"}
	got, guarded := GuardedLaunchCommand(raw, "fak", "docs", "claude", `C:\work\fak`, "")
	if !guarded {
		t.Fatalf("GuardedLaunchCommand did not guard claude command")
	}
	if got[0] != "fak" || got[1] != "guard" || got[3] != "anthropic" || got[len(got)-3] != "claude" {
		t.Fatalf("guarded command = %#v", got)
	}

	opencode, guarded := GuardedLaunchCommand([]string{"opencode", "run", "prompt"}, "fak", "docs", "opencode", "/repo", "")
	if guarded || opencode[0] != "opencode" {
		t.Fatalf("opencode without base URL must not be guarded, got %#v guarded=%v", opencode, guarded)
	}
}

func TestLaunchCommandShapeRedactsSensitiveFields(t *testing.T) {
	raw := []string{
		`C:\private\fak\fak.exe`, "guard",
		"--base-url", "https://oauth-token@node.example/v1?api_key=sk-live",
		"--api-key", "sk-live",
		"--audit", `C:\private\fak\.dispatch-runs\guard-acct-secret.audit.jsonl`,
		"--", "claude", "-p", "<resolve #1783 prompt>",
	}
	got := LaunchCommandShape(raw, `C:\private\fak`, Account{
		Tag: "acct-secret",
		Dir: `C:\Users\USER\.claude\acct-secret`,
	})
	joined := strings.Join(got, " ")
	for _, leak := range []string{`C:\private\fak`, "acct-secret", "oauth-token", "api_key", "sk-live"} {
		if strings.Contains(joined, leak) {
			t.Fatalf("launch command shape leaked %q: %#v", leak, got)
		}
	}
	for _, want := range []string{"<workspace>", "<account>", "guard", "--base-url", "https://node.example/v1", "--api-key", "<redacted>", "claude"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("launch command shape missing %q: %#v", want, got)
		}
	}
}
