package dispatchtick

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestBuildWorkerCommandClaudeFastMode(t *testing.T) {
	got, err := BuildWorkerCommand("claude", "task", WorkerLaunch{Speed: "fast"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"claude", "-p", "--permission-mode", "bypassPermissions", "--settings", `{"fastMode":true}`, "task"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("command = %#v, want %#v", got, want)
	}
}

func TestBuildWorkerCommandSpeedIgnoredForNonClaude(t *testing.T) {
	got, err := BuildWorkerCommand("codex", "task", WorkerLaunch{Speed: "fast"})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(got, "--settings") {
		t.Fatalf("non-Claude command leaked speed settings: %#v", got)
	}
}
func TestBuildWorkerCommandMatchesBackends(t *testing.T) {
	tests := []struct {
		name      string
		backend   string
		model     string
		fallback  string
		effort    string
		ultracode bool
		want      []string
	}{
		{
			name:    "claude prompt, no fallback",
			backend: "claude",
			want:    []string{"claude", "-p", "--permission-mode", "bypassPermissions", "resolve it"},
		},
		{
			// The headless worker gets --fallback-model (Claude-specific, print-mode scoped)
			// before the trailing prompt so an unattended turn degrades to the backup model
			// on a transient overload instead of dying on the walled default.
			name:     "claude prompt with fallback chain",
			backend:  "claude",
			fallback: "claude-opus-4-8,claude-sonnet-5",
			want:     []string{"claude", "-p", "--permission-mode", "bypassPermissions", "--fallback-model", "claude-opus-4-8,claude-sonnet-5", "resolve it"},
		},
		{
			// Layer 4: an explicit primary model un-blanks --model, before the prompt.
			name:    "claude prompt with primary model",
			backend: "claude",
			model:   "claude-sonnet-5",
			want:    []string{"claude", "-p", "--permission-mode", "bypassPermissions", "--model", "claude-sonnet-5", "resolve it"},
		},
		{
			// --model precedes --fallback-model (launcher ordering).
			name:     "claude primary model then fallback chain",
			backend:  "claude",
			model:    "claude-sonnet-5",
			fallback: "claude-opus-4-8",
			want:     []string{"claude", "-p", "--permission-mode", "bypassPermissions", "--model", "claude-sonnet-5", "--fallback-model", "claude-opus-4-8", "resolve it"},
		},
		{
			// Per-issue tier uplift: --effort follows --model and precedes --fallback-model.
			name:    "claude primary model with effort",
			backend: "claude",
			model:   "claude-fable-5",
			effort:  "xhigh",
			want:    []string{"claude", "-p", "--permission-mode", "bypassPermissions", "--model", "claude-fable-5", "--effort", "xhigh", "resolve it"},
		},
		{
			// Ultracode supersedes a bare --effort: emit --settings, never --effort, even when
			// both are set. --settings sits where --effort would, before --fallback-model.
			name:      "claude ultracode supersedes effort",
			backend:   "claude",
			model:     "claude-opus-4-8",
			effort:    "xhigh",
			ultracode: true,
			fallback:  "claude-sonnet-5",
			want:      []string{"claude", "-p", "--permission-mode", "bypassPermissions", "--model", "claude-opus-4-8", "--settings", `{"ultracode":true}`, "--fallback-model", "claude-sonnet-5", "resolve it"},
		},
		{
			// Effort/ultracode are Claude-only: opencode ignores both.
			name:      "opencode ignores effort and ultracode",
			backend:   "opencode",
			model:     "glm-5.2",
			effort:    "xhigh",
			ultracode: true,
			want:      []string{"opencode", "run", "--print-logs", "--dangerously-skip-permissions", "-m", "glm-5.2", OpencodePromptNotice},
		},
		{
			// --fallback-model is Claude-only: opencode pins its own model with -m and never
			// gets the Claude flag even when a fallback is passed.
			name:     "opencode ignores claude fallback",
			backend:  "opencode",
			model:    "glm-5.2",
			fallback: "claude-opus-4-8",
			want:     []string{"opencode", "run", "--print-logs", "--dangerously-skip-permissions", "-m", "glm-5.2", OpencodePromptNotice},
		},
		{
			name:    "opencode pins model",
			backend: "opencode",
			model:   "glm-5.2",
			want:    []string{"opencode", "run", "--print-logs", "--dangerously-skip-permissions", "-m", "glm-5.2", OpencodePromptNotice},
		},
		{
			name:    "codex exec",
			backend: "codex",
			model:   "gpt-5-codex",
			want: []string{
				"codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
				"--skip-git-repo-check", "-m", "gpt-5-codex", "-",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildWorkerCommand(tt.backend, "resolve it", WorkerLaunch{
				Model:     tt.model,
				Fallback:  tt.fallback,
				Effort:    tt.effort,
				Ultracode: tt.ultracode,
				AccountTag: func() string {
					if tt.backend == "opencode" {
						return "test-seat"
					}
					return ""
				}(),
				AccountDir: func() string {
					if tt.backend == "opencode" {
						return "/tmp/opencode"
					}
					return ""
				}(),
				TaskTier: func() int {
					if tt.backend == "opencode" {
						return 3
					}
					return 0
				}(),
			})
			if err != nil {
				t.Fatalf("BuildWorkerCommand: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("command = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestWorkerModelPolicyArgs(t *testing.T) {
	cases := []struct {
		name         string
		backend      string
		policy       WorkerModelPolicy
		wantModel    string
		wantFallback string
	}{
		{"claude empty floor", "claude", WorkerModelPolicy{}, "", ""},
		{"claude primary only", "claude", WorkerModelPolicy{Primary: "claude-sonnet-5"}, "claude-sonnet-5", ""},
		{"claude primary + chain deduped", "claude", WorkerModelPolicy{Primary: "fable", Chain: []string{"fable", "claude-opus-4-8", "claude-opus-4-8", "claude-sonnet-5"}}, "fable", "claude-opus-4-8,claude-sonnet-5"},
		{"claude chain element comma-split", "claude", WorkerModelPolicy{Primary: "fable", Chain: []string{"claude-opus-4-8,claude-sonnet-5"}}, "fable", "claude-opus-4-8,claude-sonnet-5"},
		{"opencode primary, no claude fallback", "opencode", WorkerModelPolicy{Primary: "glm-5.2", Chain: []string{"claude-opus-4-8"}}, "glm-5.2", ""},
		{"codex primary, no claude fallback", "codex", WorkerModelPolicy{Primary: "gpt-5-codex"}, "gpt-5-codex", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.policy.Model(); got != tc.wantModel {
				t.Fatalf("Model() = %q, want %q", got, tc.wantModel)
			}
			if got := tc.policy.FallbackModel(tc.backend); got != tc.wantFallback {
				t.Fatalf("FallbackModel(%q) = %q, want %q", tc.backend, got, tc.wantFallback)
			}
		})
	}
}

func TestWorkerStdinPayload(t *testing.T) {
	if got := WorkerStdinPayload("codex", "resolve it"); got != "resolve it" {
		t.Fatalf("codex stdin payload = %q, want prompt", got)
	}
	if got := WorkerStdinPayload("claude", "resolve it"); got != "" {
		t.Fatalf("claude stdin payload = %q, want empty", got)
	}
	if got := WorkerStdinPayload("opencode", "resolve it"); got != "resolve it" {
		t.Fatalf("opencode stdin payload = %q, want prompt", got)
	}
}

func TestOpencodePromptNoticeKeepsLivenessMarker(t *testing.T) {
	if !strings.Contains(strings.ToLower(OpencodePromptNotice), "resolve github issue #") {
		t.Fatalf("opencode prompt notice %q lost the issue-worker liveness marker", OpencodePromptNotice)
	}
	if len(OpencodePromptNotice) > 96 {
		t.Fatalf("opencode prompt notice is too long for argv safety: %d", len(OpencodePromptNotice))
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
	// #3607: a dispatch worker launches with the curated headless tool-surface profile, as a
	// guard flag immediately before the `--` command separator (never a claude argument).
	if !strings.Contains(strings.Join(got, " "), "--expose-profile headless --") {
		t.Fatalf("guarded command must carry `--expose-profile headless` before `--` (#3607): %#v", got)
	}

	opencode, guarded := GuardedLaunchCommand([]string{"opencode", "run", "prompt"}, "fak", "docs", "opencode", "/repo", "")
	if guarded || opencode[0] != "opencode" {
		t.Fatalf("opencode without base URL must not be guarded, got %#v guarded=%v", opencode, guarded)
	}

	subscriptionCodex, guarded := GuardedLaunchCommand([]string{"codex", "exec", "-"}, "fak", "docs", "codex", "/repo", "")
	if !guarded || slices.Contains(subscriptionCodex, "--provider") || slices.Contains(subscriptionCodex, "--base-url") {
		t.Fatalf("subscription Codex dispatch must defer upstream selection to guard: %v", subscriptionCodex)
	}

	if runtime.GOOS == "windows" {
		dir := t.TempDir()
		shim := filepath.Join(dir, "codex.cmd")
		if err := os.WriteFile(shim, []byte("@echo off\n"), 0755); err != nil {
			t.Fatal(err)
		}
		t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	}

	codex, guarded := GuardedLaunchCommand([]string{"codex", "exec", "-"}, "fak", "docs", "codex", "/repo", "http://127.0.0.1:18080/v1")
	if !guarded || slices.Contains(codex, "--provider") {
		t.Fatalf("codex dispatch must carry the already-evaluated loop-gate decision: %v", codex)
	}
	if runtime.GOOS == "windows" && !strings.Contains(strings.Join(codex, " "), "codex.cmd exec -") {
		t.Fatalf("Windows Codex dispatch must hand guard the concrete batch shim: %v", codex)
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

func TestBuildWorkerCommandRejectsUnboundOpenCode(t *testing.T) {
	if _, err := BuildWorkerCommand("opencode", "resolve it", WorkerLaunch{Model: "glm-5.2", RequireAccountBound: true}); err == nil || !strings.Contains(err.Error(), "account-bound") {
		t.Fatalf("err = %v, want account-bound refusal", err)
	}
}

func TestGuardedLaunchCommandEdgeAndAdversarialInputs(t *testing.T) {
	tests := []struct {
		name      string
		command   []string
		fakBin    string
		backend   string
		baseURL   string
		guarded   bool
		wantExact []string
		want      []string
		notWant   []string
	}{
		{name: "empty command refuses guard without panic", command: nil, fakBin: "fak", backend: "codex", guarded: false, wantExact: nil},
		{name: "blank fak binary preserves command", command: []string{"codex", "exec", "-"}, fakBin: " \t", backend: "codex", guarded: false, wantExact: []string{"codex", "exec", "-"}},
		{name: "non-Codex OpenAI backend requires endpoint", command: []string{"opencode", "run", "prompt"}, fakBin: "fak", backend: "opencode", guarded: false, wantExact: []string{"opencode", "run", "prompt"}},
		{name: "Codex subscription rejects whitespace endpoint without emitting it", command: []string{"codex", "exec", "--", "hostile;still-argv"}, fakBin: "fak", backend: "codex", baseURL: " \r\n", guarded: true, want: []string{"--codex-loop-gate", "off", "hostile;still-argv"}, notWant: []string{"--provider", "--base-url"}},
		{name: "Codex explicit endpoint remains one argv value", command: []string{"codex", "exec", "-"}, fakBin: "fak", backend: "codex", baseURL: "http://127.0.0.1:18080/v1?x=a;b", guarded: true, want: []string{"--base-url", "http://127.0.0.1:18080/v1?x=a;b"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, guarded := GuardedLaunchCommand(tc.command, tc.fakBin, "docs", tc.backend, "/repo", tc.baseURL)
			if guarded != tc.guarded {
				t.Fatalf("guarded = %v, want %v; argv=%q", guarded, tc.guarded, got)
			}
			if tc.wantExact != nil && !reflect.DeepEqual(got, tc.wantExact) {
				t.Fatalf("argv = %#v, want %#v", got, tc.wantExact)
			}
			for _, arg := range tc.want {
				if !slices.Contains(got, arg) {
					t.Errorf("argv %q missing exact argument %q", got, arg)
				}
			}
			for _, arg := range tc.notWant {
				if slices.Contains(got, arg) {
					t.Errorf("argv %q unexpectedly contains %q", got, arg)
				}
			}
		})
	}
}

func TestGuardedLaunchCommandDeterminism(t *testing.T) {
	command := []string{"codex", "exec", "--json", "-"}
	first, firstGuarded := GuardedLaunchCommand(command, "fak", "docs", "codex", "/repo", "")
	for i := 0; i < 100; i++ {
		got, guarded := GuardedLaunchCommand(command, "fak", "docs", "codex", "/repo", "")
		if guarded != firstGuarded || !reflect.DeepEqual(got, first) {
			t.Fatalf("run %d differs: guarded=%v argv=%#v; first guarded=%v argv=%#v", i+2, guarded, got, firstGuarded, first)
		}
	}
	first[0] = "mutated"
	if command[0] != "codex" {
		t.Fatalf("returned argv aliases caller command: %#v", command)
	}
}

func TestBuildWorkerCommandRefusalsNameRecovery(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		launch  WorkerLaunch
		want    []string
	}{
		{name: "opencode account binding", backend: "opencode", launch: WorkerLaunch{RequireAccountBound: true}, want: []string{"resolved account record", "task tier"}},
		{name: "unknown backend", backend: "bogus", want: []string{"expected", "claude", "opencode", "codex"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildWorkerCommand(tc.backend, "task", tc.launch)
			if err == nil {
				t.Fatal("expected refusal")
			}
			for _, recovery := range tc.want {
				if !strings.Contains(err.Error(), recovery) {
					t.Errorf("refusal %q does not name recovery %q", err, recovery)
				}
			}
		})
	}
}
