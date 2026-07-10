package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCodexLoopHookBlocksActiveDirectContinuation(t *testing.T) {
	home, sessionID := writeCodexHookSession(t, "openai")
	payload := `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit","turn_id":"turn-next"}`

	var stdout, stderr bytes.Buffer
	code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home})
	if code != 0 {
		t.Fatalf("hook exit = %d, want 0 with a JSON block decision; stderr=%s", code, stderr.String())
	}
	var got codexLoopHookBlock
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode hook response: %v\n%s", err, stdout.String())
	}
	if got.Decision != "block" {
		t.Fatalf("decision = %q, want block: %+v", got.Decision, got)
	}
	for _, want := range []string{
		"codex_session_bypassed_fak_guard",
		"model_provider=openai",
		"fak codex",
		"fak guard -- codex",
		codexLoopHookOverrideEnv + "=1",
	} {
		if !strings.Contains(got.Reason, want) {
			t.Errorf("block reason missing %q: %s", want, got.Reason)
		}
	}
}

func TestCodexLoopHookAllowsGuardedAndExplicitOverride(t *testing.T) {
	for _, tc := range []struct {
		name     string
		provider string
		override string
	}{
		{name: "guarded fak provider", provider: "fak"},
		{name: "explicit direct override", provider: "openai", override: "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home, sessionID := writeCodexHookSession(t, tc.provider)
			t.Setenv(codexLoopHookOverrideEnv, tc.override)
			payload := `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit"}`
			var stdout, stderr bytes.Buffer
			if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
				t.Fatalf("hook exit = %d, stderr=%s", code, stderr.String())
			}
			if stdout.Len() != 0 {
				t.Fatalf("allowed continuation emitted a block: %s", stdout.String())
			}
		})
	}
}

func TestCodexLoopHookConfigWiresUserPromptSubmit(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(repoRootFromTest(t), ".codex", "hooks.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Hooks map[string][]struct {
			Hooks []struct {
				Type    string `json:"type"`
				Command string `json:"command"`
			} `json:"hooks"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("decode .codex/hooks.json: %v", err)
	}
	groups := doc.Hooks["UserPromptSubmit"]
	if len(groups) != 1 || len(groups[0].Hooks) != 1 {
		t.Fatalf("UserPromptSubmit hook groups = %+v, want exactly one command hook", groups)
	}
	h := groups[0].Hooks[0]
	if h.Type != "command" || h.Command != "fak sessions codex-loop-hook" {
		t.Fatalf("UserPromptSubmit hook = %+v, want fak continuation gate", h)
	}
}

func writeCodexHookSession(t *testing.T, provider string) (home, sessionID string) {
	t.Helper()
	home = filepath.Join(t.TempDir(), "codex-home")
	sessionID = "019f3023-52dd-7001-b559-2818dc14ede6"
	dir := filepath.Join(home, "sessions", "2026", "07", "10")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "rollout-2026-07-10T10-00-00-"+sessionID+".jsonl")
	writeCodexLoopFixture(t, path, []string{
		`{"timestamp":"2026-07-10T17:00:00.000Z","type":"session_meta","payload":{"session_id":"` + sessionID + `","originator":"codex-tui","cli_version":"0.142.5","model_provider":"` + provider + `","git":{"branch":"main"}}}`,
	})
	return home, sessionID
}
