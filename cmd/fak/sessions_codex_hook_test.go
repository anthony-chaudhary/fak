package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unicode/utf16"
)

type codexProjectCommandHook struct {
	Type           string `json:"type"`
	Command        string `json:"command"`
	CommandWindows string `json:"commandWindows"`
}

func TestCodexLoopHookBlocksActiveDirectContinuation(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
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
	h := loadCodexProjectHook(t)
	if h.Type != "command" {
		t.Fatalf("UserPromptSubmit hook type = %q, want command", h.Type)
	}
	for name, command := range map[string]string{
		"portable": h.Command,
		"windows":  decodeCodexWindowsHook(t, h.CommandWindows),
	} {
		for _, want := range []string{
			"fak sessions codex-loop-hook",
			"codex_session_guard_unavailable",
			"fak codex",
			"fak guard -- codex",
		} {
			if !strings.Contains(command, want) {
				t.Errorf("%s UserPromptSubmit command missing %q: %s", name, want, command)
			}
		}
	}
}

func decodeCodexWindowsHook(t *testing.T, command string) string {
	t.Helper()
	const marker = "-EncodedCommand "
	i := strings.Index(command, marker)
	if i < 0 {
		t.Fatalf("Windows UserPromptSubmit hook is not shell-neutral PowerShell: %s", command)
	}
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(command[i+len(marker):]))
	if err != nil {
		t.Fatalf("decode Windows UserPromptSubmit hook: %v", err)
	}
	if len(raw)%2 != 0 {
		t.Fatalf("Windows UserPromptSubmit hook UTF-16LE bytes = %d, want even", len(raw))
	}
	units := make([]uint16, len(raw)/2)
	for i := range units {
		units[i] = uint16(raw[2*i]) | uint16(raw[2*i+1])<<8
	}
	return string(utf16.Decode(units))
}

func TestCodexLoopHookConfigFailsClosedWhenFakIsStale(t *testing.T) {
	h := loadCodexProjectHook(t)
	fakeDir := t.TempDir()
	writeCodexHookFakeFak(t, fakeDir, true)

	cmd := codexProjectHookCommand(t, h)
	cmd.Env = append(os.Environ(), "PATH="+fakeDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	cmd.Stdin = strings.NewReader("{\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"do not send\"}\n")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("stale fak hook must return a parseable block: %v; stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	var got codexLoopHookBlock
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("stale fak block is not JSON: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	for _, want := range []string{
		"codex_session_guard_unavailable",
		"Rebuild or install a current fak",
		"fak codex",
		"fak guard -- codex",
	} {
		if !strings.Contains(got.Reason, want) {
			t.Errorf("stale fak block reason missing %q: %s", want, got.Reason)
		}
	}
	if got.Decision != "block" {
		t.Errorf("stale fak decision = %q, want block", got.Decision)
	}
	if stderr.Len() != 0 {
		t.Errorf("stale fak usage leaked to stderr: %s", stderr.String())
	}
}

func TestCodexLoopHookConfigPreservesCurrentFakBlock(t *testing.T) {
	h := loadCodexProjectHook(t)
	fakeDir := t.TempDir()
	writeCodexHookFakeFak(t, fakeDir, false)

	cmd := codexProjectHookCommand(t, h)
	cmd.Env = append(os.Environ(), "PATH="+fakeDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	cmd.Stdin = strings.NewReader("{\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"do not send\"}\n")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("current fak hook: %v; stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	var got codexLoopHookBlock
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("current fak block is not JSON: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	if got.Decision != "block" || got.Reason != "current fak block" {
		t.Fatalf("current fak block = %+v, want pass-through block", got)
	}
}

func TestCodexLoopHookConfigAllowsCurrentGuardedFak(t *testing.T) {
	h := loadCodexProjectHook(t)
	fakeDir := t.TempDir()
	writeCodexHookAllowingFakeFak(t, fakeDir)

	cmd := codexProjectHookCommand(t, h)
	cmd.Env = append(os.Environ(), "PATH="+fakeDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	cmd.Stdin = strings.NewReader("{\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"send guarded turn\"}\n")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("current guarded fak hook: %v; stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	if stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatalf("current guarded fak allow must stay silent: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

func loadCodexProjectHook(t *testing.T) codexProjectCommandHook {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRootFromTest(t), ".codex", "hooks.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Hooks map[string][]struct {
			Hooks []codexProjectCommandHook `json:"hooks"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("decode .codex/hooks.json: %v", err)
	}
	groups := doc.Hooks["UserPromptSubmit"]
	if len(groups) != 1 || len(groups[0].Hooks) != 1 {
		t.Fatalf("UserPromptSubmit hook groups = %+v, want exactly one command hook", groups)
	}
	return groups[0].Hooks[0]
}

func codexProjectHookCommand(t *testing.T, h codexProjectCommandHook) *exec.Cmd {
	t.Helper()
	if runtime.GOOS == "windows" {
		if h.CommandWindows == "" {
			t.Fatal("UserPromptSubmit hook has no commandWindows fallback")
		}
		// Codex hands commandWindows to the native Windows command shell. Running
		// this fixture through cmd.exe catches the live single-quote bug that a
		// PowerShell-only test hid: cmd treats ' as a literal, not a quote.
		cmd := exec.Command("cmd.exe", "/d", "/s", "/c", h.CommandWindows)
		cmd.Dir = filepath.Join(repoRootFromTest(t), "docs")
		return cmd
	}
	cmd := exec.Command("sh", "-c", h.Command)
	cmd.Dir = filepath.Join(repoRootFromTest(t), "docs")
	return cmd
}

func writeCodexHookFakeFak(t *testing.T, dir string, stale bool) {
	t.Helper()
	if runtime.GOOS == "windows" {
		body := "@echo off\r\nmore >nul\r\n"
		if stale {
			body += "echo fak sessions: unknown subcommand codex-loop-hook 1>&2\r\nexit /b 2\r\n"
		} else {
			body += "echo {\"decision\":\"block\",\"reason\":\"current fak block\"}\r\nexit /b 0\r\n"
		}
		if err := os.WriteFile(filepath.Join(dir, "fak.cmd"), []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
		return
	}
	body := "#!/bin/sh\ncat >/dev/null\n"
	if stale {
		body += "printf '%s\\n' 'fak sessions: unknown subcommand codex-loop-hook' >&2\nexit 2\n"
	} else {
		body += "printf '%s\\n' '{\"decision\":\"block\",\"reason\":\"current fak block\"}'\nexit 0\n"
	}
	path := filepath.Join(dir, "fak")
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}

func writeCodexHookAllowingFakeFak(t *testing.T, dir string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		body := "@echo off\r\nmore >nul\r\nexit /b 0\r\n"
		if err := os.WriteFile(filepath.Join(dir, "fak.cmd"), []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
		return
	}
	path := filepath.Join(dir, "fak")
	if err := os.WriteFile(path, []byte("#!/bin/sh\ncat >/dev/null\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
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
