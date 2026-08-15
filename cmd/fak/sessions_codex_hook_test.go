package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

type codexProjectCommandHook struct {
	Type           string `json:"type"`
	Command        string `json:"command"`
	CommandWindows string `json:"commandWindows"`
	StatusMessage  string `json:"statusMessage"`
}

func TestCodexLoopHookBlocksActiveDirectContinuation(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	// An ambient FAK_GUARD_ACTIVE (set inside every `fak guard` session, so on any
	// developer box but not on a bare CI runner) short-circuits the hook to
	// allow-silently and leaves stdout empty. Neutralize it or the block assertion
	// below dies on "unexpected end of JSON input" locally while passing in CI.
	t.Setenv(guardActiveEnv, "")
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
	overrideInstruction := "prefix the Codex command with `FAK_ALLOW_DIRECT_CODEX_CONTINUE=1`"
	if runtime.GOOS == "windows" {
		overrideInstruction = "set `$env:FAK_ALLOW_DIRECT_CODEX_CONTINUE=1` in PowerShell"
	}
	for _, want := range []string{
		"codex_session_bypassed_fak_guard",
		"model_provider=openai",
		"fak codex",
		"fak guard -- codex",
		"--allow-direct",
		overrideInstruction,
	} {
		if !strings.Contains(got.Reason, want) {
			t.Errorf("block reason missing %q: %s", want, got.Reason)
		}
	}
}

func TestCodexLoopHookBlockWritesExactlyOneAuditWitness(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "openai")
	journal := filepath.Join(t.TempDir(), "audit.jsonl")
	t.Setenv(codexLoopHookAuditJournalEnv, journal)
	payload := `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit"}`

	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
		t.Fatalf("hook exit = %d; stderr=%s", code, stderr.String())
	}
	var block codexLoopHookBlock
	if err := json.Unmarshal(stdout.Bytes(), &block); err != nil || block.Decision != "block" {
		t.Fatalf("stdout must remain the Codex block JSON only: err=%v stdout=%q", err, stdout.String())
	}
	raw, err := os.ReadFile(journal)
	if err != nil {
		t.Fatalf("read audit witness: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		t.Fatalf("audit witness lines = %d, want exactly 1: %q", len(lines), raw)
	}
	var witness codexLoopHookBlockWitness
	if err := json.Unmarshal([]byte(lines[0]), &witness); err != nil {
		t.Fatalf("decode audit witness: %v", err)
	}
	if witness.Event != "codex_continuation_guard_block" || witness.SessionID != sessionID || witness.ModelProvider != "openai" || witness.Reason != "codex_session_bypassed_fak_guard" {
		t.Fatalf("unexpected audit witness: %+v", witness)
	}
}

func TestCodexLoopHookDeterminismByteIdentical(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "openai")
	payload := `{"session_id":"` + sessionID + `","turn_id":"deterministic"}`
	var outputs [2][]byte
	for i := range outputs {
		var stdout, stderr bytes.Buffer
		if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
			t.Fatalf("run %d exit=%d stderr=%s", i, code, stderr.String())
		}
		outputs[i] = append([]byte(nil), stdout.Bytes()...)
	}
	if !bytes.Equal(outputs[0], outputs[1]) {
		t.Fatalf("block envelope changed across identical runs:\n1=%s\n2=%s", outputs[0], outputs[1])
	}
}

func TestCodexLoopHookConcurrentAppendNeverAllowsDirectProvider(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "openai")
	path, err := resolveCodexLoopSessionPath(home, sessionID, "")
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	writerDone := make(chan error, 1)
	go func() {
		fh, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
		if err != nil {
			writerDone <- err
			return
		}
		defer fh.Close()
		line := []byte(`{"timestamp":"2026-07-11T23:00:00Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"total_tokens":1}}}}` + "\n")
		for {
			select {
			case <-stop:
				writerDone <- nil
				return
			default:
			}
			if _, err := fh.Write(line); err != nil {
				writerDone <- err
				return
			}
		}
	}()
	payload := `{"session_id":"` + sessionID + `"}`
	const readers = 24
	var wg sync.WaitGroup
	errs := make(chan string, readers)
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var stdout, stderr bytes.Buffer
			code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home})
			var block codexLoopHookBlock
			if code != 0 || json.Unmarshal(stdout.Bytes(), &block) != nil || block.Decision != "block" {
				errs <- fmt.Sprintf("exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
			}
		}()
	}
	wg.Wait()
	close(stop)
	if err := <-writerDone; err != nil {
		t.Fatal(err)
	}
	close(errs)
	for got := range errs {
		t.Error(got)
	}
}

func TestCodexLoopHookAllowsGuardedAndExplicitOverride(t *testing.T) {
	for _, tc := range []struct {
		name     string
		provider string
		override string
	}{
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

// TestCodexLoopHookGuardedMarkerSkipsReparse pins the #4210 guarded-path short-circuit:
// a session already wrapped by `fak guard` (which marks the child env with guardActiveEnv,
// inherited by this hook subprocess) is allowed WITHOUT opening or reparsing the transcript
// — even when the session's own model_provider would otherwise be diagnosed unguarded and
// blocked (contrast TestCodexLoopHookBlocksActiveDirectContinuation with the same fixture).
// This keeps the per-turn hook off the growing session file on the guarded dogfood path.
func TestCodexLoopHookSpoofedFakProviderBlocksWithoutGuardWitness(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "fak")
	payload := `{"session_id":"` + sessionID + `"}`
	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var block codexLoopHookBlock
	if err := json.Unmarshal(stdout.Bytes(), &block); err != nil || block.Decision != "block" {
		t.Fatalf("spoofed fak provider was not blocked: err=%v stdout=%q", err, stdout.String())
	}
}

func TestCodexLoopHookGuardMarkerPersistsWitnessThenAllowsWithoutEnv(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	home, sessionID := writeCodexHookSession(t, "fak")
	payload := `{"session_id":"` + sessionID + `"}`
	t.Setenv(guardActiveEnv, "1")
	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 || stdout.Len() != 0 {
		t.Fatalf("guarded first turn exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !codexGuardWitnessExists(home, sessionID) {
		t.Fatal("guarded turn did not persist session witness")
	}
	t.Setenv(guardActiveEnv, "")
	stdout.Reset()
	stderr.Reset()
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 || stdout.Len() != 0 {
		t.Fatalf("witnessed continuation exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func TestCodexLoopHookSessionIdentifierAliases(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	for _, tc := range []struct {
		name    string
		payload func(string) string
		setEnv  bool
	}{
		{name: "ThreadId", payload: func(id string) string { return `{"thread_id":"` + id + `"}` }},
		{name: "ConversationId", payload: func(id string) string { return `{"conversation_id":"` + id + `"}` }},
		{name: "EnvFallback", setEnv: true, payload: func(string) string { return `{}` }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home, sessionID := writeCodexHookSession(t, "openai")
			if tc.setEnv {
				t.Setenv("CODEX_THREAD_ID", sessionID)
			}
			var stdout, stderr bytes.Buffer
			if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(tc.payload(sessionID)), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
				t.Fatalf("hook exit = %d; stderr=%s", code, stderr.String())
			}
			var got codexLoopHookBlock
			if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
				t.Fatalf("decode block: %v; stdout=%q stderr=%q", err, stdout.String(), stderr.String())
			}
			if got.Decision != "block" || !strings.Contains(got.Reason, "model_provider=openai") {
				t.Fatalf("unexpected block: %+v", got)
			}
		})
	}
}

func TestCodexLoopHookLargeTranscriptUsesProviderProbeWithinBudget(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "openai")
	path, err := resolveCodexLoopSessionPath(home, sessionID, "")
	if err != nil {
		t.Fatal(err)
	}
	fh, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	chunk := bytes.Repeat([]byte(`{"timestamp":"2026-07-10T17:00:01Z","type":"response_item","payload":{"type":"message","content":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}}`+"\n"), 50000)
	if _, err := fh.Write(chunk); err != nil {
		t.Fatal(err)
	}
	if err := fh.Close(); err != nil {
		t.Fatal(err)
	}

	oldBudget := codexLoopHookBudget
	codexLoopHookBudget = 100 * time.Millisecond
	t.Cleanup(func() { codexLoopHookBudget = oldBudget })
	payload := `{"session_id":"` + sessionID + `"}`
	started := time.Now()
	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
		t.Fatalf("hook exit=%d stderr=%s", code, stderr.String())
	}
	if elapsed := time.Since(started); elapsed >= codexLoopHookBudget {
		t.Fatalf("provider probe took %s, budget=%s", elapsed, codexLoopHookBudget)
	}
	var block codexLoopHookBlock
	if err := json.Unmarshal(stdout.Bytes(), &block); err != nil || block.Decision != "block" {
		t.Fatalf("large direct transcript failed open: err=%v stdout=%q stderr=%q", err, stdout.String(), stderr.String())
	}
}

func TestCodexLoopHookGuardedMarkerSkipsReparse(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "") // no explicit direct override in play
	t.Setenv(guardActiveEnv, "1")          // as if spawned by `fak guard`
	home, sessionID := writeCodexHookSession(t, "openai")

	// The whole point of the marker is that the transcript is never reparsed: fail loudly
	// (from any goroutine — t.Error is safe) if the diagnose path is reached at all.
	original := codexLoopHookDiagnose
	codexLoopHookDiagnose = func(io.Reader, string) (codexLoopDiagnosis, error) {
		t.Error("guarded hook reparsed the session transcript; expected a marker short-circuit")
		return codexLoopDiagnosis{ModelProvider: "openai"}, nil
	}
	t.Cleanup(func() { codexLoopHookDiagnose = original })

	payload := `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit"}`
	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
		t.Fatalf("guarded marker path exit = %d, want allow (0); stderr=%s", code, stderr.String())
	}
	if stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatalf("guarded marker path must be byte-silent: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

func TestCodexLoopHookRecoversPanicAllowSilent(t *testing.T) {
	home, sessionID := writeCodexHookSession(t, "openai")
	payload := `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit"}`

	original := codexLoopHookDiagnose
	codexLoopHookDiagnose = func(io.Reader, string) (codexLoopDiagnosis, error) {
		panic("injected diagnose fault")
	}
	t.Cleanup(func() { codexLoopHookDiagnose = original })

	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
		t.Fatalf("panic path exit = %d, want allow (0)", code)
	}
	if stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatalf("panic path must be byte-silent: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

func TestCodexLoopHookTimesOutAllowSilent(t *testing.T) {
	home, sessionID := writeCodexHookSession(t, "openai")
	payload := `{"session_id":"` + sessionID + `","hook_event_name":"UserPromptSubmit"}`

	originalDiagnose := codexLoopHookDiagnose
	originalBudget := codexLoopHookBudget
	codexLoopHookBudget = 40 * time.Millisecond
	codexLoopHookDiagnose = func(io.Reader, string) (codexLoopDiagnosis, error) {
		time.Sleep(2 * time.Second)
		return codexLoopDiagnosis{ModelProvider: "openai"}, nil
	}
	t.Cleanup(func() {
		codexLoopHookDiagnose = originalDiagnose
		codexLoopHookBudget = originalBudget
	})

	started := time.Now()
	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", home}); code != 0 {
		t.Fatalf("timeout path exit = %d, want allow (0)", code)
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("timeout path took %s, want bounded well below the outer hook timeout", elapsed)
	}
	if stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatalf("timeout path must be byte-silent: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

func TestCodexLoopHookUnreadableSessionAllowsSilently(t *testing.T) {
	payload := `{"session_id":"missing-or-locked-session","hook_event_name":"UserPromptSubmit"}`

	var stdout, stderr bytes.Buffer
	if code := runSessionsWithStdin(&stdout, &stderr, strings.NewReader(payload), []string{"codex-loop-hook", "--codex-home", t.TempDir()}); code != 0 {
		t.Fatalf("unreadable session path exit = %d, want allow (0)", code)
	}
	if stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatalf("unreadable session path must be byte-silent: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

func TestCodexLoopHookConfigWiresUserPromptSubmit(t *testing.T) {
	h := loadCodexProjectHook(t)
	if h.Type != "command" {
		t.Fatalf("UserPromptSubmit hook type = %q, want command", h.Type)
	}
	for name, command := range map[string]string{
		"portable": h.Command,
		"windows":  h.CommandWindows,
	} {
		for _, want := range []string{"fak sessions codex-loop-hook"} {
			if !strings.Contains(command, want) {
				t.Errorf("%s UserPromptSubmit command missing %q: %s", name, want, command)
			}
		}
	}
	if strings.Contains(strings.ToLower(h.CommandWindows), "powershell") {
		t.Fatalf("Windows hook must invoke fak natively, not cold-start PowerShell: %s", h.CommandWindows)
	}
}

func TestCodexLoopHookConfigAllowsSilentlyWhenFakIsStale(t *testing.T) {
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
		t.Fatalf("stale fak hook must fail open: %v; stdout=%s stderr=%s", err, stdout.String(), stderr.String())
	}
	if stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatalf("stale fak allow must stay silent: stdout=%q stderr=%q", stdout.String(), stderr.String())
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
	if groups[0].Hooks[0].StatusMessage != "" {
		t.Fatalf("UserPromptSubmit leaked per-turn status %q", groups[0].Hooks[0].StatusMessage)
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
