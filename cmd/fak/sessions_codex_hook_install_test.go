package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCodexContinuationHookInstallPreservesAndIsIdempotent(t *testing.T) {
	home := t.TempDir()
	path := filepath.Join(home, "hooks.json")
	original := `{"version":1,"hooks":{"UserPromptSubmit":[{"hooks":[{"type":"command","command":"other-hook"}]}],"Stop":[{"hooks":[{"type":"command","command":"stop-hook"}]}]}}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := sessionsCodexHookInstall(&stdout, &stderr, []string{"--codex-home", home}); code != 0 {
		t.Fatalf("install exit=%d stderr=%s", code, stderr.String())
	}
	first, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := sessionsCodexHookInstall(&stdout, &stderr, []string{"--codex-home", home}); code != 0 {
		t.Fatalf("reinstall exit=%d stderr=%s", code, stderr.String())
	}
	second, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("second install changed manifest")
	}
	var manifest map[string]any
	if err := json.Unmarshal(second, &manifest); err != nil {
		t.Fatal(err)
	}
	raw := string(second)
	for _, want := range []string{"other-hook", "stop-hook", "fak sessions codex-loop-hook"} {
		if !bytes.Contains(second, []byte(want)) {
			t.Errorf("manifest missing %q: %s", want, raw)
		}
	}
	if got := bytes.Count(second, []byte("fak sessions codex-loop-hook")); got != 2 {
		t.Fatalf("command occurrences=%d, want 2 (POSIX + Windows in one entry): %s", got, raw)
	}
	if bytes.Contains(second, []byte("statusMessage")) {
		t.Fatalf("continuation hook leaked a per-turn status line: %s", raw)
	}
}

func TestCodexContinuationHookInstallDryRunDoesNotWrite(t *testing.T) {
	home := filepath.Join(t.TempDir(), "new-home")
	var stdout, stderr bytes.Buffer
	if code := sessionsCodexHookInstall(&stdout, &stderr, []string{"--codex-home", home, "--dry-run"}); code != 0 {
		t.Fatalf("dry-run exit=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte("fak sessions codex-loop-hook")) {
		t.Fatalf("projected manifest=%s", stdout.String())
	}
	if _, err := os.Stat(filepath.Join(home, "hooks.json")); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote hooks.json: %v", err)
	}
}
