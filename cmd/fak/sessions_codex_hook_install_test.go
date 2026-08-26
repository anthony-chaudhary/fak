package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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
	for _, want := range []string{"other-hook", "stop-hook", "fak sessions codex-loop-hook", guardActiveEnv, codexLoopHookHardenedEnv} {
		if !bytes.Contains(second, []byte(want)) {
			t.Errorf("manifest missing %q: %s", want, raw)
		}
	}
	if got := bytes.Count(second, []byte("fak sessions codex-loop-hook")); got != 3 {
		t.Fatalf("command occurrences=%d, want POSIX plus both cmd.exe selector arms in one entry: %s", got, raw)
	}
	if bytes.Contains(second, []byte("statusMessage")) {
		t.Fatalf("continuation hook leaked a per-turn status line: %s", raw)
	}
	if bytes.Contains(second, []byte("codex-loop-hook --hardened")) {
		t.Fatalf("default install baked hardened enforcement: %s", raw)
	}
}

func TestCodexContinuationHookInstallHardenedBakesEnforcement(t *testing.T) {
	home := t.TempDir()
	var stdout, stderr bytes.Buffer
	if code := sessionsCodexHookInstall(&stdout, &stderr, []string{"--codex-home", home, "--hardened"}); code != 0 {
		t.Fatalf("install exit=%d stderr=%s", code, stderr.String())
	}
	raw, err := os.ReadFile(filepath.Join(home, "hooks.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got := bytes.Count(raw, []byte("codex-loop-hook --hardened")); got != 2 {
		t.Fatalf("hardened command occurrences=%d, want POSIX + Windows:\n%s", got, raw)
	}
}

func TestCodexContinuationHookInstallPreservesSiblingInMixedGroup(t *testing.T) {
	home := t.TempDir()
	path := filepath.Join(home, "hooks.json")
	original := `{"hooks":{"UserPromptSubmit":[{"matcher":"mixed","hooks":[{"type":"command","command":"other-hook"},{"type":"command","command":"fak sessions codex-loop-hook --legacy"}]}]}}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := sessionsCodexHookInstall(&stdout, &stderr, []string{"--codex-home", home}); code != 0 {
		t.Fatalf("install exit=%d stderr=%s", code, stderr.String())
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Hooks map[string][]struct {
			Matcher string `json:"matcher"`
			Hooks   []struct {
				Command string `json:"command"`
			} `json:"hooks"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	groups := manifest.Hooks["UserPromptSubmit"]
	if len(groups) != 2 {
		t.Fatalf("groups=%+v, want preserved mixed group plus installed fak group", groups)
	}
	if groups[0].Matcher != "mixed" || len(groups[0].Hooks) != 1 || groups[0].Hooks[0].Command != "other-hook" {
		t.Fatalf("mixed group sibling was not preserved exactly: %+v", groups[0])
	}
}

func TestSessionsDispatchesCodexHookInstall(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	var stdout, stderr bytes.Buffer
	if code := runSessions(&stdout, &stderr, []string{"codex-hook-install", "--codex-home", home, "--dry-run"}); code != 0 {
		t.Fatalf("dispatch exit=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte("UserPromptSubmit")) {
		t.Fatalf("dispatch output=%s", stdout.String())
	}
}

func TestCodexContinuationHookWindowsCommandsUseCmdSyntax(t *testing.T) {
	commands := []string{codexContinuationHookCommandWindows, codexContinuationHookHardenedCommandWindows}
	for _, command := range commands {
		for _, forbidden := range []string{"$env:", "[Console]", "2>$null"} {
			if strings.Contains(command, forbidden) {
				t.Fatalf("Windows command contains PowerShell syntax %q: %s", forbidden, command)
			}
		}
		for _, want := range []string{"%" + codexRawRecoveryEnv + "%", "2>nul", "exit /b 0"} {
			if !strings.Contains(command, want) {
				t.Fatalf("Windows command missing cmd.exe contract %q: %s", want, command)
			}
		}
		if runtime.GOOS == "windows" {
			cmd := exec.Command("cmd.exe", "/d", "/s", "/c", command)
			cmd.Env = append(os.Environ(), codexRawRecoveryEnv+"="+codexRawRecoveryValue)
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("cmd.exe rejected command: %v\n%s", err, output)
			}
		}
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
