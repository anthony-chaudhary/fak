package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func readGuardHookSettings(t *testing.T, path string) guardPreCompactClaudeSettings {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	var s guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("parse settings: %v", err)
	}
	return s
}

func assertToolprocEvent(t *testing.T, s guardPreCompactClaudeSettings, event, kind, journal string) {
	t.Helper()
	matchers, ok := s.Hooks[event]
	if !ok || len(matchers) != 1 || len(matchers[0].Hooks) != 1 {
		t.Fatalf("%s: want exactly one matcher with one hook, got %+v", event, matchers)
	}
	h := matchers[0].Hooks[0]
	want := []string{"toolproc", "hook", kind, "--journal", journal}
	if h.Type != "command" || len(h.Args) != len(want) {
		t.Fatalf("%s hook: got %+v, want args %v", event, h, want)
	}
	for i, a := range want {
		if h.Args[i] != a {
			t.Fatalf("%s hook arg[%d]: got %q, want %q", event, i, h.Args[i], a)
		}
	}
}

// A fresh install (no prior --settings) writes its own file with the three
// observation events and injects a single --settings.
func TestInstallGuardToolprocHooksFresh(t *testing.T) {
	dir := t.TempDir()
	journal := filepath.Join(dir, "journal.jsonl")
	command, env, install, err := installGuardToolprocHooksAt(
		[]string{"claude", "--model", "opus"}, "", "", "fak", dir, journal)
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !install.Applied || install.Mode != guardToolprocModeObserve {
		t.Fatalf("install = %+v, want applied/observe", install)
	}
	if len(command) < 3 || command[1] != "--settings" || command[2] != install.SettingsPath {
		t.Fatalf("command = %v, want --settings %s injected after argv0", command, install.SettingsPath)
	}
	if len(env) != 1 || env[0][0] != guardToolprocEnvMode || env[0][1] != guardToolprocModeObserve {
		t.Fatalf("env = %v", env)
	}
	s := readGuardHookSettings(t, install.SettingsPath)
	assertToolprocEvent(t, s, "PreToolUse", "pre", journal)
	assertToolprocEvent(t, s, "PostToolUse", "post", journal)
	assertToolprocEvent(t, s, "SessionEnd", "stop", journal)
	if _, ok := s.Hooks["Stop"]; ok {
		t.Fatal("toolproc install must NOT claim the turn-end Stop event (session_end is SessionEnd)")
	}
}

// Merging into the settings file the PreCompact/Stop installers wrote
// preserves their entries and does NOT inject a second --settings.
func TestInstallGuardToolprocHooksMergesAndPreserves(t *testing.T) {
	dir := t.TempDir()
	existing := filepath.Join(dir, "settings.json")
	if err := writeGuardStopHookSettings(existing, "fak"); err != nil {
		t.Fatalf("seed stop hook: %v", err)
	}
	journal := filepath.Join(dir, "journal.jsonl")
	commandIn := []string{"claude", "--settings", existing}
	command, _, install, err := installGuardToolprocHooksAt(commandIn, "observe", existing, "fak", "", journal)
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !install.Applied || install.SettingsPath != existing {
		t.Fatalf("install = %+v, want applied into %s", install, existing)
	}
	if len(command) != len(commandIn) {
		t.Fatalf("command = %v, want unchanged (no second --settings)", command)
	}
	s := readGuardHookSettings(t, existing)
	if _, ok := s.Hooks["Stop"]; !ok {
		t.Fatal("merge dropped the pre-existing Stop hook")
	}
	assertToolprocEvent(t, s, "PreToolUse", "pre", journal)
	assertToolprocEvent(t, s, "PostToolUse", "post", journal)
	assertToolprocEvent(t, s, "SessionEnd", "stop", journal)

	// Idempotent: a second merge leaves exactly one hook per event.
	if _, _, _, err := installGuardToolprocHooksAt(command, "observe", existing, "fak", "", journal); err != nil {
		t.Fatalf("re-install: %v", err)
	}
	s = readGuardHookSettings(t, existing)
	assertToolprocEvent(t, s, "PreToolUse", "pre", journal)
}

// Off mode and a non-claude child are no-ops: command unchanged, nothing written.
func TestInstallGuardToolprocHooksNoOps(t *testing.T) {
	dir := t.TempDir()
	journal := filepath.Join(dir, "journal.jsonl")

	commandIn := []string{"claude"}
	command, env, install, err := installGuardToolprocHooksAt(commandIn, "off", "", "fak", dir, journal)
	if err != nil || install.Applied || len(env) != 0 || len(command) != 1 {
		t.Fatalf("off: command=%v env=%v install=%+v err=%v, want untouched no-op", command, env, install, err)
	}
	if install.Reason != "disabled" {
		t.Fatalf("off reason = %q", install.Reason)
	}

	command, env, install, err = installGuardToolprocHooksAt([]string{"codex"}, "observe", "", "fak", dir, journal)
	if err != nil || install.Applied || len(env) != 0 || len(command) != 1 {
		t.Fatalf("codex: command=%v env=%v install=%+v err=%v, want untouched no-op", command, env, install, err)
	}
	if install.Reason != "non-claude-child" {
		t.Fatalf("codex reason = %q", install.Reason)
	}

	if _, _, _, err := installGuardToolprocHooksAt(commandIn, "banana", "", "fak", dir, journal); err == nil {
		t.Fatal("invalid mode must refuse")
	}
}
