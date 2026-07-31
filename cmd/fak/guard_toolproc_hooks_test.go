package main

import (
	"path/filepath"
	"testing"
)

// mustReadGuardHookSettings is the fail-the-test wrapper around the production reader, so the
// suite exercises the same parse the installers use instead of keeping a second copy of it.
func mustReadGuardHookSettings(t *testing.T, path string) guardPreCompactClaudeSettings {
	t.Helper()
	s, err := readGuardHookSettings(path)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	return s
}

func assertToolprocEvent(t *testing.T, s guardPreCompactClaudeSettings, event, kind, journal string) {
	t.Helper()
	matchers, ok := s.Hooks[event]
	if !ok {
		t.Fatalf("%s: hook event missing", event)
	}
	want := []string{"toolproc", "hook", kind, "--journal", journal}
	var found *guardPreCompactClaudeCommand
	for i := range matchers {
		for j := range matchers[i].Hooks {
			h := &matchers[i].Hooks[j]
			if len(h.Args) >= 3 && h.Args[0] == "toolproc" && h.Args[1] == "hook" && h.Args[2] == kind {
				if found != nil {
					t.Fatalf("%s: duplicate toolproc %s hooks: %+v", event, kind, matchers)
				}
				found = h
			}
		}
	}
	if found == nil || found.Type != "command" || len(found.Args) != len(want) {
		t.Fatalf("%s hook: got %+v, want args %v", event, found, want)
	}
	for i, a := range want {
		if found.Args[i] != a {
			t.Fatalf("%s hook arg[%d]: got %q, want %q", event, i, found.Args[i], a)
		}
	}
	if event == "PreToolUse" {
		if len(matchers) != 2 || matchers[0].Matcher != "Bash" || len(matchers[0].Hooks) != 1 || len(matchers[0].Hooks[0].Args) == 0 || matchers[0].Hooks[0].Args[0] != "guard-commit-gate" {
			t.Fatalf("%s: commit gate not preserved before toolproc hook: %+v", event, matchers)
		}
	} else if len(matchers) != 1 {
		t.Fatalf("%s: want exactly one toolproc matcher, got %+v", event, matchers)
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
	s := mustReadGuardHookSettings(t, install.SettingsPath)
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
	s := mustReadGuardHookSettings(t, existing)
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
	s = mustReadGuardHookSettings(t, existing)
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

// TestPromptHookEventAdmissible pins the admissibility gate: a type:"prompt"
// (model-judged) hook may only be placed on the four events whose harness
// protocol lets a verdict act. On PostToolUse/SessionEnd/etc. it is REFUSED with
// the structured reason PROMPT_HOOK_EVENT_INADMISSIBLE; on PreToolUse it is
// ACCEPTED with an empty reason. No model call is wired — this is the gate only.
func TestPromptHookEventAdmissible(t *testing.T) {
	cases := []struct {
		event string
		want  bool
	}{
		{"PreToolUse", true},
		{"Stop", true},
		{"SubagentStop", true},
		{"UserPromptSubmit", true},
		{"PostToolUse", false},
		{"SessionEnd", false},
		{"PreCompact", false},
		{"Notification", false},
	}
	for _, c := range cases {
		if got := promptHookEventAdmissible(c.event); got != c.want {
			t.Errorf("promptHookEventAdmissible(%q) = %v, want %v", c.event, got, c.want)
		}
		reason, ok := validatePromptHookEvent(c.event)
		if ok != c.want {
			t.Errorf("validatePromptHookEvent(%q) ok = %v, want %v", c.event, ok, c.want)
		}
		if !c.want && reason != "PROMPT_HOOK_EVENT_INADMISSIBLE" {
			t.Errorf("validatePromptHookEvent(%q) reason = %q, want PROMPT_HOOK_EVENT_INADMISSIBLE", c.event, reason)
		}
		if c.want && reason != "" {
			t.Errorf("validatePromptHookEvent(%q) reason = %q, want empty", c.event, reason)
		}
	}
}
