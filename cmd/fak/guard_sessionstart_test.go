package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// TestGuardSessionStartEmitsAffordance asserts the #3092 affordance: the SessionStart hook
// emits a valid additionalContext envelope naming the fak entry verbs.
func TestGuardSessionStartEmitsAffordance(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "on"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
	}

	// Valid JSON with the exact Claude Code SessionStart envelope shape.
	var env struct {
		HookSpecificOutput struct {
			HookEventName     string `json:"hookEventName"`
			AdditionalContext string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("stdout is not valid JSON envelope: %v\n%s", err, out.String())
	}
	if env.HookSpecificOutput.HookEventName != "SessionStart" {
		t.Fatalf("hookEventName = %q, want SessionStart", env.HookSpecificOutput.HookEventName)
	}
	ctx := env.HookSpecificOutput.AdditionalContext
	for _, verb := range []string{"fak_index_work", "fak_admit", "fak_tools_search"} {
		if !strings.Contains(ctx, verb) {
			t.Fatalf("affordance did not name entry verb %q: %s", verb, ctx)
		}
	}
}

// TestGuardSessionStartOffSuppresses asserts the off knob emits nothing (a lean harness
// opts out).
func TestGuardSessionStartOffSuppresses(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "off"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if out.Len() != 0 {
		t.Fatalf("off mode should emit nothing, got: %q", out.String())
	}
}

// TestGuardSessionStartDefaultsOn asserts an empty mode defaults to on (the affordance is
// the fix, so it is on by default).
func TestGuardSessionStartDefaultsOn(t *testing.T) {
	if got := normalizeGuardSessionStartMode(""); got != guardSessionStartModeOn {
		t.Fatalf("empty mode = %q, want on", got)
	}
	if got := normalizeGuardSessionStartMode("OFF"); got != guardSessionStartModeOff {
		t.Fatalf("OFF (case-insensitive) = %q, want off", got)
	}
}

// TestGuardSessionStartSettingsRoundTrip asserts the settings writer emits a SessionStart
// hook entry the merge path can read back, and that the merge preserves a sibling hook.
func TestGuardSessionStartSettingsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/settings.json"

	// Seed a settings file that already carries a Stop hook (a sibling the merge must keep).
	seed := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{
		"Stop": guardStopHookMatchers("fak"),
	}}
	seedData, _ := json.MarshalIndent(seed, "", "  ")
	if err := os.WriteFile(path, seedData, 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if err := mergeGuardSessionStartIntoSettings(path, "fak"); err != nil {
		t.Fatalf("merge: %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	var got guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("parse merged: %v", err)
	}
	if _, ok := got.Hooks["SessionStart"]; !ok {
		t.Fatalf("merged settings missing SessionStart hook: %s", raw)
	}
	if _, ok := got.Hooks["Stop"]; !ok {
		t.Fatalf("merge dropped the sibling Stop hook: %s", raw)
	}
	// The SessionStart hook must invoke the guard-sessionstart verb.
	if !strings.Contains(string(raw), "guard-sessionstart") {
		t.Fatalf("SessionStart hook does not invoke guard-sessionstart: %s", raw)
	}
}

// TestInstallGuardSessionStartHookAtWiring covers the install path cmdGuard actually invokes
// (guard.go -> installGuardSessionStartHook -> installGuardSessionStartHookAt) for the #3092
// affordance. The actuator/merge tests above exercise emission and merge in isolation; this
// asserts the install BRANCHING that reaches the child — a claude launcher gets the SessionStart
// hook wired into its --settings, a non-claude child and the off knob stay no-ops, and merging
// into an existing guard settings file preserves the sibling hooks. Without this, a regression
// that stops wiring the affordance (re-inerting the fak verbs) would pass every existing test.
func TestInstallGuardSessionStartHookAtWiring(t *testing.T) {
	t.Run("claude child gets a fresh settings file and a --settings repoint", func(t *testing.T) {
		dir := t.TempDir()
		cmd := []string{"claude", "-p", "do the work"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", "fak", dir, "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if !install.Applied {
			t.Fatalf("expected Applied=true for a claude child, got %+v", install)
		}
		if install.SettingsPath == "" {
			t.Fatalf("expected a written settings path, got empty")
		}
		// The launcher must stay first and gain `--settings <path>` so Claude Code loads the hook.
		if len(out) == 0 || out[0] != "claude" {
			t.Fatalf("launcher token moved or dropped: %v", out)
		}
		if !strings.Contains(strings.Join(out, " "), "--settings "+install.SettingsPath) {
			t.Fatalf("command missing --settings %s: %v", install.SettingsPath, out)
		}
		raw, err := os.ReadFile(install.SettingsPath)
		if err != nil {
			t.Fatalf("read written settings: %v", err)
		}
		var got guardPreCompactClaudeSettings
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("parse written settings: %v", err)
		}
		if _, ok := got.Hooks["SessionStart"]; !ok {
			t.Fatalf("written settings missing SessionStart hook: %s", raw)
		}
		if !strings.Contains(string(raw), "guard-sessionstart") {
			t.Fatalf("SessionStart hook does not invoke guard-sessionstart: %s", raw)
		}
	})

	t.Run("non-claude child is a no-op", func(t *testing.T) {
		cmd := []string{"bash", "-c", "echo hi"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", "fak", t.TempDir(), "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if install.Applied {
			t.Fatalf("expected no-op for a non-claude child, got %+v", install)
		}
		if install.Reason != "non-claude-child" {
			t.Fatalf("reason = %q, want non-claude-child", install.Reason)
		}
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("non-claude command was mutated: %v", out)
		}
	})

	t.Run("off mode stays a no-op even for a claude child", func(t *testing.T) {
		cmd := []string{"claude", "-p", "x"}
		out, install, err := installGuardSessionStartHookAt(cmd, "off", "fak", t.TempDir(), "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if install.Applied {
			t.Fatalf("off mode should not apply, got %+v", install)
		}
		if install.Reason != "disabled" {
			t.Fatalf("reason = %q, want disabled", install.Reason)
		}
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("off mode mutated the command: %v", out)
		}
	})

	t.Run("merges into an existing settings file without re-pointing the command", func(t *testing.T) {
		dir := t.TempDir()
		existing := dir + "/settings.json"
		seed := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{
			"Stop": guardStopHookMatchers("fak"),
		}}
		seedData, _ := json.MarshalIndent(seed, "", "  ")
		if err := os.WriteFile(existing, seedData, 0o600); err != nil {
			t.Fatalf("seed: %v", err)
		}
		cmd := []string{"claude", "-p", "x"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", "fak", "", existing)
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if !install.Applied || install.SettingsPath != existing {
			t.Fatalf("expected merge into %s, got %+v", existing, install)
		}
		// The merge branch reuses the existing --settings file, so the command is not repointed.
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("merge branch should not append --settings again: %v", out)
		}
		raw, err := os.ReadFile(existing)
		if err != nil {
			t.Fatalf("read merged settings: %v", err)
		}
		var got guardPreCompactClaudeSettings
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("parse merged settings: %v", err)
		}
		if _, ok := got.Hooks["SessionStart"]; !ok {
			t.Fatalf("merge missing SessionStart hook: %s", raw)
		}
		if _, ok := got.Hooks["Stop"]; !ok {
			t.Fatalf("merge dropped the sibling Stop hook: %s", raw)
		}
	})
}
