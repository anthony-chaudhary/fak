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
