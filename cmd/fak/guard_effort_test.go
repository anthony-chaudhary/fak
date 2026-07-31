package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// TestGuardEffortPosture pins the attendance table: `auto` is the whole point — an attended
// interactive child gets ultracode (a human is waiting on the hard answer), a headless `-p`
// fleet worker gets xhigh (strong reasoning without the orchestration tax nobody is watching).
// An explicit mode wins over attendance in both directions, and off emits nothing.
func TestGuardEffortPosture(t *testing.T) {
	cases := []struct {
		mode        string
		interactive bool
		want        string
	}{
		{guardEffortModeAuto, true, guardEffortModeUltracode},
		{guardEffortModeAuto, false, guardEffortModeXHigh},
		// An explicit mode ignores attendance.
		{guardEffortModeUltracode, false, guardEffortModeUltracode},
		{guardEffortModeUltracode, true, guardEffortModeUltracode},
		{guardEffortModeXHigh, true, guardEffortModeXHigh},
		{guardEffortModeXHigh, false, guardEffortModeXHigh},
		// off leaves the seat's own reasoning default alone.
		{guardPreCompactModeOff, true, ""},
		{guardPreCompactModeOff, false, ""},
	}
	for _, tc := range cases {
		if got := guardEffortPosture(tc.mode, tc.interactive); got != tc.want {
			t.Errorf("guardEffortPosture(%q, interactive=%v) = %q, want %q", tc.mode, tc.interactive, got, tc.want)
		}
	}
}

func TestNormalizeGuardEffortMode(t *testing.T) {
	for _, tc := range []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", guardEffortModeAuto, false},
		{"auto", guardEffortModeAuto, false},
		{"  AUTO ", guardEffortModeAuto, false},
		{"ultracode", guardEffortModeUltracode, false},
		{"XHigh", guardEffortModeXHigh, false},
		{"off", guardPreCompactModeOff, false},
		// A typo fails LOUD rather than silently picking a posture — an operator who meant
		// `xhigh` and typed `x-high` must not get ultracode's bill by accident.
		{"x-high", "", true},
		{"max", "", true},
		{"true", "", true},
	} {
		got, err := normalizeGuardEffortMode(tc.in)
		if (err != nil) != tc.wantErr {
			t.Errorf("normalizeGuardEffortMode(%q) err = %v, wantErr = %v", tc.in, err, tc.wantErr)
			continue
		}
		if got != tc.want {
			t.Errorf("normalizeGuardEffortMode(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestGuardEffortEffectiveMode: an explicitly-set flag beats the env; otherwise
// FAK_GUARD_EFFORT speaks; otherwise the flag default stands.
func TestGuardEffortEffectiveMode(t *testing.T) {
	env := func(v string) func(string) string {
		return func(k string) string {
			if k == guardEffortEnvMode {
				return v
			}
			return ""
		}
	}
	if got := guardEffortEffectiveMode(guardEffortModeXHigh, true, env(guardEffortModeUltracode)); got != guardEffortModeXHigh {
		t.Errorf("explicit flag must beat env; got %q", got)
	}
	if got := guardEffortEffectiveMode(guardEffortModeAuto, false, env(guardEffortModeUltracode)); got != guardEffortModeUltracode {
		t.Errorf("env must speak when the flag was not set; got %q", got)
	}
	if got := guardEffortEffectiveMode(guardEffortModeAuto, false, env("")); got != guardEffortModeAuto {
		t.Errorf("flag default must stand with no env; got %q", got)
	}
}

// TestInstallGuardEffortPostureHeadlessAppendsEffortFlag: a headless `-p` worker — an AGENT
// session — gets `--effort xhigh` on its own argv. It rides a plain flag, not a settings key,
// so there is nothing to merge and no --settings to collide with.
func TestInstallGuardEffortPostureHeadlessAppendsEffortFlag(t *testing.T) {
	command := []string{"claude", "-p", "--permission-mode", "bypassPermissions", "do the thing"}
	got, install, err := installGuardEffortPosture(command, guardEffortModeAuto, "")
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !install.Applied || install.Posture != guardEffortModeXHigh {
		t.Fatalf("install = %+v, want applied xhigh", install)
	}
	want := append(append([]string{}, command...), "--effort", guardEffortLevelXHigh)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("command = %#v, want %#v", got, want)
	}
	if install.SettingsPath != "" {
		t.Fatalf("the xhigh path must not touch a settings file; got %q", install.SettingsPath)
	}
}

// TestInstallGuardEffortPostureInteractiveMergesUltracode is the load-bearing one: an attended
// interactive child gets ultracode MERGED into the hook settings file the installers already
// wrote and already put on the argv. The argv must gain NO second --settings (Claude's
// --settings is last-wins, so a second occurrence would silently discard the whole hook stack),
// and every pre-existing hook key must survive the merge.
func TestInstallGuardEffortPostureInteractiveMergesUltracode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "claude-guard-settings.json")
	if err := writeGuardPreCompactSettings(path, "/usr/local/bin/fak"); err != nil {
		t.Fatalf("seed hook settings: %v", err)
	}
	// A second installer has already merged its own event in, as the real chain does.
	if err := mergeGuardToolprocIntoSettings(path, "/usr/local/bin/fak", filepath.Join(dir, "journal.jsonl")); err != nil {
		t.Fatalf("seed toolproc hooks: %v", err)
	}

	command := appendClaudeSettingsArg([]string{"claude", "--dangerously-skip-permissions"}, path)
	got, install, err := installGuardEffortPosture(command, guardEffortModeAuto, path)
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !install.Applied || install.Posture != guardEffortModeUltracode {
		t.Fatalf("install = %+v, want applied ultracode", install)
	}
	if !reflect.DeepEqual(got, command) {
		t.Fatalf("ultracode must not change the argv (it merges into the existing --settings); got %#v want %#v", got, command)
	}
	if n := strings.Count(strings.Join(got, " "), "--settings"); n != 1 {
		t.Fatalf("argv must carry exactly ONE --settings, got %d: %#v", n, got)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read merged settings: %v", err)
	}
	var merged map[string]any
	if err := json.Unmarshal(raw, &merged); err != nil {
		t.Fatalf("merged settings is not valid JSON: %v\n%s", err, raw)
	}
	if merged["ultracode"] != true {
		t.Fatalf("merged settings missing ultracode:true\n%s", raw)
	}
	hooks, ok := merged["hooks"].(map[string]any)
	if !ok {
		t.Fatalf("merge dropped the hooks key entirely — this is the regression the merge exists to prevent\n%s", raw)
	}
	for _, event := range []string{"PreCompact", "PreToolUse", "PostToolUse", "SessionEnd"} {
		if _, ok := hooks[event]; !ok {
			t.Fatalf("merge dropped the %s hook\n%s", event, raw)
		}
	}
}

// TestInstallGuardEffortPostureWritesOwnSettingsWhenNoHookFile: with every hook installer off
// there is no file to merge into, so the ultracode posture writes its own and injects the one
// --settings itself.
func TestInstallGuardEffortPostureWritesOwnSettingsWhenNoHookFile(t *testing.T) {
	dir := t.TempDir()
	command := []string{"claude", "--dangerously-skip-permissions"}
	got, install, err := installGuardEffortPostureAt(command, guardEffortModeAuto, dir)
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !install.Applied || install.Posture != guardEffortModeUltracode {
		t.Fatalf("install = %+v, want applied ultracode", install)
	}
	want := []string{"claude", "--settings", install.SettingsPath, "--dangerously-skip-permissions"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("command = %#v, want %#v", got, want)
	}
	raw, err := os.ReadFile(install.SettingsPath)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	var settings map[string]any
	if err := json.Unmarshal(raw, &settings); err != nil {
		t.Fatalf("settings is not valid JSON: %v\n%s", err, raw)
	}
	if settings["ultracode"] != true {
		t.Fatalf("settings missing ultracode:true\n%s", raw)
	}
}

// TestInstallGuardEffortPostureNoOps covers every path that must leave the argv byte-identical:
// mode=off, a non-Claude child (--effort/--settings are Claude-specific), and a child that
// already pinned its OWN posture — an explicit per-launch choice outranks a default.
func TestInstallGuardEffortPostureNoOps(t *testing.T) {
	cases := []struct {
		name       string
		command    []string
		mode       string
		wantReason string
	}{
		{
			name:       "mode off",
			command:    []string{"claude", "--dangerously-skip-permissions"},
			mode:       guardPreCompactModeOff,
			wantReason: "disabled",
		},
		{
			name:       "non-claude child",
			command:    []string{"codex", "--dangerously-bypass-approvals-and-sandbox"},
			mode:       guardEffortModeAuto,
			wantReason: "non-claude-child",
		},
		{
			name:       "child already pinned --effort",
			command:    []string{"claude", "--effort", "max"},
			mode:       guardEffortModeAuto,
			wantReason: "child-pinned",
		},
		{
			name:       "child already pinned --effort=max",
			command:    []string{"claude", "--effort=max"},
			mode:       guardEffortModeAuto,
			wantReason: "child-pinned",
		},
		{
			// A pre-guard launcher (or an operator) handing the inline ultracode JSON has
			// already spoken; guard must not merge a second, possibly different, posture.
			name:       "child already pinned inline ultracode --settings",
			command:    []string{"claude", "-p", "--settings", ultracodeSettingsArg, "go"},
			mode:       guardEffortModeAuto,
			wantReason: "child-pinned",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, install, err := installGuardEffortPosture(tc.command, tc.mode, "")
			if err != nil {
				t.Fatalf("install: %v", err)
			}
			if install.Applied {
				t.Fatalf("install must be a no-op; got %+v", install)
			}
			if install.Reason != tc.wantReason {
				t.Fatalf("reason = %q, want %q", install.Reason, tc.wantReason)
			}
			if !reflect.DeepEqual(got, tc.command) {
				t.Fatalf("command changed: %#v, want %#v", got, tc.command)
			}
		})
	}
}

// TestInstallGuardEffortPostureRejectsBadMode: a bad posture is an error the caller surfaces,
// never a silent fallback to some default.
func TestInstallGuardEffortPostureRejectsBadMode(t *testing.T) {
	if _, _, err := installGuardEffortPosture([]string{"claude"}, "sometimes", ""); err == nil {
		t.Fatal("invalid mode must error")
	}
}

// TestGuardEffortSettingsFileIsNotAPin: guard writes its OWN --settings as a file path, so the
// installer must not read that back as a child-supplied pin (which would make a re-entrant or
// re-installed session silently skip the posture).
func TestGuardEffortSettingsFileIsNotAPin(t *testing.T) {
	if guardEffortSettingsNamesUltracode("/tmp/fak-guard/claude-settings.json") {
		t.Fatal("a settings FILE path must not count as an inline ultracode pin")
	}
	if guardEffortSettingsNamesUltracode(`{"hooks":{}}`) {
		t.Fatal("inline JSON without an ultracode key must not count as a pin")
	}
	if !guardEffortSettingsNamesUltracode(ultracodeSettingsArg) {
		t.Fatal("the inline ultracode JSON must count as a pin")
	}
	// An explicit ultracode:false is still a pin: the operator named the key.
	if !guardEffortSettingsNamesUltracode(`{"ultracode":false}`) {
		t.Fatal("an explicit ultracode:false must count as a pin")
	}
}

// TestGuardHookSettingsRoundTripPreservesPosture is the drift guard for the merge chain: every
// hook installer round-trips guardPreCompactClaudeSettings, so a posture key merged in earlier
// must survive a LATER hook merge. Without the field on the struct, json.Unmarshal drops it and
// the rewrite silently loses the posture.
func TestGuardHookSettingsRoundTripPreservesPosture(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	if err := writeGuardPreCompactSettings(path, "/usr/local/bin/fak"); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := mergeGuardUltracodeIntoSettings(path); err != nil {
		t.Fatalf("merge posture: %v", err)
	}
	// A later installer merges its own hooks into the SAME file.
	if err := mergeGuardToolprocIntoSettings(path, "/usr/local/bin/fak", filepath.Join(dir, "journal.jsonl")); err != nil {
		t.Fatalf("later hook merge: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var settings map[string]any
	if err := json.Unmarshal(raw, &settings); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	if settings["ultracode"] != true {
		t.Fatalf("a later hook merge dropped the ultracode posture\n%s", raw)
	}
}

// TestWriteGuardPreCompactSettingsOmitsUnsetPosture: the new posture fields are omitempty, so a
// hook-only settings file stays byte-identical to what the installers wrote before — no
// `"ultracode": false` appears and no seat default is silently overridden.
func TestWriteGuardPreCompactSettingsOmitsUnsetPosture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := writeGuardPreCompactSettings(path, "/usr/local/bin/fak"); err != nil {
		t.Fatalf("write: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	for _, key := range []string{"ultracode", "effortLevel"} {
		if strings.Contains(string(raw), key) {
			t.Fatalf("hook-only settings must not mention %q:\n%s", key, raw)
		}
	}
}
