package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// ultracodeConflictPayload is the inline `--settings` a launcher appends LATER on the child
// argv to raise the reasoning posture (cmd/fak/accounts_launch.go, internal/dispatchtick).
// Claude's `--settings` is last-wins, so before #5510 this payload silently discarded guard's
// entire hook-settings file.
const ultracodeConflictPayload = `{"ultracode":true}`

// countSettingsFlags reports how many `--settings` occurrences (either spelling) an argv
// carries. Exactly one is the invariant #5510 restores.
func countSettingsFlags(argv []string) int {
	n := 0
	for _, arg := range argv {
		if arg == guardClaudeSettingsFlag || strings.HasPrefix(arg, guardClaudeSettingsFlag+"=") {
			n++
		}
	}
	return n
}

// settingsKeys lists a settings map's top-level keys, so a failure message names the keys
// rather than dumping a hook tree as raw bytes.
func settingsKeys(raw map[string]json.RawMessage) []string {
	keys := make([]string, 0, len(raw))
	for key := range raw {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// readGuardSettingsRaw reads a guard settings file as bare top-level keys, so a test can see
// keys the typed struct does not model.
func readGuardSettingsRaw(t *testing.T, path string) map[string]json.RawMessage {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read settings %s: %v", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("parse settings %s: %v (%s)", path, err, data)
	}
	return raw
}

// seedGuardSettingsFile writes the PreCompact hook file the installers write, which is the
// file appendClaudeSettingsArg folds a later payload into.
func seedGuardSettingsFile(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "claude-precompact-settings.json")
	if err := writeGuardPreCompactSettings(path, "fak"); err != nil {
		t.Fatalf("seed guard settings: %v", err)
	}
	return path
}

// TestAppendClaudeSettingsArgFoldsLaterInlineSettings is the #5510 witness: a child argv that
// carries an inline `--settings` AFTER guard's injection point must come out with exactly ONE
// `--settings` — guard's file — and that file must now carry the caller's key. Before the fix
// the argv came out with two occurrences and Claude's last-wins resolution threw guard's whole
// hook stack away.
func TestAppendClaudeSettingsArgFoldsLaterInlineSettings(t *testing.T) {
	settingsPath := seedGuardSettingsFile(t)
	command := []string{"claude", "--dangerously-skip-permissions", guardClaudeSettingsFlag, ultracodeConflictPayload, "-p", "hi"}

	out := appendClaudeSettingsArg(command, settingsPath)

	if n := countSettingsFlags(out); n != 1 {
		t.Fatalf("argv carries %d %s occurrences, want exactly 1 (last-wins would disarm guard): %v", n, guardClaudeSettingsFlag, out)
	}
	if out[0] != "claude" || out[1] != guardClaudeSettingsFlag || out[2] != settingsPath {
		t.Fatalf("guard settings not injected at the head: %v", out)
	}
	for _, arg := range out {
		if arg == ultracodeConflictPayload {
			t.Fatalf("the caller's inline payload is still a separate argv token: %v", out)
		}
	}
	if got, want := strings.Join(out[3:], "\x00"), strings.Join([]string{"--dangerously-skip-permissions", "-p", "hi"}, "\x00"); got != want {
		t.Fatalf("other child args changed: %v", out)
	}

	raw := readGuardSettingsRaw(t, settingsPath)
	if string(raw["ultracode"]) != "true" {
		t.Fatalf("caller key not folded into guard's settings file: %v", settingsKeys(raw))
	}
	if _, ok := raw[guardClaudeSettingsHooksKey]; !ok {
		t.Fatalf("guard's hooks key was lost by the fold: %v", settingsKeys(raw))
	}
	var settings guardPreCompactClaudeSettings
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("parse settings: %v", err)
	}
	if len(settings.Hooks["PreCompact"]) == 0 {
		t.Fatalf("PreCompact hook missing after the fold: %+v", settings.Hooks)
	}
}

// TestAppendClaudeSettingsArgFoldsEqualsFormAndLastPayloadWins covers the `--settings=<json>`
// spelling and confirms the fold keeps Claude's own last-wins order among the caller's own
// payloads.
func TestAppendClaudeSettingsArgFoldsEqualsFormAndLastPayloadWins(t *testing.T) {
	settingsPath := seedGuardSettingsFile(t)
	command := []string{
		"claude",
		guardClaudeSettingsFlag + `={"ultracode":false,"model":"opus"}`,
		guardClaudeSettingsFlag, ultracodeConflictPayload,
		"-p", "hi",
	}

	out := appendClaudeSettingsArg(command, settingsPath)

	if n := countSettingsFlags(out); n != 1 {
		t.Fatalf("argv carries %d %s occurrences, want exactly 1: %v", n, guardClaudeSettingsFlag, out)
	}
	raw := readGuardSettingsRaw(t, settingsPath)
	if string(raw["ultracode"]) != "true" {
		t.Fatalf("last payload did not win: %v", settingsKeys(raw))
	}
	if string(raw["model"]) != `"opus"` {
		t.Fatalf("earlier payload's non-conflicting key was dropped: %v", settingsKeys(raw))
	}
}

// TestAppendClaudeSettingsArgFoldsCallerSettingsFile covers the other legal `--settings` value:
// a path to a settings file. That form disarmed guard exactly the same way.
func TestAppendClaudeSettingsArgFoldsCallerSettingsFile(t *testing.T) {
	settingsPath := seedGuardSettingsFile(t)
	callerFile := filepath.Join(t.TempDir(), "caller-settings.json")
	if err := os.WriteFile(callerFile, []byte(`{"ultracode":true}`+"\n"), 0o600); err != nil {
		t.Fatalf("write caller settings: %v", err)
	}

	out := appendClaudeSettingsArg([]string{"claude", guardClaudeSettingsFlag, callerFile}, settingsPath)

	if n := countSettingsFlags(out); n != 1 {
		t.Fatalf("argv carries %d %s occurrences, want exactly 1: %v", n, guardClaudeSettingsFlag, out)
	}
	if out[2] != settingsPath {
		t.Fatalf("argv points at %q, want guard's file %q: %v", out[2], settingsPath, out)
	}
	if raw := readGuardSettingsRaw(t, settingsPath); string(raw["ultracode"]) != "true" {
		t.Fatalf("caller settings file was not folded in: %v", settingsKeys(raw))
	}
}

// TestGuardSettingsFoldSurvivesInstallerRoundTrips is the durability half of the fix: the
// PreCompact installer folds the caller's key in FIRST, and the Stop / toolproc / SessionStart
// installers then read-modify-write the same file. Each of those round-trips through
// guardPreCompactClaudeSettings, so without unknown-key preservation the folded key is erased
// moments after it lands and the child gets a settings file with no posture again.
func TestGuardSettingsFoldSurvivesInstallerRoundTrips(t *testing.T) {
	settingsPath := seedGuardSettingsFile(t)
	appendClaudeSettingsArg([]string{"claude", guardClaudeSettingsFlag, ultracodeConflictPayload}, settingsPath)

	if err := mergeGuardStopHookIntoSettings(settingsPath, "fak"); err != nil {
		t.Fatalf("merge stop hook: %v", err)
	}
	if err := mergeGuardToolprocIntoSettings(settingsPath, "fak", filepath.Join(t.TempDir(), "journal.jsonl")); err != nil {
		t.Fatalf("merge toolproc hooks: %v", err)
	}
	if err := mergeGuardSessionStartIntoSettings(settingsPath, "fak", true, "trace-1"); err != nil {
		t.Fatalf("merge sessionstart hook: %v", err)
	}

	raw := readGuardSettingsRaw(t, settingsPath)
	if string(raw["ultracode"]) != "true" {
		t.Fatalf("the folded key did not survive the installer round-trips: %v", settingsKeys(raw))
	}
	var settings guardPreCompactClaudeSettings
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("parse settings: %v", err)
	}
	for _, event := range []string{"PreCompact", "Stop", "PreToolUse", "PostToolUse", "SessionEnd", "SessionStart"} {
		if len(settings.Hooks[event]) == 0 {
			t.Fatalf("hook event %q missing after the merges: %+v", event, settings.Hooks)
		}
	}
}

// TestInstallGuardPreCompactHookFoldsUltracodeSettings drives the real installer entry point
// with the argv shape the ultracode launchers actually build, so the witness is not confined to
// the helper.
func TestInstallGuardPreCompactHookFoldsUltracodeSettings(t *testing.T) {
	dir := t.TempDir()
	command, _, install, err := installGuardPreCompactHookAt(
		[]string{"claude", "--dangerously-skip-permissions", guardClaudeSettingsFlag, ultracodeConflictPayload, "-p", "go"},
		guardPreCompactModeShadow,
		"http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"),
		dir,
	)
	if err != nil {
		t.Fatalf("install hook: %v", err)
	}
	if !install.Applied {
		t.Fatalf("hook not applied: %+v", install)
	}
	if n := countSettingsFlags(command); n != 1 {
		t.Fatalf("installed argv carries %d %s occurrences, want exactly 1: %v", n, guardClaudeSettingsFlag, command)
	}
	if command[2] != install.SettingsPath {
		t.Fatalf("the surviving %s is not guard's file: %v", guardClaudeSettingsFlag, command)
	}
	if raw := readGuardSettingsRaw(t, install.SettingsPath); string(raw["ultracode"]) != "true" {
		t.Fatalf("ultracode posture lost by the installer: %v", settingsKeys(raw))
	}
}

// TestAppendClaudeSettingsArgLeavesCleanArgvUnchanged pins the no-conflict path: an argv with no
// later `--settings` keeps its previous shape exactly, and guard's file is not rewritten.
func TestAppendClaudeSettingsArgLeavesCleanArgvUnchanged(t *testing.T) {
	settingsPath := seedGuardSettingsFile(t)
	before := readGuardSettingsRaw(t, settingsPath)

	out := appendClaudeSettingsArg([]string{"claude", "-p", "hello"}, settingsPath)

	want := []string{"claude", guardClaudeSettingsFlag, settingsPath, "-p", "hello"}
	if strings.Join(out, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("argv = %v, want %v", out, want)
	}
	if after := readGuardSettingsRaw(t, settingsPath); len(after) != len(before) {
		t.Fatalf("settings file gained keys with no payload to fold: %v", settingsKeys(after))
	}
}

// TestFoldClaudeSettingsRefusesUnmergeablePayloads pins the loud residue. A payload guard
// cannot merge is reported by name (appendClaudeSettingsArg prints it on stderr) instead of
// being allowed to override guard's file.
func TestFoldClaudeSettingsRefusesUnmergeablePayloads(t *testing.T) {
	for _, tc := range []struct {
		name    string
		payload string
		want    string
	}{
		{name: "not json", payload: "definitely-not-a-settings-file", want: "read child settings file"},
		{name: "not an object", payload: `{"unbalanced":`, want: "parse child"},
		{name: "empty", payload: "   ", want: "empty"},
		{name: "hooks", payload: `{"hooks":{"Stop":[]}}`, want: guardClaudeSettingsHooksReason},
	} {
		t.Run(tc.name, func(t *testing.T) {
			settingsPath := seedGuardSettingsFile(t)
			err := foldClaudeSettingsIntoGuardFile(settingsPath, []string{tc.payload})
			if err == nil {
				t.Fatalf("payload %q merged silently, want a named refusal", tc.payload)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not name %q", err, tc.want)
			}
			var settings guardPreCompactClaudeSettings
			data, readErr := os.ReadFile(settingsPath)
			if readErr != nil {
				t.Fatalf("read settings: %v", readErr)
			}
			if err := json.Unmarshal(data, &settings); err != nil {
				t.Fatalf("parse settings: %v", err)
			}
			if len(settings.Hooks["PreCompact"]) == 0 {
				t.Fatalf("guard's own hooks were damaged by an unmergeable payload: %+v", settings.Hooks)
			}
		})
	}

	// Even when the payload is unmergeable the argv keeps exactly one --settings, and it is
	// guard's: an unreadable payload must never be left in place to win by last-wins.
	settingsPath := seedGuardSettingsFile(t)
	out := appendClaudeSettingsArg([]string{"claude", guardClaudeSettingsFlag, "definitely-not-a-settings-file", "-p", "x"}, settingsPath)
	if n := countSettingsFlags(out); n != 1 || out[2] != settingsPath {
		t.Fatalf("unmergeable payload left the argv disarmed: %v", out)
	}
}
