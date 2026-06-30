package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func TestGuardStopHookDecision(t *testing.T) {
	// Default ladder: warn=3, final=7, max=9.
	const (
		warn  = guardStopHookDefaultWarn  // 3
		final = guardStopHookDefaultFinal // 7
		max   = guardStopHookDefaultMax   // 9
	)
	for _, tc := range []struct {
		name        string
		consecutive int
		warnAt      int
		finalAt     int
		maxN        int
		mode        string
		wantExit    int
		wantBlock   bool
		wantStage   guardStopHookStage
	}{
		// off never blocks, but still reports the rung it WOULD be at.
		{"off-never-blocks", 5, warn, final, max, guardPreCompactModeOff, 0, false, guardStopHookWarn},
		// enforce: a clean completion (rung 0) allows; the three continue rungs all block (exit 2).
		{"enforce-clean-completion", 0, warn, final, max, guardPreCompactModeEnforce, 0, false, guardStopHookAllow},
		{"enforce-nudge-low", 1, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
		{"enforce-nudge-high", 2, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
		{"enforce-warn-low", 3, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookWarn},
		{"enforce-warn-high", 6, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookWarn},
		{"enforce-final-low", 7, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookFinal},
		{"enforce-final-at-max", 9, warn, final, max, guardPreCompactModeEnforce, 2, true, guardStopHookFinal},
		// > max is the bounded give-up: allow the stop (exit 0) so a stuck model cannot loop forever.
		{"enforce-give-up-above-max", 10, warn, final, max, guardPreCompactModeEnforce, 0, false, guardStopHookGiveUp},
		// shadow always allows (exit 0) but reports the would-be block + rung.
		{"shadow-would-block", 1, warn, final, max, guardPreCompactModeShadow, 0, true, guardStopHookNudge},
		{"shadow-clean", 0, warn, final, max, guardPreCompactModeShadow, 0, false, guardStopHookAllow},
		{"shadow-give-up", 10, warn, final, max, guardPreCompactModeShadow, 0, false, guardStopHookGiveUp},
		// Normalization clamps an INVERTED config so the rungs cannot invert: warn=5 clamps to
		// max=4, final=2 pulls up to warn -> warn=final=max=4.
		{"normalize-inverted-final", 4, 5, 2, 4, guardPreCompactModeEnforce, 2, true, guardStopHookFinal},
		{"normalize-inverted-giveup", 5, 5, 2, 4, guardPreCompactModeEnforce, 0, false, guardStopHookGiveUp},
		// A zero/garbage max falls back to the default ladder rather than wedging.
		{"normalize-zero-max", 1, 100, 1, 0, guardPreCompactModeEnforce, 2, true, guardStopHookNudge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exit, block, stage := guardStopHookDecision(tc.consecutive, tc.warnAt, tc.finalAt, tc.maxN, tc.mode)
			if exit != tc.wantExit || block != tc.wantBlock || stage != tc.wantStage {
				t.Fatalf("decision(c=%d,w=%d,f=%d,m=%d,%q) = exit %d block %v stage %s, want exit %d block %v stage %s",
					tc.consecutive, tc.warnAt, tc.finalAt, tc.maxN, tc.mode, exit, block, stage, tc.wantExit, tc.wantBlock, tc.wantStage)
			}
		})
	}
}

func TestNormalizeGuardStopHookModeDefaultsEnforce(t *testing.T) {
	got, err := normalizeGuardStopHookMode("")
	if err != nil || got != guardPreCompactModeEnforce {
		t.Fatalf("normalize(\"\") = %q, %v; want enforce", got, err)
	}
	if _, err := normalizeGuardStopHookMode("bogus"); err == nil {
		t.Fatalf("normalize(bogus) should error")
	}
}

func TestParseGuardStopHookConsecutive(t *testing.T) {
	n, err := parseGuardStopHookConsecutive("# HELP x\nfak_guard_deny_all_consecutive 2\n")
	if err != nil || n != 2 {
		t.Fatalf("parse = %d, %v; want 2", n, err)
	}
	if _, err := parseGuardStopHookConsecutive("fak_guard_deny_all_stops_total 5\n"); err == nil {
		t.Fatalf("missing gauge must error (so the hook fails open, not silently treats 0)")
	}
}

func TestReadStopHookActive(t *testing.T) {
	if !readStopHookActive(strings.NewReader(`{"stop_hook_active":true,"session_id":"s"}`)) {
		t.Fatalf("stop_hook_active true not parsed")
	}
	if readStopHookActive(strings.NewReader(`{"stop_hook_active":false}`)) {
		t.Fatalf("stop_hook_active false misread as true")
	}
	if readStopHookActive(strings.NewReader("not json")) {
		t.Fatalf("garbage stdin must read as false")
	}
	if readStopHookActive(nil) {
		t.Fatalf("nil stdin must read as false")
	}
}

func TestRunGuardStopHookEnforceBlocksOnDenyAll(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--max", "3",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (block the unchosen stop)", code)
	}
	if !strings.Contains(stderr.String(), "ALLOWED alternative") {
		t.Fatalf("stderr should carry the continuation instruction: %q", stderr.String())
	}
}

func TestRunGuardStopHookEnforceAllowsWhenNoDenyAll(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()

	code := runGuardStopHook(ioDiscard{}, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (a clean completion is a real stop)", code)
	}
}

func TestRunGuardStopHookBlocksCleanStopWithoutTaskHandoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--task-handoff-mode", guardPreCompactModeEnforce,
		"--task-handoff-file", filepath.Join(t.TempDir(), "missing-handoff.json"),
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (block clean stop until handoff exists); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "task handoff required") || !strings.Contains(stderr.String(), taskmgr.SchemaHandoff) {
		t.Fatalf("stderr should tell the agent how to write the handoff: %q", stderr.String())
	}
}

func TestRunGuardStopHookAllowsValidTaskHandoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()
	path := filepath.Join(t.TempDir(), "handoff.json")
	writeGuardStopHookHandoff(t, path, []taskmgr.HandoffNextStep{{
		Key:    "guard-test/follow-up",
		Title:  "Follow up after guarded task",
		Body:   "Pick up the remaining validation work.",
		Reason: "The completed task left a concrete verification rung.",
	}}, "")

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--task-handoff-mode", guardPreCompactModeEnforce,
		"--task-handoff-file", path,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for valid handoff; stderr=%s", code, stderr.String())
	}
}

func TestRunGuardStopHookDenyAllPrecedesTaskHandoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--task-handoff-mode", guardPreCompactModeEnforce,
		"--task-handoff-file", filepath.Join(t.TempDir(), "missing-handoff.json"),
		"--max", "3",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "ALLOWED alternative") {
		t.Fatalf("deny-all guidance should take precedence, got %q", stderr.String())
	}
	if strings.Contains(stderr.String(), "task handoff required") {
		t.Fatalf("task handoff should not mask deny-all guidance: %q", stderr.String())
	}
}

func TestRunGuardStopHookEnforceGivesUpAboveBound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 9\n"))
	}))
	defer srv.Close()

	code := runGuardStopHook(ioDiscard{}, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--max", "3",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (above the retry bound, stop looping)", code)
	}
}

func TestRunGuardStopHookShadowAllowsButLogs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeShadow,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("shadow exit = %d, want 0", code)
	}
	if !strings.Contains(stderr.String(), "would auto-continue") {
		t.Fatalf("shadow should log the would-be continue: %q", stderr.String())
	}
}

func TestRunGuardStopHookFailsOpenWhenGaugeUnavailable(t *testing.T) {
	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader("{}"), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", "http://127.0.0.1:1/metrics",
		"--timeout", "1ms",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want fail-open 0 (a Stop hook must never wedge the agent)", code)
	}
	if !strings.Contains(stderr.String(), "allowing stop") {
		t.Fatalf("stderr = %q, want fail-open log", stderr.String())
	}
}

func TestRunGuardStopHookOffIsNoOp(t *testing.T) {
	code := runGuardStopHook(ioDiscard{}, strings.NewReader("{}"), []string{"--mode", guardPreCompactModeOff})
	if code != 0 {
		t.Fatalf("off exit = %d, want 0", code)
	}
}

// TestInstallGuardStopHookMergesIntoPreCompactSettings is the load-bearing wiring test: when the
// PreCompact hook already wrote a --settings file, the Stop hook MERGES into it (both hooks
// present, a SINGLE --settings on the command) rather than injecting a second --settings that
// would clobber the first.
func TestInstallGuardStopHookMergesIntoPreCompactSettings(t *testing.T) {
	dir := t.TempDir()
	fakBin := filepath.Join(dir, "fak.exe")

	command, _, pcInstall, err := installGuardPreCompactHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeShadow, "http://127.0.0.1:4567", fakBin, dir)
	if err != nil || !pcInstall.Applied {
		t.Fatalf("precompact install: applied=%v err=%v", pcInstall.Applied, err)
	}

	command, env, stopInstall, err := installGuardStopHookAt(
		command, guardPreCompactModeEnforce, "http://127.0.0.1:4567", fakBin, "", pcInstall.SettingsPath, 3, 7, 9)
	if err != nil || !stopInstall.Applied {
		t.Fatalf("stop install: applied=%v err=%v", stopInstall.Applied, err)
	}
	if stopInstall.SettingsPath != pcInstall.SettingsPath {
		t.Fatalf("stop hook wrote a different settings file (%s) than precompact (%s) — must merge into one",
			stopInstall.SettingsPath, pcInstall.SettingsPath)
	}
	if n := strings.Count(strings.Join(command, "\x00"), "--settings"); n != 1 {
		t.Fatalf("command has %d --settings flags, want exactly 1: %v", n, command)
	}
	if stopInstall.WarnAt != 3 || stopInstall.FinalAt != 7 || stopInstall.Max != 9 {
		t.Fatalf("ladder = warn %d final %d max %d, want 3/7/9", stopInstall.WarnAt, stopInstall.FinalAt, stopInstall.Max)
	}
	var sawMode, sawWarn, sawFinal, sawMax bool
	for _, kv := range env {
		switch {
		case kv[0] == guardStopHookEnvMode && kv[1] == guardPreCompactModeEnforce:
			sawMode = true
		case kv[0] == guardStopHookEnvWarn && kv[1] == "3":
			sawWarn = true
		case kv[0] == guardStopHookEnvFinal && kv[1] == "7":
			sawFinal = true
		case kv[0] == guardStopHookEnvMax && kv[1] == "9":
			sawMax = true
		}
	}
	if !sawMode || !sawWarn || !sawFinal || !sawMax {
		t.Fatalf("missing stop-hook env: mode=%v warn=%v final=%v max=%v from %v", sawMode, sawWarn, sawFinal, sawMax, env)
	}

	// The single settings file now carries BOTH hooks.
	data, err := os.ReadFile(stopInstall.SettingsPath)
	if err != nil {
		t.Fatalf("read merged settings: %v", err)
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("unmarshal merged settings: %v\n%s", err, data)
	}
	if len(settings.Hooks["PreCompact"]) != 1 {
		t.Fatalf("merged file lost the PreCompact hook: %s", data)
	}
	stop := settings.Hooks["Stop"]
	if len(stop) != 1 || len(stop[0].Hooks) != 1 {
		t.Fatalf("merged file missing the Stop hook: %s", data)
	}
	if stop[0].Matcher != "" {
		t.Fatalf("Stop hook must carry no matcher, got %q", stop[0].Matcher)
	}
	if got := stop[0].Hooks[0].Args; len(got) != 1 || got[0] != "guard-stophook" {
		t.Fatalf("Stop hook args = %v, want [guard-stophook]", got)
	}
}

// TestInstallGuardStopHookCreatesOwnSettingsWhenPreCompactOff covers the path where PreCompact is
// off: the Stop hook writes its own settings file and injects the single --settings itself.
func TestInstallGuardStopHookCreatesOwnSettingsWhenPreCompactOff(t *testing.T) {
	dir := t.TempDir()
	command, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9)
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	if command[1] != "--settings" || command[2] != install.SettingsPath {
		t.Fatalf("stop hook did not inject its own --settings: %v", command)
	}
	if got := strings.Join(command[3:], "\x00"); got != strings.Join([]string{"-p", "hi"}, "\x00") {
		t.Fatalf("user args changed: %v", command)
	}
	if len(env) == 0 {
		t.Fatalf("expected stop-hook env vars")
	}
}

func TestInstallGuardStopHookInjectsTaskHandoffEnv(t *testing.T) {
	dir := t.TempDir()
	handoffPath := filepath.Join(dir, "handoff.json")
	_, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9,
		guardTaskHandoffConfig{Mode: guardPreCompactModeEnforce, File: handoffPath, Repo: "owner/repo", Live: true})
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	got := map[string]string{}
	for _, kv := range env {
		got[kv[0]] = kv[1]
	}
	if got[guardTaskHandoffEnvMode] != guardPreCompactModeEnforce ||
		got[guardTaskHandoffEnvFile] != handoffPath ||
		got[guardTaskHandoffFileEnv] != handoffPath ||
		got[guardTaskHandoffEnvRepo] != "owner/repo" ||
		got[guardTaskHandoffEnvLive] != "1" {
		t.Fatalf("task handoff env missing: %+v", got)
	}
}

func TestInstallGuardStopHookSkipsOffAndNonClaude(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mode    string
		command []string
	}{
		{"off", guardPreCompactModeOff, []string{"claude"}},
		{"non-claude", guardPreCompactModeEnforce, []string{"codex"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			command, env, install, err := installGuardStopHookAt(tc.command, tc.mode, "http://127.0.0.1:4567", "fak", dir, "", 3, 7, 9)
			if err != nil {
				t.Fatalf("install: %v", err)
			}
			if install.Applied {
				t.Fatalf("hook applied unexpectedly: %+v", install)
			}
			if len(env) != 0 {
				t.Fatalf("env = %v, want none", env)
			}
			if strings.Join(command, "\x00") != strings.Join(tc.command, "\x00") {
				t.Fatalf("command changed: %v -> %v", tc.command, command)
			}
		})
	}
}

func writeGuardStopHookHandoff(t *testing.T, path string, steps []taskmgr.HandoffNextStep, noNext string) {
	t.Helper()
	h := taskmgr.Handoff{
		Schema:       taskmgr.SchemaHandoff,
		CurrentState: "The guarded task completed and the remaining state is documented.",
		Task: taskmgr.HandoffTask{
			TaskID: "guard-test",
			Title:  "guard test",
			State:  taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{
				VerifiedState: taskmgr.VerifiedDone,
				Source:        "test",
				SHA:           "deadbeef",
			},
		},
		NextSteps:        steps,
		NoNextStepReason: noNext,
	}
	b, err := json.MarshalIndent(h, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}
