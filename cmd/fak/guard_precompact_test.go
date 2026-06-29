package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compactcohere"
)

func TestGuardPreCompactInstallsShadowClaudeHook(t *testing.T) {
	dir := t.TempDir()
	command, env, install, err := installGuardPreCompactHookAt(
		[]string{"claude", "-p", "hello"},
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
	if install.Mode != guardPreCompactModeShadow {
		t.Fatalf("mode = %q, want shadow", install.Mode)
	}
	if got, want := command[1], "--settings"; got != want {
		t.Fatalf("command missing settings flag: %v", command)
	}
	if got, want := command[2], install.SettingsPath; got != want {
		t.Fatalf("settings path = %q, want %q", got, want)
	}
	if got, want := strings.Join(command[3:], "\x00"), strings.Join([]string{"-p", "hello"}, "\x00"); got != want {
		t.Fatalf("user args changed or settings were appended after prompt args: %v", command)
	}
	wantEnv := map[string]string{
		guardPreCompactEnvMode:       guardPreCompactModeShadow,
		guardPreCompactEnvMetricsURL: "http://127.0.0.1:4567/metrics",
	}
	for _, kv := range env {
		if wantEnv[kv[0]] == kv[1] {
			delete(wantEnv, kv[0])
		}
	}
	if len(wantEnv) != 0 {
		t.Fatalf("missing hook env: %+v from %v", wantEnv, env)
	}

	data, err := os.ReadFile(install.SettingsPath)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("unmarshal settings: %v\n%s", err, data)
	}
	matchers := settings.Hooks["PreCompact"]
	if len(matchers) != 1 || matchers[0].Matcher != "auto" {
		t.Fatalf("PreCompact matchers = %+v, want one auto matcher", matchers)
	}
	if len(matchers[0].Hooks) != 1 {
		t.Fatalf("PreCompact hooks = %+v, want one command", matchers[0].Hooks)
	}
	hook := matchers[0].Hooks[0]
	if hook.Type != "command" {
		t.Fatalf("hook type = %q, want command", hook.Type)
	}
	if got, want := hook.Command, filepath.Join(dir, "fak.exe"); got != want {
		t.Fatalf("hook command = %q, want %q", got, want)
	}
	if len(hook.Args) != 1 || hook.Args[0] != "guard-precompact" {
		t.Fatalf("hook args = %v, want [guard-precompact]", hook.Args)
	}
}

func TestGuardPreCompactSkipsOffAndNonClaude(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mode    string
		command []string
	}{
		{name: "off", mode: guardPreCompactModeOff, command: []string{"claude"}},
		{name: "non-claude", mode: guardPreCompactModeShadow, command: []string{"codex"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			command, env, install, err := installGuardPreCompactHookAt(tc.command, tc.mode, "http://127.0.0.1:4567", "fak", dir)
			if err != nil {
				t.Fatalf("install hook: %v", err)
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

func TestGuardPreCompactParsesMetricsPosture(t *testing.T) {
	block, err := parseGuardPreCompactMetricsPosture(`# HELP fak_harness_coherence_posture posture
fak_harness_coherence_posture 1
`)
	if err != nil {
		t.Fatalf("parse block: %v", err)
	}
	if block != compactcohere.PostureBlock {
		t.Fatalf("block posture = %q", block)
	}
	allow, err := parseGuardPreCompactMetricsPosture(`fak_harness_coherence_posture 0`)
	if err != nil {
		t.Fatalf("parse allow: %v", err)
	}
	if allow != compactcohere.PostureAllow {
		t.Fatalf("allow posture = %q", allow)
	}
}

func TestRunGuardPreCompactShadowLogsWouldBlockButAllows(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			t.Fatalf("path = %q, want /metrics", r.URL.Path)
		}
		_, _ = w.Write([]byte("fak_harness_coherence_posture 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardPreCompact(nil, &stderr, []string{
		"--mode", guardPreCompactModeShadow,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(stderr.String(), "shadow would block") {
		t.Fatalf("stderr = %q, want shadow block log", stderr.String())
	}
}

func TestRunGuardPreCompactEnforceReturnsPostureExitCode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_harness_coherence_posture 1\n"))
	}))
	defer srv.Close()

	code := runGuardPreCompact(nil, ioDiscard{}, []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
}

func TestRunGuardPreCompactEnforceAllowsWhenPostureAllows(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_harness_coherence_posture 0\n"))
	}))
	defer srv.Close()

	code := runGuardPreCompact(nil, ioDiscard{}, []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
}

func TestRunGuardPreCompactFailsOpenWhenPostureUnavailable(t *testing.T) {
	var stderr strings.Builder
	code := runGuardPreCompact(nil, &stderr, []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", "http://127.0.0.1:1/metrics",
		"--timeout", "1ms",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want fail-open 0", code)
	}
	if !strings.Contains(stderr.String(), "allowing Claude auto-compaction") {
		t.Fatalf("stderr = %q, want fail-open log", stderr.String())
	}
}

type ioDiscard struct{}

func (ioDiscard) Write(p []byte) (int, error) { return len(p), nil }
