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
	code := runGuardPreCompact(nil, &stderr, nil, []string{
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

	code := runGuardPreCompact(nil, ioDiscard{}, nil, []string{
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

	code := runGuardPreCompact(nil, ioDiscard{}, nil, []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
}

func TestRunGuardPreCompactFailsOpenWhenPostureUnavailable(t *testing.T) {
	var stderr strings.Builder
	code := runGuardPreCompact(nil, &stderr, nil, []string{
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

// TestRelayPrecompactShadowSurfacesArmedWithoutCompaction is the #1869 witness:
// the relay would-rotate signal, carried on the gateway /metrics scrape the
// PreCompact hook already reads, surfaces on the SAME shadow seam as the
// compaction posture — and never triggers compaction (the hook stays exit 0 in
// shadow mode regardless of the relay state).
func TestRelayPrecompactShadowSurfacesArmedWithoutCompaction(t *testing.T) {
	for _, tc := range []struct {
		name       string
		body       string
		wantSubstr string
		wantAbsent string
	}{
		{
			name:       "armed surfaces RELAY_ARMED alongside an allow posture",
			body:       "fak_harness_coherence_posture 0\nfak_relay_would_rotate 1\n",
			wantSubstr: "relay would-rotate signal armed (reason=RELAY_ARMED)",
		},
		{
			name:       "armed surfaces alongside a block posture too",
			body:       "fak_harness_coherence_posture 1\nfak_relay_would_rotate 1\n",
			wantSubstr: "relay would-rotate signal armed (reason=RELAY_ARMED)",
		},
		{
			name:       "disarmed gauge surfaces disarmed, still shadow-only",
			body:       "fak_harness_coherence_posture 1\nfak_relay_would_rotate 0\n",
			wantSubstr: "relay would-rotate signal disarmed",
		},
		{
			name:       "absent gauge keeps the seam silent (non-relay session)",
			body:       "fak_harness_coherence_posture 1\n",
			wantAbsent: "relay would-rotate",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			var stderr strings.Builder
			code := runGuardPreCompact(nil, &stderr, nil, []string{
				"--mode", guardPreCompactModeShadow,
				"--metrics-url", srv.URL + "/metrics",
			})
			if code != 0 {
				t.Fatalf("exit = %d, want 0 (shadow never triggers compaction)", code)
			}
			// The compaction posture line must always still be present — the relay
			// signal rides the same seam, it does not replace it.
			if !strings.Contains(stderr.String(), "fak guard PreCompact: shadow would") {
				t.Fatalf("stderr = %q, want the compaction shadow posture line", stderr.String())
			}
			if tc.wantSubstr != "" && !strings.Contains(stderr.String(), tc.wantSubstr) {
				t.Fatalf("stderr = %q, want relay substring %q", stderr.String(), tc.wantSubstr)
			}
			if tc.wantAbsent != "" && strings.Contains(stderr.String(), tc.wantAbsent) {
				t.Fatalf("stderr = %q, want NO relay line %q", stderr.String(), tc.wantAbsent)
			}
		})
	}
}

// TestRunGuardPreCompactExplicitMetricsURLOutranksAmbientIPC pins the signal-source
// precedence. `fak guard` exports FAK_GUARD_LIFECYCLE_SOCKET/TOKEN into everything it
// launches, so any nested `fak guard-precompact --metrics-url …` inherits a live
// supervisor IPC. The #3305 IPC preference used to win even there, which read the
// SUPERVISOR's posture (default "block", relay gauge unseen) and never dialled the URL
// the caller named — turning an allow posture into exit 2 and swallowing the relay
// shadow line. An explicitly-passed --metrics-url must pin the source to HTTP.
func TestRunGuardPreCompactExplicitMetricsURLOutranksAmbientIPC(t *testing.T) {
	_, ipc := newGuardLifecycleTestServer(t)
	t.Setenv(guardLifecycleSocketEnv, ipc.path)
	t.Setenv(guardLifecycleTokenEnv, ipc.token)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("fak_harness_coherence_posture 0\nfak_relay_would_rotate 1\n"))
	}))
	defer srv.Close()

	var stderr strings.Builder
	code := runGuardPreCompact(nil, &stderr, nil, []string{
		"--mode", guardPreCompactModeShadow,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(stderr.String(), "signals=http") {
		t.Fatalf("stderr = %q, want signals=http (explicit --metrics-url pins the source)", stderr.String())
	}
	if !strings.Contains(stderr.String(), "shadow would allow") {
		t.Fatalf("stderr = %q, want the allow posture read from the named URL", stderr.String())
	}
	if !strings.Contains(stderr.String(), "relay would-rotate signal armed (reason=RELAY_ARMED)") {
		t.Fatalf("stderr = %q, want the relay gauge from the named URL", stderr.String())
	}

	// Enforce mode must honour the same pin: an unreachable NAMED url fails open
	// rather than silently succeeding against the ambient supervisor IPC.
	var enforceErr strings.Builder
	if code := runGuardPreCompact(nil, &enforceErr, nil, []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", "http://127.0.0.1:1/metrics",
		"--timeout", "1ms",
	}); code != 0 {
		t.Fatalf("enforce exit = %d, want fail-open 0; stderr = %q", code, enforceErr.String())
	}
}

// TestRelayPrecompactShadowParsesGauge covers the pure relay-gauge parser: a
// present gauge reports its armed bit, a labeled series matches, and an absent
// gauge reports not-present so the shadow seam stays silent.
func TestRelayPrecompactShadowParsesGauge(t *testing.T) {
	for _, tc := range []struct {
		name        string
		body        string
		wantArmed   bool
		wantPresent bool
	}{
		{name: "armed", body: "fak_relay_would_rotate 1\n", wantArmed: true, wantPresent: true},
		{name: "disarmed", body: "fak_relay_would_rotate 0\n", wantArmed: false, wantPresent: true},
		{name: "labeled", body: `fak_relay_would_rotate{trace="abc"} 1` + "\n", wantArmed: true, wantPresent: true},
		{name: "absent", body: "fak_harness_coherence_posture 1\n", wantArmed: false, wantPresent: false},
		{name: "comment-and-malformed-skipped", body: "# HELP fak_relay_would_rotate x\nfak_relay_would_rotate nope\nfak_relay_would_rotate 1\n", wantArmed: true, wantPresent: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			armed, present := parseGuardPreCompactRelayArmed(tc.body)
			if armed != tc.wantArmed || present != tc.wantPresent {
				t.Fatalf("parse(%q) = (armed=%v present=%v), want (armed=%v present=%v)", tc.body, armed, present, tc.wantArmed, tc.wantPresent)
			}
		})
	}
}

type ioDiscard struct{}

func (ioDiscard) Write(p []byte) (int, error) { return len(p), nil }
