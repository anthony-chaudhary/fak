package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// agentFailLoudHelperTimeout bounds the re-invoked bare-agent child so a hang
// fails this fixture on its own deadline instead of consuming the whole package
// test budget and reporting only a package timeout.
const agentFailLoudHelperTimeout = 20 * time.Second

// TestAgentDemoRunsOfflineDemo pins that `fak agentdemo` still runs the full
// offline mock A/B: the demo moved off `fak agent`, but the verb is unchanged.
func TestAgentDemoRunsOfflineDemo(t *testing.T) {
	out := filepath.Join(t.TempDir(), "demo-report.json")

	_, stderr := captureAgentStdio(t, func() {
		cmdAgentDemo([]string{"--out", out, "--code-tools=false", "--sys-tools=false"})
	})

	if strings.Contains(stderr, "no model endpoint") {
		t.Fatalf("fak agentdemo must run the offline demo, not fail loud.\ngot stderr:\n%s", stderr)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("fak agentdemo must write %s: %v", out, err)
	}
	if len(body) == 0 {
		t.Fatal("fak agentdemo wrote an empty report")
	}
	var res map[string]any
	if err := json.Unmarshal(body, &res); err != nil {
		t.Fatalf("report did not parse as JSON: %v\nbody:\n%s", err, body)
	}
	if len(res) == 0 {
		t.Fatalf("report parsed to an empty JSON object: %s", body)
	}
}

// TestAgentNoEndpointFailsLoud pins that a bare `fak agent` with no model
// endpoint fails loud (exit 2) with the guidance block, never silently running
// the offline demo. The three provider base-url env vars are cleared because
// this appliance may carry an ambient OPENAI_BASE_URL, which would otherwise
// make the bare run resolve a live endpoint and attempt a network call. The
// router origin is also pinned to a dead port: the loopback family fallback
// would otherwise find an ambient `fak serve` (e.g. exposed on ::1 only) and
// make the bare run auto-connect instead of failing loud.
func TestAgentNoEndpointFailsLoud(t *testing.T) {
	if os.Getenv("TEST_AGENT_FAILLOUD_HANG") == "1" {
		// Deterministic never-return used by the fixture deadline regression:
		// the parent must bound and kill this helper rather than wait forever.
		time.Sleep(time.Hour)
		return
	}
	if os.Getenv("TEST_AGENT_FAILLOUD_HELPER") == "1" {
		for i, arg := range os.Args {
			if arg == "--" {
				runAgent(os.Args[i+1:])
				return
			}
		}
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), agentFailLoudHelperTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.run=TestAgentNoEndpointFailsLoud", "--")
	cmd.WaitDelay = 3 * time.Second
	env := make([]string, 0, len(os.Environ()))
	for _, kv := range os.Environ() {
		key, _, _ := strings.Cut(kv, "=")
		switch {
		case strings.EqualFold(key, "OPENAI_BASE_URL"),
			strings.EqualFold(key, "ANTHROPIC_BASE_URL"),
			strings.EqualFold(key, "GOOGLE_GEMINI_BASE_URL"),
			strings.EqualFold(key, "TEST_AGENT_FAILLOUD_HELPER"):
			continue
		}
		env = append(env, kv)
	}
	cmd.Env = append(env, "TEST_AGENT_FAILLOUD_HELPER=1", "FAK_AGENT_ROUTER_ORIGIN=http://127.0.0.1:1")
	out, err := cmd.CombinedOutput()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		t.Fatalf("bare `fak agent` helper exceeded its %s bound; output:\n%s", agentFailLoudHelperTimeout, out)
	}
	if errors.Is(err, exec.ErrWaitDelay) {
		t.Fatalf("bare `fak agent` helper left output pipes open after exit: %v\noutput:\n%s", err, out)
	}

	var ee *exec.ExitError
	if !errors.As(err, &ee) {
		t.Fatalf("bare `fak agent` must exit non-zero: err=%v\noutput:\n%s", err, out)
	}
	if got := ee.ExitCode(); got != 2 {
		t.Fatalf("bare `fak agent` exit code = %d, want 2\noutput:\n%s", got, out)
	}
	for _, want := range []string{"no model endpoint configured", "fak agentdemo", "--base-url"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("fail-loud guidance must contain %q.\ngot output:\n%s", want, out)
		}
	}
}

// TestAgentNoEndpointFailsLoudDeadlineExpiry proves the fixture's own deadlock
// guard: a helper that never returns must be killed at the fixture bound and
// reported as a deadline failure, never left to consume the package budget.
func TestAgentNoEndpointFailsLoudDeadlineExpiry(t *testing.T) {
	if os.Getenv("TEST_AGENT_FAILLOUD_HANG") == "1" {
		time.Sleep(time.Hour)
		return
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	const bound = 2 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.run=TestAgentNoEndpointFailsLoud")
	cmd.WaitDelay = time.Second
	cmd.Env = append(os.Environ(), "TEST_AGENT_FAILLOUD_HANG=1")
	start := time.Now()
	_, runErr := cmd.CombinedOutput()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		if elapsed := time.Since(start); elapsed > 10*time.Second {
			t.Fatalf("fixture deadline fired after %s, far past its %s bound", elapsed, bound)
		}
		return
	}
	t.Fatalf("a never-returning helper must hit the fixture deadline; ctx=%v runErr=%v", ctx.Err(), runErr)
}

// TestAgentOfflineFlagStillRunsDemo is the regression guard that the explicit
// --offline opt-in still runs the demo in-process (no guidance, exit 0).
func TestAgentOfflineFlagStillRunsDemo(t *testing.T) {
	out := filepath.Join(t.TempDir(), "offline-report.json")

	_, stderr := captureAgentStdio(t, func() {
		cmdAgent([]string{"--offline", "--out", out, "--code-tools=false", "--sys-tools=false"})
	})

	if strings.Contains(stderr, "no model endpoint") {
		t.Fatalf("--offline must run the demo, not fail loud.\ngot stderr:\n%s", stderr)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("fak agent --offline must write %s: %v", out, err)
	}
	if len(body) == 0 {
		t.Fatal("fak agent --offline wrote an empty report")
	}
	var res map[string]any
	if err := json.Unmarshal(body, &res); err != nil {
		t.Fatalf("report did not parse as JSON: %v\nbody:\n%s", err, body)
	}
	if len(res) == 0 {
		t.Fatalf("report parsed to an empty JSON object: %s", body)
	}
}
