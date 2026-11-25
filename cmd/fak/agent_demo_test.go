package main

import (
	"bytes"
	"context"
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

func TestAgentDemoRetiredFromDispatch(t *testing.T) {
	if dispatchCoreVerbA("agentdemo", nil) {
		t.Fatal("retired agentdemo still dispatches")
	}
	var out bytes.Buffer
	printUnknownVerb(&out, "agentdemo")
	for _, want := range []string{"unknown verb", "scripted demo was retired", "fak up", "fak agent"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("retirement guidance missing %q:\n%s", want, out.String())
		}
	}
}

// TestAgentNoEndpointFailsLoud pins that a bare `fak agent` with no model
// endpoint fails loud (exit 2) with the guidance block, never silently running
// a scripted planner. The three provider base-url env vars are cleared because
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
	for _, want := range []string{"no model endpoint configured", "fak up", "--base-url"} {
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

func TestAgentOfflineFlagRetiredBeforeEndpointResolution(t *testing.T) {
	if os.Getenv("TEST_AGENT_OFFLINE_RETIRED_HELPER") == "1" {
		runAgent([]string{"--offline", "--out", os.Getenv("TEST_AGENT_RETIRED_OUT")})
		return
	}
	outPath := filepath.Join(t.TempDir(), "must-not-exist.json")
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=TestAgentOfflineFlagRetiredBeforeEndpointResolution")
	cmd.Env = append(os.Environ(),
		"TEST_AGENT_OFFLINE_RETIRED_HELPER=1",
		"TEST_AGENT_RETIRED_OUT="+outPath,
	)
	output, runErr := cmd.CombinedOutput()
	var ee *exec.ExitError
	if !errors.As(runErr, &ee) || ee.ExitCode() != 2 {
		t.Fatalf("retired --offline exit = %v, want 2\n%s", runErr, output)
	}
	if !strings.Contains(string(output), "flag provided but not defined: -offline") {
		t.Fatalf("retired --offline was not rejected by flag parsing:\n%s", output)
	}
	if strings.Contains(string(output), "turn-use vs now") {
		t.Fatalf("retired --offline ran the scripted demo:\n%s", output)
	}
	if _, statErr := os.Stat(outPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("retired --offline created a report: %v", statErr)
	}
}
