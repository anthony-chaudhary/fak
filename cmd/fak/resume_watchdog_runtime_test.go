package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resumeactuator"
)

// TestResumeWatchdogHarnessClassification pins the watchdog's harness normalization
// (#14330): codex/codex_exec fold to codex, opencode keeps its own identity instead of
// being misrouted through Claude's child environment, and every other value stays on
// the legacy claude default.
func TestResumeWatchdogHarnessClassification(t *testing.T) {
	tests := []struct {
		name    string
		harness string
		want    string
	}{
		{name: "claude", harness: "claude", want: "claude"},
		{name: "blank", harness: "", want: "claude"},
		{name: "whitespace", harness: "   ", want: "claude"},
		{name: "codex", harness: "codex", want: "codex"},
		{name: "codex_exec", harness: "codex_exec", want: "codex"},
		{name: "opencode", harness: "opencode", want: "opencode"},
		{name: "OpenCode-case", harness: "OpenCode", want: "opencode"},
		{name: "OPENCODE-trimmed", harness: " OPENCODE ", want: "opencode"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := rwHarness(resume.WatchdogPlanRow{Harness: tt.harness}); got != tt.want {
				t.Fatalf("rwHarness(harness=%q) = %q, want %q", tt.harness, got, tt.want)
			}
		})
	}
}

func rwEnvHasNone(t *testing.T, harness string, env map[string]string, keys ...string) {
	t.Helper()
	for _, k := range keys {
		if v, ok := env[k]; ok {
			t.Fatalf("%s child env must not carry %s (got %q)", harness, k, v)
		}
	}
}

func rwEnvWants(t *testing.T, harness, key string, env map[string]string, want string) {
	t.Helper()
	if got := env[key]; got != want {
		t.Fatalf("%s child env[%s] = %q, want %q", harness, key, got, want)
	}
}

// TestResumeWatchdogChildEnvironmentByHarness pins the three-way child-env routing
// (#14330): a codex row keeps the codex env shape, a claude row keeps today's claude
// env shape (seat pinned + affinity carried), and an opencode row gets a NEUTRAL env —
// no CLAUDE_CONFIG_DIR pin at all — while still carrying the harness-neutral relaunch
// cache affinity key when the session yields one.
func TestResumeWatchdogChildEnvironmentByHarness(t *testing.T) {
	t.Setenv("RW_WATCHDOG_ENV_PROBE", "keep-me")
	session := "30000000-0000-4000-8000-000000000001"
	affinity := resume.RelaunchCacheAffinityKey(session)
	seat := t.TempDir() // stand-in for the target account's CLAUDE_CONFIG_DIR

	codex := rwChildEnvironment(resume.WatchdogPlanRow{Harness: "codex", Session: session}, seat)
	rwEnvHasNone(t, "codex", codex, "CLAUDE_CONFIG_DIR", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_SESSION_ID")
	rwEnvWants(t, "codex", "RW_WATCHDOG_ENV_PROBE", codex, "keep-me")

	claude := rwChildEnvironment(resume.WatchdogPlanRow{Session: session}, seat)
	rwEnvWants(t, "claude", "CLAUDE_CONFIG_DIR", claude, seat)
	rwEnvWants(t, "claude", resume.RelaunchCacheAffinityEnv, claude, affinity)
	rwEnvHasNone(t, "claude", claude, "ANTHROPIC_BASE_URL", "CLAUDE_CODE_SESSION_ID")
	rwEnvWants(t, "claude", "RW_WATCHDOG_ENV_PROBE", claude, "keep-me")

	opencode := rwChildEnvironment(resume.WatchdogPlanRow{Harness: "opencode", Session: session}, seat)
	rwEnvHasNone(t, "opencode", opencode, "CLAUDE_CONFIG_DIR", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_SESSION_ID")
	rwEnvWants(t, "opencode", resume.RelaunchCacheAffinityEnv, opencode, affinity)
	rwEnvWants(t, "opencode", "RW_WATCHDOG_ENV_PROBE", opencode, "keep-me")

	// A blank session names no transcript, so no affinity route is derived.
	blank := rwChildEnvironment(resume.WatchdogPlanRow{Harness: "opencode"}, "")
	rwEnvHasNone(t, "opencode-blank-session", blank, resume.RelaunchCacheAffinityEnv)
}

// TestResumeWatchdogBrokerAttemptRoutesOpenCodeRows pins the broker-side half of
// #14330: an opencode plan row names the opencode backend (not "claude"), fronts the
// managed envelope with the opencode continuation, and carries the neutral opencode
// child env — no claude seat pin — while keeping the harness-neutral affinity key.
func TestResumeWatchdogBrokerAttemptRoutesOpenCodeRows(t *testing.T) {
	session := "40000000-0000-4000-8000-000000000001"
	p := resume.WatchdogPlanRow{Harness: resumeactuator.HarnessOpenCode, Session: session}
	attempt := rwResumeBrokerAttempt("fak-bin", "claude-bin", p, t.TempDir(), nil)
	if attempt.Backend != resumeactuator.HarnessOpenCode {
		t.Fatalf("attempt.Backend = %q, want %q", attempt.Backend, resumeactuator.HarnessOpenCode)
	}
	if joined := strings.Join(attempt.Argv, " "); !strings.Contains(joined, "opencode run --session "+session) {
		t.Fatalf("argv = %q, want the managed envelope fronting the opencode continuation", joined)
	}
	if got := attempt.Env["CLAUDE_CONFIG_DIR"]; got != "" {
		t.Fatalf("opencode attempt env pins CLAUDE_CONFIG_DIR=%q; an opencode child has no claude seat", got)
	}
	if got := attempt.Env[resume.RelaunchCacheAffinityEnv]; got != resume.RelaunchCacheAffinityKey(session) {
		t.Fatalf("opencode attempt env[%s] = %q, want the derived affinity key", resume.RelaunchCacheAffinityEnv, got)
	}
}

// TestResumeWatchdogOpenCodeExeResolution pins FLEET_OPENCODE_EXE resolution and that
// the actuator request carries it (symmetry with CodexExe).
func TestResumeWatchdogOpenCodeExeResolution(t *testing.T) {
	if got := rwOpenCodeExe(); got != "opencode" {
		t.Fatalf("rwOpenCodeExe() = %q, want the default %q", got, "opencode")
	}
	t.Setenv("FLEET_OPENCODE_EXE", "/opt/opencode/bin/opencode")
	if got := rwOpenCodeExe(); got != "/opt/opencode/bin/opencode" {
		t.Fatalf("rwOpenCodeExe() = %q, want the FLEET_OPENCODE_EXE override", got)
	}
	req := rwActuatorRequest(resume.WatchdogPlanRow{Harness: resumeactuator.HarnessOpenCode, Session: "50000000-0000-4000-8000-000000000001"}, "claude-bin")
	if req.OpenCodeExe != "/opt/opencode/bin/opencode" {
		t.Fatalf("request.OpenCodeExe = %q, want the resolved override", req.OpenCodeExe)
	}
}
