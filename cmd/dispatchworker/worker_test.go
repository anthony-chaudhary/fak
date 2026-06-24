// worker_test.go — parity witness for the Go dispatch-worker port, ported 1:1 from
// tools/dispatch_worker_test.py. Hermetic: the launch path uses an injected runner.
package main

import (
	"os"
	"testing"
)

func TestResolveBackendFlagBeatsEnvBeatsDefault(t *testing.T) {
	// Empty/nil env falls through to the process env (Python's `env or os.environ`),
	// so unset FLEET_WORKER_BACKEND for the default assertion.
	saved, had := os.LookupEnv("FLEET_WORKER_BACKEND")
	_ = os.Unsetenv("FLEET_WORKER_BACKEND")
	defer func() {
		if had {
			_ = os.Setenv("FLEET_WORKER_BACKEND", saved)
		}
	}()

	if b, _ := resolveBackend("claude", map[string]string{"FLEET_WORKER_BACKEND": "opencode"}); b != "claude" {
		t.Errorf("flag must beat env: got %q", b)
	}
	if b, _ := resolveBackend("", map[string]string{"FLEET_WORKER_BACKEND": "opencode"}); b != "opencode" {
		t.Errorf("env must be used: got %q", b)
	}
	if b, err := resolveBackend("", map[string]string{}); err != nil || b != "claude" {
		t.Errorf("empty env -> default claude: got %q err=%v", b, err)
	}
}

func TestResolveBackendRejectsUnknown(t *testing.T) {
	if _, err := resolveBackend("cursor", nil); err == nil {
		t.Error("unknown explicit backend must error")
	}
	if _, err := resolveBackend("", map[string]string{"FLEET_WORKER_BACKEND": "nope"}); err == nil {
		t.Error("unknown env backend must error")
	}
}

func TestClaudeCommandShapeMatchesDosTomlReference(t *testing.T) {
	cmd, err := buildCommand("adjudicator", "claude")
	if err != nil {
		t.Fatal(err)
	}
	if cmd[0] != "claude" || cmd[1] != "-p" || cmd[2] != "--permission-mode" || cmd[3] != "bypassPermissions" {
		t.Errorf("claude prefix wrong: %v", cmd)
	}
	if cmd[4] != "/dos-kernel:dos-dispatch-loop --lane adjudicator" {
		t.Errorf("claude prompt = %q", cmd[4])
	}
}

func TestOpencodeCommandUsesDispatchAgentAndSkipPermissions(t *testing.T) {
	cmd, err := buildCommand("agent", "opencode")
	if err != nil {
		t.Fatal(err)
	}
	if cmd[0] != "opencode" {
		t.Errorf("opencode[0] = %q", cmd[0])
	}
	if !contains(cmd, "--dangerously-skip-permissions") {
		t.Error("opencode must skip permissions")
	}
	if idx := indexOf(cmd, "--agent"); idx < 0 || cmd[idx+1] != "dos-dispatch" {
		t.Errorf("opencode agent wrong: %v", cmd)
	}
	if cmd[len(cmd)-1] != "dispatch lane agent" {
		t.Errorf("opencode message = %q", cmd[len(cmd)-1])
	}
}

func TestBuildCommandRejectsEmptyLane(t *testing.T) {
	if _, err := buildCommand("", "claude"); err == nil {
		t.Error("empty lane must error")
	}
}

func TestChildEnvStampsAssignmentAndPassesThrough(t *testing.T) {
	env := childEnv("canon", "claude", "C:/work/fleet", map[string]string{"PATH": "x", "KEEP_ME": "1"})
	if env["DISPATCH_LANE"] != "canon" || env["DISPATCH_BACKEND"] != "claude" || env["DISPATCH_WORKSPACE"] != "C:/work/fleet" {
		t.Errorf("assignment not stamped: %v", env)
	}
	if env["KEEP_ME"] != "1" {
		t.Error("base env must be preserved")
	}
}

func TestNormalizeTimeoutCapsByDefaultAndOptsOutAtZero(t *testing.T) {
	if d, bounded := normalizeTimeout(defaultTimeoutS); !bounded || int(d.Seconds()) != defaultTimeoutS {
		t.Errorf("default must be bounded: %v bounded=%v", d, bounded)
	}
	if d, bounded := normalizeTimeout(60); !bounded || int(d.Seconds()) != 60 {
		t.Errorf("60 -> bounded 60s: %v bounded=%v", d, bounded)
	}
	for _, v := range []int{0, -5} {
		if _, bounded := normalizeTimeout(v); bounded {
			t.Errorf("%d must be unbounded", v)
		}
	}
	if defaultTimeoutS <= 0 {
		t.Error("default must be a real positive bound")
	}
}

func TestLiveLaunchCallsRunnerWithResolvedCommandAndEnv(t *testing.T) {
	var seen [][]string
	runner := func(cmd []string, cwd string, env map[string]string) launchResult {
		seen = append(seen, cmd)
		if env["DISPATCH_LANE"] != "recall" || env["DISPATCH_BACKEND"] != "opencode" {
			t.Errorf("runner env wrong: %v", env)
		}
		return launchResult{ReturnCode: 0}
	}
	command, _ := buildCommand("recall", "opencode")
	env := childEnv("recall", "opencode", "C:/work/fleet", map[string]string{})
	res := launch(command, "C:/work/fleet", env, runner, 0, false)
	if res.ReturnCode != 0 || len(seen) != 1 || seen[0][0] != "opencode" {
		t.Errorf("runner not called as expected: res=%v seen=%v", res, seen)
	}
}

func TestLiveNonzeroReturncodePropagatesToPayloadOkFalse(t *testing.T) {
	runner := func(_ []string, _ string, _ map[string]string) launchResult {
		return launchResult{ReturnCode: 1, Stderr: "boom"}
	}
	cmd, _ := buildCommand("x", "claude")
	res := launch(cmd, ".", map[string]string{}, runner, 0, false)
	p := buildPayload("x", "claude", ".", false, &res, "")
	if p.OK {
		t.Error("nonzero returncode must make payload ok=false")
	}
}

func TestResolveExeFallsBackToNameWhenNotFound(t *testing.T) {
	if got := resolveExe("definitely-not-a-real-backend-xyz"); got != "definitely-not-a-real-backend-xyz" {
		t.Errorf("unresolvable name must fall back to itself, got %q", got)
	}
}

func TestDryRunPayloadDoesNotLaunch(t *testing.T) {
	p := buildPayload("docs", "claude", "C:/work/fleet", true, nil, "")
	if !p.OK || !p.DryRun || p.Result != nil || p.Backend != "claude" {
		t.Errorf("dry-run payload wrong: %+v", p)
	}
}

func contains(xs []string, want string) bool {
	return indexOf(xs, want) >= 0
}

func indexOf(xs []string, want string) int {
	for i, x := range xs {
		if x == want {
			return i
		}
	}
	return -1
}
