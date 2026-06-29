package main

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

// guard_login_e2e_test.go — the END-TO-END witness for the "fak guard gets stuck on login
// sometimes" class. The unit tests in guard_test.go check the DECISION (resolveGuardUpstream
// flags, the term seam); this file checks the BEHAVIOR of the real cmdGuard entry point,
// including the os.Exit(2) the unit tests cannot reach in-process.
//
// Why this exists (the DOS lesson): the first ship of this fix was `dos commit-audit`
// diff-witnessed yet WRONG — the headless gate did not fire on Windows. A green commit audit
// proves the diff "does the KIND of thing claimed", NOT that the symptom is gone. The only
// witness for "the hang no longer happens" is running the real binary headless and asserting it
// EXITS with guidance instead of blocking. That witness must live in CI, not in a transcript of
// someone eyeballing stderr once — otherwise "field-verified" evaporates with the context.

// guardE2EHelperEnv, when set, makes TestMain run cmdGuard with the helper's args and exit,
// so a test can exec THIS test binary as a real `fak guard` invocation and observe the exit
// code + stderr the way a user would — a non-TTY stdin included (exec.Command pipes stdin).
const guardE2EHelperEnv = "FAK_GUARD_E2E_HELPER"

func TestMain(m *testing.M) {
	if argv := os.Getenv(guardE2EHelperEnv); argv != "" {
		// Helper mode: behave as `fak guard <argv...>`. The split is on spaces because the
		// args this test passes are simple flags + a trivial child command.
		cmdGuard(strings.Fields(argv))
		// cmdGuard returns only if it neither exited nor the child ran to a normal finish;
		// either way, a clean return is a 0 exit.
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// runGuardE2E execs this test binary in helper mode as `fak guard <args>` under env, with a
// hard deadline so a HANG (the bug) is a test failure, not an infinite block. It returns the
// exit code, the combined stderr+stdout, and whether the run timed out.
//
// The child's stdin is the real os.DevNull device — NOT a strings.Reader pipe. This is the
// load-bearing choice: on Windows os.DevNull (NUL) reports as a CHARACTER DEVICE, the exact
// shape the original os.ModeCharDevice gate mishandled (it called NUL "interactive" and let
// the headless run hang). A pipe reports as ModeNamedPipe, which even the BROKEN check
// classified correctly — so a pipe-stdin test would pass on both the broken and fixed code and
// witness nothing. Driving stdin from DevNull makes this test fail on the broken check and pass
// on term.IsTerminal, which is what makes it a real witness for the Windows bug.
func runGuardE2E(t *testing.T, args string, env map[string]string) (exitCode int, out string, timedOut bool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatalf("open %s: %v", os.DevNull, err)
	}
	defer devNull.Close()

	cmd := exec.CommandContext(ctx, os.Args[0])
	cmd.Env = append(os.Environ(), guardE2EHelperEnv+"="+args)
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.Stdin = devNull // headless: DevNull is a char device on Windows (the bug condition).

	b, runErr := cmd.CombinedOutput()
	out = string(b)
	if ctx.Err() == context.DeadlineExceeded {
		return -1, out, true
	}
	if runErr == nil {
		return 0, out, false
	}
	if ee, ok := runErr.(*exec.ExitError); ok {
		return ee.ExitCode(), out, false
	}
	t.Fatalf("guard e2e run failed to start: %v\n%s", runErr, out)
	return -1, out, false
}

func guardE2EExitZeroCommand() string {
	if runtime.GOOS == "windows" {
		return "cmd /c exit 0"
	}
	return "sh -c true"
}

// TestGuardHeadlessNoTokenFailsLoudNotHang is the symptom witness: with NO subscription token
// anywhere, NO ANTHROPIC_API_KEY, and a non-TTY stdin (the headless/automation shape), real
// `fak guard -- claude` must EXIT with the setup guidance — never block on a login the wrapped
// agent cannot complete. A timeout here means the hang regressed.
func TestGuardHeadlessNoTokenFailsLoudNotHang(t *testing.T) {
	emptyCfg := t.TempDir()
	code, out, timedOut := runGuardE2E(t,
		"--provider anthropic -- claude",
		map[string]string{
			"CLAUDE_CONFIG_DIR":       emptyCfg,
			"ANTHROPIC_API_KEY":       "",
			"CLAUDE_CODE_OAUTH_TOKEN": "",
		},
	)
	if timedOut {
		t.Fatalf("REGRESSION: `fak guard -- claude` HUNG on a headless no-token launch instead of failing loud.\noutput:\n%s", out)
	}
	if code != 2 {
		t.Fatalf("want exit 2 (fail loud before spawn) on headless no-token; got %d.\noutput:\n%s", code, out)
	}
	if !strings.Contains(out, "refusing to spawn a headless agent") {
		t.Fatalf("want the headless-login refusal guidance on stderr; got:\n%s", out)
	}
	// The guidance must point home so an operator can recover.
	for _, want := range []string{"setup-token", "ANTHROPIC_API_KEY"} {
		if !strings.Contains(out, want) {
			t.Fatalf("refusal guidance missing %q (operator can't recover); got:\n%s", want, out)
		}
	}
}

// TestGuardHeadlessWithAPIKeySpawnsNotRefused is the false-positive guard: a headless run with
// NO subscription token but a real ANTHROPIC_API_KEY is a legitimate API-billing passthrough
// (the child's own key flows upstream). guard must NOT fail loud here — it must spawn. The
// trivial child (`printf`) runs to a clean exit, so a 0 exit proves guard got past the gate.
func TestGuardHeadlessWithAPIKeySpawnsNotRefused(t *testing.T) {
	emptyCfg := t.TempDir()
	code, out, timedOut := runGuardE2E(t,
		"--provider anthropic --quiet -- "+guardE2EExitZeroCommand(),
		map[string]string{
			"CLAUDE_CONFIG_DIR":       emptyCfg,
			"ANTHROPIC_API_KEY":       "sk-ant-api03-e2e-test",
			"CLAUDE_CODE_OAUTH_TOKEN": "",
		},
	)
	if timedOut {
		t.Fatalf("guard hung on a headless API-key passthrough launch.\noutput:\n%s", out)
	}
	if code == 2 || strings.Contains(out, "refusing to spawn a headless agent") {
		t.Fatalf("REGRESSION: guard wrongly REFUSED a legitimate ANTHROPIC_API_KEY passthrough; want spawn, got exit %d.\noutput:\n%s", code, out)
	}
}
