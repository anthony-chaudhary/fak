package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
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

func writeGuardE2EFakeGit(t *testing.T) (binDir string, logPath string) {
	t.Helper()
	dir := t.TempDir()
	logPath = filepath.Join(dir, "git-calls.log")
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, "git.bat")
		body := `@echo off
echo %*>>"%FAK_GUARD_GIT_CALL_LOG%"
if "%1"=="rev-parse" (
  echo 0123456789abcdef0123456789abcdef01234567
  exit /b 0
)
if "%1"=="hash-object" (
  echo abcdef0123456789abcdef0123456789abcdef01
  exit /b 0
)
exit /b 0
`
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("write fake git: %v", err)
		}
		return dir, logPath
	}
	path := filepath.Join(dir, "git")
	body := `#!/bin/sh
printf '%s\n' "$*" >> "$FAK_GUARD_GIT_CALL_LOG"
case "$1" in
  rev-parse) echo 0123456789abcdef0123456789abcdef01234567 ;;
  hash-object) echo abcdef0123456789abcdef0123456789abcdef01 ;;
esac
exit 0
`
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake git: %v", err)
	}
	return dir, logPath
}

func writeGuardE2ENoopChild(t *testing.T, sleep bool) string {
	t.Helper()
	dir := t.TempDir()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, "guard-noop-child.bat")
		body := "@echo off\r\n"
		if sleep {
			body += "ping -n 3 127.0.0.1 >NUL\r\n"
		}
		body += "exit /b 0\r\n"
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("write noop child: %v", err)
		}
		return path
	}
	path := filepath.Join(dir, "guard-noop-child")
	body := "#!/bin/sh\n"
	if sleep {
		body += "sleep 1\n"
	}
	body += "exit 0\n"
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write noop child: %v", err)
	}
	return path
}

func guardE2EGitEnv(binDir, logPath, registryPath string) map[string]string {
	return map[string]string{
		"PATH":                   binDir + string(os.PathListSeparator) + os.Getenv("PATH"),
		"FAK_GUARD_GIT_CALL_LOG": logPath,
		sessionRegistryEnv:       registryPath,
	}
}

func readGuardE2EGitCalls(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return ""
	}
	if err != nil {
		t.Fatalf("read fake git call log: %v", err)
	}
	return string(b)
}

// TestGuardDefaultLaunchDoesNotSpawnGit is the #1833 regression witness: a plain
// `fak guard -- <agent>` used to pay sessionDescriptorMeta + PublishSession on the
// critical path, which meant at least three serial `git` subprocesses before the child could
// run. With a fake git first on PATH, the default launch must produce no git call log at all.
func TestGuardDefaultLaunchDoesNotSpawnGit(t *testing.T) {
	binDir, logPath := writeGuardE2EFakeGit(t)
	child := writeGuardE2ENoopChild(t, false)
	registryPath := filepath.Join(t.TempDir(), "session-registry.json")
	env := guardE2EGitEnv(binDir, logPath, registryPath)

	code, out, timedOut := runGuardE2E(t,
		"--provider openai --base-url http://127.0.0.1:9 --quiet --no-audit -- "+child,
		env,
	)
	if timedOut {
		t.Fatalf("guard default launch timed out.\noutput:\n%s", out)
	}
	if code != 0 {
		t.Fatalf("guard default launch exit=%d, want 0.\noutput:\n%s", code, out)
	}
	if calls := readGuardE2EGitCalls(t, logPath); strings.TrimSpace(calls) != "" {
		t.Fatalf("default guard launch spawned git despite no durability opt-in:\n%s\noutput:\n%s", calls, out)
	}
}

// TestGuardDurableLaunchStillPublishes proves the #1833 gate did not remove durability:
// an explicit session id opts in, so the deferred post-ready path still resolves git HEAD and
// publishes the live session side ref while the child remains alive.
func TestGuardDurableLaunchStillPublishes(t *testing.T) {
	binDir, logPath := writeGuardE2EFakeGit(t)
	child := writeGuardE2ENoopChild(t, true)
	registryPath := filepath.Join(t.TempDir(), "session-registry.json")
	env := guardE2EGitEnv(binDir, logPath, registryPath)

	code, out, timedOut := runGuardE2E(t,
		"--provider openai --base-url http://127.0.0.1:9 --quiet --no-audit --session-id e2e-durable -- "+child,
		env,
	)
	if timedOut {
		t.Fatalf("guard durable launch timed out.\noutput:\n%s", out)
	}
	if code != 0 {
		t.Fatalf("guard durable launch exit=%d, want 0.\noutput:\n%s", code, out)
	}
	calls := readGuardE2EGitCalls(t, logPath)
	for _, want := range []string{
		"rev-parse HEAD",
		"hash-object -w",
		"update-ref refs/fak/locks/session-e2e-durable",
	} {
		if !strings.Contains(calls, want) {
			t.Fatalf("durable guard launch did not publish %q.\ngit calls:\n%s\noutput:\n%s", want, calls, out)
		}
	}
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

// TestGuardEmptyNamedAPIKeyFailsLoud is the witness for the empty-`--api-key-env` accident:
// naming --api-key-env ANTHROPIC_API_KEY is the explicit opt-in to API billing, so an EMPTY
// value (a typo, a sudo-stripped env, a CI secret that did not inject) must EXIT with guidance
// rather than silently demote to the subscription pin (which bills the wrong account). The run
// also has no subscription token, so the only way to leave 0/2 is the empty-key gate firing
// before the headless-no-token path — which is what proves the new gate, not the old one.
func TestGuardEmptyNamedAPIKeyFailsLoud(t *testing.T) {
	emptyCfg := t.TempDir()
	code, out, timedOut := runGuardE2E(t,
		"--provider anthropic --api-key-env ANTHROPIC_API_KEY -- claude",
		map[string]string{
			"CLAUDE_CONFIG_DIR":       emptyCfg,
			"ANTHROPIC_API_KEY":       "", // named but empty: the accident this gate refuses.
			"CLAUDE_CODE_OAUTH_TOKEN": "",
		},
	)
	if timedOut {
		t.Fatalf("guard hung on an empty named --api-key-env instead of failing loud.\noutput:\n%s", out)
	}
	if code != 2 {
		t.Fatalf("want exit 2 (fail loud) on an empty named --api-key-env; got %d.\noutput:\n%s", code, out)
	}
	if !strings.Contains(out, "is set but that env var is empty") {
		t.Fatalf("want the empty-named-key refusal guidance on stderr; got:\n%s", out)
	}
}

// TestGuardHeadlessWithAPIKeySpawnsNotRefused is the false-positive guard: a headless run with
// NO subscription token but a real ANTHROPIC_API_KEY is a legitimate API-billing passthrough
// (the child's own key flows upstream). guard must NOT fail loud here — it must spawn the
// trivial child (`cmd /c exit 0` on Windows, `sh -c true` on Unix) and run it to a clean exit.
//
// The witness is NOT a bare 0 exit (a future early-return that never spawns would also exit 0
// and silently pass). It is the guard EXIT SUMMARY — `fak guard: N kernel decision(s)` — which
// finishGuardChildAndReport prints ONLY after child.Run() returns. So asserting the summary
// line appears proves the child actually spawned AND ran to completion past the headless gate,
// which a 0-exit early-return cannot fake. --quiet is dropped here on purpose so the summary
// (and the kernel-adjudicated banner) reach the captured output.
func TestGuardHeadlessWithAPIKeySpawnsNotRefused(t *testing.T) {
	emptyCfg := t.TempDir()
	code, out, timedOut := runGuardE2E(t,
		"--provider anthropic -- "+guardE2EExitZeroCommand(),
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
	// The exit summary only prints after the child ran — it is the spawn-and-complete witness a
	// bare 0 exit is not.
	if !strings.Contains(out, "kernel decision(s)") {
		t.Fatalf("want the guard exit summary (proof the child actually spawned and ran), got exit %d.\noutput:\n%s", code, out)
	}
}
