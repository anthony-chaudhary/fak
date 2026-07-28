package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

// TestDockerRunArgs pins the isolation contract of the sandbox invocation: the
// container is throwaway (--rm), network-isolated (--network none), reads stdin
// (-i), and the candidate source is forwarded through the LCB_CODE environment
// variable — never placed on argv, where quoting or length could break it.
func TestDockerRunArgs(t *testing.T) {
	args := dockerRunArgs("python:3.11-slim", "fak-lcb-test-1")
	joined := strings.Join(args, " ")

	for _, want := range []string{"run", "--rm", "-i", "--network none", "-e LCB_CODE", "python:3.11-slim"} {
		if !strings.Contains(joined, want) {
			t.Errorf("dockerRunArgs missing %q; got %v", want, args)
		}
	}
	if args[0] != "run" {
		t.Errorf("first arg = %q, want \"run\"", args[0])
	}
	// The source must ride on the environment, not argv: no element may carry an
	// inline LCB_CODE= value.
	for _, a := range args {
		if strings.HasPrefix(a, "LCB_CODE=") {
			t.Errorf("candidate source leaked onto argv: %q", a)
		}
	}
	// The bootstrap that execs the env-carried source must be the -c payload.
	if !strings.Contains(joined, "os.environ['LCB_CODE']") {
		t.Errorf("bootstrap does not exec LCB_CODE from env; got %v", args)
	}
}

// TestDockerRunArgsCapsTheRunaway pins the resource ceilings. Docker's defaults
// for memory, pids, and cpu are all UNLIMITED, so a dropped flag here does not
// fail loudly — it silently hands an unbounded host to model-generated code, and
// the symptom (a wedged machine) surfaces far from the cause. Hence an explicit
// per-flag assertion rather than a smoke check.
func TestDockerRunArgsCapsTheRunaway(t *testing.T) {
	args := dockerRunArgs("python:3.11-slim", "fak-lcb-test-1")

	// Paired flags must be checked as adjacent VALUES, not substrings: asserting
	// on the joined string would pass if a value were attached to the wrong flag.
	pairs := map[string]string{
		"--memory":       sandboxMemoryLimit,
		"--pids-limit":   sandboxPidsLimit,
		"--cpus":         sandboxCPULimit,
		"--cap-drop":     "ALL",
		"--security-opt": "no-new-privileges",
		"--network":      "none",
		"--name":         "fak-lcb-test-1",
	}
	for flag, want := range pairs {
		idx := -1
		for i, a := range args {
			if a == flag {
				idx = i
				break
			}
		}
		if idx < 0 {
			t.Errorf("missing %s — an uncapped container is Docker's DEFAULT, so this cannot be caught downstream; got %v", flag, args)
			continue
		}
		if idx+1 >= len(args) || args[idx+1] != want {
			t.Errorf("%s value = %q, want %q", flag, args[idx+1], want)
		}
	}

	// The caps must precede the IMAGE: everything after the image name is argv for
	// the container, so a flag that drifts past it becomes a python argument and
	// silently stops constraining anything.
	image := -1
	for i, a := range args {
		if a == "python:3.11-slim" {
			image = i
			break
		}
	}
	if image < 0 {
		t.Fatalf("image not in argv: %v", args)
	}
	for flag := range pairs {
		for i, a := range args {
			if a == flag && i > image {
				t.Errorf("%s appears AFTER the image (index %d > %d) — it is a container argument there, not a limit", flag, i, image)
			}
		}
	}
}

// TestSandboxContainerNameIsUnique: the deadline path reaps by name, so two
// candidates sharing a name would let one grading timeout kill another
// candidate's still-running container and score it as a spurious TLE.
func TestSandboxContainerNameIsUnique(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 100; i++ {
		n := sandboxContainerName()
		if seen[n] {
			t.Fatalf("duplicate container name %q on iteration %d", n, i)
		}
		seen[n] = true
	}
}

// sandboxHelperEnv turns a re-executed copy of THIS test binary into a stand-in
// sandbox CLI: when the variable names a directory, the copy records the argv it was
// invoked with under <dir>/<verb>.argv and exits NON-ZERO. Every invocation fails,
// which is the whole point of the fixture — the reap contract is about what happens
// to a FAILING cleanup. Only a `go test` process ever sets it (init below is inert
// otherwise), and it is read in _test.go, which the CONFIG_NOT_ENV ratchet excludes.
const sandboxHelperEnv = "LCB_TEST_SANDBOX_WITNESS_DIR"

func init() {
	dir := os.Getenv(sandboxHelperEnv)
	if dir == "" {
		return // the ordinary `go test` process: not a sandbox stand-in
	}
	verb := "none"
	if len(os.Args) > 1 {
		verb = filepath.Base(os.Args[1])
	}
	_ = os.WriteFile(filepath.Join(dir, verb+".argv"), []byte(strings.Join(os.Args[1:], "\x00")), 0o600)
	os.Exit(3) // a sandbox CLI that FAILS
}

// sandboxHelperArgv returns the argv the stand-in sandbox recorded for verb, or nil
// when it was never invoked with that verb.
func sandboxHelperArgv(t *testing.T, dir, verb string) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, verb+".argv"))
	if err != nil {
		return nil
	}
	return strings.Split(string(b), "\x00")
}

// TestReapSandboxContainerSwallowsFailure: cleanup must never fail a grading run.
// Removing an already-exited container is an ERROR from docker and is also the
// common case (--rm usually wins the race), so the reap has to tolerate it.
//
// "Swallows" is not observable from reapSandboxContainer's own signature — it returns
// nothing, so calling it can only witness the absence of a panic. The contract lives at
// the CALLER: the deadline branch of dockerExecRunner reaps and must still grade the
// sample a clean TLE ("", true, nil). So this drives that branch with a sandbox CLI
// that is GUARANTEED to fail (the re-executed test binary, exit 3) and asserts both
// halves — that the reap really ran and really failed (its argv is on disk), and that
// its failure reached the graded result as exactly nothing.
func TestReapSandboxContainerSwallowsFailure(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Skipf("cannot locate the test binary to use as a failing sandbox: %v", err)
	}
	witness := t.TempDir()
	t.Setenv(sandboxHelperEnv, witness)

	// 1. The reap itself: a sandbox CLI that exits non-zero must leave the caller with
	//    nothing to handle, and must not wedge on its own 10s bound.
	done := make(chan struct{})
	go func() {
		defer close(done)
		reapSandboxContainer(self, "fak-lcb-test-nonexistent")
	}()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("reapSandboxContainer did not return: cleanup must be bounded, a grading run cannot block on it")
	}
	got := sandboxHelperArgv(t, witness, "rm")
	want := []string{"rm", "-f", "fak-lcb-test-nonexistent"}
	if !slices.Equal(got, want) {
		t.Fatalf("reap argv = %q, want %q (the failing reap must actually have been attempted)", got, want)
	}

	// 2. The contract the name states, at the only place it is observable. An expired
	//    deadline forces the reap branch; the reap then fails (exit 3, as above). The
	//    graded result must be a clean TLE: a cleanup failure is NOT a candidate error
	//    and NOT an infra error, or every timed-out sample would be regraded by the
	//    weather on the docker socket.
	if err := os.Remove(filepath.Join(witness, "rm.argv")); err != nil {
		t.Fatalf("clearing the reap witness: %v", err)
	}
	expired, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Hour))
	defer cancel()
	stdout, timedOut, err := dockerExecRunner(self, "python:3.11-slim")(expired, "print(1)", "", time.Second)
	if !timedOut {
		t.Fatalf("an expired deadline must grade as a timeout; timedOut=%v err=%v", timedOut, err)
	}
	if err != nil {
		t.Fatalf("the reap failed and its failure LEAKED into the graded result: err = %v, want nil", err)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty on a timeout", stdout)
	}
	reaped := sandboxHelperArgv(t, witness, "rm")
	if len(reaped) != 3 || reaped[0] != "rm" || reaped[1] != "-f" || !strings.HasPrefix(reaped[2], "fak-lcb-") {
		t.Fatalf("deadline path reap argv = %q, want [rm -f fak-lcb-*]: the timed-out container was never reaped", reaped)
	}
}

func TestSandboxPreflight(t *testing.T) {
	echo := func(_ context.Context, _, stdin string, _ time.Duration) (string, bool, error) {
		return stdin, false, nil // a live sandbox echoes stdin verbatim
	}
	if err := sandboxPreflight(echo); err != nil {
		t.Fatalf("echoing sandbox should pass preflight, got %v", err)
	}

	cases := map[string]livecodebench.ExecRunner{
		"wrong-output": func(_ context.Context, _, _ string, _ time.Duration) (string, bool, error) {
			return "not-the-input", false, nil
		},
		"exec-error": func(_ context.Context, _, _ string, _ time.Duration) (string, bool, error) {
			return "", false, errors.New("daemon down")
		},
		"timeout": func(_ context.Context, _, _ string, _ time.Duration) (string, bool, error) {
			return "", true, nil
		},
	}
	for name, run := range cases {
		t.Run(name, func(t *testing.T) {
			if err := sandboxPreflight(run); err == nil {
				t.Fatalf("preflight must fail for %s, got nil", name)
			}
		})
	}
}

// TestDockerExecRunnerInfraFailure exercises the real runner against a sandbox
// binary that does not exist: it must surface a non-timeout error (an infra
// failure, not a candidate RE) so the preflight can abstain. No Docker needed.
func TestDockerExecRunnerInfraFailure(t *testing.T) {
	run := dockerExecRunner("fak-no-such-sandbox-binary-xyz", "python:3.11-slim")
	stdout, timedOut, err := run(context.Background(), "print(1)", "", time.Second)
	if err == nil {
		t.Fatal("running a nonexistent sandbox binary must error")
	}
	if timedOut {
		t.Error("a missing binary is not a timeout")
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if !strings.Contains(err.Error(), "sandbox exec failed") {
		t.Errorf("infra failure should be reported as a sandbox failure, got %v", err)
	}
}

// TestResolveCodegenGraderAbsentSandbox: with no sandbox on PATH the resolver
// returns a nil grader and an honest GATED_UNGRADED note, so the seam abstains
// rather than fabricating a delta.
func TestResolveCodegenGraderAbsentSandbox(t *testing.T) {
	grade, note := resolveCodegenGrader("fak-no-such-sandbox-binary-xyz", "python:3.11-slim", 4)
	if grade != nil {
		t.Fatal("absent sandbox must resolve to a nil grader")
	}
	for _, want := range []string{"no code-execution sandbox", "GATED_UNGRADED"} {
		if !strings.Contains(note, want) {
			t.Errorf("note %q missing %q", note, want)
		}
	}
}
