package main

import (
	"context"
	"errors"
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

// TestReapSandboxContainerSwallowsFailure: cleanup must never fail a grading run.
// Removing an already-exited container is an ERROR from docker and is also the
// common case (--rm usually wins the race), so the reap has to tolerate it.
func TestReapSandboxContainerSwallowsFailure(t *testing.T) {
	// A binary that does not exist is the strongest form of the failure.
	reapSandboxContainer("fak-no-such-sandbox-binary-xyz", "fak-lcb-test-nonexistent")
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
