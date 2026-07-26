package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

// execrunner.go plugs a real code-execution sandbox behind the injected
// livecodebench.ExecRunner seam (grade.go). It closes the always-nil gap in
// resolveCodegenGrader so `livecodebench ab-graded` can produce a LOCAL
// pass-rate signal instead of abstaining on every host — while keeping the
// honesty fences intact: the runner is Docker-isolated, and grading is gated on
// a preflight that proves the sandbox can actually execute a candidate before it
// is trusted to grade one (an unusable sandbox abstains, it never scores every
// sample a fabricated RE). The delta this feeds remains local-ungraded and
// never backs a claim (that path runs through the official lcb_runner).

// pyBootstrap execs the candidate program out of the LCB_CODE environment
// variable. Passing the source via the environment (not argv) keeps it off the
// command line — no quoting or length pitfalls — and leaves the container's
// stdin free to carry the test-case input the program reads.
const pyBootstrap = "import os;exec(compile(os.environ['LCB_CODE'],'<candidate>','exec'))"

// Resource ceilings for one candidate container. A LiveCodeBench candidate is
// model-generated code of unknown quality, so the failure mode to design for is
// not malice but a runaway: an accidental fork bomb, an unbounded allocation, a
// busy loop across every core. Docker's defaults for all three are UNLIMITED, so
// without these a single bad generation can saturate the host — and on this fleet
// that is a measured hazard, not a hypothetical one (host lockups here trace to
// scheduler/memory-manager pressure from spawn bursts, not to steady load).
//
// None of these can change a CORRECT program's verdict. They bound only programs
// that were already going to fail: an allocation past the cap is a candidate
// error, a fork past the pid cap is a candidate error, and the cpu cap makes the
// existing wall-clock TLE MORE comparable across hosts rather than less, since a
// candidate can no longer buy time by spreading across cores.
const (
	sandboxMemoryLimit = "512m" // generous for an algorithm answer, fatal to a leak
	sandboxPidsLimit   = "128"  // a solution needs a handful; a fork bomb needs many
	sandboxCPULimit    = "1"    // one core, so the wall-clock deadline means one thing
)

// dockerRunArgs builds the `<sandbox> run` argv that executes one candidate
// program in a throwaway, network-isolated, resource-capped container: --rm reaps
// it, -i keeps stdin open for the test input, --network none denies the untrusted
// program any egress, -e LCB_CODE forwards ONLY the source variable into the
// container, --name gives the deadline path a handle to kill (see
// dockerExecRunner), and python runs the bootstrap that execs it. It is a pure
// function so the argv shape is unit-tested without a live daemon.
//
// Two hardening flags are deliberately NOT here, because each can fail a correct
// candidate and a false wrong-answer corrupts the benchmark signal this whole
// path exists to produce:
//   - --read-only: a legitimate solution may use a temp file.
//   - --user nobody: depends on the image's ownership; python:3.11-slim is not
//     guaranteed to be writable-where-needed for an arbitrary uid.
//
// They are worth revisiting behind a flag once there is a graded run to compare
// against, so a regression in pass-rate would be attributable.
func dockerRunArgs(image, name string) []string {
	return []string{
		"run", "--rm", "-i",
		"--network", "none",
		"--name", name,
		"--cpus", sandboxCPULimit,
		"--memory", sandboxMemoryLimit,
		"--pids-limit", sandboxPidsLimit,
		"--cap-drop", "ALL",
		"--security-opt", "no-new-privileges",
		"-e", "LCB_CODE",
		image,
		"python3", "-c", pyBootstrap,
	}
}

// sandboxRunSeq numbers containers within this process so each candidate gets a
// distinct --name. Process id plus a counter is enough: the name only has to be
// unique among the containers this run could have alive at once, and it is not a
// security boundary.
var sandboxRunSeq atomic.Uint64

func sandboxContainerName() string {
	return fmt.Sprintf("fak-lcb-%d-%d", os.Getpid(), sandboxRunSeq.Add(1))
}

// reapSandboxContainer force-removes a named container, best-effort and bounded.
//
// This exists because CommandContext's deadline kills the docker CLIENT, not the
// container the daemon is running. Without this, every timed-out candidate leaves
// a live container behind — still holding its memory and cpu share — and a
// benchmark run times out many candidates by design, so the orphans accumulate
// for the whole run. --rm only fires when the container itself exits.
//
// Failure is ignored on purpose: the container may have already exited (the
// common case, and `rm -f` on a missing name is an error), and a grading run must
// never fail because cleanup did.
func reapSandboxContainer(sandboxCmd, name string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = exec.CommandContext(ctx, sandboxCmd, "rm", "-f", name).Run()
}

// dockerExecRunner returns a livecodebench.ExecRunner that grades one candidate
// program by running it in a `<sandboxCmd> run` container (Docker-compatible
// CLI). It maps a per-test wall-clock deadline to timedOut=true (graded TLE) and
// a non-timeout container/interpreter failure to a non-nil err (graded RE). The
// source is injected through the LCB_CODE env var; stdin carries the test input;
// only stdout is compared (stderr is captured for RE diagnostics).
//
// The ExitError=>RE mapping is only sound once a candidate is known to be
// runnable, which is why the CLI gates this runner behind sandboxPreflight: a
// daemon-down / image-missing failure trips the preflight and abstains, so it is
// never miscounted as a per-sample runtime error inside a grading run.
func dockerExecRunner(sandboxCmd, image string) livecodebench.ExecRunner {
	return func(ctx context.Context, code, stdin string, timeout time.Duration) (string, bool, error) {
		if timeout <= 0 {
			timeout = livecodebench.DefaultGradeTimeout
		}
		runCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		name := sandboxContainerName()
		cmd := exec.CommandContext(runCtx, sandboxCmd, dockerRunArgs(image, name)...)
		cmd.Env = append(os.Environ(), "LCB_CODE="+code)
		cmd.Stdin = strings.NewReader(stdin)
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr

		err := cmd.Run()
		if runCtx.Err() == context.DeadlineExceeded {
			// The deadline fired: a time-limit-exceeded, distinct from a wrong
			// answer or a crash. Killing the client did NOT stop the container, so
			// reap it by name before returning — otherwise a run that times out
			// many candidates (which grading does by design) leaves a live,
			// resource-holding container behind for each one.
			reapSandboxContainer(sandboxCmd, name)
			return "", true, nil
		}
		if err != nil {
			var ee *exec.ExitError
			if errors.As(err, &ee) {
				// The container ran and the interpreter exited non-zero: a
				// candidate runtime error (graded RE by GradeCode).
				return "", false, fmt.Errorf("candidate exited nonzero: %s", tail(stderr.String(), 400))
			}
			// The sandbox itself could not run the command (daemon down, binary
			// missing). Surfaced as an error; the preflight refuses to grade when
			// this happens to a smoke program, so it abstains rather than scoring
			// a fabricated RE.
			return "", false, fmt.Errorf("sandbox exec failed: %w", err)
		}
		return stdout.String(), false, nil
	}
}

// sandboxPreflight proves the sandbox can actually execute a candidate before it
// is trusted to grade real generations. It runs a trivial program that echoes
// its stdin; a live sandbox returns the input verbatim. Any failure — exec
// error, timeout, or a mismatched echo — means the sandbox is not usable, and
// the caller abstains (GATED_UNGRADED) rather than grade against a broken host.
func sandboxPreflight(run livecodebench.ExecRunner) error {
	const want = "fak-lcb-preflight-ok"
	const echo = "import sys;sys.stdout.write(sys.stdin.read())"
	stdout, timedOut, err := run(context.Background(), echo, want, 30*time.Second)
	switch {
	case err != nil:
		return fmt.Errorf("smoke exec failed: %w", err)
	case timedOut:
		return errors.New("smoke exec timed out")
	case strings.TrimSpace(stdout) != want:
		return fmt.Errorf("smoke exec produced %q, want %q", strings.TrimSpace(stdout), want)
	default:
		return nil
	}
}

// tail returns the last n bytes of s (trimmed), prefixed with an ellipsis when
// truncated, so a captured stderr snippet stays bounded in diagnostics.
func tail(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return "…" + s[len(s)-n:]
}
