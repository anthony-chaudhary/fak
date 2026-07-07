package microagent_test

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/microagent"

	// The microagent-minimal registration set (#2009): wires the Ref resolver +
	// the registered adjudication floor so a bare kernel.New folds the REAL
	// process-global monitor chain, not an empty one — the floor the deny test
	// must witness gating exec.
	_ "github.com/anthony-chaudhary/fak/internal/registrations/microagent"
)

// helperMode selects what a re-exec'd copy of the test binary does. Guarding on
// this env var keeps the function a passing no-op under a normal `go test` run
// and turns it into the requested helper only when Run re-execs os.Args[0].
const helperModeEnv = "MICROAGENT_TOOLEXEC_HELPER"

// helperMarkerEnv carries the path a surviving grandchild would write to — the
// tree-kill witness (its ABSENCE proves the grandchild was reaped).
const helperMarkerEnv = "MICROAGENT_TOOLEXEC_MARKER"

// TestHelperProcess is the subprocess body Run execs. It is not a real test: it
// dispatches on helperModeEnv and exits, so it stays an inert pass when the env
// is unset (the normal suite run).
func TestHelperProcess(t *testing.T) {
	switch os.Getenv(helperModeEnv) {
	case "":
		return // ordinary `go test` — inert.
	case "echo":
		// Capture witness: known bytes on both streams, clean exit.
		os.Stdout.WriteString("out-marker")
		os.Stderr.WriteString("err-marker")
		os.Exit(0)
	case "runaway":
		// Fork a grandchild that INHERITS this process's stdout (the captured
		// pipe), then park well past the action timeout. The whole tree must die
		// on the timeout.
		gc := exec.Command(os.Args[0], "-test.run=^TestHelperProcess$")
		gc.Env = append(os.Environ(), helperModeEnv+"=grandchild")
		gc.Stdout = os.Stdout
		gc.Stderr = os.Stderr
		_ = gc.Start()
		time.Sleep(60 * time.Second)
		os.Exit(0)
	case "grandchild":
		// Survive a grace window comfortably longer than the action timeout, THEN
		// write the marker. If the tree kill reached us we die before this line and
		// the marker never appears; if only the direct child was killed we live to
		// write it and the test fails.
		time.Sleep(5 * time.Second)
		if p := os.Getenv(helperMarkerEnv); p != "" {
			_ = os.WriteFile(p, []byte("grandchild-survived"), 0o600)
		}
		time.Sleep(60 * time.Second)
		os.Exit(0)
	}
}

// denyKernel folds a fail-closed empty policy: nothing is affirmatively allowed,
// so every action DEFAULT_DENYs. Injected explicitly so the test does not depend
// on the shared adjudicator.Default's mutable policy.
func denyKernel() *kernel.Kernel {
	return kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{adjudicator.New(adjudicator.Policy{})}))
}

// allowKernel folds a policy that affirmatively allows exactly the test's action
// tool, so the exec path is reached.
func allowKernel(tool string) *kernel.Kernel {
	return kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{
		adjudicator.New(adjudicator.Policy{Allow: map[string]bool{tool: true}}),
	}))
}

// TestToolExecKernelAdjudicationGatesExec is the #2014 acceptance half "kernel
// adjudication still fires before exec": a denied action never spawns a process.
// The action's Path points at the re-exec helper in a mode that would write a
// marker file; because the floor DENIES it, exec never happens and the marker is
// absent. It also witnesses the REAL registered floor (empty policy → default
// deny) refusing, not just an injected stub.
func TestToolExecKernelAdjudicationGatesExec(t *testing.T) {
	te, err := microagent.NewToolExec(denyKernel())
	if err != nil {
		t.Fatalf("NewToolExec: %v", err)
	}
	marker := filepath.Join(t.TempDir(), "should-not-exist")
	act := microagent.ToolAction{
		Tool:    "run_shell", // not in the empty policy's allow set → default deny
		Path:    os.Args[0],
		Argv:    []string{"-test.run=^TestHelperProcess$"},
		Timeout: 5 * time.Second,
	}
	// Point the (never-reached) helper at the marker so a spawn would be visible.
	t.Setenv(helperModeEnv, "grandchild")
	t.Setenv(helperMarkerEnv, marker)

	res, err := te.Run(context.Background(), act)
	if !errors.Is(err, microagent.ErrActionDenied) {
		t.Fatalf("Run on a denied action = %v, want ErrActionDenied", err)
	}
	if res.Ran {
		t.Fatal("Ran=true on a denied action — the subprocess must never start")
	}
	if res.Verdict.Kind != abi.VerdictDeny {
		t.Errorf("verdict kind = %v, want VerdictDeny", res.Verdict.Kind)
	}
	if _, statErr := os.Stat(marker); statErr == nil {
		t.Fatal("marker exists — the denied action spawned a process (adjudication did NOT gate exec)")
	}
}

// TestToolExecRunawayKilledTreeWide is the #2014 acceptance half "a runaway
// action is killed tree-wide on timeout". The allowed action re-execs a helper
// that forks a grandchild and parks for 60s; the 2s per-action timeout must reap
// the WHOLE tree via the reused procguard reaper. The grandchild's post-grace
// marker (written only if it outlives the kill) must be absent.
func TestToolExecRunawayKilledTreeWide(t *testing.T) {
	te, err := microagent.NewToolExec(allowKernel("run_shell"))
	if err != nil {
		t.Fatalf("NewToolExec: %v", err)
	}
	marker := filepath.Join(t.TempDir(), "grandchild-marker")
	act := microagent.ToolAction{
		Tool:    "run_shell",
		Path:    os.Args[0],
		Argv:    []string{"-test.run=^TestHelperProcess$"},
		Timeout: 2 * time.Second,
	}
	t.Setenv(helperModeEnv, "runaway")
	t.Setenv(helperMarkerEnv, marker)

	start := time.Now()
	res, err := te.Run(context.Background(), act)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	elapsed := time.Since(start)
	if !res.Ran {
		t.Fatal("Ran=false — an allowed action must start the subprocess")
	}
	if !res.TimedOut || !res.Killed {
		t.Fatalf("TimedOut=%v Killed=%v, want both true (the runaway was not reaped on timeout)", res.TimedOut, res.Killed)
	}
	// The reap must be prompt: ~timeout + at most WaitDelay, never the 60s park.
	if elapsed > 30*time.Second {
		t.Fatalf("Run took %v — the process tree was not killed promptly on timeout", elapsed)
	}
	// Tree-wide witness: wait past the grandchild's 5s grace window, then assert it
	// never wrote its marker — i.e. the kill reached the grandchild, not just the
	// direct child.
	time.Sleep(6 * time.Second)
	if _, statErr := os.Stat(marker); statErr == nil {
		t.Fatal("grandchild marker exists — the timeout killed the direct child but NOT the tree")
	}
}

// TestToolExecAllowRunsAndCaptures pins the happy path: an allowed action runs to
// a clean exit with stdout/stderr captured and no timeout.
func TestToolExecAllowRunsAndCaptures(t *testing.T) {
	te, err := microagent.NewToolExec(allowKernel("run_shell"))
	if err != nil {
		t.Fatalf("NewToolExec: %v", err)
	}
	act := microagent.ToolAction{
		Tool:    "run_shell",
		Path:    os.Args[0],
		Argv:    []string{"-test.run=^TestHelperProcess$"},
		Timeout: 30 * time.Second,
	}
	t.Setenv(helperModeEnv, "echo")

	res, err := te.Run(context.Background(), act)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !res.Ran || res.TimedOut || res.Killed {
		t.Fatalf("Ran=%v TimedOut=%v Killed=%v, want Ran=true and no kill", res.Ran, res.TimedOut, res.Killed)
	}
	if res.ExitCode != 0 {
		t.Errorf("exit code = %d, want 0", res.ExitCode)
	}
	if string(res.Stdout) != "out-marker" {
		t.Errorf("stdout = %q, want %q", res.Stdout, "out-marker")
	}
	if string(res.Stderr) != "err-marker" {
		t.Errorf("stderr = %q, want %q", res.Stderr, "err-marker")
	}
	if res.Verdict.Kind != abi.VerdictAllow {
		t.Errorf("verdict = %v, want VerdictAllow", res.Verdict.Kind)
	}
}

// TestToolExecRefusals pins the constructor + no-program edges.
func TestToolExecRefusals(t *testing.T) {
	if _, err := microagent.NewToolExec(nil); !errors.Is(err, microagent.ErrNilFloor) {
		t.Fatalf("NewToolExec(nil) = %v, want ErrNilFloor", err)
	}
	te, err := microagent.NewToolExec(allowKernel("run_shell"))
	if err != nil {
		t.Fatalf("NewToolExec: %v", err)
	}
	// Allowed by the floor but no Path to exec → ErrNoProgram, never a spawn.
	res, err := te.Run(context.Background(), microagent.ToolAction{Tool: "run_shell"})
	if !errors.Is(err, microagent.ErrNoProgram) {
		t.Fatalf("Run with no Path = %v, want ErrNoProgram", err)
	}
	if res.Ran {
		t.Fatal("Ran=true with no Path")
	}
}
