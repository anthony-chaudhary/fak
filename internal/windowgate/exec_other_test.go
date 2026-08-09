//go:build !windows

package windowgate

import (
	"context"
	"os/exec"
	"testing"
)

// TestWorkerJobShimIsNoopOffWindows asserts the tree-teardown surface compiles
// and is a benign no-op off Windows (architest tier unchanged): ConfigureWorker
// does not panic, AssignToNewJobObject returns (nil, nil), and Close is safe.
func TestWorkerJobShimIsNoopOffWindows(t *testing.T) {
	cmd := exec.Command("true")
	ConfigureWorkerCommand(cmd) // must not panic; no SysProcAttr requirement off Windows

	job, err := AssignToNewJobObject(cmd)
	if err != nil {
		t.Fatalf("AssignToNewJobObject shim returned error: %v", err)
	}
	if job != nil {
		t.Fatalf("AssignToNewJobObject shim returned non-nil job: %v", job)
	}
	if err := job.Close(); err != nil { // nil receiver Close must be safe
		t.Fatalf("nil JobObject.Close returned error: %v", err)
	}
}

func TestRunInNewJobPreservesExitErrorOffWindows(t *testing.T) {
	cmd := exec.Command("sh", "-c", "exit 7")
	err := RunInNewJob(cmd)
	exitErr, ok := err.(*exec.ExitError)
	if !ok || exitErr.ExitCode() != 7 {
		t.Fatalf("RunInNewJob error = %T %v, want *exec.ExitError code 7", err, err)
	}
}

func TestCommandConstructorsPortable(t *testing.T) {
	if got := Command("go", "version"); got.Path == "" {
		t.Fatal("Command returned empty path")
	}
	if got := CommandContext(context.Background(), "go", "version"); got.Path == "" {
		t.Fatal("CommandContext returned empty path")
	}
}
