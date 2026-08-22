package windowgate

import (
	"errors"
	"os"
	"os/exec"
	"testing"
	"time"
)

type cleanupRecorder struct {
	callbacks []func()
}

func (c *cleanupRecorder) Cleanup(fn func()) {
	c.callbacks = append(c.callbacks, fn)
}

func (c *cleanupRecorder) Run() {
	for i := len(c.callbacks) - 1; i >= 0; i-- {
		c.callbacks[i]()
	}
	c.callbacks = nil
}

func TestStartOwnedProcessRequiresCleanupOwner(t *testing.T) {
	if _, err := StartOwnedProcess(nil, exec.Command(os.Args[0])); err == nil {
		t.Fatal("StartOwnedProcess accepted a nil cleanup owner")
	}
}

func TestOwnedProcessCleanupJoinsRoot(t *testing.T) {
	if os.Getenv("FAK_WINDOWGATE_OWNED_ROOT_HELPER") == "1" {
		time.Sleep(2 * time.Minute)
		return
	}

	owner := &cleanupRecorder{}
	cmd := exec.Command(os.Args[0], "-test.run=^TestOwnedProcessCleanupJoinsRoot$")
	cmd.Env = append(os.Environ(), "FAK_WINDOWGATE_OWNED_ROOT_HELPER=1")
	process, err := StartOwnedProcess(owner, cmd)
	if err != nil {
		t.Fatalf("StartOwnedProcess: %v", err)
	}
	if process.PID() <= 0 {
		t.Fatalf("PID = %d, want a started root", process.PID())
	}

	owner.Run() // models testing.T cleanup after Fatal/FailNow
	if cmd.ProcessState == nil {
		t.Fatalf("cleanup returned before root was joined: state=%v", cmd.ProcessState)
	}
	if err := process.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestOwnedProcessWaitPreservesExitError(t *testing.T) {
	if os.Getenv("FAK_WINDOWGATE_OWNED_EXIT_HELPER") == "1" {
		os.Exit(7)
	}

	owner := &cleanupRecorder{}
	cmd := exec.Command(os.Args[0], "-test.run=^TestOwnedProcessWaitPreservesExitError$")
	cmd.Env = append(os.Environ(), "FAK_WINDOWGATE_OWNED_EXIT_HELPER=1")
	process, err := StartOwnedProcess(owner, cmd)
	if err != nil {
		t.Fatalf("StartOwnedProcess: %v", err)
	}
	defer owner.Run()

	err = process.Wait()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 7 {
		t.Fatalf("Wait error = %T %v, want *exec.ExitError code 7", err, err)
	}
}
