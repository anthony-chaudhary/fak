package procguard

import (
	"errors"
	"os/exec"
	"runtime"
	"syscall"
	"testing"
)

func startLongRunningHelper() (*exec.Cmd, error) {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("ping", "127.0.0.1", "-n", "10")
	} else {
		cmd = exec.Command("sleep", "10")
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

func TestSuspendProcessInvalidPIDs(t *testing.T) {
	invalidPIDs := []int{0, -1, -999}
	for _, pid := range invalidPIDs {
		if err := SuspendProcess(pid); err == nil {
			t.Errorf("SuspendProcess(%d) expected error, got nil", pid)
		}
		if err := ResumeProcess(pid); err == nil {
			t.Errorf("ResumeProcess(%d) expected error, got nil", pid)
		}
	}
}

func TestSuspendProcessNonExistentPID(t *testing.T) {
	nonExistentPID := 2147483640
	err := SuspendProcess(nonExistentPID)
	if err == nil {
		t.Fatal("SuspendProcess(nonExistentPID) expected error, got nil")
	}
	if !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("SuspendProcess(nonExistentPID) error = %v, want ESRCH", err)
	}

	err = ResumeProcess(nonExistentPID)
	if err == nil {
		t.Fatal("ResumeProcess(nonExistentPID) expected error, got nil")
	}
	if !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("ResumeProcess(nonExistentPID) error = %v, want ESRCH", err)
	}
}

func TestSuspendResumeSubprocess(t *testing.T) {
	cmd, err := startLongRunningHelper()
	if err != nil {
		t.Fatalf("failed to start test helper: %v", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	pid := cmd.Process.Pid
	if pid <= 0 {
		t.Fatalf("invalid subprocess pid: %d", pid)
	}

	if err := SuspendProcess(pid); err != nil {
		t.Fatalf("SuspendProcess(%d) failed: %v", pid, err)
	}

	if err := ResumeProcess(pid); err != nil {
		t.Fatalf("ResumeProcess(%d) failed: %v", pid, err)
	}
}
