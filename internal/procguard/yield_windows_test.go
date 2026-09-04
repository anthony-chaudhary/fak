//go:build windows

package procguard

import (
	"os"
	"os/exec"
	"syscall"
	"testing"
)

func TestWindowsEmptyWorkingSetCurrentProcess(t *testing.T) {
	h, err := syscall.GetCurrentProcess()
	if err != nil {
		t.Fatalf("syscall.GetCurrentProcess() failed: %v", err)
	}
	if h == 0 {
		t.Fatal("GetCurrentProcess returned 0 handle")
	}
	if !emptyWorkingSet(h) {
		t.Fatal("emptyWorkingSet failed on current process")
	}
}

func TestWindowsEmptyWorkingSetInvalidHandle(t *testing.T) {
	if emptyWorkingSet(0) {
		t.Fatal("emptyWorkingSet(0) should return false")
	}
	// An invalid handle like ^uintptr(0)-1 should fail
	if emptyWorkingSet(syscall.Handle(^uintptr(0) - 1)) {
		t.Fatal("emptyWorkingSet on invalid handle should return false")
	}
}

func TestWindowsYieldWorkingSetsChildProcess(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/c", "ping -n 4 127.0.0.1 >nul")
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start helper child process: %v", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	childPID := cmd.Process.Pid
	if childPID <= 0 {
		t.Fatalf("invalid child PID: %d", childPID)
	}

	// YieldMemory targeting the child process
	YieldMemory(childPID)

	// Also call yieldWorkingSets directly with the child and verify
	yieldWorkingSets(childPID)
}

func TestWindowsYieldWorkingSetsExitedPID(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/c", "exit 0")
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed to run helper process: %v", err)
	}
	exitedPID := cmd.Process.Pid

	// YieldMemory on an exited process must succeed silently without panic
	YieldMemory(exitedPID)
	yieldWorkingSets(exitedPID)
}

func TestWindowsYieldWorkingSetsDuplicatesAndCurrentPID(t *testing.T) {
	currentPID := os.Getpid()
	// Should handle duplicates and current PID gracefully
	YieldMemory(currentPID, currentPID, currentPID)
	yieldWorkingSets(currentPID, currentPID)
}
