//go:build windows

package procguard

import (
	"errors"
	"os/exec"
	"syscall"
	"testing"
)

func TestSuspendWindowsConstants(t *testing.T) {
	if processSuspendResume != 0x0800 {
		t.Fatalf("processSuspendResume = %#x, want 0x0800", processSuspendResume)
	}
	if statusSuccess != 0x00000000 {
		t.Fatalf("statusSuccess = %#x, want 0x00000000", statusSuccess)
	}
}

func TestSuspendWindowsProcessPID4(t *testing.T) {
	if err := SuspendProcess(4); err == nil {
		t.Fatal("SuspendProcess(4) expected error for System PID 4, got nil")
	}
	if err := ResumeProcess(4); err == nil {
		t.Fatal("ResumeProcess(4) expected error for System PID 4, got nil")
	}
}

func TestSuspendWindowsResumePingSubprocess(t *testing.T) {
	cmd := exec.Command("ping", "127.0.0.1", "-n", "10")
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start ping: %v", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	pid := cmd.Process.Pid
	if pid <= 0 {
		t.Fatalf("invalid ping pid: %d", pid)
	}

	if err := SuspendProcess(pid); err != nil {
		t.Fatalf("NtSuspendProcess on ping failed: %v", err)
	}

	if err := ResumeProcess(pid); err != nil {
		t.Fatalf("NtResumeProcess on ping failed: %v", err)
	}
}

func TestSuspendWindowsResumeExitedProcess(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/c", "exit 0")
	if err := cmd.Run(); err != nil {
		t.Fatalf("failed to run helper process: %v", err)
	}
	exitedPID := cmd.Process.Pid

	err := SuspendProcess(exitedPID)
	if err == nil {
		t.Fatal("SuspendProcess(exitedPID) expected error, got nil")
	}
	if !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("SuspendProcess(exitedPID) error = %v, want ESRCH", err)
	}

	err = ResumeProcess(exitedPID)
	if err == nil {
		t.Fatal("ResumeProcess(exitedPID) expected error, got nil")
	}
	if !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("ResumeProcess(exitedPID) error = %v, want ESRCH", err)
	}
}
