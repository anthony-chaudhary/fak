//go:build windows

package processalive

import (
	"os"
	"os/exec"
	"syscall"
	"testing"
)

func TestCheckDistinguishesLiveAndExitedProcesses(t *testing.T) {
	if !Check(os.Getpid()) {
		t.Fatal("Check(self) = false, want true")
	}
	if Check(0) || Check(-1) {
		t.Fatal("Check(non-positive pid) = true, want false")
	}

	cmd := exec.Command("cmd", "/c", "exit", "0")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if Check(pid) {
		t.Fatalf("Check(exited pid %d) = true, want false", pid)
	}
}

func TestCheckWindowsRegressionMatrix(t *testing.T) {
	origOpenProcess := openProcess
	t.Cleanup(func() {
		openProcess = origOpenProcess
	})

	// a) Access denied: stub openProcess returning syscall.ERROR_ACCESS_DENIED, assert Check(pid) == true.
	openProcess = func(da uint32, inherit bool, pid uint32) (syscall.Handle, error) {
		return 0, syscall.ERROR_ACCESS_DENIED
	}
	if !Check(12345) {
		t.Fatal("Check(access_denied) = false, want true")
	}

	// b) Invalid parameter: stub openProcess returning ERROR_INVALID_PARAMETER, assert Check(pid) == false.
	openProcess = func(da uint32, inherit bool, pid uint32) (syscall.Handle, error) {
		return 0, ERROR_INVALID_PARAMETER
	}
	if Check(12345) {
		t.Fatal("Check(invalid_parameter) = true, want false")
	}

	// c) Unexpected failure: stub openProcess returning an unexpected syscall.Errno(0x1234), assert Check(pid) == true.
	openProcess = func(da uint32, inherit bool, pid uint32) (syscall.Handle, error) {
		return 0, syscall.Errno(0x1234)
	}
	if !Check(12345) {
		t.Fatal("Check(unexpected_failure) = false, want true")
	}

	// Restore real openProcess for live, exited, and system process probes.
	openProcess = origOpenProcess

	// d) Live process (self os.Getpid()): Check(os.Getpid()) == true.
	if !Check(os.Getpid()) {
		t.Fatal("Check(os.Getpid()) = false, want true")
	}

	// e) Exited process (cmd /c exit 0): Check(pid) == false.
	cmd := exec.Command("cmd", "/c", "exit", "0")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if Check(pid) {
		t.Fatalf("Check(exited pid %d) = true, want false", pid)
	}

	// f) Real Windows System process (PID 4): Check(4) == true (PID 4 on Windows is always running and inaccessible).
	if !Check(4) {
		t.Fatal("Check(4) = false, want true (Windows System process should be preserved as alive)")
	}
}
