//go:build windows

package serverlifecycle

import (
	"os"
	"os/exec"
	"testing"
	"time"
)

// fak-test:justify why=regression when=changed:internal/serverlifecycle/lifecycle.go witness=internal/serverlifecycle/lifecycle.go#func waitForOriginalProcess
// fak-test:runtime fast est=1s lane=default
func TestWaitForOriginalProcessAcceptsExitedProcessWithRetainedHandle(t *testing.T) {
	if os.Getenv("FAK_SERVERLIFECYCLE_PROCESS_HELPER") == "1" {
		time.Sleep(30 * time.Second)
		return
	}

	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(executable, "-test.run=^TestWaitForOriginalProcessAcceptsExitedProcessWithRetainedHandle$")
	cmd.Env = append(os.Environ(), "FAK_SERVERLIFECYCLE_PROCESS_HELPER=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}()

	identity, err := waitForProcessIdentity(cmd.Process.Pid, 3*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if processDefinitelyGone(cmd.Process.Pid) {
		t.Fatal("live matching process reported gone")
	}
	if !processMatches(cmd.Process.Pid, identity) {
		t.Fatal("live process did not retain its observed start identity")
	}

	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for !processDefinitelyGone(cmd.Process.Pid) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !processDefinitelyGone(cmd.Process.Pid) {
		t.Fatal("killed process did not report an exit while its handle was retained")
	}
	if !processMatches(cmd.Process.Pid, identity) {
		t.Fatal("retained handle no longer exposed the killed process start identity")
	}
	if !waitForOriginalProcess(cmd.Process.Pid, identity, 3*time.Second) {
		t.Fatal("terminated process remained live while its process handle was retained")
	}
}
