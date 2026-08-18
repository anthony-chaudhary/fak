//go:build !windows

package processalive

import (
	"os"
	"os/exec"
	"testing"
)

func TestCheckDistinguishesLiveAndExitedProcesses(t *testing.T) {
	if !Check(os.Getpid()) {
		t.Fatal("Check(self) = false, want true")
	}
	if Check(0) || Check(-1) {
		t.Fatal("Check(non-positive pid) = true, want false")
	}

	cmd := exec.Command("sh", "-c", "exit 0")
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
