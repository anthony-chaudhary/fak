//go:build windows

package windowgate

import (
	"bufio"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestConfigureBackgroundCommandSetsWindowsNoWindow(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	ConfigureBackgroundCommand(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Fatal("HideWindow = false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow == 0 {
		t.Fatalf("CreationFlags=%#x missing CREATE_NO_WINDOW", cmd.SysProcAttr.CreationFlags)
	}
}

// TestConfigureWorkerCommandSetsProcessGroupFlag asserts the worker launch path
// carries the no-window flags AND CREATE_NEW_PROCESS_GROUP (acceptance: the
// spawned worker's SysProcAttr carries the group flag).
func TestConfigureWorkerCommandSetsProcessGroupFlag(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	ConfigureWorkerCommand(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Error("HideWindow = false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow == 0 {
		t.Errorf("CreationFlags=%#x missing CREATE_NO_WINDOW", cmd.SysProcAttr.CreationFlags)
	}
	if cmd.SysProcAttr.CreationFlags&CreateNewProcessGroup == 0 {
		t.Errorf("CreationFlags=%#x missing CREATE_NEW_PROCESS_GROUP", cmd.SysProcAttr.CreationFlags)
	}
}

// processAlive reports whether pid names a running process. A terminated process
// reads either as "cannot open" or GetExitCodeProcess != STILL_ACTIVE.
func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	const processQueryLimitedInformation = 0x1000
	const stillActive = 259
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil {
		return false
	}
	defer syscall.CloseHandle(h)
	var code uint32
	if err := syscall.GetExitCodeProcess(h, &code); err != nil {
		return false
	}
	return code == stillActive
}

// TestAssignToNewJobObjectKillsProcessTree is the load-bearing acceptance test:
// spawn a shell that forks a long-lived grandchild, record both PIDs, close the
// job handle, and assert BOTH processes are gone — proving KILL_ON_JOB_CLOSE
// reaps the descendant tree in one syscall (kill == kill the tree).
func TestAssignToNewJobObjectKillsProcessTree(t *testing.T) {
	ps, err := exec.LookPath("powershell.exe")
	if err != nil {
		t.Skipf("powershell.exe not on PATH: %v", err)
	}

	// Parent shell: start a detached grandchild that sleeps, print its PID, then
	// sleep itself so both stay resident until we close the job. The grandchild,
	// spawned by a job member without breakaway, inherits the job.
	const script = `$c = Start-Process -FilePath powershell.exe ` +
		`-ArgumentList '-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 120' ` +
		`-PassThru; Write-Output $c.Id; Start-Sleep -Seconds 120`
	cmd := exec.Command(ps, "-NoProfile", "-NonInteractive", "-Command", script)
	ConfigureWorkerCommand(cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	parentPID := cmd.Process.Pid

	job, err := AssignToNewJobObject(cmd)
	if err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		t.Fatalf("AssignToNewJobObject: %v", err)
	}

	// Read the grandchild PID the parent reports.
	grandchildPID := 0
	sc := bufio.NewScanner(stdout)
	done := make(chan struct{})
	go func() {
		if sc.Scan() {
			grandchildPID, _ = strconv.Atoi(strings.TrimSpace(sc.Text()))
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		_ = job.Close()
		t.Fatal("timed out waiting for grandchild PID from parent shell")
	}
	if grandchildPID == 0 {
		_ = job.Close()
		t.Fatal("parent shell did not report a grandchild PID")
	}

	// Both must be alive before we close the job, or the test proves nothing.
	if !waitFor(func() bool { return processAlive(parentPID) && processAlive(grandchildPID) }, 10*time.Second) {
		_ = job.Close()
		t.Fatalf("parent(%d) or grandchild(%d) not alive before job close", parentPID, grandchildPID)
	}

	// Close the job handle: KILL_ON_JOB_CLOSE must reap the whole tree.
	if err := job.Close(); err != nil {
		t.Fatalf("job.Close: %v", err)
	}
	_, _ = cmd.Process.Wait()

	if !waitFor(func() bool { return !processAlive(parentPID) && !processAlive(grandchildPID) }, 15*time.Second) {
		t.Fatalf("tree survived job close: parentAlive=%v grandchildAlive=%v",
			processAlive(parentPID), processAlive(grandchildPID))
	}
}

// waitFor polls cond until it is true or the deadline elapses.
func waitFor(cond func() bool, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for {
		if cond() {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(100 * time.Millisecond)
	}
}
