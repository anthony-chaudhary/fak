//go:build windows

package windowgate

import (
	"bufio"
	"errors"
	"fmt"
	"os"
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

func TestRunInNewJobPreservesExitErrorOnWindows(t *testing.T) {
	cmd := exec.Command("cmd.exe", "/c", "exit", "7")
	err := RunInNewJob(cmd)
	exitErr, ok := err.(*exec.ExitError)
	if !ok || exitErr.ExitCode() != 7 {
		t.Fatalf("RunInNewJob error = %T %v, want *exec.ExitError code 7", err, err)
	}
}

func TestStartInNewJobAssignsBeforeChildExecutes(t *testing.T) {
	marker := t.TempDir() + `\child-ran`
	old := assignToNewJobObject
	assignmentObserved := false
	childRanEarly := false
	assignToNewJobObject = func(cmd *exec.Cmd) (*JobObject, error) {
		assignmentObserved = true
		if _, err := os.Stat(marker); err == nil {
			childRanEarly = true
		}
		return AssignToNewJobObject(cmd)
	}
	t.Cleanup(func() { assignToNewJobObject = old })

	script := fmt.Sprintf(`Set-Content -LiteralPath %q -Value ran`, marker)
	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
	job, err := StartInNewJob(cmd)
	if err != nil {
		t.Fatalf("StartInNewJob: %v", err)
	}
	if !assignmentObserved {
		t.Fatal("job assignment seam was not reached")
	}
	if childRanEarly {
		t.Fatal("child executed before job assignment")
	}
	if err := cmd.Wait(); err != nil {
		_ = job.Close()
		t.Fatalf("contained child Wait: %v", err)
	}
	if err := job.Close(); err != nil {
		t.Fatalf("job Close: %v", err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("resumed child did not execute after assignment: %v", err)
	}
}

func TestStartInNewJobReapsChildWhenAssignmentFails(t *testing.T) {
	old := assignToNewJobObject
	assignToNewJobObject = func(*exec.Cmd) (*JobObject, error) {
		return nil, errors.New("forced assignment failure")
	}
	t.Cleanup(func() { assignToNewJobObject = old })

	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 120")
	job, err := StartInNewJob(cmd)
	if err == nil || !strings.Contains(err.Error(), "forced assignment failure") {
		t.Fatalf("StartInNewJob error = %v, want forced assignment failure", err)
	}
	if job != nil {
		t.Fatalf("StartInNewJob job = %#v, want nil on assignment failure", job)
	}
	if cmd.Process == nil || cmd.Process.Pid <= 0 {
		t.Fatal("assignment-failure witness never started a child")
	}
	if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
		t.Fatalf("started child was not waited after containment failure: state=%v", cmd.ProcessState)
	}
	if processAlive(cmd.Process.Pid) {
		t.Fatalf("started child pid %d survived containment failure", cmd.Process.Pid)
	}
}

// TestRunInNewJobReapsTreeWhenOwnerIsTerminated models the Windows Terminal
// tab-close failure mode. The helper process owns the job handle and blocks in
// RunInNewJob; the parent force-terminates that owner without running defers,
// then proves Windows closed the handle and reaped both child generations.
func TestRunInNewJobReapsTreeWhenOwnerIsTerminated(t *testing.T) {
	if os.Getenv("FAK_WINDOWGATE_TAB_CLOSE_HELPER") == "1" {
		runTabCloseHelper(t)
		return
	}
	ps, err := exec.LookPath("powershell.exe")
	if err != nil {
		t.Skipf("powershell.exe not on PATH: %v", err)
	}
	_ = ps // The helper performs the same availability check in its process.

	helper := exec.Command(os.Args[0], "-test.run=^TestRunInNewJobReapsTreeWhenOwnerIsTerminated$")
	helper.Env = append(os.Environ(), "FAK_WINDOWGATE_TAB_CLOSE_HELPER=1")
	stdout, err := helper.StdoutPipe()
	if err != nil {
		t.Fatalf("helper StdoutPipe: %v", err)
	}
	helper.Stderr = os.Stderr
	if err := helper.Start(); err != nil {
		t.Fatalf("helper Start: %v", err)
	}
	defer func() {
		_ = helper.Process.Kill()
		_, _ = helper.Process.Wait()
	}()

	sc := bufio.NewScanner(stdout)
	pids := make([]int, 0, 2)
	deadline := time.After(30 * time.Second)
	lines := make(chan string)
	go func() {
		for sc.Scan() {
			lines <- strings.TrimSpace(sc.Text())
		}
		close(lines)
	}()
	for len(pids) < 2 {
		select {
		case line, ok := <-lines:
			if !ok {
				t.Fatalf("helper output closed before PIDs: %v", sc.Err())
			}
			if pid, err := strconv.Atoi(line); err == nil && pid > 0 {
				pids = append(pids, pid)
			}
		case <-deadline:
			t.Fatal("timed out waiting for contained child PIDs")
		}
	}
	if !waitFor(func() bool { return processAlive(pids[0]) && processAlive(pids[1]) }, 10*time.Second) {
		t.Fatalf("contained tree not alive before owner termination: pids=%v", pids)
	}

	// Process.Kill bypasses helper defers, matching a terminal host terminating
	// fak during CTRL_CLOSE_EVENT teardown.
	if err := helper.Process.Kill(); err != nil {
		t.Fatalf("terminate job owner: %v", err)
	}
	_, _ = helper.Process.Wait()
	if !waitFor(func() bool { return !processAlive(pids[0]) && !processAlive(pids[1]) }, 15*time.Second) {
		t.Fatalf("guard child tree survived owner termination: childAlive=%v grandchildAlive=%v pids=%v",
			processAlive(pids[0]), processAlive(pids[1]), pids)
	}
}

func runTabCloseHelper(t *testing.T) {
	ps, err := exec.LookPath("powershell.exe")
	if err != nil {
		t.Fatalf("powershell.exe not on PATH: %v", err)
	}
	const script = `$PID; $c = Start-Process -FilePath powershell.exe ` +
		`-ArgumentList '-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 120' ` +
		`-PassThru; Write-Output $c.Id; Start-Sleep -Seconds 120`
	cmd := exec.Command(ps, "-NoProfile", "-NonInteractive", "-Command", script)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := RunInNewJob(cmd); err != nil {
		fmt.Fprintf(os.Stderr, "RunInNewJob: %v\n", err)
		os.Exit(2)
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
