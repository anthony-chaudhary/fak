//go:build windows

package windowgate

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

// Env keys the console probe below uses to re-exec this test binary as its own
// child, matching the FAK_WINDOWGATE_TAB_CLOSE_HELPER pattern further down.
const (
	consoleProbeEnv       = "FAK_WINDOWGATE_CONSOLE_PROBE"
	consoleProbeReportEnv = "FAK_WINDOWGATE_CONSOLE_PROBE_REPORT"
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

// TestConfigureDetachedCommandClearsNoWindowAndDetaches pins the mutual-exclusion
// invariant ConfigureDetachedCommand exists to encode (#3597). The production
// sequence applies it OVER an already-background command (spawnDispatchIssueWorker
// calls configureDispatchSpawn first), so the pre-set CREATE_NO_WINDOW below is the
// real starting state, not a contrived one. Windows silently ignores
// CREATE_NO_WINDOW when DETACHED_PROCESS is present, so leaving both set would
// encode a contradiction that reads as though the window flag still did something.
func TestConfigureDetachedCommandClearsNoWindowAndDetaches(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "exit", "0")
	ConfigureBackgroundCommand(cmd) // the production predecessor
	ConfigureDetachedCommand(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Error("HideWindow = false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&DetachedProcess == 0 {
		t.Errorf("CreationFlags=%#x missing DETACHED_PROCESS (%#x)",
			cmd.SysProcAttr.CreationFlags, DetachedProcess)
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow != 0 {
		t.Errorf("CreationFlags=%#x still carries CREATE_NO_WINDOW (%#x); DETACHED_PROCESS must clear it",
			cmd.SysProcAttr.CreationFlags, CreateNoWindow)
	}
	ConfigureDetachedCommand(nil) // must not panic on a nil command
}

// consoleAttachedProcessCount returns how many processes share THIS process's
// console, via GetConsoleProcessList. It is 0 exactly when the caller has no
// console at all — which is the only direct, host-independent way to tell
// DETACHED_PROCESS apart from CREATE_NO_WINDOW.
//
// GetConsoleWindow is NOT usable here: CREATE_NO_WINDOW allocates a console whose
// WINDOW is suppressed, so it returns NULL for both flags even though only one of
// them actually declined the console (and its conhost.exe host process).
func consoleAttachedProcessCount() uint32 {
	buf := make([]uint32, 64)
	r, _, _ := kernel32.NewProc("GetConsoleProcessList").Call(
		uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	return uint32(r)
}

// TestDetachedSpawnHasNoConsoleWhereBackgroundSpawnDoes is #3597's load-bearing
// witness: it measures the thing the issue is actually about — whether the spawned
// child owns a console (and therefore a conhost.exe/OpenConsole.exe host process,
// the per-worker cost #2340 measured at 87 panes / 2,829 threads / 54k handles /
// 2 GB and #3405 showed scaling linearly with fleet size).
//
// Asserting both flags in ONE test is deliberate: the background spawn is the
// control. Without it a passing detached assertion could just mean the test host
// never allocates consoles, and the regression this guards against —
// silently reverting to CREATE_NO_WINDOW — would still read green.
func TestDetachedSpawnHasNoConsoleWhereBackgroundSpawnDoes(t *testing.T) {
	if os.Getenv(consoleProbeEnv) == "1" {
		report := os.Getenv(consoleProbeReportEnv)
		if report == "" {
			os.Exit(3)
		}
		if err := os.WriteFile(report, []byte(strconv.FormatUint(uint64(consoleAttachedProcessCount()), 10)), 0o644); err != nil {
			os.Exit(4)
		}
		os.Exit(0)
	}

	probe := func(configure func(*exec.Cmd)) uint32 {
		t.Helper()
		report := filepath.Join(t.TempDir(), "console-count")
		cmd := exec.Command(os.Args[0], "-test.run=^TestDetachedSpawnHasNoConsoleWhereBackgroundSpawnDoes$")
		cmd.Env = append(os.Environ(), consoleProbeEnv+"=1", consoleProbeReportEnv+"="+report)
		// Redirect to files, matching the dispatched worker's transcript binding: the
		// child has no use for a console because its output is already captured.
		sink, err := os.Create(filepath.Join(filepath.Dir(report), "sink"))
		if err != nil {
			t.Fatalf("create sink: %v", err)
		}
		defer sink.Close()
		cmd.Stdout, cmd.Stderr = sink, sink
		configure(cmd)
		if err := cmd.Run(); err != nil {
			t.Fatalf("probe child: %v", err)
		}
		b, err := os.ReadFile(report)
		if err != nil {
			t.Fatalf("probe child wrote no report: %v", err)
		}
		n, err := strconv.ParseUint(strings.TrimSpace(string(b)), 10, 32)
		if err != nil {
			t.Fatalf("probe report %q: %v", b, err)
		}
		return uint32(n)
	}

	// Control: CREATE_NO_WINDOW is invisible but still OWNS a console.
	if got := probe(ConfigureBackgroundCommand); got == 0 {
		t.Fatalf("background spawn reported %d console processes, want >0; "+
			"the detached assertion below would be vacuous on this host", got)
	}
	// Acceptance: DETACHED_PROCESS owns no console, so there is no host process to pay for.
	if got := probe(ConfigureDetachedCommand); got != 0 {
		t.Fatalf("detached spawn reported %d console processes, want 0 (#3597): "+
			"the child still owns a console and therefore a conhost/OpenConsole host process", got)
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
