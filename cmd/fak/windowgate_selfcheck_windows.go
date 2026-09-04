//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

var desktopConsoleDLL = syscall.NewLazyDLL("kernel32.dll")

func runDesktopConsoleSelfcheckChild(stdout, stderr io.Writer) (int, bool) {
	switch os.Getenv(desktopConsoleSelfcheckRoleEnv) {
	case "controller":
		return runDesktopConsoleSelfcheckController(stdout, stderr), true
	case "root":
		return runDesktopConsoleSelfcheckRoot(stdout, stderr), true
	case "child":
		return runDesktopConsoleSelfcheckDescendant(stderr), true
	default:
		return 0, false
	}
}

func runDesktopConsoleSelfcheck(stdout, stderr io.Writer, asJSON bool) int {
	tmp, err := os.MkdirTemp("", "fak-windowgate-selfcheck-")
	if err != nil {
		return failDesktopConsoleSelfcheck(stderr, err)
	}
	defer os.RemoveAll(tmp)
	release := filepath.Join(tmp, "release")
	exe, err := os.Executable()
	if err != nil {
		return failDesktopConsoleSelfcheck(stderr, err)
	}
	env := envMap(os.Environ())
	env[desktopConsoleSelfcheckRoleEnv] = "controller"
	env[desktopConsoleSelfcheckDirEnv] = tmp
	env[desktopConsoleSelfcheckReleaseEnv] = release
	logPath := filepath.Join(tmp, "controller.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return failDesktopConsoleSelfcheck(stderr, err)
	}
	controller := exec.Command(exe, "windowgate", "--selfcheck")
	controller.Env = envSliceFromMap(env)
	controller.Dir = tmp
	controller.Stdout, controller.Stderr = logFile, logFile
	// The outer controller has no console but keeps inherited file handles. This
	// makes an unconfigured attended console root allocate a visible HWND, so the
	// witness deterministically fails on the pre-#8853 branch.
	windowgate.ConfigureDetachedCommand(controller)

	beforeWindows := codexMCPVisibleConsoleWindows()
	var visibleMu sync.Mutex
	visibleHandles := map[uintptr]bool{}
	stopCapture := make(chan struct{})
	captureDone := make(chan struct{})
	go func() {
		defer close(captureDone)
		ticker := time.NewTicker(15 * time.Millisecond)
		defer ticker.Stop()
		capture := func() {
			visibleMu.Lock()
			defer visibleMu.Unlock()
			for hwnd := range codexMCPVisibleConsoleWindows() {
				if !beforeWindows[hwnd] {
					visibleHandles[hwnd] = true
				}
			}
		}
		for {
			select {
			case <-ticker.C:
				capture()
			case <-stopCapture:
				capture()
				return
			}
		}
	}()
	captureStopped := false
	stopWindowCapture := func() {
		if captureStopped {
			return
		}
		captureStopped = true
		close(stopCapture)
		<-captureDone
	}
	defer stopWindowCapture()

	if err := controller.Start(); err != nil {
		_ = logFile.Close()
		return failDesktopConsoleSelfcheck(stderr, err)
	}
	_ = logFile.Close()
	defer releaseDesktopConsoleSelfcheck(controller.Process.Pid, release)

	labels := []string{"codex-root", "pwsh", "node", "fak-mcp"}
	var transcript string
	if !waitForDesktopConsoleSelfcheck(30*time.Second, func() bool {
		b, _ := os.ReadFile(logPath)
		transcript = string(b)
		for _, label := range labels {
			if !strings.Contains(transcript, `"label":"`+label+`"`) {
				return false
			}
		}
		return true
	}) {
		fmt.Fprintf(stderr, "fak windowgate --selfcheck: timed out waiting for descendant reports\n%s", transcript)
		return 1
	}
	stopWindowCapture()

	processes := parseDesktopConsoleSelfcheckProcesses(transcript)
	rootPID := 0
	seen := map[string]bool{}
	shared := true
	for _, p := range processes {
		seen[p.Label] = true
		if p.Label == "codex-root" {
			rootPID = p.PID
		}
		if p.WindowHandle != 0 {
			visibleHandles[p.WindowHandle] = true
		}
	}
	if rootPID == 0 {
		shared = false
	}
	for _, label := range labels {
		if !seen[label] {
			shared = false
		}
	}
	for _, p := range processes {
		if !containsSelfcheckPID(p.ConsolePIDs, uint32(p.PID)) ||
			!containsSelfcheckPID(p.ConsolePIDs, uint32(rootPID)) {
			shared = false
		}
	}
	visibleMu.Lock()
	visible := len(visibleHandles)
	visibleMu.Unlock()
	ok := shared && visible == 0
	reason := "attended managed Codex plus pwsh/node/fak-mcp descendants share one hidden console; 15ms capture saw zero visible console windows"
	if !ok {
		reason = "the attended managed Codex root or a representative descendant did not share one hidden console without a visible HWND"
	}
	rep := desktopConsoleSelfcheckReport{
		Schema: desktopConsoleSelfcheckSchema, OK: ok, Applicable: true, Platform: runtime.GOOS,
		Backend: "codex", RootPID: rootPID, VisibleWindows: visible, Processes: processes, Reason: reason,
	}
	if shared {
		rep.SharedHiddenConsoles = 1
	}
	if err := writeDesktopConsoleSelfcheck(stdout, rep, asJSON); err != nil {
		return failDesktopConsoleSelfcheck(stderr, err)
	}
	if !ok {
		return 1
	}
	return 0
}

// runDesktopConsoleSelfcheckController launches the Codex-like root through the
// real attended guard-child seam. Its own DETACHED_PROCESS parent still supplies
// stdout/stderr files, matching an attended harness with usable inherited handles
// but no console available for ordinary console descendants to inherit.
func runDesktopConsoleSelfcheckController(stdout, stderr io.Writer) int {
	dir := os.Getenv(desktopConsoleSelfcheckDirEnv)
	release := os.Getenv(desktopConsoleSelfcheckReleaseEnv)
	if dir == "" || release == "" {
		fmt.Fprintln(stderr, "windowgate selfcheck controller is missing its private paths")
		return 2
	}
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	codexPath := filepath.Join(dir, "codex.exe")
	if err := os.Link(exe, codexPath); err != nil {
		fmt.Fprintf(stderr, "link codex selfcheck: %v\n", err)
		return 2
	}
	command := []string{codexPath, "exec", "windowgate", "--selfcheck"}
	meta := guardChildSpawnMetadata{
		AgentRunID:   "windowgate-selfcheck",
		ToolCallID:   "guard-child:windowgate-selfcheck",
		PolicyDigest: "sha256:windowgate-selfcheck",
		Backend:      "codex",
		Envelope: toolprocgate.CapabilityEnvelope{
			Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn},
		},
		RegistryPath: filepath.Join(dir, "guard-child-registry.jsonl"),
		LaunchPlan:   newGuardLaunchPlan(command),
	}
	_, child, err := launchGuardChildWithBroker(
		command, nil, false, meta, toolprocgate.NewSpawnBroker(), nil,
		[2]string{desktopConsoleSelfcheckRoleEnv, "root"},
		[2]string{desktopConsoleSelfcheckDirEnv, dir},
		[2]string{desktopConsoleSelfcheckReleaseEnv, release},
		// The worker may itself be running beneath a registered agent. This
		// self-contained witness uses its private registry, so clear ambient
		// lineage rather than naming a parent absent from that store.
		[2]string{"FAK_REGISTRATION_ID", ""},
		[2]string{"FAK_PARENT_REGISTRATION_ID", ""},
		[2]string{"FAK_ATTEMPT_ID", ""},
		[2]string{"FAK_PARENT_ATTEMPT_ID", ""},
		[2]string{"FAK_ROOT_REGISTRATION_ID", ""},
	)
	if err != nil {
		fmt.Fprintf(stderr, "prepare attended Codex selfcheck root: %v\n", err)
		return 2
	}
	job, err := windowgate.StartInNewJob(child)
	if err != nil {
		fmt.Fprintf(stderr, "start attended Codex selfcheck root: %v\n", err)
		return 2
	}
	waitErr := child.Wait()
	_ = job.Close()
	if waitErr != nil {
		fmt.Fprintf(stderr, "wait attended Codex selfcheck root: %v\n", waitErr)
		return 2
	}
	return 0
}

func runDesktopConsoleSelfcheckRoot(stdout, stderr io.Writer) int {
	dir := os.Getenv(desktopConsoleSelfcheckDirEnv)
	release := os.Getenv(desktopConsoleSelfcheckReleaseEnv)
	if dir == "" || release == "" {
		fmt.Fprintln(stderr, "windowgate selfcheck root is missing its private paths")
		return 2
	}
	if err := emitDesktopConsoleSelfcheckProcess(stdout, "codex-root"); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	type child struct {
		cmd    *exec.Cmd
		report string
	}
	children := make([]child, 0, 3)
	for _, label := range []string{"pwsh", "node", "fak-mcp"} {
		path := filepath.Join(dir, label+".exe")
		if err := os.Link(exe, path); err != nil {
			fmt.Fprintf(stderr, "link %s selfcheck: %v\n", label, err)
			return 2
		}
		reportPath := filepath.Join(dir, label+".json")
		childEnv := envMap(os.Environ())
		childEnv[desktopConsoleSelfcheckRoleEnv] = "child"
		childEnv[desktopConsoleSelfcheckLabelEnv] = label
		childEnv[desktopConsoleSelfcheckDirEnv] = dir
		cmd := exec.Command(path, "windowgate", "--selfcheck")
		cmd.Env = envSliceFromMap(childEnv)
		cmd.Stdout, cmd.Stderr = stdout, stderr
		if err := cmd.Start(); err != nil {
			fmt.Fprintf(stderr, "start %s selfcheck: %v\n", label, err)
			return 2
		}
		children = append(children, child{cmd: cmd, report: reportPath})
	}
	for _, c := range children {
		if !waitForDesktopConsoleSelfcheck(15*time.Second, func() bool {
			_, err := os.Stat(c.report)
			return err == nil
		}) {
			fmt.Fprintf(stderr, "timed out waiting for %s\n", c.report)
			return 2
		}
		b, err := os.ReadFile(c.report)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		_, _ = stdout.Write(append(b, '\n'))
	}
	waitForDesktopConsoleSelfcheck(45*time.Second, func() bool {
		_, err := os.Stat(release)
		return err == nil
	})
	for _, c := range children {
		_ = c.cmd.Wait()
	}
	return 0
}

func runDesktopConsoleSelfcheckDescendant(stderr io.Writer) int {
	dir := os.Getenv(desktopConsoleSelfcheckDirEnv)
	label := os.Getenv(desktopConsoleSelfcheckLabelEnv)
	if dir == "" || label == "" {
		fmt.Fprintln(stderr, "windowgate selfcheck descendant is missing its label or report directory")
		return 2
	}
	path := filepath.Join(dir, label+".json")
	row := captureDesktopConsoleSelfcheckProcess(label)
	b, err := json.Marshal(row)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	release := os.Getenv(desktopConsoleSelfcheckReleaseEnv)
	waitForDesktopConsoleSelfcheck(45*time.Second, func() bool {
		_, err := os.Stat(release)
		return err == nil
	})
	return 0
}

func emitDesktopConsoleSelfcheckProcess(w io.Writer, label string) error {
	return json.NewEncoder(w).Encode(captureDesktopConsoleSelfcheckProcess(label))
}

func captureDesktopConsoleSelfcheckProcess(label string) desktopConsoleSelfcheckProcess {
	buf := make([]uint32, 64)
	n, _, _ := desktopConsoleDLL.NewProc("GetConsoleProcessList").Call(
		uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	if n > uintptr(len(buf)) {
		n = uintptr(len(buf))
	}
	pids := append([]uint32(nil), buf[:n]...)
	sort.Slice(pids, func(i, j int) bool { return pids[i] < pids[j] })
	hwnd, _, _ := desktopConsoleDLL.NewProc("GetConsoleWindow").Call()
	return desktopConsoleSelfcheckProcess{Label: label, PID: os.Getpid(), ConsolePIDs: pids, WindowHandle: hwnd}
}

func parseDesktopConsoleSelfcheckProcesses(transcript string) []desktopConsoleSelfcheckProcess {
	var out []desktopConsoleSelfcheckProcess
	for _, line := range strings.Split(transcript, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var row desktopConsoleSelfcheckProcess
		if json.Unmarshal([]byte(line), &row) == nil && row.Label != "" {
			out = append(out, row)
		}
	}
	return out
}

func containsSelfcheckPID(pids []uint32, want uint32) bool {
	for _, pid := range pids {
		if pid == want {
			return true
		}
	}
	return false
}

func releaseDesktopConsoleSelfcheck(pid int, release string) {
	_ = os.WriteFile(release, []byte("release"), 0o600)
	if waitForDesktopConsoleSelfcheck(10*time.Second, func() bool { return !dispatchPIDAlive(pid) }) {
		return
	}
	if p, err := os.FindProcess(pid); err == nil {
		_ = p.Kill()
		_, _ = p.Wait()
	}
}

func waitForDesktopConsoleSelfcheck(timeout time.Duration, ok func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if ok() {
			return true
		}
		time.Sleep(25 * time.Millisecond)
	}
	return ok()
}
