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
	"syscall"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

var desktopConsoleDLL = syscall.NewLazyDLL("kernel32.dll")

func runDesktopConsoleSelfcheckChild(stdout, stderr io.Writer) (int, bool) {
	switch os.Getenv(desktopConsoleSelfcheckRoleEnv) {
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
	env[desktopConsoleSelfcheckRoleEnv] = "root"
	env[desktopConsoleSelfcheckDirEnv] = tmp
	env[desktopConsoleSelfcheckReleaseEnv] = release
	spawned, err := spawnDispatchIssueWorker(
		[]string{exe, "windowgate", "--selfcheck"}, env, tmp, tmp,
		8252, "cmd", "codex", "windowgate-selfcheck", []string{"cmd/fak/**"},
		dispatchtick.Account{}, nil, "", "", 0,
	)
	if err != nil {
		return failDesktopConsoleSelfcheck(stderr, err)
	}
	defer releaseDesktopConsoleSelfcheck(spawned.PID, release)

	labels := []string{"codex-root", "pwsh", "node", "fak-mcp"}
	var transcript string
	if !waitForDesktopConsoleSelfcheck(30*time.Second, func() bool {
		b, _ := os.ReadFile(spawned.Log)
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

	processes := parseDesktopConsoleSelfcheckProcesses(transcript)
	rootPID := spawned.PID
	seen := map[string]bool{}
	visible := 0
	shared := true
	for _, p := range processes {
		seen[p.Label] = true
		if p.WindowHandle != 0 {
			visible++
		}
		if !containsSelfcheckPID(p.ConsolePIDs, uint32(p.PID)) ||
			!containsSelfcheckPID(p.ConsolePIDs, uint32(rootPID)) {
			shared = false
		}
	}
	for _, label := range labels {
		if !seen[label] {
			shared = false
		}
	}
	ok := shared && visible == 0
	reason := "Codex plus pwsh/node/fak-mcp descendants share one hidden console; zero visible console windows"
	if !ok {
		reason = "a representative Codex descendant did not inherit the hidden console or gained a visible window"
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
