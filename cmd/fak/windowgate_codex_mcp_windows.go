//go:build windows

package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

var (
	codexMCPWindowUser32             = syscall.NewLazyDLL("user32.dll")
	codexMCPWindowEnumWindows        = codexMCPWindowUser32.NewProc("EnumWindows")
	codexMCPWindowIsWindowVisible    = codexMCPWindowUser32.NewProc("IsWindowVisible")
	codexMCPWindowGetWindowProcessID = codexMCPWindowUser32.NewProc("GetWindowThreadProcessId")
	codexMCPWindowGetClassName       = codexMCPWindowUser32.NewProc("GetClassNameW")
)

func captureCodexMCPWindowSelfcheck(config, server string, timeout time.Duration) (codexMCPWindowSelfcheckReport, error) {
	rep := codexMCPWindowSelfcheckReport{
		Schema: codexMCPWindowSelfcheckSchema, Applicable: true, Platform: "windows", Server: server,
	}
	if abs, err := filepath.Abs(config); err == nil {
		config = abs
	}
	command, args, _, err := resolveCodexMCPEntry(config, server)
	if err != nil {
		return rep, fmt.Errorf("resolve configured Codex MCP entry: %w", err)
	}
	command, err = exec.LookPath(command)
	if err != nil {
		return rep, fmt.Errorf("resolve configured Codex MCP executable: %w", err)
	}
	if abs, absErr := filepath.Abs(command); absErr == nil {
		command = abs
	}
	rep.Command = filepath.Base(command)
	if rep.CommandSHA256, err = sha256File(command); err != nil {
		return rep, fmt.Errorf("hash configured Codex MCP executable: %w", err)
	}
	rep.ConfigState = classifyCodexMCPStatus(config, server, timeout, false).State

	codexPID := codexMCPAncestorPID(os.Getpid())
	for _, route := range codexMCPWindowSelfcheckRoutes {
		rep.Routes = append(rep.Routes, captureCodexMCPWindowRoute(route, codexPID, command, args, timeout))
	}
	return rep, nil
}

func captureCodexMCPWindowRoute(route string, codexPID int, command string, args []string, timeout time.Duration) codexMCPWindowRouteReport {
	row := codexMCPWindowRouteReport{Route: route, CodexAncestorPID: codexPID, ParentPID: os.Getpid(), ExitCode: -1}
	beforeWindows := codexMCPVisibleConsoleWindows()
	cmd := exec.Command(command, args...)
	if route == "managed" {
		windowgate.ConfigureBackgroundCommand(cmd)
	}
	registry := filepath.Join(os.TempDir(), fmt.Sprintf("fak-windowgate-mcp-%s-%d.json", route, os.Getpid()))
	defer os.Remove(registry)
	cmd.Env = append(os.Environ(), sessionRegistryEnv+"="+registry)
	in, err := cmd.StdinPipe()
	if err != nil {
		row.Failure = "stdin_pipe"
		return row
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		row.Failure = "stdout_pipe"
		return row
	}
	errPipe, err := cmd.StderrPipe()
	if err != nil {
		row.Failure = "stderr_pipe"
		return row
	}
	if err := cmd.Start(); err != nil {
		row.Failure = "start"
		return row
	}
	row.PID = cmd.Process.Pid
	time.Sleep(75 * time.Millisecond)
	row.ObservedParent = codexMCPObservedParent(row.PID)
	row.ConsolePIDs = codexMCPCurrentConsolePIDs()
	row.ConsoleMember = containsSelfcheckPID(row.ConsolePIDs, uint32(row.PID))
	row.WindowHandle = codexMCPNewVisibleWindow(beforeWindows, row.PID)

	stderrDone := make(chan []byte, 1)
	go func() {
		b, _ := io.ReadAll(io.LimitReader(errPipe, 4096))
		stderrDone <- b
	}()
	request := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"fak-windowgate","version":"1"}}}` + "\n"
	if _, err := io.WriteString(in, request); err != nil {
		row.Failure = "initialize_write"
		_ = in.Close()
		return finishCodexMCPWindowRoute(cmd, row, timeout)
	}
	type lineResult struct {
		line string
		err  error
	}
	response := make(chan lineResult, 1)
	go func() {
		line, readErr := bufio.NewReader(out).ReadString('\n')
		response <- lineResult{line: line, err: readErr}
	}()
	select {
	case got := <-response:
		var rpc struct {
			JSONRPC string          `json:"jsonrpc"`
			ID      json.RawMessage `json:"id"`
			Result  json.RawMessage `json:"result"`
			Error   json.RawMessage `json:"error"`
		}
		if got.err != nil && strings.TrimSpace(got.line) == "" {
			row.Failure = "initialize_read"
		} else if json.Unmarshal([]byte(strings.TrimSpace(got.line)), &rpc) != nil || rpc.JSONRPC != "2.0" || len(rpc.ID) == 0 || (len(rpc.Error) > 0 && string(rpc.Error) != "null") {
			row.Failure = "initialize_response"
		} else {
			row.InitializeOK = true
			_, _ = io.WriteString(in, `{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}`+"\n")
		}
	case <-time.After(timeout):
		row.Failure = "initialize_timeout"
	}
	_ = in.Close()
	row = finishCodexMCPWindowRoute(cmd, row, timeout)
	select {
	case <-stderrDone:
	default:
	}
	return row
}

func finishCodexMCPWindowRoute(cmd *exec.Cmd, row codexMCPWindowRouteReport, timeout time.Duration) codexMCPWindowRouteReport {
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var waitErr error
	select {
	case waitErr = <-done:
	case <-time.After(timeout):
		row.Failure = "exit_timeout"
		_ = cmd.Process.Kill()
		waitErr = <-done
	}
	if cmd.ProcessState != nil {
		row.ExitCode = cmd.ProcessState.ExitCode()
	}
	if waitErr == nil && row.ExitCode == 0 {
		row.ExitState = "clean"
	} else {
		row.ExitState = "failed"
		if row.Failure == "" {
			var exitErr *exec.ExitError
			if errors.As(waitErr, &exitErr) {
				row.Failure = "nonzero_exit"
			} else {
				row.Failure = "wait"
			}
		}
	}
	return row
}

func codexMCPCurrentConsolePIDs() []uint32 {
	buf := make([]uint32, 64)
	n, _, _ := desktopConsoleDLL.NewProc("GetConsoleProcessList").Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	if n > uintptr(len(buf)) {
		n = uintptr(len(buf))
	}
	out := append([]uint32(nil), buf[:n]...)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

type codexMCPProcessRelation struct {
	parent int
	name   string
}

func codexMCPProcessRelations() map[int]codexMCPProcessRelation {
	handle, _, _ := desktopConsoleSoakCreateSnapshot.Call(desktopConsoleSoakSnapshotProcess, 0)
	if handle == ^uintptr(0) {
		return nil
	}
	defer desktopConsoleSoakCloseHandle.Call(handle)
	relations := map[int]codexMCPProcessRelation{}
	var entry desktopConsoleSoakProcessEntry
	entry.Size = uint32(unsafe.Sizeof(entry))
	ok, _, _ := desktopConsoleSoakProcessFirst.Call(handle, uintptr(unsafe.Pointer(&entry)))
	for ok != 0 {
		relations[int(entry.ProcessID)] = codexMCPProcessRelation{
			parent: int(entry.ParentProcessID),
			name:   strings.ToLower(syscall.UTF16ToString(entry.ExeFile[:])),
		}
		ok, _, _ = desktopConsoleSoakProcessNext.Call(handle, uintptr(unsafe.Pointer(&entry)))
	}
	return relations
}

func codexMCPObservedParent(pid int) int {
	return codexMCPProcessRelations()[pid].parent
}

func codexMCPAncestorPID(pid int) int {
	relations := codexMCPProcessRelations()
	seen := map[int]bool{}
	for i := 0; pid > 0 && i < 64 && !seen[pid]; i++ {
		seen[pid] = true
		relation, ok := relations[pid]
		if !ok {
			return 0
		}
		if strings.HasPrefix(relation.name, "codex") && strings.HasSuffix(relation.name, ".exe") {
			return pid
		}
		pid = relation.parent
	}
	return 0
}

var (
	codexMCPWindowMu        sync.Mutex
	codexMCPVisibleOut      map[uintptr]bool
	codexMCPVisibleCallback = syscall.NewCallback(codexMCPEnumVisibleCallback)
	codexMCPDirectTargetPID int
	codexMCPDirectFoundHWND uintptr
	codexMCPDirectCallback  = syscall.NewCallback(codexMCPEnumDirectCallback)
)

func codexMCPEnumVisibleCallback(hwnd, _ uintptr) uintptr {
	visible, _, _ := codexMCPWindowIsWindowVisible.Call(hwnd)
	if visible == 0 {
		return 1
	}
	name := make([]uint16, 128)
	n, _, _ := codexMCPWindowGetClassName.Call(hwnd, uintptr(unsafe.Pointer(&name[0])), uintptr(len(name)))
	class := strings.ToLower(syscall.UTF16ToString(name[:n]))
	if strings.Contains(class, "console") || strings.Contains(class, "cascadia") {
		if codexMCPVisibleOut != nil {
			codexMCPVisibleOut[hwnd] = true
		}
	}
	return 1
}

func codexMCPEnumDirectCallback(hwnd, _ uintptr) uintptr {
	visible, _, _ := codexMCPWindowIsWindowVisible.Call(hwnd)
	if visible == 0 {
		return 1
	}
	var owner uint32
	codexMCPWindowGetWindowProcessID.Call(hwnd, uintptr(unsafe.Pointer(&owner)))
	if int(owner) == codexMCPDirectTargetPID {
		codexMCPDirectFoundHWND = hwnd
		return 0
	}
	return 1
}

func codexMCPVisibleConsoleWindows() map[uintptr]bool {
	codexMCPWindowMu.Lock()
	defer codexMCPWindowMu.Unlock()
	out := map[uintptr]bool{}
	codexMCPVisibleOut = out
	codexMCPWindowEnumWindows.Call(codexMCPVisibleCallback, 0)
	codexMCPVisibleOut = nil
	return out
}

func codexMCPNewVisibleWindow(before map[uintptr]bool, pid int) uintptr {
	codexMCPWindowMu.Lock()
	codexMCPDirectTargetPID = pid
	codexMCPDirectFoundHWND = 0
	codexMCPWindowEnumWindows.Call(codexMCPDirectCallback, 0)
	direct := codexMCPDirectFoundHWND
	codexMCPWindowMu.Unlock()
	if direct != 0 {
		return direct
	}
	after := codexMCPVisibleConsoleWindows()
	handles := make([]uintptr, 0, len(after))
	for hwnd := range after {
		if !before[hwnd] {
			handles = append(handles, hwnd)
		}
	}
	sort.Slice(handles, func(i, j int) bool { return handles[i] < handles[j] })
	if len(handles) > 0 {
		return handles[0]
	}
	return 0
}
