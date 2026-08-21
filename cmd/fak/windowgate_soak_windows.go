//go:build windows

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

const desktopConsoleSoakSnapshotProcess = 0x00000002

type desktopConsoleSoakProcessEntry struct {
	Size              uint32
	Usage             uint32
	ProcessID         uint32
	DefaultHeapID     uintptr
	ModuleID          uint32
	Threads           uint32
	ParentProcessID   uint32
	PriorityClassBase int32
	Flags             uint32
	ExeFile           [syscall.MAX_PATH]uint16
}

var (
	desktopConsoleSoakCreateSnapshot = desktopConsoleDLL.NewProc("CreateToolhelp32Snapshot")
	desktopConsoleSoakProcessFirst   = desktopConsoleDLL.NewProc("Process32FirstW")
	desktopConsoleSoakProcessNext    = desktopConsoleDLL.NewProc("Process32NextW")
	desktopConsoleSoakCloseHandle    = desktopConsoleDLL.NewProc("CloseHandle")
)

func snapshotDesktopConsoleSoakProcesses() (desktopConsoleSoakProcessCounts, error) {
	exe, err := os.Executable()
	if err != nil {
		return desktopConsoleSoakProcessCounts{}, err
	}
	watched := map[string]string{
		strings.ToLower(filepath.Base(exe)): "codex-root",
		"pwsh.exe":                          "pwsh",
		"node.exe":                          "node",
		"fak-mcp.exe":                       "fak-mcp",
		"conhost.exe":                       "conhost",
		"openconsole.exe":                   "openconsole",
	}
	handle, _, callErr := desktopConsoleSoakCreateSnapshot.Call(desktopConsoleSoakSnapshotProcess, 0)
	if handle == ^uintptr(0) {
		return desktopConsoleSoakProcessCounts{}, fmt.Errorf("CreateToolhelp32Snapshot: %v", callErr)
	}
	defer desktopConsoleSoakCloseHandle.Call(handle)
	counts := desktopConsoleSoakProcessCounts{Families: map[string]int{}}
	var entry desktopConsoleSoakProcessEntry
	entry.Size = uint32(unsafe.Sizeof(entry))
	ok, _, firstErr := desktopConsoleSoakProcessFirst.Call(handle, uintptr(unsafe.Pointer(&entry)))
	if ok == 0 {
		return desktopConsoleSoakProcessCounts{}, fmt.Errorf("Process32FirstW: %v", firstErr)
	}
	for {
		name := strings.ToLower(syscall.UTF16ToString(entry.ExeFile[:]))
		if family, found := watched[name]; found {
			counts.Families[family]++
			counts.TrackedTotal++
		}
		next, _, _ := desktopConsoleSoakProcessNext.Call(handle, uintptr(unsafe.Pointer(&entry)))
		if next == 0 {
			break
		}
	}
	for _, family := range []string{"codex-root", "pwsh", "node", "fak-mcp", "conhost", "openconsole"} {
		if _, ok := counts.Families[family]; !ok {
			counts.Families[family] = 0
		}
	}
	return counts, nil
}

func waitDesktopConsoleSoakProcessBaseline(before desktopConsoleSoakProcessCounts, timeout time.Duration) (desktopConsoleSoakProcessCounts, map[string]int, error) {
	deadline := time.Now().Add(timeout)
	for {
		after, err := snapshotDesktopConsoleSoakProcesses()
		if err != nil {
			return desktopConsoleSoakProcessCounts{}, nil, err
		}
		increases := desktopConsoleSoakConsoleHostIncreases(before, after)
		if len(increases) == 0 || time.Now().After(deadline) {
			return after, increases, nil
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func desktopConsoleSoakConsoleHostIncreases(before, after desktopConsoleSoakProcessCounts) map[string]int {
	increases := map[string]int{}
	for _, family := range []string{"conhost", "openconsole"} {
		if delta := after.Families[family] - before.Families[family]; delta > 0 {
			increases[family] = delta
		}
	}
	if len(increases) == 0 {
		return nil
	}
	return increases
}

func waitDesktopConsoleSoakSurvivors(pids []int, timeout time.Duration) []int {
	deadline := time.Now().Add(timeout)
	for {
		alive := desktopConsoleSoakAlivePIDs(pids)
		if len(alive) == 0 || time.Now().After(deadline) {
			return alive
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func desktopConsoleSoakAlivePIDs(pids []int) []int {
	alive := make([]int, 0, len(pids))
	for _, pid := range pids {
		if pid > 0 && dispatchPIDAlive(pid) {
			alive = append(alive, pid)
		}
	}
	return alive
}
