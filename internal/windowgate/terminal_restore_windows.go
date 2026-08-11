//go:build windows

package windowgate

import (
	"os"
	"syscall"
	"time"
	"unsafe"
)

const (
	swRestore       = 9
	th32csSnapProc  = 0x00000002
	maxAncestorScan = 64
)

var (
	user32                       = syscall.NewLazyDLL("user32.dll")
	kernel32                     = syscall.NewLazyDLL("kernel32.dll")
	procEnumWindows              = user32.NewProc("EnumWindows")
	procGetWindowThreadProcessID = user32.NewProc("GetWindowThreadProcessId")
	procIsWindowVisible          = user32.NewProc("IsWindowVisible")
	procIsIconic                 = user32.NewProc("IsIconic")
	procShowWindowAsync          = user32.NewProc("ShowWindowAsync")
	procSetForegroundWindow      = user32.NewProc("SetForegroundWindow")
	procGetForegroundWindow      = user32.NewProc("GetForegroundWindow")
	procGetConsoleWindow         = kernel32.NewProc("GetConsoleWindow")
	procCreateToolhelp32Snapshot = kernel32.NewProc("CreateToolhelp32Snapshot")
	procProcess32First           = kernel32.NewProc("Process32FirstW")
	procProcess32Next            = kernel32.NewProc("Process32NextW")
)

var (
	resolveTerminalWindow          = terminalWindow
	restoreResolvedTerminalWindow  = restoreTerminalWindow
	isResolvedTerminalWindowIconic = isTerminalWindowIconic
)

type processEntry32 struct {
	Size            uint32
	CntUsage        uint32
	ProcessID       uint32
	DefaultHeapID   uintptr
	ModuleID        uint32
	CntThreads      uint32
	ParentProcessID uint32
	PriClassBase    int32
	Flags           uint32
	ExeFile         [260]uint16
}

// RestoreTerminalWindow restores and foregrounds the visible top-level window owned by the
// current process's ancestor chain. This mirrors the attended PowerShell Codex wrapper's
// focus repair, but works when fak execs Codex directly and therefore bypasses functions in
// the user's profile.
func RestoreTerminalWindow() bool {
	return restoreTerminalWindow(terminalWindow())
}

// terminalWindow resolves the attended terminal once. Classic conhost windows are
// normally owned by the current process or one of its ancestors. Windows Terminal's
// visible HWND is different: the console client talks through ConPTY and the HWND is
// owned by WindowsTerminal.exe, which is not in that ancestry. In that case the
// foreground HWND at launch is the only stable attended-window identity available.
func terminalWindow() uintptr {
	for _, pid := range ancestorPIDs(uint32(os.Getpid())) {
		if hwnd := firstVisibleWindowForPID(pid); hwnd != 0 {
			return hwnd
		}
	}
	if hwnd, _, _ := procGetConsoleWindow.Call(); hwnd != 0 {
		if visible, _, _ := procIsWindowVisible.Call(hwnd); visible != 0 {
			return hwnd
		}
	}
	hwnd, _, _ := procGetForegroundWindow.Call()
	return hwnd
}

func isTerminalWindowIconic(hwnd uintptr) bool {
	if hwnd == 0 {
		return false
	}
	iconic, _, _ := procIsIconic.Call(hwnd)
	return iconic != 0
}

func restoreTerminalWindow(hwnd uintptr) bool {
	if hwnd == 0 {
		return false
	}
	procShowWindowAsync.Call(hwnd, swRestore)
	procSetForegroundWindow.Call(hwnd)
	return true
}

// StartTerminalRestorePulse repeatedly restores the owning terminal for a short startup
// window. A single restore can miss late plugin-load focus/minimize races.
func StartTerminalRestorePulse(duration, interval time.Duration) {
	if duration <= 0 {
		return
	}
	if interval <= 0 {
		interval = 500 * time.Millisecond
	}
	// Pin the HWND before the child can steal focus or minimize the terminal. Re-resolving
	// on every tick would select whichever unrelated window became foreground after the
	// terminal disappeared, making the pulse permanently miss its intended target.
	hwnd := resolveTerminalWindow()
	go func() {
		deadline := time.NewTimer(duration)
		defer deadline.Stop()
		tick := time.NewTicker(interval)
		defer tick.Stop()
		restoreIfMinimized := func() {
			// Repair only an actual minimize race. Re-foregrounding an already visible
			// terminal steals focus back from a window the operator deliberately chose
			// during this startup window, producing a visible focus bounce.
			if isResolvedTerminalWindowIconic(hwnd) {
				restoreResolvedTerminalWindow(hwnd)
			}
		}
		restoreIfMinimized()
		for {
			select {
			case <-deadline.C:
				return
			case <-tick.C:
				restoreIfMinimized()
			}
		}
	}()
}

func ancestorPIDs(pid uint32) []uint32 {
	parents := parentPIDMap()
	if len(parents) == 0 || pid == 0 {
		return nil
	}
	out := make([]uint32, 0, 8)
	seen := map[uint32]bool{}
	for i := 0; pid != 0 && i < maxAncestorScan && !seen[pid]; i++ {
		seen[pid] = true
		out = append(out, pid)
		pid = parents[pid]
	}
	return out
}

func parentPIDMap() map[uint32]uint32 {
	h, _, _ := procCreateToolhelp32Snapshot.Call(th32csSnapProc, 0)
	if h == 0 || h == uintptr(syscall.InvalidHandle) {
		return nil
	}
	defer syscall.CloseHandle(syscall.Handle(h))
	var pe processEntry32
	pe.Size = uint32(unsafe.Sizeof(pe))
	ok, _, _ := procProcess32First.Call(h, uintptr(unsafe.Pointer(&pe)))
	if ok == 0 {
		return nil
	}
	parents := make(map[uint32]uint32)
	for {
		parents[pe.ProcessID] = pe.ParentProcessID
		ok, _, _ = procProcess32Next.Call(h, uintptr(unsafe.Pointer(&pe)))
		if ok == 0 {
			break
		}
	}
	return parents
}

func firstVisibleWindowForPID(pid uint32) uintptr {
	if pid == 0 {
		return 0
	}
	var found uintptr
	cb := syscall.NewCallback(func(hwnd, lparam uintptr) uintptr {
		_ = lparam
		var windowPID uint32
		procGetWindowThreadProcessID.Call(hwnd, uintptr(unsafe.Pointer(&windowPID)))
		if windowPID == pid {
			visible, _, _ := procIsWindowVisible.Call(hwnd)
			if visible != 0 {
				found = hwnd
				return 0
			}
		}
		return 1
	})
	procEnumWindows.Call(cb, 0)
	return found
}
