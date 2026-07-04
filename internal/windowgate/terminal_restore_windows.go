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
	procShowWindowAsync          = user32.NewProc("ShowWindowAsync")
	procSetForegroundWindow      = user32.NewProc("SetForegroundWindow")
	procCreateToolhelp32Snapshot = kernel32.NewProc("CreateToolhelp32Snapshot")
	procProcess32First           = kernel32.NewProc("Process32FirstW")
	procProcess32Next            = kernel32.NewProc("Process32NextW")
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
	for _, pid := range ancestorPIDs(uint32(os.Getpid())) {
		hwnd := firstVisibleWindowForPID(pid)
		if hwnd == 0 {
			continue
		}
		procShowWindowAsync.Call(hwnd, swRestore)
		procSetForegroundWindow.Call(hwnd)
		return true
	}
	return false
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
	go func() {
		deadline := time.NewTimer(duration)
		defer deadline.Stop()
		tick := time.NewTicker(interval)
		defer tick.Stop()
		RestoreTerminalWindow()
		for {
			select {
			case <-deadline.C:
				return
			case <-tick.C:
				RestoreTerminalWindow()
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
