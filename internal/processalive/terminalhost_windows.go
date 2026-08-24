//go:build windows

package processalive

import (
	"strings"
	"syscall"
	"unsafe"
)

// TerminalHostPID returns the Windows Terminal/OpenConsole ancestor that owns
// pid's interactive process tree. The bounded walk fails closed on a missing
// process, a cycle, or a process that is not terminal-hosted.
func TerminalHostPID(pid int) (int, bool) {
	parents, names, ok := processTree()
	return terminalHostPID(pid, parents, names, ok)
}

func terminalHostPID(pid int, parents map[int]int, names map[int]string, ok bool) (int, bool) {
	if pid <= 0 {
		return 0, false
	}
	if !ok {
		return 0, false
	}
	seen := make(map[int]bool, 16)
	for depth := 0; pid > 0 && depth < 64 && !seen[pid]; depth++ {
		seen[pid] = true
		switch strings.ToLower(names[pid]) {
		case "windowsterminal.exe", "openconsole.exe":
			return pid, true
		}
		pid = parents[pid]
	}
	return 0, false
}

func processTree() (map[int]int, map[int]string, bool) {
	h, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil, nil, false
	}
	defer syscall.CloseHandle(h)
	parents := make(map[int]int)
	names := make(map[int]string)
	var e syscall.ProcessEntry32
	e.Size = uint32(unsafe.Sizeof(e))
	if err := syscall.Process32First(h, &e); err != nil {
		return nil, nil, false
	}
	for {
		pid := int(e.ProcessID)
		parents[pid] = int(e.ParentProcessID)
		names[pid] = syscall.UTF16ToString(e.ExeFile[:])
		if err := syscall.Process32Next(h, &e); err != nil {
			break
		}
	}
	return parents, names, true
}
