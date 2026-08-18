//go:build windows

package procguard

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"sort"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

const (
	th32csSnapProcess                = 0x00000002
	processTerminate                 = 0x0001
	processVMRead                    = 0x0010
	processQueryLimitedInformation   = 0x1000
	processEntryExeFileMax           = 260
	stillActiveExitCode              = 259
	ntProcessBasicInformationClass   = 0
	maxRemoteCommandLineBytes        = 64 * 1024
	maxRemoteCommandLineDisplayBytes = 8192
)

var (
	kernel32Procguard                  = syscall.NewLazyDLL("kernel32.dll")
	ntdllProcguard                     = syscall.NewLazyDLL("ntdll.dll")
	procCreateToolhelp32SnapshotGuard  = kernel32Procguard.NewProc("CreateToolhelp32Snapshot")
	procProcess32FirstWGuard           = kernel32Procguard.NewProc("Process32FirstW")
	procProcess32NextWGuard            = kernel32Procguard.NewProc("Process32NextW")
	procGetProcessTimesGuard           = kernel32Procguard.NewProc("GetProcessTimes")
	procGetExitCodeProcessGuard        = kernel32Procguard.NewProc("GetExitCodeProcess")
	procTerminateProcessGuard          = kernel32Procguard.NewProc("TerminateProcess")
	procReadProcessMemoryGuard         = kernel32Procguard.NewProc("ReadProcessMemory")
	procNtQueryInformationProcessGuard = ntdllProcguard.NewProc("NtQueryInformationProcess")
)

type processEntry32WGuard struct {
	Size            uint32
	CntUsage        uint32
	ProcessID       uint32
	DefaultHeapID   uintptr
	ModuleID        uint32
	CntThreads      uint32
	ParentProcessID uint32
	PriClassBase    int32
	Flags           uint32
	ExeFile         [processEntryExeFileMax]uint16
}

type processBasicInformationGuard struct {
	Reserved1       uintptr
	PebBaseAddress  uintptr
	Reserved2       [2]uintptr
	UniqueProcessID uintptr
	Reserved3       uintptr
}

type unicodeStringGuard struct {
	Length        uint16
	MaximumLength uint16
	Buffer        uintptr
}

func collectWindowsRelationsNative() ([]Proc, bool, string) {
	rows, err := snapshotWindowsProcessesNative(true)
	if err != "" {
		return nil, false, err
	}
	return rows, true, ""
}

func snapshotWindowsProcessesNative(withCommandLine bool) ([]Proc, string) {
	h, _, err := procCreateToolhelp32SnapshotGuard.Call(th32csSnapProcess, 0)
	if h == uintptr(syscall.InvalidHandle) || h == 0 {
		return nil, fmt.Sprintf("CreateToolhelp32Snapshot: %v", err)
	}
	defer syscall.CloseHandle(syscall.Handle(h))

	var pe processEntry32WGuard
	pe.Size = uint32(unsafe.Sizeof(pe))
	ok, _, err := procProcess32FirstWGuard.Call(h, uintptr(unsafe.Pointer(&pe)))
	if ok == 0 {
		return nil, fmt.Sprintf("Process32FirstW: %v", err)
	}

	now := time.Now()
	out := []Proc{}
	for {
		name := stripExe(syscall.UTF16ToString(pe.ExeFile[:]))
		p := Proc{
			PID:     int(pe.ProcessID),
			Name:    name,
			PPID:    IntPtr(int(pe.ParentProcessID)),
			Threads: IntPtr(int(pe.CntThreads)),
		}
		if start, ok := processStartTime(p.PID); ok {
			p.Start = start.UTC().Format(time.RFC3339Nano)
			age := int(now.Sub(start).Seconds())
			if age >= 0 {
				p.AgeSec = IntPtr(age)
			}
		}
		if withCommandLine {
			p.Cmdline = processCommandLine(p.PID)
		}
		out = append(out, p)

		ok, _, _ = procProcess32NextWGuard.Call(h, uintptr(unsafe.Pointer(&pe)))
		if ok == 0 {
			break
		}
	}
	return out, ""
}

func processStartTime(pid int) (time.Time, bool) {
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		return time.Time{}, false
	}
	defer syscall.CloseHandle(h)

	var create, exit, kernel, user syscall.Filetime
	r, _, _ := procGetProcessTimesGuard.Call(
		uintptr(h),
		uintptr(unsafe.Pointer(&create)),
		uintptr(unsafe.Pointer(&exit)),
		uintptr(unsafe.Pointer(&kernel)),
		uintptr(unsafe.Pointer(&user)),
	)
	if r == 0 {
		return time.Time{}, false
	}
	return time.Unix(0, create.Nanoseconds()), true
}

func processCommandLine(pid int) string {
	if pid <= 0 {
		return ""
	}
	h, err := syscall.OpenProcess(processQueryLimitedInformation|processVMRead, false, uint32(pid))
	if err != nil || h == 0 {
		return ""
	}
	defer syscall.CloseHandle(h)

	var pbi processBasicInformationGuard
	status, _, _ := procNtQueryInformationProcessGuard.Call(
		uintptr(h),
		uintptr(ntProcessBasicInformationClass),
		uintptr(unsafe.Pointer(&pbi)),
		unsafe.Sizeof(pbi),
		0,
	)
	if status != 0 || pbi.PebBaseAddress == 0 {
		return ""
	}

	var params uintptr
	if !readRemoteValue(h, pbi.PebBaseAddress+pebProcessParametersOffset(), unsafe.Pointer(&params), unsafe.Sizeof(params)) || params == 0 {
		return ""
	}

	var us unicodeStringGuard
	if !readRemoteValue(h, params+rtlCommandLineOffset(), unsafe.Pointer(&us), unsafe.Sizeof(us)) {
		return ""
	}
	if us.Length == 0 || us.Buffer == 0 || int(us.Length) > maxRemoteCommandLineBytes {
		return ""
	}
	buf := make([]uint16, int(us.Length)/2)
	if len(buf) == 0 {
		return ""
	}
	if !readRemoteValue(h, us.Buffer, unsafe.Pointer(&buf[0]), uintptr(us.Length)) {
		return ""
	}
	cmd := syscall.UTF16ToString(buf)
	if len(cmd) > maxRemoteCommandLineDisplayBytes {
		return cmd[:maxRemoteCommandLineDisplayBytes]
	}
	return cmd
}

func readRemoteValue(h syscall.Handle, addr uintptr, dst unsafe.Pointer, size uintptr) bool {
	if addr == 0 || dst == nil || size == 0 {
		return false
	}
	var n uintptr
	r, _, _ := procReadProcessMemoryGuard.Call(
		uintptr(h),
		addr,
		uintptr(dst),
		size,
		uintptr(unsafe.Pointer(&n)),
	)
	return r != 0 && n == size
}

func pebProcessParametersOffset() uintptr {
	if unsafe.Sizeof(uintptr(0)) == 8 {
		return 0x20
	}
	return 0x10
}

func rtlCommandLineOffset() uintptr {
	if unsafe.Sizeof(uintptr(0)) == 8 {
		return 0x70
	}
	return 0x40
}

func killTreeWindowsNative(pid int) (bool, string, bool) {
	if pid <= 0 {
		return false, "invalid pid", true
	}
	rows, err := snapshotWindowsProcessesNative(false)
	if err != "" {
		return false, err, false
	}
	byPID := map[int]Proc{}
	children := map[int][]Proc{}
	for _, p := range rows {
		byPID[p.PID] = p
		if p.PPID != nil {
			children[*p.PPID] = append(children[*p.PPID], p)
		}
	}
	if _, ok := byPID[pid]; !ok {
		return true, "process already exited", true
	}

	order := killPostorder(pid, byPID, children)
	killed := 0
	failures := []string{}
	for _, target := range order {
		if target <= 0 || target == 4 {
			continue
		}
		ok, detail := terminatePIDNative(target)
		if ok {
			killed++
			continue
		}
		failures = append(failures, fmt.Sprintf("pid %d: %s", target, detail))
	}
	if len(failures) > 0 {
		return false, trimTo(strings.Join(failures, "; "), 200), true
	}
	return true, fmt.Sprintf("terminated %d process(es) via native process tree", killed), true
}

func killPostorder(root int, byPID map[int]Proc, children map[int][]Proc) []int {
	order := []int{}
	visiting := map[int]bool{}
	var walk func(int)
	walk = func(cur int) {
		if visiting[cur] {
			return
		}
		visiting[cur] = true
		kids := append([]Proc{}, children[cur]...)
		sort.Slice(kids, func(i, j int) bool { return kids[i].PID < kids[j].PID })
		parent := byPID[cur]
		for _, child := range kids {
			if child.PID == cur || childPredatesParent(child, parent) {
				continue
			}
			walk(child.PID)
		}
		order = append(order, cur)
	}
	walk(root)
	return order
}

func childPredatesParent(child, parent Proc) bool {
	if child.Start == "" || parent.Start == "" {
		return false
	}
	childStart, err1 := time.Parse(time.RFC3339Nano, child.Start)
	parentStart, err2 := time.Parse(time.RFC3339Nano, parent.Start)
	if err1 != nil || err2 != nil {
		return false
	}
	return childStart.Before(parentStart)
}

func terminatePIDNative(pid int) (bool, string) {
	h, err := syscall.OpenProcess(processTerminate|processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		if !pidAliveNative(pid) {
			return true, "process already exited"
		}
		return false, err.Error()
	}
	defer syscall.CloseHandle(h)

	r, _, callErr := procTerminateProcessGuard.Call(uintptr(h), 1)
	if r == 0 {
		return false, callErr.Error()
	}
	return true, "terminated"
}

func pidAliveNative(pid int) bool { return processalive.Check(pid) }
