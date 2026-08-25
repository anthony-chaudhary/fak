//go:build windows

package procguard

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	psapiCommit              = syscall.NewLazyDLL("psapi.dll")
	procGetPerformanceInfo   = psapiCommit.NewProc("GetPerformanceInfo")
	procGetProcessMemoryInfo = psapiCommit.NewProc("GetProcessMemoryInfo")
)

type performanceInformation struct {
	CB, CommitTotal, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable uintptr
	SystemCache, KernelTotal, KernelPaged, KernelNonpaged, PageSize            uintptr
	HandleCount, ProcessCount, ThreadCount                                     uint32
}

type processMemoryCountersCommit struct {
	CB                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
}

func collectCommitSnapshot(rootPID int) (CommitSnapshot, bool, string) {
	rows, err := snapshotWindowsProcessesNative(true)
	if err != "" {
		return CommitSnapshot{RootPID: rootPID}, true, err
	}
	children := make(map[int][]int)
	byPID := make(map[int]Proc, len(rows))
	for _, row := range rows {
		byPID[row.PID] = row
		if row.PPID != nil {
			children[*row.PPID] = append(children[*row.PPID], row.PID)
		}
	}
	owned := map[int]bool{}
	queue := []int{rootPID}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if owned[pid] {
			continue
		}
		owned[pid] = true
		queue = append(queue, children[pid]...)
	}
	s := CommitSnapshot{RootPID: rootPID}
	for pid := range owned {
		h, openErr := syscall.OpenProcess(processQueryLimitedInformation|processVMRead, false, uint32(pid))
		if openErr != nil {
			return s, true, fmt.Sprintf("OpenProcess pid %d: %v", pid, openErr)
		}
		var counters processMemoryCountersCommit
		counters.CB = uint32(unsafe.Sizeof(counters))
		r, _, _ := procGetProcessMemoryInfo.Call(uintptr(h), uintptr(unsafe.Pointer(&counters)), uintptr(counters.CB))
		_ = syscall.CloseHandle(h)
		if r == 0 {
			return s, true, fmt.Sprintf("GetProcessMemoryInfo pid %d failed", pid)
		}
		row := byPID[pid]
		ppid := 0
		if row.PPID != nil {
			ppid = *row.PPID
		}
		commit := uint64(counters.PagefileUsage)
		s.TreeCommitBytes += commit
		s.Processes = append(s.Processes, CommitProcess{PID: pid, PPID: ppid, Name: row.Name, CommandLine: row.Cmdline, CommitBytes: commit})
	}
	var perf performanceInformation
	perf.CB = unsafe.Sizeof(perf)
	r, _, callErr := procGetPerformanceInfo.Call(uintptr(unsafe.Pointer(&perf)), perf.CB)
	if r == 0 {
		return s, true, fmt.Sprintf("GetPerformanceInfo: %v", callErr)
	}
	s.SystemCommitBytes = uint64(perf.CommitTotal) * uint64(perf.PageSize)
	s.SystemCommitLimit = uint64(perf.CommitLimit) * uint64(perf.PageSize)
	return s, true, ""
}
