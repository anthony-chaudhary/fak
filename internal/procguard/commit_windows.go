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

func collectMemorySnapshot(rootPID int) (MemorySnapshot, bool, string) {
	rows, err := snapshotWindowsProcessesNative(true)
	if err != "" {
		return MemorySnapshot{Metric: MemoryMetricCommit, RootPID: rootPID}, true, err
	}
	children := make(map[int][]Proc)
	byPID := make(map[int]Proc, len(rows))
	for _, row := range rows {
		byPID[row.PID] = row
		if row.PPID != nil {
			children[*row.PPID] = append(children[*row.PPID], row)
		}
	}
	owned := windowsCommitOwnedPIDs(rootPID, byPID, children)
	s := MemorySnapshot{Metric: MemoryMetricCommit, RootPID: rootPID}
	for pid := range owned {
		h, openErr := syscall.OpenProcess(processQueryLimitedInformation|processVMRead, false, uint32(pid))
		if openErr != nil {
			if processExitedDuringSnapshot(openErr) {
				continue
			}
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
		s.TreeBytes += commit
		s.Processes = append(s.Processes, MemoryProcess{PID: pid, PPID: ppid, Name: row.Name, CommandLine: row.Cmdline, Bytes: commit})
	}
	var perf performanceInformation
	perf.CB = unsafe.Sizeof(perf)
	r, _, callErr := procGetPerformanceInfo.Call(uintptr(unsafe.Pointer(&perf)), perf.CB)
	if r == 0 {
		return s, true, fmt.Sprintf("GetPerformanceInfo: %v", callErr)
	}
	s.SystemBytes = uint64(perf.CommitTotal) * uint64(perf.PageSize)
	s.SystemLimit = uint64(perf.CommitLimit) * uint64(perf.PageSize)
	return s, true, ""
}

// windowsCommitOwnedPIDs rejects PID-reuse edges before memory accounting.
// A process that started before its reported parent cannot be that parent's child.
func windowsCommitOwnedPIDs(rootPID int, byPID map[int]Proc, children map[int][]Proc) map[int]bool {
	owned := map[int]bool{}
	queue := []int{rootPID}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if owned[pid] {
			continue
		}
		owned[pid] = true
		parent := byPID[pid]
		for _, child := range children[pid] {
			if child.PID == pid || childPredatesParent(child, parent) {
				continue
			}
			queue = append(queue, child.PID)
		}
	}
	return owned
}

func hostPhysicalMemoryBytes() (uint64, string) {
	var perf performanceInformation
	perf.CB = unsafe.Sizeof(perf)
	r, _, callErr := procGetPerformanceInfo.Call(uintptr(unsafe.Pointer(&perf)), perf.CB)
	if r == 0 {
		return 0, fmt.Sprintf("GetPerformanceInfo: %v", callErr)
	}
	return uint64(perf.PhysicalTotal) * uint64(perf.PageSize), ""
}

// processExitedDuringSnapshot recognizes the Windows result for a PID that vanished
// after enumeration. Other errors remain fatal so the resource guard never mistakes
// denied or broken telemetry for a safe process tree.
func processExitedDuringSnapshot(err error) bool {
	return err == syscall.Errno(87)
}
