//go:build windows

package modelperfobs

import (
	"syscall"
	"time"
	"unsafe"
)

type memoryStatusEx struct {
	Length                                                                                               uint32
	MemoryLoad                                                                                           uint32
	TotalPhys, AvailPhys, TotalPageFile, AvailPageFile, TotalVirtual, AvailVirtual, AvailExtendedVirtual uint64
}
type processMemoryCounters struct {
	CB, PageFaultCount                                                                                                                                                     uint32
	PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage uintptr
}
type ioCounters struct{ ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount uint64 }

func collectHostSnapshot() (hostSnapshot, error) {
	s := hostSnapshot{at: time.Now(), collector: "windows-kernel32"}
	k := syscall.NewLazyDLL("kernel32.dll")
	m := memoryStatusEx{Length: uint32(unsafe.Sizeof(memoryStatusEx{}))}
	if r, _, _ := k.NewProc("GlobalMemoryStatusEx").Call(uintptr(unsafe.Pointer(&m))); r != 0 {
		s.host.PhysicalTotalBytes = &m.TotalPhys
		s.host.PhysicalAvailableBytes = &m.AvailPhys
		s.availability.PhysicalMemory = true
		// Despite its API field name, TotalPageFile is the system commit
		// limit (RAM plus paging files), not paging-file capacity or I/O.
		commitLimit := m.TotalPageFile
		commitUsed := uint64(0)
		if m.TotalPageFile >= m.AvailPageFile {
			commitUsed = m.TotalPageFile - m.AvailPageFile
		}
		s.host.CommitLimitBytes = &commitLimit
		s.host.CommitUsedBytes = &commitUsed
		s.availability.MemoryPressure = true
	}
	h, _, _ := k.NewProc("GetCurrentProcess").Call()
	p := processMemoryCounters{CB: uint32(unsafe.Sizeof(processMemoryCounters{}))}
	if r, _, _ := k.NewProc("K32GetProcessMemoryInfo").Call(h, uintptr(unsafe.Pointer(&p)), uintptr(p.CB)); r != 0 {
		v := uint64(p.WorkingSetSize)
		s.host.ProcessResidentBytes = &v
		faults := uint64(p.PageFaultCount)
		s.host.ProcessPageFaults = &faults
		s.availability.ProcessMemory = true
		s.availability.MemoryPressure = true
	}
	var io ioCounters
	if r, _, _ := k.NewProc("GetProcessIoCounters").Call(h, uintptr(unsafe.Pointer(&io))); r != 0 {
		s.host.ProcessReadBytes = &io.ReadTransferCount
		s.host.ProcessWriteBytes = &io.WriteTransferCount
		s.host.ProcessIOScope = "process-io-all-devices-not-dram"
		s.availability.ProcessIO = true
	}
	return s, nil
}
