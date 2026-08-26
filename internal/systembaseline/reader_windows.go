//go:build windows

package systembaseline

import (
	"syscall"
	"time"
	"unsafe"
)

var (
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	psapi                    = syscall.NewLazyDLL("psapi.dll")
	getSystemTimes           = kernel32.NewProc("GetSystemTimes")
	globalMemoryStatusEx     = kernel32.NewProc("GlobalMemoryStatusEx")
	createToolhelp32Snapshot = kernel32.NewProc("CreateToolhelp32Snapshot")
	process32FirstW          = kernel32.NewProc("Process32FirstW")
	process32NextW           = kernel32.NewProc("Process32NextW")
	openProcess              = kernel32.NewProc("OpenProcess")
	getProcessTimes          = kernel32.NewProc("GetProcessTimes")
	closeHandle              = kernel32.NewProc("CloseHandle")
	getProcessMemoryInfo     = psapi.NewProc("GetProcessMemoryInfo")
)

type filetime struct{ Low, High uint32 }

func (f filetime) uint64() uint64 { return uint64(f.High)<<32 | uint64(f.Low) }

type memoryStatus struct {
	Length, MemoryLoad                                                                    uint32
	TotalPhys, AvailPhys, TotalPage, AvailPage, TotalVirtual, AvailVirtual, AvailExtended uint64
}
type processEntry struct {
	Size, CntUsage, PID          uint32
	DefaultHeapID                uintptr
	ModuleID, Threads, ParentPID uint32
	PriClassBase                 int32
	Flags                        uint32
	ExeFile                      [260]uint16
}
type processMemory struct {
	CB, PageFaultCount                                                                                                                                                     uint32
	PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage uintptr
}

func capturePlatform() Snapshot {
	s := Snapshot{At: time.Now().UTC(), Host: readWindowsHost()}
	h, _, _ := createToolhelp32Snapshot.Call(2, 0)
	if h == 0 || h == ^uintptr(0) {
		s.ProcessNote = "Toolhelp process census unavailable"
		return s
	}
	defer closeHandle.Call(h)
	var entry processEntry
	entry.Size = uint32(unsafe.Sizeof(entry))
	ok, _, _ := process32FirstW.Call(h, uintptr(unsafe.Pointer(&entry)))
	if ok == 0 {
		s.ProcessNote = "Toolhelp process census unreadable"
		return s
	}
	s.ProcessEnumerationOK = true
	for {
		p := readWindowsProcess(int(entry.PID), int(entry.ParentPID), syscall.UTF16ToString(entry.ExeFile[:]))
		if !p.CPUAvailable && !p.RSSAvailable {
			s.ProcessUnreadable++
		}
		s.Processes = append(s.Processes, p)
		entry.Size = uint32(unsafe.Sizeof(entry))
		result, _, callErr := process32NextW.Call(h, uintptr(unsafe.Pointer(&entry)))
		errorCode := uintptr(0)
		if errno, isErrno := callErr.(syscall.Errno); isErrno {
			errorCode = uintptr(errno)
		}
		done, truncated := classifyWindowsProcessAdvance(result, errorCode)
		if truncated {
			s.AttributionIncomplete = true
			s.ProcessNote = "Toolhelp process census truncated unexpectedly"
		}
		if done {
			break
		}
	}
	return s
}

func readWindowsHost() HostSample {
	var h HostSample
	var idle, kernel, user filetime
	if ok, _, _ := getSystemTimes.Call(uintptr(unsafe.Pointer(&idle)), uintptr(unsafe.Pointer(&kernel)), uintptr(unsafe.Pointer(&user))); ok != 0 {
		h.CPUAvailable = true
		h.TotalCPUNS = (kernel.uint64() + user.uint64()) * 100
		h.BusyCPUNS = (kernel.uint64() + user.uint64() - idle.uint64()) * 100
	}
	m := memoryStatus{Length: uint32(unsafe.Sizeof(memoryStatus{}))}
	if ok, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&m))); ok != 0 {
		h.MemoryAvailable = true
		h.MemoryTotal = m.TotalPhys
		h.MemoryFree = m.AvailPhys
	}
	return h
}

func readWindowsProcess(pid, ppid int, image string) ProcessSample {
	const (
		queryLimitedInformationAndVMRead = 0x1010
		queryLimitedInformation          = 0x1000
	)
	p := ProcessSample{PID: pid, PPID: ppid, Image: image}
	h, _, _ := openProcess.Call(queryLimitedInformationAndVMRead, 0, uintptr(pid))
	if h == 0 {
		h, _, _ = openProcess.Call(queryLimitedInformation, 0, uintptr(pid))
		if h == 0 {
			return p
		}
	}
	defer closeHandle.Call(h)
	var create, exit, kernel, user filetime
	if ok, _, _ := getProcessTimes.Call(h, uintptr(unsafe.Pointer(&create)), uintptr(unsafe.Pointer(&exit)), uintptr(unsafe.Pointer(&kernel)), uintptr(unsafe.Pointer(&user))); ok != 0 {
		p.StartID = create.uint64()
		p.CPUAvailable = true
		p.CPUNS = (kernel.uint64() + user.uint64()) * 100
	}
	m := processMemory{CB: uint32(unsafe.Sizeof(processMemory{}))}
	if ok, _, _ := getProcessMemoryInfo.Call(h, uintptr(unsafe.Pointer(&m)), uintptr(m.CB)); ok != 0 {
		p.RSSAvailable = true
		p.RSSBytes = uint64(m.WorkingSetSize)
	}
	return p
}
