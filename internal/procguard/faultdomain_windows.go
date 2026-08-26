//go:build windows

package procguard

import (
	"fmt"
	"syscall"
	"time"
	"unsafe"
)

const (
	jobObjectExtendedLimitInformation   = 9
	jobObjectCpuRateControlInformation  = 15
	jobObjectBasicAccountingInformation = 1
	jobObjectBasicProcessIDList         = 3
	jobObjectLimitActiveProcess         = 0x00000008
	jobObjectLimitJobTime               = 0x00000004
	jobObjectLimitJobMemory             = 0x00000200
	jobObjectLimitKillOnJobClose        = 0x00002000
	jobObjectCpuRateControlEnable       = 0x1
	jobObjectCpuRateControlHardCap      = 0x4
	processSetQuota                     = 0x0100
	jobProcessTerminate                 = 0x0001
)

var (
	pgKernel32                  = syscall.NewLazyDLL("kernel32.dll")
	pgCreateJobObject           = pgKernel32.NewProc("CreateJobObjectW")
	pgSetInformationJobObject   = pgKernel32.NewProc("SetInformationJobObject")
	pgAssignProcessToJobObject  = pgKernel32.NewProc("AssignProcessToJobObject")
	pgOpenProcess               = pgKernel32.NewProc("OpenProcess")
	pgCloseHandle               = pgKernel32.NewProc("CloseHandle")
	pgQueryInformationJobObject = pgKernel32.NewProc("QueryInformationJobObject")
)

type jobBasicLimit struct {
	PerProcessUserTimeLimit, PerJobUserTimeLimit int64
	LimitFlags                                   uint32
	MinimumWorkingSetSize, MaximumWorkingSetSize uintptr
	ActiveProcessLimit                           uint32
	Affinity                                     uintptr
	PriorityClass, SchedulingClass               uint32
}
type ioCounters struct{ ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount uint64 }
type jobExtendedLimit struct {
	BasicLimitInformation                                                        jobBasicLimit
	IoInfo                                                                       ioCounters
	ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed uintptr
}
type jobCPUControl struct{ ControlFlags, CPURate uint32 }
type jobAccounting struct {
	TotalUserTime, TotalKernelTime, ThisPeriodTotalUserTime, ThisPeriodTotalKernelTime int64
	TotalPageFaultCount, TotalProcesses, ActiveProcesses, TotalTerminatedProcesses     uint32
}

type windowsFaultDomain struct{ job syscall.Handle }

func newNativeFaultDomain(owner string, e ResourceEnvelope) (nativeFaultDomain, FaultDomainReceipt, error) {
	h, _, callErr := pgCreateJobObject.Call(0, 0)
	if h == 0 {
		return nil, FaultDomainReceipt{}, fmt.Errorf("CreateJobObjectW: %w", callErr)
	}
	w := &windowsFaultDomain{job: syscall.Handle(h)}
	fail := func(err error) (nativeFaultDomain, FaultDomainReceipt, error) {
		_ = w.close()
		return nil, FaultDomainReceipt{}, err
	}
	info := jobExtendedLimit{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	enforced := map[string]string{}
	if e.MemoryBytes > 0 {
		info.BasicLimitInformation.LimitFlags |= jobObjectLimitJobMemory
		info.JobMemoryLimit = uintptr(e.MemoryBytes)
		enforced["memory"] = "JobMemoryLimit"
	}
	if e.ProcessCount > 0 {
		info.BasicLimitInformation.LimitFlags |= jobObjectLimitActiveProcess
		info.BasicLimitInformation.ActiveProcessLimit = e.ProcessCount
		enforced["processes"] = "ActiveProcessLimit"
	}
	if e.CPUTime > 0 {
		info.BasicLimitInformation.LimitFlags |= jobObjectLimitJobTime
		info.BasicLimitInformation.PerJobUserTimeLimit = e.CPUTime.Nanoseconds() / 100
		enforced["cpu_time"] = "PerJobUserTimeLimit"
	}
	ok, _, ce := pgSetInformationJobObject.Call(h, jobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&info)), unsafe.Sizeof(info))
	if ok == 0 {
		return fail(fmt.Errorf("SetInformationJobObject limits: %w", ce))
	}
	if e.CPUPercent > 0 {
		cpu := jobCPUControl{ControlFlags: jobObjectCpuRateControlEnable | jobObjectCpuRateControlHardCap, CPURate: e.CPUPercent * 100}
		ok, _, ce = pgSetInformationJobObject.Call(h, jobObjectCpuRateControlInformation, uintptr(unsafe.Pointer(&cpu)), unsafe.Sizeof(cpu))
		if ok == 0 {
			return fail(fmt.Errorf("SetInformationJobObject CPU: %w", ce))
		}
		enforced["cpu_share"] = "JobObjectCpuRateControlInformation hard cap"
	}
	limits := requestedSupport(e, enforced)
	return w, FaultDomainReceipt{Mode: modeFor(limits), Primitive: "windows-job-object", Limits: limits}, nil
}

func (w *windowsFaultDomain) bindCurrent() error {
	p, _, e := pgOpenProcess.Call(processSetQuota|jobProcessTerminate, 0, uintptr(syscall.Getpid()))
	if p == 0 {
		return e
	}
	defer pgCloseHandle.Call(p)
	ok, _, e := pgAssignProcessToJobObject.Call(uintptr(w.job), p)
	if ok == 0 {
		return e
	}
	return nil
}
func (w *windowsFaultDomain) usage() (ResourceUsage, error) {
	var a jobAccounting
	var n uint32
	ok, _, e := pgQueryInformationJobObject.Call(uintptr(w.job), jobObjectBasicAccountingInformation, uintptr(unsafe.Pointer(&a)), unsafe.Sizeof(a), uintptr(unsafe.Pointer(&n)))
	if ok == 0 {
		return ResourceUsage{}, e
	}
	var ext jobExtendedLimit
	ok, _, e = pgQueryInformationJobObject.Call(uintptr(w.job), jobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&ext)), unsafe.Sizeof(ext), uintptr(unsafe.Pointer(&n)))
	if ok == 0 {
		return ResourceUsage{}, e
	}
	return ResourceUsage{MemoryBytes: uint64(ext.PeakJobMemoryUsed), CPUTime: time100ns(a.TotalUserTime + a.TotalKernelTime), Processes: uint64(a.ActiveProcesses)}, nil
}
func time100ns(v int64) time.Duration { return time.Duration(v * 100) }
func (w *windowsFaultDomain) close() error {
	if w.job == 0 {
		return nil
	}
	ok, _, e := pgCloseHandle.Call(uintptr(w.job))
	w.job = 0
	if ok == 0 {
		return e
	}
	return nil
}
