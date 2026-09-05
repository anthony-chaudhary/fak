//go:build windows

package sandbox

import (
	"fmt"
	"os/exec"
	"syscall"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	jobObjectExtendedLimitInformationClass = 9
	jobObjectBasicAccountingInformation    = 1
	jobObjectLimitJobMemory                = 0x00000200
	jobObjectLimitKillOnJobClose           = 0x00002000
	processSetQuota                        = 0x0100
	processTerminate                       = 0x0001
	createNewProcessGroup                  = 0x00000200
	createNoWindow                         = 0x08000000
)

var (
	kernel32DLL                   = syscall.NewLazyDLL("kernel32.dll")
	procCreateJobObjectW          = kernel32DLL.NewProc("CreateJobObjectW")
	procSetInformationJobObject   = kernel32DLL.NewProc("SetInformationJobObject")
	procAssignProcessToJobObject  = kernel32DLL.NewProc("AssignProcessToJobObject")
	procQueryInformationJobObject = kernel32DLL.NewProc("QueryInformationJobObject")
	procCloseHandle               = kernel32DLL.NewProc("CloseHandle")
)

type jobBasicLimit struct {
	PerProcessUserTimeLimit, PerJobUserTimeLimit int64
	LimitFlags                                   uint32
	MinimumWorkingSetSize, MaximumWorkingSetSize uintptr
	ActiveProcessLimit                           uint32
	Affinity                                     uintptr
	PriorityClass, SchedulingClass               uint32
}

type ioCounters struct {
	ReadOperationCount, WriteOperationCount, OtherOperationCount uint64
	ReadTransferCount, WriteTransferCount, OtherTransferCount    uint64
}

type jobExtendedLimit struct {
	BasicLimitInformation                                                        jobBasicLimit
	IoInfo                                                                       ioCounters
	ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed uintptr
}

type windowsConfinement struct {
	spec   Spec
	job    syscall.Handle
	closed bool
}

func newOSConfinement(spec Spec) (osConfinement, error) {
	h, _, callErr := procCreateJobObjectW.Call(0, 0)
	if h == 0 {
		return nil, fmt.Errorf("CreateJobObjectW: %w", callErr)
	}
	w := &windowsConfinement{
		spec: spec,
		job:  syscall.Handle(h),
	}

	info := jobExtendedLimit{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	if spec.MemoryLimitBytes > 0 {
		info.BasicLimitInformation.LimitFlags |= jobObjectLimitJobMemory
		info.JobMemoryLimit = uintptr(spec.MemoryLimitBytes)
	}

	ok, _, ce := procSetInformationJobObject.Call(
		h,
		jobObjectExtendedLimitInformationClass,
		uintptr(unsafe.Pointer(&info)),
		unsafe.Sizeof(info),
	)
	if ok == 0 {
		_ = w.Close()
		return nil, fmt.Errorf("SetInformationJobObject: %w", ce)
	}

	return w, nil
}

func (w *windowsConfinement) PrepareCommand(cmd *exec.Cmd, req ExecutionRequest) error {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= createNewProcessGroup
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		killCmd := exec.Command("taskkill.exe", "/T", "/F", "/PID", fmt.Sprint(cmd.Process.Pid))
		windowgate.ConfigureBackgroundCommand(killCmd)
		if killCmd.SysProcAttr == nil {
			killCmd.SysProcAttr = &syscall.SysProcAttr{}
		}
		killCmd.SysProcAttr.HideWindow = true
		killCmd.SysProcAttr.CreationFlags |= createNoWindow
		_ = killCmd.Run()
		return cmd.Process.Kill()
	}
	cmd.WaitDelay = 2 * time.Second
	return nil
}

func (w *windowsConfinement) OnProcessStart(pid int) error {
	if w.job == 0 || pid <= 0 {
		return nil
	}
	hProc, err := syscall.OpenProcess(processSetQuota|processTerminate, false, uint32(pid))
	if err != nil {
		return fmt.Errorf("OpenProcess(%d): %w", pid, err)
	}
	defer procCloseHandle.Call(uintptr(hProc))

	ok, _, ce := procAssignProcessToJobObject.Call(uintptr(w.job), uintptr(hProc))
	if ok == 0 {
		return fmt.Errorf("AssignProcessToJobObject: %w", ce)
	}
	return nil
}

func (w *windowsConfinement) PostProcess(res *ExecutionResult) error {
	if w.job == 0 {
		return nil
	}
	var ext jobExtendedLimit
	var n uint32
	ok, _, _ := procQueryInformationJobObject.Call(
		uintptr(w.job),
		jobObjectExtendedLimitInformationClass,
		uintptr(unsafe.Pointer(&ext)),
		unsafe.Sizeof(ext),
		uintptr(unsafe.Pointer(&n)),
	)
	if ok != 0 && ext.PeakJobMemoryUsed > 0 {
		res.MemoryBytes = int64(ext.PeakJobMemoryUsed)
	}
	return nil
}

func (w *windowsConfinement) Close() error {
	if w.closed || w.job == 0 {
		return nil
	}
	w.closed = true
	h := w.job
	w.job = 0
	ok, _, ce := procCloseHandle.Call(uintptr(h))
	if ok == 0 {
		return fmt.Errorf("CloseHandle: %w", ce)
	}
	return nil
}
