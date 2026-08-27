//go:build windows

package systembaseline

import (
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

const (
	createSuspended                        = 0x00000004
	jobObjectLimitKillOnJobClose           = 0x00002000
	jobObjectBasicProcessIDListClass       = 3
	jobObjectBasicAndIOAccountingClass     = 8
	jobObjectExtendedLimitInformationClass = 9
	processTerminate                       = 0x0001
	processSetQuota                        = 0x0100
	processQueryLimitedInformation         = 0x1000
	processSuspendResume                   = 0x0800
	threadSuspendResume                    = 0x0002
	threadSnapshot                         = 0x00000004
	windowsJobCleanupTimeout               = 2 * time.Second
)

var (
	createJobObjectW             = kernel32.NewProc("CreateJobObjectW")
	setInformationJobObject      = kernel32.NewProc("SetInformationJobObject")
	queryInformationJobObject    = kernel32.NewProc("QueryInformationJobObject")
	procAssignProcessToJobObject = kernel32.NewProc("AssignProcessToJobObject")
	isProcessInJob               = kernel32.NewProc("IsProcessInJob")
	terminateJobObject           = kernel32.NewProc("TerminateJobObject")
	openThread                   = kernel32.NewProc("OpenThread")
	resumeThread                 = kernel32.NewProc("ResumeThread")
	thread32First                = kernel32.NewProc("Thread32First")
	thread32Next                 = kernel32.NewProc("Thread32Next")
)

var assignWindowsProcessToJob = func(job, process syscall.Handle) error {
	ok, _, callErr := procAssignProcessToJobObject.Call(uintptr(job), uintptr(process))
	if ok == 0 {
		return fmt.Errorf("AssignProcessToJobObject: %v", callErr)
	}
	return nil
}

type windowsJobBasicLimitInformation struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type windowsIOCounters struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

type windowsJobExtendedLimitInformation struct {
	BasicLimitInformation windowsJobBasicLimitInformation
	IOInfo                windowsIOCounters
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

type windowsJobBasicAccountingInformation struct {
	TotalUserTime             int64
	TotalKernelTime           int64
	ThisPeriodTotalUserTime   int64
	ThisPeriodTotalKernelTime int64
	TotalPageFaultCount       uint32
	TotalProcesses            uint32
	ActiveProcesses           uint32
	TotalTerminatedProcesses  uint32
}

type windowsJobBasicAndIOAccountingInformation struct {
	BasicInfo windowsJobBasicAccountingInformation
	IOInfo    windowsIOCounters
}

type windowsThreadEntry struct {
	Size           uint32
	Usage          uint32
	ThreadID       uint32
	OwnerProcessID uint32
	BasePri        int32
	DeltaPri       int32
	Flags          uint32
}

type windowsCommandAttributor struct {
	job            syscall.Handle
	result         WindowsJobObject
	configured     bool
	startProcessed bool
	assigned       bool
	exact          bool
	finished       bool
}

func newCommandAttributorPlatform() commandAttributorPlatform {
	return newWindowsCommandAttributor()
}

func newWindowsCommandAttributor() *windowsCommandAttributor {
	unavailable := func(reason string) *windowsCommandAttributor {
		return &windowsCommandAttributor{result: unavailableWindowsJob(reason), finished: true}
	}
	h, _, callErr := createJobObjectW.Call(0, 0)
	if h == 0 {
		return unavailable(fmt.Sprintf("CreateJobObjectW: %v", callErr))
	}
	w := &windowsCommandAttributor{job: syscall.Handle(h), result: unavailableWindowsJob("command has not been assigned to the Windows Job Object")}
	info := windowsJobExtendedLimitInformation{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	ok, _, callErr := setInformationJobObject.Call(h, jobObjectExtendedLimitInformationClass, uintptr(unsafe.Pointer(&info)), unsafe.Sizeof(info))
	if ok == 0 {
		reason := fmt.Sprintf("SetInformationJobObject(KILL_ON_JOB_CLOSE): %v", callErr)
		w.closeUnavailable(reason, true)
		return w
	}
	var readback windowsJobExtendedLimitInformation
	if err := queryWindowsJob(w.job, jobObjectExtendedLimitInformationClass, unsafe.Pointer(&readback), unsafe.Sizeof(readback)); err != nil || readback.BasicLimitInformation.LimitFlags&jobObjectLimitKillOnJobClose == 0 {
		reason := "KILL_ON_JOB_CLOSE readback failed"
		if err != nil {
			reason += ": " + err.Error()
		}
		w.closeUnavailable(reason, true)
		return w
	}
	return w
}

func (w *windowsCommandAttributor) active() bool {
	return w != nil && !w.finished && w.job != 0 && (!w.startProcessed || w.exact)
}

func (w *windowsCommandAttributor) configure(cmd *exec.Cmd) bool {
	if w == nil || w.finished || w.job == 0 || cmd == nil {
		return false
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= createSuspended
	w.configured = true
	return true
}

func (w *windowsCommandAttributor) started(pid int) error {
	if w == nil || w.finished || !w.configured || w.job == 0 {
		return nil
	}
	w.startProcessed = true
	var process syscall.Handle
	var placementErr error
	if pid <= 0 {
		placementErr = errors.New("invalid child PID")
	} else {
		var err error
		process, err = syscall.OpenProcess(processTerminate|processSetQuota|processQueryLimitedInformation|processSuspendResume, false, uint32(pid))
		if err != nil {
			placementErr = fmt.Errorf("OpenProcess(%d): %w", pid, err)
		}
	}
	var rootStartID uint64
	if placementErr == nil {
		rootStartID, placementErr = windowsProcessStartID(process)
	}
	if placementErr == nil {
		if placementErr = assignWindowsProcessToJob(w.job, process); placementErr == nil {
			w.assigned = true
		}
	}
	var members uint32
	if placementErr == nil {
		members, placementErr = w.verifyAssignment(process, pid)
	}
	if placementErr == nil {
		w.exact = true
		w.result.Membership = WindowsJobMembership{
			AtomicPlacement: true,
			RootPID:         pid,
			RootStartID:     rootStartID,
			AfterStart:      available(float64(members), "processes", "Windows JobObjectBasicProcessIdList readback"),
			AfterWait:       unavailable("processes", "command has not completed"),
			PlacementSource: "CREATE_SUSPENDED -> AssignProcessToJobObject -> JobObjectBasicProcessIdList/IsProcessInJob readback -> ResumeThread",
			IdentitySource:  "GetProcessTimes creation FILETIME read through the held process handle",
		}
	} else {
		w.exact = false
		w.result = unavailableWindowsJob("Windows Job Object placement unavailable; using sampled PID/PPID coverage: " + placementErr.Error())
		if !w.assigned {
			w.closeUnavailable(w.result.Reason, true)
		}
	}

	// Assignment failures are an explicit sampled fallback, not a command-launch
	// failure. The child was created suspended, so it must be resumed on every
	// nonfatal path. If it was assigned before readback failed, retain the job
	// handle for kill-on-close cleanup even though exact coverage is withheld.
	resumeErr := resumeWindowsProcess(pid)
	if resumeErr != nil && process != 0 {
		// NtResumeProcess is the race-free fallback when the primary thread handle
		// is not exposed by os/exec and Toolhelp races with helper-thread churn.
		ntdll := syscall.NewLazyDLL("ntdll.dll")
		status, _, callErr := ntdll.NewProc("NtResumeProcess").Call(uintptr(process))
		if status == 0 {
			resumeErr = nil
		} else {
			resumeErr = fmt.Errorf("%w; NtResumeProcess status=%#x call=%v", resumeErr, status, callErr)
		}
	}
	if process != 0 {
		_ = syscall.CloseHandle(process)
	}
	if resumeErr == nil {
		return nil
	}
	if w.job != 0 {
		w.result.Cleanup.Attempted = true
		if err := syscall.CloseHandle(w.job); err != nil {
			w.result.Cleanup.Reason = err.Error()
		} else {
			w.result.Cleanup.Closed = true
		}
		w.job = 0
	}
	w.exact = false
	w.finished = true
	w.result.State = WindowsJobStateUnavailable
	w.result.Reason = "resume suspended child: " + resumeErr.Error()
	w.result.Membership.UnavailableCause = w.result.Reason
	return fmt.Errorf("systembaseline: %w", errors.New(w.result.Reason))
}

func (w *windowsCommandAttributor) verifyAssignment(process syscall.Handle, pid int) (uint32, error) {
	var inJob int32
	ok, _, callErr := isProcessInJob.Call(uintptr(process), uintptr(w.job), uintptr(unsafe.Pointer(&inJob)))
	if ok == 0 {
		return 0, fmt.Errorf("IsProcessInJob: %v", callErr)
	}
	if inJob == 0 {
		return 0, errors.New("IsProcessInJob readback did not contain the root process")
	}
	const idCapacity = 16
	buf := make([]byte, 8+idCapacity*int(unsafe.Sizeof(uintptr(0))))
	if err := queryWindowsJob(w.job, jobObjectBasicProcessIDListClass, unsafe.Pointer(&buf[0]), uintptr(len(buf))); err != nil {
		return 0, fmt.Errorf("JobObjectBasicProcessIdList: %w", err)
	}
	assigned := *(*uint32)(unsafe.Pointer(&buf[0]))
	inList := *(*uint32)(unsafe.Pointer(&buf[4]))
	ids := unsafe.Slice((*uintptr)(unsafe.Pointer(&buf[8])), idCapacity)
	found := false
	for i := uint32(0); i < inList && i < idCapacity; i++ {
		if ids[i] == uintptr(pid) {
			found = true
			break
		}
	}
	if assigned == 0 || inList == 0 || !found {
		return 0, fmt.Errorf("JobObjectBasicProcessIdList did not read back root PID %d (assigned=%d listed=%d)", pid, assigned, inList)
	}
	return assigned, nil
}

func (w *windowsCommandAttributor) launchFailed(startErr error) {
	if w == nil || w.finished {
		return
	}
	reason := "suspended command launch failed"
	if startErr != nil {
		reason += ": " + startErr.Error()
	}
	w.closeUnavailable(reason, true)
}

func (w *windowsCommandAttributor) finish() CgroupV2 {
	_ = w.finishWindowsJob()
	return unavailableCgroup("cgroup v2 attribution is unavailable on Windows")
}

func (w *windowsCommandAttributor) finishAttribution() CommandAttribution {
	job := w.finishWindowsJob()
	return CommandAttribution{WindowsJobObject: &job}
}

func (w *windowsCommandAttributor) finishWindowsJob() WindowsJobObject {
	if w == nil {
		return unavailableWindowsJob("Windows Job Object attribution adapter unavailable")
	}
	if w.finished {
		return w.result.clone()
	}
	w.finished = true
	if w.job == 0 {
		return w.result.clone()
	}
	w.result.Cleanup.Attempted = true

	atCommandEnd, endErr := readWindowsJobAccounting(w.job)
	if endErr == nil {
		w.result.Membership.AfterWait = available(float64(atCommandEnd.BasicInfo.ActiveProcesses), "processes", "Windows JobObjectBasicAccountingInformation after command wait")
	} else {
		w.result.Membership.AfterWait = unavailable("processes", endErr.Error())
	}
	if endErr == nil && atCommandEnd.BasicInfo.ActiveProcesses > 0 {
		w.result.Cleanup.KilledRemaining = true
		ok, _, callErr := terminateJobObject.Call(uintptr(w.job), 1)
		if ok == 0 {
			w.addCleanupReason(fmt.Sprintf("TerminateJobObject: %v", callErr))
		} else if err := w.waitEmpty(windowsJobCleanupTimeout); err != nil {
			w.addCleanupReason(err.Error())
		} else {
			w.result.Cleanup.Empty = true
		}
	} else if endErr == nil {
		w.result.Cleanup.Empty = true
	} else {
		w.addCleanupReason(endErr.Error())
		// Counter readback failure cannot weaken teardown. Terminate the bounded
		// job anyway, then use a fresh accounting read to prove it became empty.
		w.result.Cleanup.KilledRemaining = true
		ok, _, callErr := terminateJobObject.Call(uintptr(w.job), 1)
		if ok == 0 {
			w.addCleanupReason(fmt.Sprintf("TerminateJobObject after accounting failure: %v", callErr))
		} else if err := w.waitEmpty(windowsJobCleanupTimeout); err != nil {
			w.addCleanupReason(err.Error())
		} else {
			w.result.Cleanup.Empty = true
		}
	}

	final, finalErr := readWindowsJobAccounting(w.job)
	extended, extendedErr := readWindowsJobExtended(w.job)
	if finalErr == nil {
		w.foldAccounting(final)
	} else {
		w.addCleanupReason(finalErr.Error())
	}
	w.foldMemory(extended, extendedErr)
	if err := syscall.CloseHandle(w.job); err != nil {
		w.addCleanupReason("CloseHandle(job): " + err.Error())
	} else {
		w.result.Cleanup.Closed = true
	}
	w.job = 0

	if w.exact && finalErr == nil && w.result.CPU.Available && w.result.Processes.Available && w.result.IO.Available && w.result.Cleanup.Empty && w.result.Cleanup.Closed {
		w.result.State = WindowsJobStateMeasured
		w.result.Reason = ""
		w.result.Membership.UnavailableCause = ""
	} else {
		reason := strings.TrimSpace(w.result.Reason)
		if reason == "" || reason == "command has not been assigned to the Windows Job Object" {
			reason = "Windows Job Object exact attribution was not fully verified"
		}
		if w.result.Cleanup.Reason != "" {
			reason += ": " + w.result.Cleanup.Reason
		}
		w.result.State = WindowsJobStateUnavailable
		w.result.Reason = reason
		w.result.Membership.UnavailableCause = reason
	}
	return w.result.clone()
}

func (w *windowsCommandAttributor) waitEmpty(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		accounting, err := readWindowsJobAccounting(w.job)
		if err != nil {
			return err
		}
		if accounting.BasicInfo.ActiveProcesses == 0 {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("Windows Job Object still contains %d process(es) after cleanup timeout", accounting.BasicInfo.ActiveProcesses)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (w *windowsCommandAttributor) foldAccounting(info windowsJobBasicAndIOAccountingInformation) {
	if info.BasicInfo.TotalUserTime < 0 || info.BasicInfo.TotalKernelTime < 0 {
		reason := "Windows Job Object returned negative CPU accounting"
		w.result.CPU = unavailableCounterSet(reason)
		w.result.Processes = unavailableCounterSet(reason)
		w.result.IO = unavailableCounterSet(reason)
		return
	}
	user := uint64(info.BasicInfo.TotalUserTime)
	kernel := uint64(info.BasicInfo.TotalKernelTime)
	if ^uint64(0)-user < kernel {
		reason := "Windows Job Object CPU accounting overflow"
		w.result.CPU = unavailableCounterSet(reason)
		w.result.Processes = unavailableCounterSet(reason)
		w.result.IO = unavailableCounterSet(reason)
		return
	}
	w.result.CPU = availableCounterSet(map[string]uint64{
		"user_100ns":   user,
		"kernel_100ns": kernel,
		"usage_100ns":  user + kernel,
	}, "Windows JobObjectBasicAndIoAccountingInformation")
	w.result.Processes = availableCounterSet(map[string]uint64{
		"total_processes":            uint64(info.BasicInfo.TotalProcesses),
		"active_processes":           uint64(info.BasicInfo.ActiveProcesses),
		"limit_terminated_processes": uint64(info.BasicInfo.TotalTerminatedProcesses),
		"page_faults":                uint64(info.BasicInfo.TotalPageFaultCount),
	}, "Windows JobObjectBasicAndIoAccountingInformation")
	w.result.IO = availableCounterSet(map[string]uint64{
		"read_operations":  info.IOInfo.ReadOperationCount,
		"write_operations": info.IOInfo.WriteOperationCount,
		"other_operations": info.IOInfo.OtherOperationCount,
		"read_bytes":       info.IOInfo.ReadTransferCount,
		"write_bytes":      info.IOInfo.WriteTransferCount,
		"other_bytes":      info.IOInfo.OtherTransferCount,
	}, "Windows JobObjectBasicAndIoAccountingInformation")
}

func (w *windowsCommandAttributor) foldMemory(extended windowsJobExtendedLimitInformation, extendedErr error) {
	if extendedErr == nil {
		w.result.Memory.PeakJobCommitBytes = available(float64(extended.PeakJobMemoryUsed), "bytes", "Windows JobObjectExtendedLimitInformation")
		w.result.Memory.PeakProcessCommitBytes = available(float64(extended.PeakProcessMemoryUsed), "bytes", "Windows JobObjectExtendedLimitInformation")
	} else {
		w.result.Memory.PeakJobCommitBytes = unavailable("bytes", extendedErr.Error())
		w.result.Memory.PeakProcessCommitBytes = unavailable("bytes", extendedErr.Error())
	}
}

func (w *windowsCommandAttributor) closeUnavailable(reason string, empty bool) {
	w.result = unavailableWindowsJob(reason)
	w.result.Cleanup.Attempted = true
	w.result.Cleanup.Empty = empty
	if w.job != 0 {
		if err := syscall.CloseHandle(w.job); err != nil {
			w.result.Cleanup.Empty = false
			w.result.Cleanup.Reason = err.Error()
		} else {
			w.result.Cleanup.Closed = true
		}
		w.job = 0
	}
	w.exact = false
	w.finished = true
}

func (w *windowsCommandAttributor) addCleanupReason(reason string) {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return
	}
	if w.result.Cleanup.Reason == "" {
		w.result.Cleanup.Reason = reason
		return
	}
	w.result.Cleanup.Reason += "; " + reason
}

func queryWindowsJob(job syscall.Handle, class uintptr, out unsafe.Pointer, size uintptr) error {
	ok, _, callErr := queryInformationJobObject.Call(uintptr(job), class, uintptr(out), size, 0)
	if ok == 0 {
		return fmt.Errorf("QueryInformationJobObject(class=%d): %v", class, callErr)
	}
	return nil
}

func readWindowsJobAccounting(job syscall.Handle) (windowsJobBasicAndIOAccountingInformation, error) {
	var out windowsJobBasicAndIOAccountingInformation
	err := queryWindowsJob(job, jobObjectBasicAndIOAccountingClass, unsafe.Pointer(&out), unsafe.Sizeof(out))
	return out, err
}

func readWindowsJobExtended(job syscall.Handle) (windowsJobExtendedLimitInformation, error) {
	var out windowsJobExtendedLimitInformation
	err := queryWindowsJob(job, jobObjectExtendedLimitInformationClass, unsafe.Pointer(&out), unsafe.Sizeof(out))
	return out, err
}

func windowsProcessStartID(process syscall.Handle) (uint64, error) {
	var create, exit, kernel, user filetime
	ok, _, callErr := getProcessTimes.Call(uintptr(process), uintptr(unsafe.Pointer(&create)), uintptr(unsafe.Pointer(&exit)), uintptr(unsafe.Pointer(&kernel)), uintptr(unsafe.Pointer(&user)))
	if ok == 0 {
		return 0, fmt.Errorf("GetProcessTimes: %v", callErr)
	}
	if create.uint64() == 0 {
		return 0, errors.New("GetProcessTimes returned an empty creation identity")
	}
	return create.uint64(), nil
}

func resumeWindowsProcess(pid int) error {
	if pid <= 0 {
		return errors.New("invalid suspended process PID")
	}
	snapshot, err := syscall.CreateToolhelp32Snapshot(threadSnapshot, 0)
	if err != nil {
		return fmt.Errorf("snapshot threads: %w", err)
	}
	defer syscall.CloseHandle(snapshot)
	entry := windowsThreadEntry{Size: uint32(unsafe.Sizeof(windowsThreadEntry{}))}
	ok, _, callErr := thread32First.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&entry)))
	if ok == 0 {
		return fmt.Errorf("Thread32First: %v", callErr)
	}
	for {
		if entry.OwnerProcessID == uint32(pid) {
			h, _, callErr := openThread.Call(threadSuspendResume, 0, uintptr(entry.ThreadID))
			if h != 0 {
				previous, _, resumeErr := resumeThread.Call(h)
				_ = syscall.CloseHandle(syscall.Handle(h))
				if previous == 0xffffffff {
					return fmt.Errorf("ResumeThread(%d): %v", entry.ThreadID, resumeErr)
				}
				if previous > 0 {
					return nil
				}
			} else if !errors.Is(callErr, syscall.Errno(87)) {
				return fmt.Errorf("OpenThread(%d): %v", entry.ThreadID, callErr)
			}
			// A process may create helper threads before the snapshot becomes visible;
			// skip vanished or already-running threads and find the suspended primary.
		}
		entry.Size = uint32(unsafe.Sizeof(windowsThreadEntry{}))
		ok, _, callErr = thread32Next.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&entry)))
		if ok == 0 {
			return fmt.Errorf("suspended process %d thread not found: %v", pid, callErr)
		}
	}
}
