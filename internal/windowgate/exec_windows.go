//go:build windows

package windowgate

import (
	"errors"
	"fmt"
	"os/exec"
	"syscall"
	"unsafe"
)

// CreateNoWindow is the Windows process creation flag that prevents a console
// child spawned by a windowless parent from allocating a visible conhost window.
const CreateNoWindow = 0x08000000

// CreateNewProcessGroup is CREATE_NEW_PROCESS_GROUP: the spawned worker becomes
// the root of a fresh process group so a group-directed console signal
// (CTRL_BREAK_EVENT) reaches the whole tree instead of only the top process.
const CreateNewProcessGroup = 0x00000200

// jobObjectLimitKillOnJobClose is JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: when the
// last handle to the job closes — normal owner exit OR abnormal teardown — every
// process still assigned to the job is terminated in one syscall. This is the
// seam that makes `kill == kill the tree` on Windows, where killing a worker's
// top process otherwise strands its conhost/node/python descendants.
const jobObjectLimitKillOnJobClose = 0x00002000

// jobObjectExtendedLimitInformationClass is the JOBOBJECTINFOCLASS selector
// (JobObjectExtendedLimitInformation) passed to SetInformationJobObject.
const jobObjectExtendedLimitInformationClass = 9

// Access rights AssignProcessToJobObject requires on the target process handle.
const (
	processTerminate = 0x0001
	processSetQuota  = 0x0100
)

// Kept on syscall.NewLazyDLL rather than a typed wrapper so this package stays
// standard-library-only, matching the house pattern in conptyver_windows.go and
// internal/compute/hostmem_windows.go (golang.org/x/sys is only an indirect dep).
// The kernel32 handle is declared once in terminal_restore_windows.go.
var (
	procCreateJobObjectW        = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJobObject = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJobObj   = kernel32.NewProc("AssignProcessToJobObject")
)

// jobObjectBasicLimitInformation mirrors JOBOBJECT_BASIC_LIMIT_INFORMATION.
// Field widths and order match the Win32 struct so the amd64 layout (with the
// natural 8-byte alignment padding after the uint32 words) is byte-for-byte
// identical to what SetInformationJobObject reads.
type jobObjectBasicLimitInformation struct {
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

// ioCounters mirrors IO_COUNTERS (embedded in the extended limit struct).
type ioCounters struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

// jobObjectExtendedLimitInformation mirrors JOBOBJECT_EXTENDED_LIMIT_INFORMATION.
type jobObjectExtendedLimitInformation struct {
	BasicLimitInformation jobObjectBasicLimitInformation
	IoInfo                ioCounters
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

// JobObject owns a Windows Job Object created with KILL_ON_JOB_CLOSE. The owner
// keeps it for the worker's lifetime; Close (deferred, or on any teardown path)
// reaps the entire assigned descendant tree in one syscall.
type JobObject struct {
	handle syscall.Handle
}

// Close closes the job handle. Because the job carries KILL_ON_JOB_CLOSE, closing
// the last handle terminates every process still assigned to it. Safe on a nil or
// already-closed receiver.
func (j *JobObject) Close() error {
	if j == nil || j.handle == 0 {
		return nil
	}
	h := j.handle
	j.handle = 0
	return syscall.CloseHandle(h)
}

// ConfigureBackgroundCommand marks cmd as a background/helper process. It must
// not create a user-visible console window when the parent has no console of its
// own, which is the common shape for scheduled fak maintenance tasks.
func ConfigureBackgroundCommand(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags |= CreateNoWindow
}

// ConfigureWorkerCommand prepares a long-lived dispatched-worker / loop-child
// spawn: the no-window background flags PLUS CREATE_NEW_PROCESS_GROUP so a
// group-directed signal reaches the worker's whole tree. Pair it with
// AssignToNewJobObject after Start for reliable KILL_ON_JOB_CLOSE teardown.
func ConfigureWorkerCommand(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	ConfigureBackgroundCommand(cmd)
	cmd.SysProcAttr.CreationFlags |= CreateNewProcessGroup
}

// AssignToNewJobObject creates a fresh Job Object with KILL_ON_JOB_CLOSE and
// assigns the freshly-started child (and thereby every descendant it later
// spawns) to it. It must be called after cmd.Start. The returned *JobObject is
// the teardown handle: hold it for the worker's lifetime and Close it — on
// normal exit or abnormal teardown — to reap the whole tree.
//
// On failure the caller still owns the started process and should terminate it.
func AssignToNewJobObject(cmd *exec.Cmd) (*JobObject, error) {
	if cmd == nil || cmd.Process == nil {
		return nil, errors.New("windowgate: AssignToNewJobObject requires a started process")
	}
	hJob, _, callErr := procCreateJobObjectW.Call(0, 0)
	if hJob == 0 {
		return nil, fmt.Errorf("windowgate: CreateJobObject: %w", callErr)
	}
	job := &JobObject{handle: syscall.Handle(hJob)}

	info := jobObjectExtendedLimitInformation{}
	info.BasicLimitInformation.LimitFlags = jobObjectLimitKillOnJobClose
	ok, _, callErr := procSetInformationJobObject.Call(
		hJob,
		uintptr(jobObjectExtendedLimitInformationClass),
		uintptr(unsafe.Pointer(&info)),
		unsafe.Sizeof(info),
	)
	if ok == 0 {
		job.Close()
		return nil, fmt.Errorf("windowgate: SetInformationJobObject: %w", callErr)
	}

	hProc, err := syscall.OpenProcess(processSetQuota|processTerminate, false, uint32(cmd.Process.Pid))
	if err != nil {
		job.Close()
		return nil, fmt.Errorf("windowgate: OpenProcess(%d): %w", cmd.Process.Pid, err)
	}
	defer syscall.CloseHandle(hProc)

	ok, _, callErr = procAssignProcessToJobObj.Call(hJob, uintptr(hProc))
	if ok == 0 {
		job.Close()
		return nil, fmt.Errorf("windowgate: AssignProcessToJobObject: %w", callErr)
	}
	return job, nil
}
