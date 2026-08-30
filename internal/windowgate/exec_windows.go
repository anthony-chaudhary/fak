//go:build windows

package windowgate

import (
	"context"
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

// DetachedProcess is DETACHED_PROCESS: the child neither inherits the parent's
// console nor gets one of its own.
//
// This is the flag that actually removes the per-worker console COST (#3597).
// CREATE_NO_WINDOW is often mistaken for it, but the two differ in what they
// suppress: CREATE_NO_WINDOW suppresses the console's WINDOW while still
// allocating the console, and on modern Windows every console is hosted by its
// own conhost.exe/OpenConsole.exe process. So a CREATE_NO_WINDOW child is
// invisible yet still pays the full per-console price the fleet audit measured —
// #2340 found 87 accumulated console hosts costing 2,829 threads / 54k handles /
// 2 GB, and #3405 confirmed that price scales linearly with fleet size
// (microsoft/terminal#15976). DETACHED_PROCESS allocates no console at all, so
// there is no host process to pay for.
const DetachedProcess = 0x00000008

// createSuspended is CREATE_SUSPENDED. StartInNewJob uses it so no child code
// can run (or fork an escaping descendant) between CreateProcess and assignment
// to the job object. The primary thread is resumed only after assignment.
const createSuspended = 0x00000004

// jobObjectLimitKillOnJobClose is JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: when the
// last handle to the job closes — normal owner exit OR abnormal teardown — every
// process still assigned to the job is terminated in one syscall. This is the
// seam that makes `kill == kill the tree` on Windows, where killing a worker's
// top process otherwise strands its conhost/node/python descendants.
const (
	jobObjectLimitKillOnJobClose = 0x00002000
	jobObjectLimitJobMemory      = 0x00000200
)

// jobObjectExtendedLimitInformationClass is the JOBOBJECTINFOCLASS selector
// (JobObjectExtendedLimitInformation) passed to SetInformationJobObject.
const jobObjectExtendedLimitInformationClass = 9

// Access rights AssignProcessToJobObject requires on the target process handle.
const (
	processTerminate    = 0x0001
	processSetQuota     = 0x0100
	threadSuspendResume = 0x0002
)

// Kept on syscall.NewLazyDLL rather than a typed wrapper so this package stays
// standard-library-only, matching the house pattern in conptyver_windows.go and
// internal/compute/hostmem_windows.go (golang.org/x/sys is only an indirect dep).
// The kernel32 handle is declared once in terminal_restore_windows.go.
var (
	procCreateJobObjectW        = kernel32.NewProc("CreateJobObjectW")
	procSetInformationJobObject = kernel32.NewProc("SetInformationJobObject")
	procAssignProcessToJobObj   = kernel32.NewProc("AssignProcessToJobObject")
	procOpenThread              = kernel32.NewProc("OpenThread")
	procResumeThread            = kernel32.NewProc("ResumeThread")
	procThread32First           = kernel32.NewProc("Thread32First")
	procThread32Next            = kernel32.NewProc("Thread32Next")
)

// assignToNewJobObject is the one injectable syscall seam StartInNewJob uses.
// Production always points at AssignToNewJobObject; tests replace it to prove
// the fail-closed branch reaps a child that started but could not be contained.
var assignToNewJobObject = AssignToNewJobObject

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

// threadEntry32 mirrors THREADENTRY32 for the one suspended primary thread
// StartInNewJob resumes after job assignment. This is the same Toolhelp seam the
// Go runtime's Windows process tests use for CREATE_SUSPENDED children.
type threadEntry32 struct {
	Size           uint32
	Usage          uint32
	ThreadID       uint32
	OwnerProcessID uint32
	BasePri        int32
	DeltaPri       int32
	Flags          uint32
}

// JobObject owns a Windows Job Object created with KILL_ON_JOB_CLOSE. The owner
// keeps it for the worker's lifetime; Close (deferred, or on any teardown path)
// reaps the entire assigned descendant tree in one syscall.
type JobObject struct {
	handle syscall.Handle
}

// ManagedJobConfig carries the explicit aggregate Job Object memory ceiling.
// A zero value preserves the historical 64 GiB managed-agent default.
type ManagedJobConfig struct {
	MemoryLimitBytes uint64
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
// Command constructs a subprocess that inherits its standard handles while never
// allocating a user-visible console window. Use it for short-lived helper tools
// (including the Go toolchain) launched by fak control-plane code.
func Command(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	ConfigureBackgroundCommand(cmd)
	return cmd
}

// CommandContext is Command with cancellation.
func CommandContext(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	ConfigureBackgroundCommand(cmd)
	return cmd
}

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

// ConfigureDetachedCommand prepares a spawn that must not own a console AT ALL:
// a fully unattended child whose stdout/stderr are already redirected to files or
// pipes, so it has no use for one. Use it instead of ConfigureBackgroundCommand
// only when nobody — neither an operator nor the child itself — reads a console:
// a detached child cannot be reached by a console control event (CTRL_BREAK), and
// a descendant that probes for a console sees none.
//
// It CLEARS CreateNoWindow rather than OR-ing alongside it. Windows treats the two
// as mutually exclusive and silently ignores CREATE_NO_WINDOW when DETACHED_PROCESS
// is present, so leaving both set would encode a contradiction that reads as though
// the window flag still did something.
func ConfigureDetachedCommand(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags &^= CreateNoWindow
	cmd.SysProcAttr.CreationFlags |= DetachedProcess
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

// StartInNewJob starts cmd as the root of a KILL_ON_JOB_CLOSE job. The caller
// owns the returned job until cmd.Wait completes. Windows closes that handle
// even when the caller is force-terminated (for example when a Windows Terminal
// tab is closed), so a stopped or otherwise non-cooperative child cannot outlive
// its guard session.
//
// This intentionally does not call ConfigureWorkerCommand: guard children are
// interactive and must retain their inherited terminal handles and window mode.
func StartInNewJob(cmd *exec.Cmd) (*JobObject, error) {
	return startInNewJob(cmd, 0)
}

// StartManagedAgentInNewJob starts a child-agent tree with both kill-on-close
// ownership and the aggregate commit ceiling. Callers opt in explicitly so an
// unrelated command-line argument cannot accidentally select or evade the cap.
func StartManagedAgentInNewJob(cmd *exec.Cmd, config ManagedJobConfig) (*JobObject, error) {
	return startInNewJob(cmd, managedJobMemoryLimitBytes(config))
}

func startInNewJob(cmd *exec.Cmd, memoryLimit uint64) (*JobObject, error) {
	if cmd == nil {
		return nil, errors.New("windowgate: StartInNewJob requires a command")
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= createSuspended
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	var job *JobObject
	var err error
	if memoryLimit > 0 {
		job, err = assignProcessToNewJobObject(cmd, memoryLimit)
	} else {
		job, err = assignToNewJobObject(cmd)
	}
	if err != nil {
		// Assignment is the teardown invariant. Do not leave an uncontained child
		// running when that invariant cannot be established.
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("windowgate: contain process tree: %w", err)
	}
	if err := resumeSuspendedProcess(cmd.Process.Pid); err != nil {
		// A contained but permanently suspended child is just as broken as an
		// uncontained one. Closing the job reaps the whole (still inert) tree.
		_ = job.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("windowgate: resume contained process: %w", err)
	}
	return job, nil
}

// RunInNewJob is the synchronous StartInNewJob/Wait lifecycle. Keeping the
// asynchronous start seam separate lets supervisors retain their select loop
// without weakening the same job-object containment invariant.
func RunInNewJob(cmd *exec.Cmd) error {
	job, err := StartInNewJob(cmd)
	if err != nil {
		return err
	}
	defer job.Close()
	return cmd.Wait()
}

func resumeSuspendedProcess(pid int) error {
	if pid <= 0 {
		return errors.New("invalid suspended process pid")
	}
	snapshot, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPTHREAD, 0)
	if err != nil {
		return fmt.Errorf("snapshot threads: %w", err)
	}
	defer syscall.CloseHandle(snapshot)

	entry := threadEntry32{Size: uint32(unsafe.Sizeof(threadEntry32{}))}
	ok, _, callErr := procThread32First.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&entry)))
	if ok == 0 {
		return fmt.Errorf("Thread32First: %w", callErr)
	}
	for {
		if entry.OwnerProcessID == uint32(pid) {
			h, _, callErr := procOpenThread.Call(threadSuspendResume, 0, uintptr(entry.ThreadID))
			if h == 0 {
				return fmt.Errorf("OpenThread(%d): %w", entry.ThreadID, callErr)
			}
			defer syscall.CloseHandle(syscall.Handle(h))
			previous, _, callErr := procResumeThread.Call(h)
			if previous == 0xffffffff {
				return fmt.Errorf("ResumeThread(%d): %w", entry.ThreadID, callErr)
			}
			return nil
		}
		ok, _, callErr = procThread32Next.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&entry)))
		if ok == 0 {
			return fmt.Errorf("suspended process %d thread not found: %w", pid, callErr)
		}
	}
}

// AssignToNewJobObject creates a fresh Job Object with KILL_ON_JOB_CLOSE and
// assigns the freshly-started child (and thereby every descendant it later
// spawns) to it. It must be called after cmd.Start. The returned *JobObject is
// the teardown handle: hold it for the worker's lifetime and Close it — on
// normal exit or abnormal teardown — to reap the whole tree.
//
// On failure the caller still owns the started process and should terminate it.
func AssignToNewJobObject(cmd *exec.Cmd) (*JobObject, error) {
	return assignProcessToNewJobObject(cmd, 0)
}

func assignProcessToNewJobObject(cmd *exec.Cmd, memoryLimit uint64) (*JobObject, error) {
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
	if memoryLimit > 0 {
		info.BasicLimitInformation.LimitFlags |= jobObjectLimitJobMemory
		info.JobMemoryLimit = uintptr(memoryLimit)
	}
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

func managedJobMemoryLimitBytes(config ManagedJobConfig) uint64 {
	const defaultLimit = uint64(64) << 30
	if config.MemoryLimitBytes == 0 {
		return defaultLimit
	}
	return config.MemoryLimitBytes
}
