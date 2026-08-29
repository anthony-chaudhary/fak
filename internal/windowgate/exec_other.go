//go:build !windows

package windowgate

import (
	"context"
	"errors"
	"os/exec"
)

// ConfigureBackgroundCommand is a no-op off Windows. POSIX helpers do not create
// Windows console windows, and callers keep their ordinary process semantics.
// Command constructs a short-lived helper subprocess. On Windows the matching
// implementation also suppresses console-window allocation.
func Command(name string, args ...string) *exec.Cmd { return exec.Command(name, args...) }

// CommandContext is Command with cancellation.
func CommandContext(ctx context.Context, name string, args ...string) *exec.Cmd {
	return exec.CommandContext(ctx, name, args...)
}

func ConfigureBackgroundCommand(_ *exec.Cmd) {}

// ConfigureDetachedCommand is a no-op off Windows. There is no console object to
// decline: the dispatch spawn already calls setsid, which is the POSIX equivalent
// of dropping the controlling terminal.
func ConfigureDetachedCommand(_ *exec.Cmd) {}

// JobObject is a no-op placeholder off Windows. POSIX teardown uses ordinary
// process-group semantics (Setpgid + a group signal), so there is no job handle
// to own; Close is always a nil-error no-op. Kept so callers can hold a
// *JobObject uniformly across build tags.
type JobObject struct{}

// ManagedJobConfig mirrors the Windows aggregate Job Object limit input.
type ManagedJobConfig struct {
	MemoryLimitBytes uint64
}

// Close is a no-op off Windows.
func (j *JobObject) Close() error { return nil }

// ConfigureWorkerCommand is a no-op off Windows: the tree-teardown guarantee is a
// Windows Job Object concern. POSIX callers keep their existing process-group
// behavior, applied where they already spawn (dispatch process-group setup).
func ConfigureWorkerCommand(_ *exec.Cmd) {}

// StartInNewJob preserves the ordinary asynchronous exec lifecycle off Windows.
// The nil job handle is safe to close after Wait.
func StartInNewJob(cmd *exec.Cmd) (*JobObject, error) {
	if cmd == nil {
		return nil, errors.New("windowgate: StartInNewJob requires a command")
	}
	return nil, cmd.Start()
}

// StartManagedAgentInNewJob preserves the ordinary process lifecycle off Windows.
func StartManagedAgentInNewJob(cmd *exec.Cmd, _ ManagedJobConfig) (*JobObject, error) {
	return StartInNewJob(cmd)
}

// RunInNewJob is the cross-platform guard-child runner. Non-Windows platforms
// use the ordinary exec lifecycle; Windows supplies job-object containment.
func RunInNewJob(cmd *exec.Cmd) error {
	job, err := StartInNewJob(cmd)
	if err != nil {
		return err
	}
	defer job.Close()
	return cmd.Wait()
}

// AssignToNewJobObject is a no-op off Windows, returning a nil job and nil error.
// There is nothing to reap via a job handle; POSIX teardown signals the group.
func AssignToNewJobObject(_ *exec.Cmd) (*JobObject, error) { return nil, nil }
