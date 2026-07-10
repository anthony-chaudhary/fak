//go:build !windows

package windowgate

import "os/exec"

// ConfigureBackgroundCommand is a no-op off Windows. POSIX helpers do not create
// Windows console windows, and callers keep their ordinary process semantics.
func ConfigureBackgroundCommand(_ *exec.Cmd) {}

// JobObject is a no-op placeholder off Windows. POSIX teardown uses ordinary
// process-group semantics (Setpgid + a group signal), so there is no job handle
// to own; Close is always a nil-error no-op. Kept so callers can hold a
// *JobObject uniformly across build tags.
type JobObject struct{}

// Close is a no-op off Windows.
func (j *JobObject) Close() error { return nil }

// ConfigureWorkerCommand is a no-op off Windows: the tree-teardown guarantee is a
// Windows Job Object concern. POSIX callers keep their existing process-group
// behavior, applied where they already spawn (dispatch process-group setup).
func ConfigureWorkerCommand(_ *exec.Cmd) {}

// AssignToNewJobObject is a no-op off Windows, returning a nil job and nil error.
// There is nothing to reap via a job handle; POSIX teardown signals the group.
func AssignToNewJobObject(_ *exec.Cmd) (*JobObject, error) { return nil, nil }
