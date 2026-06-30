//go:build !windows

package windowgate

import "os/exec"

// ConfigureBackgroundCommand is a no-op off Windows. POSIX helpers do not create
// Windows console windows, and callers keep their ordinary process semantics.
func ConfigureBackgroundCommand(_ *exec.Cmd) {}
