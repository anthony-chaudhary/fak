//go:build !windows

package studylink

import "os/exec"

// ConfigureBackgroundCommand is a no-op on non-Windows platforms.
func ConfigureBackgroundCommand(_ *exec.Cmd) {}
