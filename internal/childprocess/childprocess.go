// Package childprocess normalizes subprocess exit status.
package childprocess

import (
	"errors"
	"os/exec"
)

// ExitCode returns the child's status, or launchFailure when the process never started.
func ExitCode(err error, launchFailure int) int {
	if err == nil {
		return 0
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode()
	}
	return launchFailure
}
