//go:build windows

package devcmd

import (
	"errors"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"io"
	"os/exec"
	"syscall"
)

// The existing job runner starts suspended, contains the child before resuming,
// waits for the direct child, then closes the kill-on-close job before returning.
func runOwnedCompanionProbe(cmd *exec.Cmd) error {
	return windowgate.RunInNewJob(cmd)
}

func companionProbeSocketClosed(err error) bool {
	return errors.Is(err, io.EOF) || errors.Is(err, syscall.WSAECONNRESET)
}
