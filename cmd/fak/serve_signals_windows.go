//go:build windows

package main

import (
	"os"
	"syscall"
)

// terminatingSignals (Windows): the OS has no SIGHUP, and Go's runtime only ever delivers
// os.Interrupt (Ctrl-C / Ctrl-Break) and syscall.SIGTERM here, so registering SIGHUP would
// be dead weight. The graceful drain + session-state flush (#1359) still fires on both of
// the signals Windows can actually raise.
func terminatingSignals() []os.Signal {
	return []os.Signal{os.Interrupt, syscall.SIGTERM}
}
