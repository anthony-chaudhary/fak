//go:build !windows

package main

import (
	"os"
	"syscall"
)

// terminatingSignals are the signals that trigger `fak serve`'s graceful drain + final
// session-state flush (#1359). On a real OS, three reach us: SIGINT (Ctrl-C), SIGTERM (an
// orchestrator stop), and SIGHUP (the controlling terminal closed — the single most common
// "I accidentally closed it" case). All three must flush identically; only the uncatchable
// SIGKILL escapes, which is the write-ahead journal's gap to close (#1363).
func terminatingSignals() []os.Signal {
	return []os.Signal{os.Interrupt, syscall.SIGTERM, syscall.SIGHUP}
}
