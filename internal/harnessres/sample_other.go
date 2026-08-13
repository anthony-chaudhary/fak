//go:build !linux && !darwin && !windows

package harnessres

import "os"

// readProcSelf on an unsupported platform reports no OS-level axes; the portable
// runtime metrics (goroutines, Go heap, CPU count) still populate the Snapshot.
func readProcSelf() procSample { return procSample{} }

// foldChildRusage is a no-op where no per-platform child accounting is wired.
func foldChildRusage(h *Half, ps *os.ProcessState) {}

// readProcPID has no reader on an unsupported platform, so every walked process comes
// back unreadable — the fleet rollup then reports a process count with n/a byte and CPU
// axes rather than a fabricated zero-cost fleet (#6557).
func readProcPID(pid int) (procSample, bool) { return procSample{}, false }
