//go:build !linux && !darwin && !windows

package harnessres

import "os"

// readProcSelf on an unsupported platform reports no OS-level axes; the portable
// runtime metrics (goroutines, Go heap, CPU count) still populate the Snapshot.
func readProcSelf() procSample { return procSample{} }

// foldChildRusage is a no-op where no per-platform child accounting is wired.
func foldChildRusage(h *Half, ps *os.ProcessState) {}
