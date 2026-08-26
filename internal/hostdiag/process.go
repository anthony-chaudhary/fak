package hostdiag

import "time"

// ProcessImage identifies a live process executable without exposing its command line.
func ProcessImage(pid int) (string, bool) { return processImage(pid) }

// ProcessStartedAt returns the kernel creation time for PID-reuse-safe correlation.
func ProcessStartedAt(pid int) (time.Time, bool) { return processStartedAt(pid) }
