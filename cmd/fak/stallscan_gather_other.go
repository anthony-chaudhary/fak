//go:build !windows

package main

// stallscan_gather_other.go — non-Windows fallback for `fak stallscan`. The
// churn signals it reads (demand-zero/transition fault split, system-wide
// context-switch and syscall rates, disk-queue length) are gathered here via
// Windows perf counters; the fleet's stall problem is on the Windows host. On
// other platforms the verb reports cleanly that it is unsupported rather than
// emitting a misleading all-zero fingerprint. (A Linux gatherer reading
// /proc/vmstat + /proc/stat could be added later behind this same seam.)

import "github.com/anthony-chaudhary/fak/internal/stallscan"

func gatherStallSample(topN int) (stallscan.Sample, string) {
	return stallscan.Sample{}, "fak stallscan is only implemented on Windows (the churn-stall host); no-op here"
}
