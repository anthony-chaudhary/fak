//go:build linux

package compute

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// hostSystemMemory reports host physical memory. `free` is what a new allocation can ACTUALLY
// claim, which on Linux is MemAvailable (free pages PLUS reclaimable page cache + slab), not
// Sysinfo.Freeram (strictly-free pages only). The distinction is load-bearing for the
// CPU-offload fit check on a box that has just read hundreds of GB off disk: those bytes sit
// in the page cache, so Freeram collapses to a small number while MemAvailable still reflects
// the ~2 TB the kernel will hand out (evicting clean cache on demand). Using Freeram made the
// GLM-5.2 expert offload (~417 GiB host) wrongly refuse on a 2 TB box whose Freeram had dropped
// to ~100 GiB after staging the 434 GB shards. Fall back to Sysinfo (Freeram) only when
// /proc/meminfo is unreadable.
func hostSystemMemory() (total, free int64, known bool) {
	if t, a, ok := procMeminfoAvailable(); ok {
		return t, a, true
	}
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil {
		return 0, FreeUnknown, false
	}
	unit := uint64(si.Unit)
	if unit == 0 {
		unit = 1
	}
	total = scaleHostMem(uint64(si.Totalram), unit)
	free = scaleHostMem(uint64(si.Freeram), unit)
	return total, free, total > 0
}

// procMeminfoAvailable reads MemTotal + MemAvailable (in kB) from /proc/meminfo. MemAvailable is
// the kernel's own estimate of allocatable memory (free + reclaimable), present on every Linux
// since 3.14. Returns ok=false if the file is missing or either field is absent/unparseable.
func procMeminfoAvailable() (total, avail int64, ok bool) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0, false
	}
	defer f.Close()
	var haveTotal, haveAvail bool
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "MemTotal:"):
			if v, ok := parseMeminfoKB(line); ok {
				total = v
				haveTotal = true
			}
		case strings.HasPrefix(line, "MemAvailable:"):
			if v, ok := parseMeminfoKB(line); ok {
				avail = v
				haveAvail = true
			}
		}
		if haveTotal && haveAvail {
			break
		}
	}
	if !haveTotal || !haveAvail {
		return 0, 0, false
	}
	return total, avail, true
}

// parseMeminfoKB parses a "Field:   <N> kB" line into N*1024 bytes.
func parseMeminfoKB(line string) (int64, bool) {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return 0, false
	}
	kb, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil || kb < 0 || kb > (1<<63-1)/1024 {
		return 0, false
	}
	return kb * 1024, true
}

func scaleHostMem(v, unit uint64) int64 {
	if unit == 0 {
		unit = 1
	}
	if v > maxInt64Uint64/unit {
		return int64(maxInt64Uint64)
	}
	return int64(v * unit)
}
