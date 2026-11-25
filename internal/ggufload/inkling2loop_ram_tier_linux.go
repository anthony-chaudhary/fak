//go:build linux

package ggufload

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"unsafe"

	"golang.org/x/sys/unix"
)

// inkling2RAMAdvise queues readahead for every RAM-tier region up front, so the
// NVMe sees one deep queue instead of the touch loop's serial page faults.
// Advice is a hint; a refused call only loses the overlap.
func inkling2RAMAdvise(regions [][]byte) {
	page := unix.Getpagesize()
	for _, r := range regions {
		if len(r) == 0 {
			continue
		}
		// madvise needs a page-aligned start and a stack may begin mid-page;
		// skipping to the next boundary loses at most one page of the hint.
		skip := (page - int(uintptr(unsafe.Pointer(&r[0])))%page) % page
		if skip >= len(r) {
			continue
		}
		_ = unix.Madvise(r[skip:], unix.MADV_WILLNEED)
	}
}

func inkling2RAMMlock(b []byte) error { return unix.Mlock(b) }

// inkling2CgroupClamped reports whether the probe's avail sits materially
// under /proc/meminfo MemAvailable, i.e. the compute probe lowered it to a
// cgroup's headroom. The slack absorbs drift between the two reads.
func inkling2CgroupClamped(avail int64) bool {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return false
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil || kb < 0 || kb > (1<<63-1)/1024 {
			return false
		}
		const slack = int64(256) << 20
		return avail+slack < kb*1024
	}
	return false
}
