//go:build linux

package ggufload

import (
	"bufio"
	"os"
	"runtime"
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

const inkling2UploadAdviseOn = true

// inkling2UploadPopulate faults one upcoming upload chunk into the page cache
// and returns once it is read. MADV_WILLNEED is no substitute: on recent
// kernels it starts only a few MiB of readahead per call, whatever the length.
// madvise needs a page-aligned start, so the range is widened down to its page
// as raw addresses; a Go pointer below b's start would be invalid. No
// MADV_SEQUENTIAL: it would outlive the load on the split VMA.
func inkling2UploadPopulate(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	start := uintptr(unsafe.Pointer(&b[0]))
	addr := start &^ uintptr(unix.Getpagesize()-1)
	n := uintptr(len(b)) + (start - addr)
	_, _, errno := unix.Syscall(unix.SYS_MADVISE, addr, n, unix.MADV_POPULATE_READ)
	runtime.KeepAlive(b)
	if errno != 0 {
		return errno
	}
	return nil
}

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
