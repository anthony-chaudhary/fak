//go:build windows

package safecommit

import (
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

// processImageName resolves a live PID's executable base name (lowercased, without a
// trailing ".exe"), reporting ok=false when the image cannot be read. It powers the
// PID-reuse guard: a lock whose recorded PID is alive but whose image is provably not a
// committer is a reused PID number, safe to break (issue #2339).
//
// On Windows it opens the process with PROCESS_QUERY_LIMITED_INFORMATION (the same
// right the liveness probe uses) and calls QueryFullProcessImageNameW. Any failure —
// the process exited, or we lack the right — yields ok=false, so an unidentifiable
// live holder is treated as committer-like and never reaped.
func processImageName(pid int) (string, bool) {
	if pid <= 0 {
		return "", false
	}
	const processQueryLimitedInformation = 0x1000
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil || h == 0 {
		return "", false
	}
	defer syscall.CloseHandle(h)

	buf := make([]uint16, syscall.MAX_PATH)
	size := uint32(len(buf))
	r, _, _ := procQueryFullProcessImageNameW.Call(
		uintptr(h), 0, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)))
	if r == 0 || size == 0 {
		return "", false
	}
	full := syscall.UTF16ToString(buf[:size])
	base := strings.ToLower(filepath.Base(full))
	base = strings.TrimSuffix(base, ".exe")
	if base == "" {
		return "", false
	}
	return base, true
}

var procQueryFullProcessImageNameW = modkernel32.NewProc("QueryFullProcessImageNameW")
