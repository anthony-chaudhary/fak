//go:build darwin

package compute

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

func hostSystemMemory() (total, free int64, known bool) {
	raw, err := syscall.Sysctl("hw.memsize")
	if err != nil || raw == "" {
		return 0, FreeUnknown, false
	}
	var buf [8]byte
	copy(buf[:], raw)
	totalBytes := binary.LittleEndian.Uint64(buf[:])
	if totalBytes == 0 {
		return 0, FreeUnknown, false
	}
	if out, err := exec.Command("vm_stat").Output(); err == nil {
		if freeBytes, ok := parseVMStat(out); ok && freeBytes > 0 {
			return uint64ToCapInt64(totalBytes), int64(freeBytes), true
		}
	}
	return uint64ToCapInt64(totalBytes), FreeUnknown, true
}

func parseVMStat(out []byte) (int64, bool) {
	pageSize := uint64(4096)
	if runtime.GOARCH == "arm64" {
		pageSize = 16384
	}

	var pagesFree, pagesSpeculative, pagesInactive uint64
	var foundFree bool

	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if idx := strings.Index(line, "page size of "); idx != -1 {
			rest := line[idx+len("page size of "):]
			if end := strings.Index(rest, " bytes"); end != -1 {
				if sz, err := strconv.ParseUint(strings.TrimSpace(rest[:end]), 10, 64); err == nil && sz > 0 {
					pageSize = sz
				}
			}
		}

		if val, ok := parseVMStatField(line, "Pages free:"); ok {
			pagesFree = val
			foundFree = true
		} else if val, ok := parseVMStatField(line, "Pages speculative:"); ok {
			pagesSpeculative = val
		} else if val, ok := parseVMStatField(line, "Pages inactive:"); ok {
			pagesInactive = val
		}
	}

	if !foundFree {
		return 0, false
	}

	availablePages := pagesFree + pagesSpeculative + pagesInactive
	if pageSize > 0 && availablePages > maxInt64Uint64/pageSize {
		return int64(maxInt64Uint64), true
	}
	freeBytes := availablePages * pageSize
	if freeBytes == 0 {
		return 0, false
	}
	return int64(freeBytes), true
}

func parseVMStatField(line, prefix string) (uint64, bool) {
	if !strings.HasPrefix(line, prefix) {
		return 0, false
	}
	val := strings.TrimPrefix(line, prefix)
	val = strings.TrimSpace(val)
	val = strings.TrimSuffix(val, ".")
	cnt, err := strconv.ParseUint(val, 10, 64)
	if err != nil {
		return 0, false
	}
	return cnt, true
}
