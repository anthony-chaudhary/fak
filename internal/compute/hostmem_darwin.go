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
	if v := strings.TrimSpace(syscallGetenv("FAK_UP_MEMORY_BYTES")); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil && n > 0 {
			if avail := strings.TrimSpace(syscallGetenv("FAK_UP_AVAILABLE_BYTES")); avail != "" {
				if a, err := strconv.ParseInt(avail, 10, 64); err == nil && a > 0 {
					return uint64ToCapInt64(n), a, true
				}
			}
			return uint64ToCapInt64(n), int64((n * 80) / 100), true
		}
	}
	if avail := strings.TrimSpace(syscallGetenv("FAK_UP_AVAILABLE_BYTES")); avail != "" {
		if a, err := strconv.ParseInt(avail, 10, 64); err == nil && a > 0 {
			raw, err := syscall.Sysctl("hw.memsize")
			if err == nil && raw != "" {
				var buf [8]byte
				copy(buf[:], raw)
				totalBytes := binary.LittleEndian.Uint64(buf[:])
				if totalBytes > 0 {
					return uint64ToCapInt64(totalBytes), a, true
				}
			}
			return a, a, true
		}
	}

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
	cmd := exec.Command("/usr/bin/vm_stat")
	out, err := cmd.Output()
	if err != nil {
		out, err = exec.Command("vm_stat").Output()
	}
	if err == nil {
		if freeBytes, ok := parseVMStat(out); ok && freeBytes > 0 {
			return uint64ToCapInt64(totalBytes), int64(freeBytes), true
		}
	}
	return uint64ToCapInt64(totalBytes), FreeUnknown, true
}

func syscallGetenv(key string) string {
	v, _ := syscall.Getenv(key)
	return v
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
