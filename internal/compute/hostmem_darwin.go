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
		p := parseVMStatFields(out)
		if freeBytes, ok := p.availableBytes(int64(totalBytes)); ok && freeBytes > 0 {
			return uint64ToCapInt64(totalBytes), freeBytes, true
		}
	}
	return uint64ToCapInt64(totalBytes), FreeUnknown, true
}

func syscallGetenv(key string) string {
	v, _ := syscall.Getenv(key)
	return v
}

// vmStatPages is the classified page counts parsed from vm_stat output.
type vmStatPages struct {
	pageSize    uint64
	free        uint64
	speculative uint64
	inactive    uint64
	purgeable   uint64
	compressor  uint64
	wired       uint64
	foundFree   bool
}

// compressorReclaimFactor discounts compressor-held pages: each stored (compressed)
// page backs multiple logical pages, so counting every stored page as free would
// double-count. Half is a deliberately conservative reclaimable share.
const compressorReclaimFactor = 0.5

// parseVMStatFields scans vm_stat text into a classified page count. Pure.
func parseVMStatFields(out []byte) vmStatPages {
	p := vmStatPages{pageSize: 4096}
	if runtime.GOARCH == "arm64" {
		p.pageSize = 16384
	}

	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if idx := strings.Index(line, "page size of "); idx != -1 {
			rest := line[idx+len("page size of "):]
			if end := strings.Index(rest, " bytes"); end != -1 {
				if sz, err := strconv.ParseUint(strings.TrimSpace(rest[:end]), 10, 64); err == nil && sz > 0 {
					p.pageSize = sz
				}
			}
		}

		if val, ok := parseVMStatField(line, "Pages free:"); ok {
			p.free = val
			p.foundFree = true
		} else if val, ok := parseVMStatField(line, "Pages speculative:"); ok {
			p.speculative = val
		} else if val, ok := parseVMStatField(line, "Pages inactive:"); ok {
			p.inactive = val
		} else if val, ok := parseVMStatField(line, "Pages purgeable:"); ok {
			p.purgeable = val
		} else if val, ok := parseVMStatField(line, "Pages stored in compressor:"); ok {
			p.compressor = val
		} else if val, ok := parseVMStatField(line, "Pages wired down:"); ok {
			p.wired = val
		}
	}
	return p
}

// availablePages is the reclaimable page count: pages that are free, speculative,
// inactive, or purgeable, plus a discounted share of compressor-held pages. The
// kernel reclaims all of these under pressure; compressor pages are discounted by
// compressorReclaimFactor because each stored page backs multiple logical pages.
func (p vmStatPages) availablePages() uint64 {
	avail := satAddUint64(p.free, p.speculative)
	avail = satAddUint64(avail, p.inactive)
	avail = satAddUint64(avail, p.purgeable)
	avail = satAddUint64(avail, uint64(float64(p.compressor)*compressorReclaimFactor))
	return avail
}

// availableBytes converts the classified counts to a byte estimate, clamped so it
// can never exceed total - wired: wired memory is non-reclaimable, and reporting
// more than the machine's non-wired capacity would overcount and risk OOM. The
// clamp applies only when total > 0 and wired is known; otherwise the raw estimate
// stands. ok is false only when Pages free: was absent (vm_stat unavailable/short).
func (p vmStatPages) availableBytes(total int64) (int64, bool) {
	if !p.foundFree || p.pageSize == 0 {
		return 0, false
	}
	pages := p.availablePages()
	if pages > maxInt64Uint64/p.pageSize {
		return int64(maxInt64Uint64), true
	}
	freeBytes := pages * p.pageSize
	if total > 0 && p.wired > 0 {
		nonWired := uint64(total)
		if p.wired > maxInt64Uint64/p.pageSize {
			return 0, false
		}
		if wiredBytes := p.wired * p.pageSize; wiredBytes < nonWired {
			nonWired -= wiredBytes
		} else {
			nonWired = 0
		}
		if freeBytes > nonWired {
			freeBytes = nonWired
		}
	}
	if freeBytes == 0 {
		return 0, false
	}
	return int64(freeBytes), true
}

func satAddUint64(a, b uint64) uint64 {
	if a >= maxInt64Uint64 || b >= maxInt64Uint64 || a > maxInt64Uint64-b {
		return maxInt64Uint64
	}
	return a + b
}

func parseVMStat(out []byte) (int64, bool) {
	return parseVMStatFields(out).availableBytes(0)
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
