package compute

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// DedicatedVRAMReporter is the optional backend probe for a DEDICATED device-memory carve-out
// on an integrated tier: physical DRAM the firmware reserves for the GPU and that the OS does
// NOT count in MemTotal (amdgpu mem_info_vram_total on a Strix Halo with a BIOS UMA carve-out).
// It is distinct from the GTT window (mem_info_gtt_total), which is an aperture over system RAM
// and therefore already inside MemTotal. known=false means the carve-out is unreadable or
// ambiguous, and callers must judge physical capacity by MemTotal alone (fak#13668).
type DedicatedVRAMReporter interface {
	DedicatedVRAMCarveout() (bytes int64, known bool)
}

// DedicatedVRAMCarveoutInfo returns a backend's dedicated VRAM carve-out when it advertises
// DedicatedVRAMReporter, and known=false otherwise.
func DedicatedVRAMCarveoutInfo(b Backend) (bytes int64, known bool) {
	r, ok := b.(DedicatedVRAMReporter)
	if !ok {
		return 0, false
	}
	bytes, known = r.DedicatedVRAMCarveout()
	if !known || bytes <= 0 {
		return 0, false
	}
	return bytes, true
}

const amdPCIVendorID = "0x1002"

// hostDedicatedVRAMCarveout reads the host's amdgpu dedicated VRAM carve-out from sysfs.
func hostDedicatedVRAMCarveout() (int64, bool) {
	if runtime.GOOS != "linux" {
		return 0, false
	}
	return dedicatedVRAMCarveoutFromSysfs("/sys/class/drm", filepath.Glob, os.ReadFile)
}

// dedicatedVRAMCarveoutFromSysfs returns mem_info_vram_total of EXACTLY ONE amdgpu card under
// drmRoot. Zero or several AMD cards, or an unreadable or non-positive value, is known=false:
// the carve-out is never guessed, so an ambiguous host keeps the MemTotal-only bound.
func dedicatedVRAMCarveoutFromSysfs(drmRoot string, glob func(string) ([]string, error), readFile func(string) ([]byte, error)) (int64, bool) {
	cards, err := glob(filepath.Join(drmRoot, "card*"))
	if err != nil {
		return 0, false
	}
	var found int64
	n := 0
	for _, card := range cards {
		if strings.Contains(filepath.Base(card), "-") {
			continue
		}
		vendor, err := readFile(filepath.Join(card, "device", "vendor"))
		if err != nil || !strings.EqualFold(strings.TrimSpace(string(vendor)), amdPCIVendorID) {
			continue
		}
		n++
		raw, err := readFile(filepath.Join(card, "device", "mem_info_vram_total"))
		if err != nil {
			return 0, false
		}
		v, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
		if err != nil || v <= 0 {
			return 0, false
		}
		found = v
	}
	if n != 1 {
		return 0, false
	}
	return found, true
}
