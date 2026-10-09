package compute

import (
	"os"
	"path"
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

// hostDedicatedVRAMCarveout observes only the selected integrated AMD device.
func hostDedicatedVRAMCarveout(backend Backend) (int64, bool) {
	return dedicatedVRAMCarveoutFromSysfs(backend, hostEnvironmentDeps{
		goos: runtime.GOOS, readFile: os.ReadFile, evalSymlinks: filepath.EvalSymlinks,
	})
}

// dedicatedVRAMCarveoutFromSysfs reads only the selected device's dedicated VRAM;
// GTT and other cards never contribute capacity. The caller owns the current
// initialized-device lifetime. Rechecking identity detects observed changes, but
// is not a lifetime lease or an atomic guarantee against hot unplug.
func dedicatedVRAMCarveoutFromSysfs(backend Backend, deps hostEnvironmentDeps) (int64, bool) {
	if deps.goos != "linux" {
		return 0, false
	}
	typed, ok := backend.(interface{ VulkanPhysicalDeviceType() uint32 })
	if !ok || typed.VulkanPhysicalDeviceType() != 1 {
		return 0, false
	}
	snapshot, available, err := CaptureBackendExecutionSnapshot(backend)
	if err != nil || !available || snapshot.Identity.Backend != "vulkan" {
		return 0, false
	}
	vendor, device, err := parseBackendPCIIdentity(snapshot.Identity.Driver)
	if err != nil || vendor != amdPCIVendorID {
		return 0, false
	}
	binding, err := bindSelectedVulkanDRMDevice(backend, deps, vendor, device)
	if err != nil {
		return 0, false
	}
	raw, err := deps.readFile(path.Join(binding.devicePath, "mem_info_vram_total"))
	if err != nil {
		return 0, false
	}
	bytes, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil || bytes <= 0 {
		return 0, false
	}
	closingBinding, err := bindSelectedVulkanDRMDevice(backend, deps, vendor, device)
	if err != nil || closingBinding != binding {
		return 0, false
	}
	closingSnapshot, available, err := CaptureBackendExecutionSnapshot(backend)
	if err != nil || !available || closingSnapshot.Identity != snapshot.Identity || typed.VulkanPhysicalDeviceType() != 1 {
		return 0, false
	}
	return bytes, true
}
