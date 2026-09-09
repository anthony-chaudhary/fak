// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"fmt"
	"os"
	"unsafe"
)

const (
	// DefaultDRMRenderPath defines the primary Linux DRM render node for AMDGPU GFX1151.
	DefaultDRMRenderPath = "/dev/dri/renderD128"

	// DefaultDRMCard0Path defines the primary DRM card node.
	DefaultDRMCard0Path = "/dev/dri/card0"

	// DefaultDRMCard1Path defines the secondary DRM card node.
	DefaultDRMCard1Path = "/dev/dri/card1"

	// DRM ioctl encoding constants matching Linux <drm/drm.h> and <drm/amdgpu_drm.h>.
	DRM_IOCTL_BASE   = 'd'  // 0x64
	DRM_COMMAND_BASE = 0x40 // 64

	// DRM AMDGPU ioctl command offsets.
	DRM_AMDGPU_GEM_CREATE = 0x00
	DRM_AMDGPU_GEM_MMAP   = 0x01

	// Precomputed Linux ioctl command numbers (_IOWR).
	// DRM_IOCTL_AMDGPU_GEM_CREATE = _IOWR('d', 0x40, sizeof(drm_amdgpu_gem_create)=32) = 0xc0206440
	DRM_IOCTL_AMDGPU_GEM_CREATE uintptr = 0xc0206440

	// DRM_IOCTL_AMDGPU_GEM_MMAP = _IOWR('d', 0x41, sizeof(drm_amdgpu_gem_mmap)=8) = 0xc0086441
	DRM_IOCTL_AMDGPU_GEM_MMAP uintptr = 0xc0086441

	// AMDGPU GEM memory domains matching Linux <drm/amdgpu_drm.h>.
	AMDGPU_GEM_DOMAIN_CPU  uint64 = 0x1
	AMDGPU_GEM_DOMAIN_VRAM uint64 = 0x4
	AMDGPU_GEM_DOMAIN_GDS  uint64 = 0x8
	AMDGPU_GEM_DOMAIN_GWS  uint64 = 0x10
	AMDGPU_GEM_DOMAIN_OA   uint64 = 0x20

	// AMDGPU GEM allocation flags matching Linux <drm/amdgpu_drm.h>.
	AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED uint64 = (1 << 0)
	AMDGPU_GEM_CREATE_NO_CPU_ACCESS       uint64 = (1 << 1)
	AMDGPU_GEM_CREATE_VM_ALWAYS_VALID     uint64 = (1 << 9)
	AMDGPU_GEM_CREATE_EXPLICIT_SYNC       uint64 = (1 << 10)
	AMDGPU_GEM_CREATE_ENCRYPTED           uint64 = (1 << 11)
	AMDGPU_GEM_CREATE_TGID_AFFINITY       uint64 = (1 << 12)
)

// DRMAMDGPUGEMCreateIn matches struct drm_amdgpu_gem_create_in (32 bytes).
type DRMAMDGPUGEMCreateIn struct {
	BOSize      uint64 `json:"bo_size"`
	Alignment   uint64 `json:"alignment"`
	Domains     uint64 `json:"domains"`
	DomainFlags uint64 `json:"domain_flags"`
}

// DRMAMDGPUGEMCreateOut matches struct drm_amdgpu_gem_create_out (8 bytes).
type DRMAMDGPUGEMCreateOut struct {
	Handle uint32 `json:"handle"`
	Pad    uint32 `json:"_pad"`
}

// DRMAMDGPUGEMCreateArgs represents union drm_amdgpu_gem_create (32 bytes).
// In Linux C ABI:
//
//	union drm_amdgpu_gem_create {
//	    struct drm_amdgpu_gem_create_in  in;
//	    struct drm_amdgpu_gem_create_out out;
//	};
type DRMAMDGPUGEMCreateArgs struct {
	BOSize      uint64 `json:"bo_size"`
	Alignment   uint64 `json:"alignment"`
	Domains     uint64 `json:"domains"`
	DomainFlags uint64 `json:"domain_flags"`
}

// Handle extracts the returned GEM handle from union offset 0 upon successful ioctl.
func (a *DRMAMDGPUGEMCreateArgs) Handle() uint32 {
	if a == nil {
		return 0
	}
	return *(*uint32)(unsafe.Pointer(&a.BOSize))
}

// SetHandle sets the returned GEM handle for mock/simulation testing.
func (a *DRMAMDGPUGEMCreateArgs) SetHandle(handle uint32) {
	if a != nil {
		*(*uint32)(unsafe.Pointer(&a.BOSize)) = handle
	}
}

// DRMAMDGPUGEMMmapIn matches struct drm_amdgpu_gem_mmap_in (8 bytes).
type DRMAMDGPUGEMMmapIn struct {
	Handle uint32 `json:"handle"`
	Pad    uint32 `json:"_pad"`
}

// DRMAMDGPUGEMMmapOut matches struct drm_amdgpu_gem_mmap_out (8 bytes).
type DRMAMDGPUGEMMmapOut struct {
	AddrPtr uint64 `json:"addr_ptr"`
}

// DRMAMDGPUGEMMmapArgs represents union drm_amdgpu_gem_mmap (8 bytes).
// In Linux C ABI:
//
//	union drm_amdgpu_gem_mmap {
//	    struct drm_amdgpu_gem_mmap_in  in;
//	    struct drm_amdgpu_gem_mmap_out out;
//	};
type DRMAMDGPUGEMMmapArgs struct {
	Handle uint32 `json:"handle"`
	Pad    uint32 `json:"_pad"`
}

// AddrPtr extracts the returned mmap offset from union offset 0 upon successful ioctl.
func (m *DRMAMDGPUGEMMmapArgs) AddrPtr() uint64 {
	if m == nil {
		return 0
	}
	return *(*uint64)(unsafe.Pointer(m))
}

// SetAddrPtr sets the returned mmap offset for mock/simulation testing.
func (m *DRMAMDGPUGEMMmapArgs) SetAddrPtr(addr uint64) {
	if m != nil {
		*(*uint64)(unsafe.Pointer(m)) = addr
	}
}

// KFDAllocMemoryOfGPUArgs matches struct kfd_ioctl_alloc_memory_of_gpu_args (40 bytes)
// from Linux <linux/kfd_ioctl.h>.
type KFDAllocMemoryOfGPUArgs struct {
	VAAddr     uint64 `json:"va_addr"`
	Size       uint64 `json:"size"`
	Handle     uint64 `json:"handle"`
	MmapOffset uint64 `json:"mmap_offset"`
	GPUID      uint32 `json:"gpu_id"`
	Flags      uint32 `json:"flags"`
}

// DRMAccessProbeResult contains telemetry from checking access permissions
// to Linux DRM device nodes.
type DRMAccessProbeResult struct {
	Path           string `json:"path"`
	Available      bool   `json:"available"`
	IsDeviceNode   bool   `json:"is_device_node"`
	Readable       bool   `json:"readable"`
	Writable       bool   `json:"writable"`
	Error          string `json:"error,omitempty"`
	Recommendation string `json:"recommendation,omitempty"`
}

// ProbeDRMRenderNode checks whether the specified DRM render node exists and
// is accessible for read/write ioctl operations. If inaccessible, provides
// actionable remediation guidance.
func ProbeDRMRenderNode(path string) DRMAccessProbeResult {
	if path == "" {
		path = DefaultDRMRenderPath
	}

	res := DRMAccessProbeResult{Path: path}
	fi, err := os.Stat(path)
	if err != nil {
		res.Error = err.Error()
		res.Recommendation = fmt.Sprintf("DRM node %s missing; verify AMDGPU kernel driver is loaded (modprobe amdgpu) or run in simulation mode", path)
		return res
	}

	if fi.IsDir() {
		res.Error = fmt.Sprintf("%s is a directory, not a device node", path)
		res.Recommendation = "specify valid DRM character device path such as /dev/dri/renderD128"
		return res
	}

	if (fi.Mode() & os.ModeDevice) != 0 {
		res.IsDeviceNode = true
	}

	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		res.Error = err.Error()
		res.Recommendation = "add current user to video/render group: sudo usermod -aG render $USER"
		return res
	}
	_ = f.Close()

	res.Available = true
	res.Readable = true
	res.Writable = true
	return res
}
