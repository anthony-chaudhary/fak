package compute

import (
	"errors"
	"fmt"
	"strings"
)

// Host-VA / device-VA equivalence invariant for pinned host slabs.
//
// Root-cause discipline adapted from SGLang PR #38967 ("[ROCm] Fix HiCache
// kernel backend GPU fault by using hipHostMalloc for host KV pool"): on
// HIP/ROCm, hipHostRegister maps host memory at a device virtual address
// that may differ from the host VA, so a device-visible pointer table that
// stores the host data_ptr() faults when device code dereferences it;
// allocations backed by hipHostMalloc guarantee the same VA on both sides.
// CUDA is unaffected: cudaHostAlloc and cudaHostRegister already expose a
// unified VA.
//
// Invariant: any host memory whose pointer is stored in a device-visible
// table and dereferenced by device code must be backed by the allocation
// primitive that guarantees host/device VA equivalence on that platform —
// DeviceDerefHostPrimitive names it per platform. This covers fak's pinned
// host surfaces: the AllocPinnedHost ledger kind, KFD non-pageable APU
// allocations, DirectSlabAllocator registrations, and StorageModeHostAlloc
// doorbell buffers. No allocation behavior changes on any platform; this
// leaf only states and machine-checks the contract.

// HostVAClass classifies whether an allocation primitive guarantees that the
// host virtual address of the backing memory equals the virtual address
// observed by device code.
type HostVAClass int

const (
	// HostVADivergent is the zero value: absent proof of equivalence, a host
	// pointer must be treated as device-dereference-unsafe.
	HostVADivergent HostVAClass = iota
	// HostVAEquivalent: the host pointer value is the device pointer value;
	// device-visible tables may store it and device code may dereference it.
	HostVAEquivalent
)

func (c HostVAClass) String() string {
	if c == HostVAEquivalent {
		return "equivalent"
	}
	return "divergent"
}

// Platform spellings the closed contract recognizes (case-insensitive).
const (
	PlatformCUDA = "cuda"
	PlatformHIP  = "hip"
	PlatformROCm = "rocm"
)

var (
	// ErrHostVAContractUnknown reports a platform/primitive pair the closed
	// contract does not cover. Unknown is never treated as safe.
	ErrHostVAContractUnknown = errors.New("compute: host VA equivalence contract unknown")

	// ErrHostVADivergentBacking reports that a named backing primitive does
	// not guarantee host/device VA equivalence on the named platform, so it
	// must not back host memory whose pointer is dereferenced by device code.
	ErrHostVADivergentBacking = errors.New("compute: host backing primitive diverges from device VA")
)

func normalizeHostVAPlatform(platform string) string {
	switch strings.ToLower(strings.TrimSpace(platform)) {
	case PlatformCUDA:
		return PlatformCUDA
	case PlatformHIP, PlatformROCm:
		return PlatformHIP
	default:
		return ""
	}
}

// HostVAEquivalence reports the VA-equivalence class of allocating host
// memory with the named primitive on the named platform. Pairs outside the
// closed contract return ErrHostVAContractUnknown.
func HostVAEquivalence(platform, primitive string) (HostVAClass, error) {
	p := normalizeHostVAPlatform(platform)
	if p == "" {
		return 0, fmt.Errorf("%w: platform %q", ErrHostVAContractUnknown, platform)
	}
	// Primitive names are exact C API spellings; only surrounding
	// whitespace is tolerated.
	switch strings.TrimSpace(primitive) {
	case "cudaHostAlloc", "cudaHostRegister":
		if p != PlatformCUDA {
			return 0, fmt.Errorf("%w: %s on %s", ErrHostVAContractUnknown, primitive, p)
		}
		return HostVAEquivalent, nil
	case "hipHostMalloc":
		if p != PlatformHIP {
			return 0, fmt.Errorf("%w: %s on %s", ErrHostVAContractUnknown, primitive, p)
		}
		return HostVAEquivalent, nil
	case "hipHostRegister":
		if p != PlatformHIP {
			return 0, fmt.Errorf("%w: %s on %s", ErrHostVAContractUnknown, primitive, p)
		}
		return HostVADivergent, nil
	default:
		return 0, fmt.Errorf("%w: %s on %s", ErrHostVAContractUnknown, primitive, p)
	}
}

// DeviceDerefHostPrimitive names the single allocation primitive that must
// back host memory whose pointer is stored in a device-visible table and
// dereferenced by device code on the named platform. This is the invariant's
// per-platform allocator-selection default.
func DeviceDerefHostPrimitive(platform string) (string, error) {
	switch normalizeHostVAPlatform(platform) {
	case PlatformCUDA:
		return "cudaHostAlloc", nil
	case PlatformHIP:
		return "hipHostMalloc", nil
	default:
		return "", fmt.Errorf("%w: platform %q", ErrHostVAContractUnknown, platform)
	}
}

// CheckDeviceVisibleHostBacking verifies that every named backing primitive
// guarantees host/device VA equivalence on the named platform. It returns
// ErrHostVADivergentBacking when any primitive is known divergent and
// ErrHostVAContractUnknown when the platform or any primitive is not covered
// by the closed contract. An empty list is vacuously safe.
func CheckDeviceVisibleHostBacking(platform string, primitives ...string) error {
	for _, primitive := range primitives {
		class, err := HostVAEquivalence(platform, primitive)
		if err != nil {
			return err
		}
		if class == HostVADivergent {
			return fmt.Errorf("%w: %s on %s", ErrHostVADivergentBacking, primitive, platform)
		}
	}
	return nil
}
