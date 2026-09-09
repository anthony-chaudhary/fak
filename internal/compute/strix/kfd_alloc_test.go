package strix

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

func TestKFDTTMAllocation(t *testing.T) {
	const allocSize = 4 * 1024 * 1024 // 4 MiB test buffer

	buf, err := AllocateGTT(allocSize)
	if err != nil {
		t.Fatalf("expected AllocateGTT to succeed, got %v", err)
	}
	defer func() {
		_ = buf.Free()
	}()

	// 1. Verify 2MB hugepage alignment on Zen 5 MMU
	if !buf.IsHugepageAligned() {
		t.Errorf("expected buffer host pointer (0x%x) to be 2MB aligned", buf.HostPtr)
	}

	// 2. Verify strict zero-copy pointer identity (HostPtr == DevicePtr)
	if !buf.IsZeroCopy() {
		t.Errorf("expected zero-copy pointer identity: hostPtr=0x%x != devPtr=0x%x", buf.HostPtr, buf.DevicePtr)
	}

	// 3. Verify eviction protection flags
	if !buf.IsPinned() {
		t.Errorf("expected buffer to be eviction-pinned with AMDGPU_GEM_CREATE_NO_EVICT")
	}

	if (buf.Flags & AMDGPU_GEM_CREATE_CPU_GTT_USWC) == 0 {
		t.Errorf("expected AMDGPU_GEM_CREATE_CPU_GTT_USWC flag to be set")
	}

	if (buf.KFDFlags & KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE) == 0 {
		t.Errorf("expected KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE flag to be set")
	}

	// 4. Verify read/write memory coherency
	slice := buf.Slice()
	if len(slice) != allocSize {
		t.Fatalf("expected slice length %d, got %d", allocSize, len(slice))
	}

	testPattern := []byte{0xde, 0xad, 0xbe, 0xef, 0x42}
	copy(slice[:len(testPattern)], testPattern)

	for i, b := range testPattern {
		if slice[i] != b {
			t.Errorf("byte mismatch at %d: expected 0x%x, got 0x%x", i, b, slice[i])
		}
	}

	// 5. Verify UnsafePointer and Len
	if buf.UnsafePointer() == nil {
		t.Errorf("expected non-nil UnsafePointer")
	}
	if buf.Len() != allocSize {
		t.Errorf("expected Len %d, got %d", allocSize, buf.Len())
	}

	// 6. Verify clean free
	if err := buf.Free(); err != nil {
		t.Fatalf("unexpected error on Free: %v", err)
	}
	if buf.Slice() != nil {
		t.Errorf("expected nil slice after free")
	}
	if buf.UnsafePointer() != nil {
		t.Errorf("expected nil UnsafePointer after free")
	}

	// Double-free should return ErrGTTBufferClosed
	if err := buf.Free(); !errors.Is(err, ErrGTTBufferClosed) {
		t.Errorf("expected ErrGTTBufferClosed on double free, got %v", err)
	}
}

func TestKFDTTMAllocation_Boundaries(t *testing.T) {
	// Size <= 0 must fail
	if _, err := AllocateGTT(0); !errors.Is(err, ErrInvalidGTTSize) {
		t.Errorf("expected ErrInvalidGTTSize for size 0, got %v", err)
	}
	if _, err := AllocateGTT(-1024); !errors.Is(err, ErrInvalidGTTSize) {
		t.Errorf("expected ErrInvalidGTTSize for negative size, got %v", err)
	}

	// Size exceeding 116 GiB must fail
	if _, err := AllocateGTT(MaxGTTAllocPoolBytes + 1); !errors.Is(err, ErrInvalidGTTSize) {
		t.Errorf("expected ErrInvalidGTTSize for > 116 GiB, got %v", err)
	}
}

func TestKFDTTMAllocation_UnpinnedOption(t *testing.T) {
	buf, err := AllocateGTTPinned(1024*1024, false)
	if err != nil {
		t.Fatalf("failed to allocate unpinned GTT: %v", err)
	}
	defer func() { _ = buf.Free() }()

	if buf.IsPinned() {
		t.Errorf("expected unpinned buffer when noEvict=false")
	}
	if (buf.Flags & AMDGPU_GEM_CREATE_NO_EVICT) != 0 {
		t.Errorf("expected AMDGPU_GEM_CREATE_NO_EVICT to NOT be set when noEvict=false")
	}
}

func TestKFDIoctlDefinitions(t *testing.T) {
	// Verify Linux ioctl numbers computed via _IOWR / _IOW on 'K' = 0x4B
	const expectedAlloc = uintptr(0xc0284b16)
	const expectedFree = uintptr(0x40084b17)
	const expectedMap = uintptr(0xc0184b18)

	if AMDKFD_IOCTL_ALLOC_MEM_OF_GPU != expectedAlloc {
		t.Errorf("expected AMDKFD_IOCTL_ALLOC_MEM_OF_GPU = 0x%x, got 0x%x", expectedAlloc, AMDKFD_IOCTL_ALLOC_MEM_OF_GPU)
	}
	if AMDKFD_IOCTL_FREE_MEM != expectedFree {
		t.Errorf("expected AMDKFD_IOCTL_FREE_MEM = 0x%x, got 0x%x", expectedFree, AMDKFD_IOCTL_FREE_MEM)
	}
	if AMDKFD_IOCTL_MAP_MEMORY_TO_GPU != expectedMap {
		t.Errorf("expected AMDKFD_IOCTL_MAP_MEMORY_TO_GPU = 0x%x, got 0x%x", expectedMap, AMDKFD_IOCTL_MAP_MEMORY_TO_GPU)
	}

	// Verify struct sizes matching Linux kernel C structs byte-for-byte
	if sz := unsafe.Sizeof(KFDAllocMemArgs{}); sz != 40 {
		t.Errorf("expected sizeof(KFDAllocMemArgs) == 40, got %d", sz)
	}
	if sz := unsafe.Sizeof(KFDFreeMemArgs{}); sz != 8 {
		t.Errorf("expected sizeof(KFDFreeMemArgs) == 8, got %d", sz)
	}
	if sz := unsafe.Sizeof(KFDMapMemoryToGPUArgs{}); sz != 24 {
		t.Errorf("expected sizeof(KFDMapMemoryToGPUArgs) == 24, got %d", sz)
	}

	// Verify aperture constants
	const expectedAperture int64 = 120 * 1024 * 1024 * 1024
	const expectedMin64GB int64 = 64 * 1024 * 1024 * 1024

	if StrixHaloTotalApertureBytes != expectedAperture {
		t.Errorf("expected StrixHaloTotalApertureBytes = %d, got %d", expectedAperture, StrixHaloTotalApertureBytes)
	}
	if MinUnifiedMapping64GB != expectedMin64GB {
		t.Errorf("expected MinUnifiedMapping64GB = %d, got %d", expectedMin64GB, MinUnifiedMapping64GB)
	}
}

func TestValidateKFDUnifiedMemoryMapping_64GB(t *testing.T) {
	res, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
	if err != nil {
		t.Fatalf("expected ValidateKFDUnifiedMemoryMapping(64GB) to succeed, got %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil KFDProbeResult")
	}

	if !res.AllocSucceeded {
		t.Errorf("expected AllocSucceeded to be true")
	}
	if res.AllocSizeBytes != MinUnifiedMapping64GB {
		t.Errorf("expected AllocSizeBytes %d, got %d", MinUnifiedMapping64GB, res.AllocSizeBytes)
	}
	if res.ApertureBytes != StrixHaloTotalApertureBytes {
		t.Errorf("expected ApertureBytes %d, got %d", StrixHaloTotalApertureBytes, res.ApertureBytes)
	}

	// Zero-copy verification: HostPtr == DevicePtr
	if !res.ZeroCopyVerified {
		t.Errorf("expected ZeroCopyVerified to be true")
	}
	if res.HostPtr == 0 || res.HostPtr != res.DevicePtr {
		t.Errorf("expected hostPtr (0x%x) == devicePtr (0x%x) != 0", res.HostPtr, res.DevicePtr)
	}

	// 2MB hugepage alignment
	if !res.HugepageAligned {
		t.Errorf("expected HugepageAligned to be true")
	}
	if (res.HostPtr % uintptr(HugepageSize2MB)) != 0 {
		t.Errorf("expected hostPtr (0x%x) to be 2MB aligned", res.HostPtr)
	}

	// Eviction pinning flags
	if !res.NoEvictPinned {
		t.Errorf("expected NoEvictPinned to be true")
	}

	// No page faults or TTM migration stalls across unified APU DRAM
	if !res.NoPageFaults {
		t.Errorf("expected NoPageFaults to be true")
	}
	if !res.NoTTMMigrationStalls {
		t.Errorf("expected NoTTMMigrationStalls to be true")
	}

	if !res.IsValid() {
		t.Errorf("expected IsValid() to return true")
	}
}

func TestValidateKFDUnifiedMemoryMapping_120GB_Aperture(t *testing.T) {
	// Test allocation against the complete 120GB aperture
	res, err := ValidateKFDUnifiedMemoryMapping(StrixHaloTotalApertureBytes)
	if err != nil {
		t.Fatalf("expected allocation against 120GB aperture to succeed, got %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil KFDProbeResult")
	}
	if !res.AllocSucceeded {
		t.Errorf("expected AllocSucceeded to be true")
	}
	if res.AllocSizeBytes != StrixHaloTotalApertureBytes {
		t.Errorf("expected AllocSizeBytes %d, got %d", StrixHaloTotalApertureBytes, res.AllocSizeBytes)
	}
	if !res.ZeroCopyVerified || !res.HugepageAligned || !res.NoEvictPinned {
		t.Errorf("unified memory invariants failed for 120GB mapping")
	}
	if !res.NoPageFaults || !res.NoTTMMigrationStalls {
		t.Errorf("expected no page faults and no TTM migration stalls")
	}
}

func TestValidateKFDUnifiedMemoryMapping_Boundaries(t *testing.T) {
	testCases := []struct {
		name      string
		sizeBytes int64
	}{
		{"zero size", 0},
		{"negative size -1", -1},
		{"negative size -1MB", -1024 * 1024},
		{"exceeds 120GB aperture by 1 byte", StrixHaloTotalApertureBytes + 1},
		{"exceeds 120GB aperture by 1GB", StrixHaloTotalApertureBytes + 1024*1024*1024},
		{"exceeds 120GB aperture by 120GB", StrixHaloTotalApertureBytes * 2},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := ValidateKFDUnifiedMemoryMapping(tc.sizeBytes)
			if err == nil {
				t.Fatalf("expected error for size %d, got nil (res: %+v)", tc.sizeBytes, res)
			}
			if !errors.Is(err, ErrInvalidGTTSize) {
				t.Errorf("expected error wrapping ErrInvalidGTTSize, got %v", err)
			}
			if res != nil {
				t.Errorf("expected nil result on boundary rejection, got %+v", res)
			}
		})
	}
}

func TestValidateKFDUnifiedMemoryMapping_Invariants(t *testing.T) {
	t.Run("ZeroCopyMismatch", func(t *testing.T) {
		SetKFDAllocHook(func(args *KFDAllocMemArgs) (uintptr, uintptr, error) {
			hPtr := SimulatedUnifiedVABase
			dPtr := SimulatedUnifiedVABase + 4096 // Diverging device pointer
			return hPtr, dPtr, nil
		})
		defer ResetKFDAllocHook()

		res, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
		if err == nil {
			t.Fatal("expected error on zero-copy mismatch, got nil")
		}
		if !errors.Is(err, ErrZeroCopyPointerMismatch) {
			t.Errorf("expected error wrapping ErrZeroCopyPointerMismatch, got %v", err)
		}
		if res == nil {
			t.Fatal("expected non-nil result")
		}
		if res.ZeroCopyVerified {
			t.Errorf("expected ZeroCopyVerified to be false")
		}
		if res.NoPageFaults {
			t.Errorf("expected NoPageFaults to be false on zero-copy mismatch")
		}
		if res.NoTTMMigrationStalls {
			t.Errorf("expected NoTTMMigrationStalls to be false on zero-copy mismatch")
		}
		if res.IsValid() {
			t.Errorf("expected IsValid() to be false")
		}
	})

	t.Run("HugepageMisaligned", func(t *testing.T) {
		SetKFDAllocHook(func(args *KFDAllocMemArgs) (uintptr, uintptr, error) {
			hPtr := SimulatedUnifiedVABase + 4096 // Not 2MB aligned
			return hPtr, hPtr, nil
		})
		defer ResetKFDAllocHook()

		res, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
		if err == nil {
			t.Fatal("expected error on hugepage misaligned pointer, got nil")
		}
		if !strings.Contains(err.Error(), "hugepage aligned") {
			t.Errorf("expected error about hugepage alignment, got %v", err)
		}
		if res.HugepageAligned {
			t.Errorf("expected HugepageAligned to be false")
		}
		if res.NoPageFaults {
			t.Errorf("expected NoPageFaults to be false on unaligned mapping")
		}
	})

	t.Run("MissingEvictionPinning", func(t *testing.T) {
		SetKFDAllocHook(func(args *KFDAllocMemArgs) (uintptr, uintptr, error) {
			args.Flags &^= KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE // Strip eviction protection
			return SimulatedUnifiedVABase, SimulatedUnifiedVABase, nil
		})
		defer ResetKFDAllocHook()

		res, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
		if err == nil {
			t.Fatal("expected error on missing eviction pinning, got nil")
		}
		if !strings.Contains(err.Error(), "eviction protection") {
			t.Errorf("expected error about eviction protection, got %v", err)
		}
		if res.NoEvictPinned {
			t.Errorf("expected NoEvictPinned to be false")
		}
	})

	t.Run("AllocFailure", func(t *testing.T) {
		SetKFDAllocHook(func(args *KFDAllocMemArgs) (uintptr, uintptr, error) {
			return 0, 0, errors.New("simulated ioctl ENOMEM")
		})
		defer ResetKFDAllocHook()

		res, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
		if err == nil {
			t.Fatal("expected error on alloc failure, got nil")
		}
		if !strings.Contains(err.Error(), "simulated ioctl ENOMEM") {
			t.Errorf("expected error to contain inner message, got %v", err)
		}
		if res.AllocSucceeded {
			t.Errorf("expected AllocSucceeded to be false")
		}
	})
}

func TestValidateKFDUnifiedMemoryMapping_CustomValidator(t *testing.T) {
	called := false
	mockValidator := KFDMappingValidatorFunc(func(sizeBytes int64) (*KFDProbeResult, error) {
		called = true
		return &KFDProbeResult{
			AllocSizeBytes:       sizeBytes,
			ApertureBytes:        StrixHaloTotalApertureBytes,
			AllocSucceeded:       true,
			ZeroCopyVerified:     true,
			HugepageAligned:      true,
			NoEvictPinned:        true,
			NoPageFaults:         true,
			NoTTMMigrationStalls: true,
			HostPtr:              0x1000000000,
			DevicePtr:            0x1000000000,
		}, nil
	})

	SetKFDMappingValidator(mockValidator)
	defer ResetKFDMappingValidator()

	res, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
	if err != nil {
		t.Fatalf("expected custom validator to succeed, got %v", err)
	}
	if !called {
		t.Errorf("expected custom validator to be invoked")
	}
	if res.HostPtr != 0x1000000000 {
		t.Errorf("expected HostPtr from custom validator, got 0x%x", res.HostPtr)
	}

	// Reset and verify default validator works
	ResetKFDMappingValidator()
	resDefault, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
	if err != nil {
		t.Fatalf("expected default validator to succeed, got %v", err)
	}
	if resDefault.HostPtr != SimulatedUnifiedVABase {
		t.Errorf("expected default SimulatedUnifiedVABase, got 0x%x", resDefault.HostPtr)
	}
}

func TestCheckHSAInitAvailable(t *testing.T) {
	t.Run("DefaultSearch", func(t *testing.T) {
		// Should execute cleanly without panic
		avail, paths := CheckHSAInitAvailable()
		t.Logf("CheckHSAInitAvailable default: available=%v, paths=%v", avail, paths)
	})

	t.Run("DirectorySearch", func(t *testing.T) {
		tmpDir := t.TempDir()
		libDir := filepath.Join(tmpDir, "lib")
		if err := os.MkdirAll(libDir, 0755); err != nil {
			t.Fatalf("failed to create temp lib dir: %v", err)
		}

		hsaLib := filepath.Join(libDir, "libhsa-runtime64.so.1")
		hipLib := filepath.Join(libDir, "libamdhip64.so.6")

		if err := os.WriteFile(hsaLib, []byte("stub-hsa"), 0644); err != nil {
			t.Fatalf("failed to write stub hsa: %v", err)
		}
		if err := os.WriteFile(hipLib, []byte("stub-hip"), 0644); err != nil {
			t.Fatalf("failed to write stub hip: %v", err)
		}

		t.Setenv("ROCM_PATH", tmpDir)

		avail, paths := CheckHSAInitAvailable()
		if !avail {
			t.Errorf("expected CheckHSAInitAvailable to return true with staged ROCM_PATH")
		}
		if len(paths) < 2 {
			t.Errorf("expected at least 2 found paths, got %v", paths)
		}
	})

	t.Run("HookInjection", func(t *testing.T) {
		mockPaths := []string{"/opt/rocm/lib/libhsa-runtime64.so.1", "/opt/rocm/lib/libamdhip64.so.6"}
		SetHSAProbeHook(func() (bool, []string) {
			return true, mockPaths
		})
		defer ResetHSAProbeHook()

		avail, paths := CheckHSAInitAvailable()
		if !avail {
			t.Errorf("expected hook available to be true")
		}
		if len(paths) != 2 || paths[0] != mockPaths[0] {
			t.Errorf("expected hook paths %v, got %v", mockPaths, paths)
		}

		ResetHSAProbeHook()
	})
}

func TestCheckKFDAvailability(t *testing.T) {
	t.Run("DefaultPath", func(t *testing.T) {
		avail, err := CheckKFDAvailability("")
		t.Logf("CheckKFDAvailability default: avail=%v, err=%v", avail, err)
	})

	t.Run("NonExistentPath", func(t *testing.T) {
		avail, err := CheckKFDAvailability(filepath.Join(t.TempDir(), "nonexistent_dev_kfd"))
		if avail {
			t.Errorf("expected non-existent path to return available=false")
		}
		if err == nil {
			t.Errorf("expected error for non-existent path")
		}
	})

	t.Run("ExistingFile", func(t *testing.T) {
		tmpFile := filepath.Join(t.TempDir(), "test_kfd_node")
		if err := os.WriteFile(tmpFile, []byte("stub"), 0644); err != nil {
			t.Fatalf("failed to create temp file: %v", err)
		}
		avail, err := CheckKFDAvailability(tmpFile)
		if !avail || err != nil {
			t.Errorf("expected accessible temp file to return true, got avail=%v, err=%v", avail, err)
		}
	})

	t.Run("HookInjection", func(t *testing.T) {
		SetKFDAvailabilityHook(func(path string) (bool, error) {
			if path == "/mock/kfd" {
				return true, nil
			}
			return false, errors.New("mock not found")
		})
		defer ResetKFDAvailabilityHook()

		avail, err := CheckKFDAvailability("/mock/kfd")
		if !avail || err != nil {
			t.Errorf("expected hook to return true, got avail=%v, err=%v", avail, err)
		}

		availBad, errBad := CheckKFDAvailability("/other/path")
		if availBad || errBad == nil {
			t.Errorf("expected hook to return false, got avail=%v, err=%v", availBad, errBad)
		}

		ResetKFDAvailabilityHook()
	})
}

func TestKFDAllocMemArgs_JSONSerialization(t *testing.T) {
	args := KFDAllocMemArgs{
		VAAddr:     0x7f0000000000,
		Size:       uint64(MinUnifiedMapping64GB),
		Handle:     0x10001,
		MmapOffset: 0x200000,
		GPUID:      0,
		Flags:      KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE,
	}

	data, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("failed to marshal KFDAllocMemArgs: %v", err)
	}

	var unmarshaled KFDAllocMemArgs
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("failed to unmarshal KFDAllocMemArgs: %v", err)
	}

	if unmarshaled != args {
		t.Errorf("unmarshaled args mismatch: got %+v, want %+v", unmarshaled, args)
	}

	// Verify KFDProbeResult JSON round-trip
	res, err := ValidateKFDUnifiedMemoryMapping(MinUnifiedMapping64GB)
	if err != nil {
		t.Fatalf("failed to validate: %v", err)
	}

	resData, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("failed to marshal KFDProbeResult: %v", err)
	}

	var unmarshaledRes KFDProbeResult
	if err := json.Unmarshal(resData, &unmarshaledRes); err != nil {
		t.Fatalf("failed to unmarshal KFDProbeResult: %v", err)
	}

	if unmarshaledRes.AllocSizeBytes != MinUnifiedMapping64GB {
		t.Errorf("expected %d, got %d", MinUnifiedMapping64GB, unmarshaledRes.AllocSizeBytes)
	}
	if !unmarshaledRes.ZeroCopyVerified || !unmarshaledRes.HugepageAligned {
		t.Errorf("unmarshaled result lost verified attributes")
	}
}

func TestKFDGEM2MBAllocator(t *testing.T) {
	t.Run("AlignmentEnforcement", func(t *testing.T) {
		alloc := NewKFDGEM2MBAllocator("/dev/null/nonexistent", "/dev/null/nonexistent")

		// Negative, zero, or excessive sizes must fail with ErrInvalidGEMSize
		if _, err := alloc.Allocate(0); !errors.Is(err, ErrInvalidGEMSize) {
			t.Errorf("expected ErrInvalidGEMSize for size 0, got %v", err)
		}
		if _, err := alloc.Allocate(-1024); !errors.Is(err, ErrInvalidGEMSize) {
			t.Errorf("expected ErrInvalidGEMSize for negative size, got %v", err)
		}
		if _, err := alloc.Allocate(MaxGTTAllocPoolBytes + 1); !errors.Is(err, ErrInvalidGEMSize) {
			t.Errorf("expected ErrInvalidGEMSize for > 116 GiB, got %v", err)
		}

		// Unaligned sizes must fail with ErrUnaligned2MB
		unalignedSizes := []int64{
			1024,
			4096,
			HugepageSize2MB - 1,
			HugepageSize2MB + 1,
			HugepageSize2MB + 4096,
			3 * 1024 * 1024, // 3MB is not a multiple of 2MB
		}
		for _, sz := range unalignedSizes {
			if _, err := alloc.Allocate(sz); !errors.Is(err, ErrUnaligned2MB) {
				t.Errorf("expected ErrUnaligned2MB for unaligned size %d, got %v", sz, err)
			}
		}

		// Valid 2MB-aligned sizes must succeed
		alignedSizes := []int64{
			HugepageSize2MB,
			2 * HugepageSize2MB,
			8 * HugepageSize2MB,
		}
		for _, sz := range alignedSizes {
			buf, err := alloc.Allocate(sz)
			if err != nil {
				t.Fatalf("expected allocation to succeed for %d bytes, got %v", sz, err)
			}
			if !buf.IsHugepageAligned() {
				t.Errorf("expected buffer host pointer (0x%x) and size (%d) to be 2MB aligned", buf.HostPtr, buf.Size)
			}
			if !buf.IsZeroCopy() {
				t.Errorf("expected zero-copy pointer identity: hostPtr=0x%x != devPtr=0x%x", buf.HostPtr, buf.DevicePtr)
			}
			if buf.Len() != sz {
				t.Errorf("expected buffer length %d, got %d", sz, buf.Len())
			}
			_ = buf.Free()
		}
	})

	t.Run("MockIoctlHarnessValidation", func(t *testing.T) {
		alloc := NewKFDGEM2MBAllocator(DefaultDRMRenderPath, DefaultKFDPath)

		createCalled := false
		mmapCalled := false
		const expectedHandle uint32 = 0xbeef01
		const expectedMmapOffset uint64 = 0x800000000
		const allocSize = 4 * 1024 * 1024 // 4 MiB (2x 2MB hugepage)

		mockDRM := func(fd uintptr, cmd uintptr, arg unsafe.Pointer) error {
			switch cmd {
			case DRM_IOCTL_AMDGPU_GEM_CREATE:
				createCalled = true
				args := (*DRMAMDGPUGEMCreateArgs)(arg)
				if args.Domains != AMDGPU_GEM_DOMAIN_GTT {
					t.Errorf("expected AMDGPU_GEM_DOMAIN_GTT (0x2), got 0x%x", args.Domains)
				}
				if (args.DomainFlags & AMDGPU_GEM_CREATE_CPU_GTT_USWC) == 0 {
					t.Errorf("expected AMDGPU_GEM_CREATE_CPU_GTT_USWC flag to be set")
				}
				if (args.DomainFlags & AMDGPU_GEM_CREATE_NO_EVICT) == 0 {
					t.Errorf("expected AMDGPU_GEM_CREATE_NO_EVICT flag to be set")
				}
				if args.Alignment != uint64(HugepageSize2MB) {
					t.Errorf("expected alignment %d, got %d", HugepageSize2MB, args.Alignment)
				}
				if args.BOSize != uint64(allocSize) {
					t.Errorf("expected bo_size %d, got %d", allocSize, args.BOSize)
				}
				args.SetHandle(expectedHandle)
				return nil

			case DRM_IOCTL_AMDGPU_GEM_MMAP:
				mmapCalled = true
				args := (*DRMAMDGPUGEMMmapArgs)(arg)
				if args.Handle != expectedHandle {
					t.Errorf("expected handle %d in mmap ioctl, got %d", expectedHandle, args.Handle)
				}
				args.SetAddrPtr(expectedMmapOffset)
				return nil

			default:
				return fmt.Errorf("unexpected ioctl cmd: 0x%x", cmd)
			}
		}

		rawMem := make([]byte, allocSize+HugepageSize2MB)
		baseAddr := uintptr(unsafe.Pointer(&rawMem[0]))
		offset := int((uintptr(HugepageSize2MB) - (baseAddr % uintptr(HugepageSize2MB))) % uintptr(HugepageSize2MB))
		alignedMem := rawMem[offset : offset+allocSize : offset+allocSize]

		mockMmap := func(fd int, off int64, sz int) ([]byte, error) {
			if off != int64(expectedMmapOffset) {
				t.Errorf("expected mmap offset 0x%x, got 0x%x", expectedMmapOffset, off)
			}
			return alignedMem, nil
		}

		unmapped := false
		mockMunmap := func(b []byte) error {
			unmapped = true
			return nil
		}

		alloc.SetMockIoctl(mockDRM, mockMmap, mockMunmap)

		buf, err := alloc.Allocate(allocSize)
		if err != nil {
			t.Fatalf("expected Allocate to succeed with mock ioctls, got %v", err)
		}
		if !createCalled {
			t.Errorf("expected DRM_IOCTL_AMDGPU_GEM_CREATE to be called")
		}
		if !mmapCalled {
			t.Errorf("expected DRM_IOCTL_AMDGPU_GEM_MMAP to be called")
		}
		if buf.Handle != expectedHandle {
			t.Errorf("expected buf.Handle=%d, got %d", expectedHandle, buf.Handle)
		}
		if buf.MmapOffset != expectedMmapOffset {
			t.Errorf("expected buf.MmapOffset=0x%x, got 0x%x", expectedMmapOffset, buf.MmapOffset)
		}
		if !buf.IsZeroCopy() {
			t.Errorf("expected zero-copy pointer identity")
		}
		if !buf.IsHugepageAligned() {
			t.Errorf("expected 2MB hugepage aligned pointer")
		}
		if !buf.IsPinned() {
			t.Errorf("expected eviction-pinned buffer")
		}
		if buf.Len() != allocSize {
			t.Errorf("expected buffer length %d, got %d", allocSize, buf.Len())
		}

		// Read / write test pattern through slice
		slice := buf.Slice()
		testPattern := []byte{0x5a, 0xa5, 0x11, 0x22, 0x33, 0x44}
		copy(slice[:len(testPattern)], testPattern)
		for i, b := range testPattern {
			if slice[i] != b {
				t.Errorf("byte mismatch at index %d: expected 0x%x, got 0x%x", i, b, slice[i])
			}
		}

		// Free should invoke unmap
		if err := alloc.Free(buf); err != nil {
			t.Fatalf("unexpected error freeing buffer: %v", err)
		}
		if !unmapped {
			t.Errorf("expected mockMunmap to be called on Free")
		}
	})

	t.Run("SimulationFallback", func(t *testing.T) {
		alloc := NewKFDGEM2MBAllocator("/dev/nonexistent_drm_node", "/dev/nonexistent_kfd_node")
		const allocSize = 2 * 1024 * 1024 // 2MB

		buf, err := alloc.Allocate(allocSize)
		if err != nil {
			t.Fatalf("expected fallback simulation to succeed, got %v", err)
		}
		defer func() { _ = alloc.Free(buf) }()

		if !alloc.FallbackEngaged() {
			t.Errorf("expected FallbackEngaged() to be true")
		}
		if !buf.IsSimulated {
			t.Errorf("expected buf.IsSimulated to be true")
		}
		if !buf.IsZeroCopy() {
			t.Errorf("expected zero-copy pointer identity in simulation: host=0x%x, dev=0x%x", buf.HostPtr, buf.DevicePtr)
		}
		if !buf.IsHugepageAligned() {
			t.Errorf("expected 2MB hugepage alignment in simulation: host=0x%x", buf.HostPtr)
		}
		if buf.Len() != allocSize {
			t.Errorf("expected length %d, got %d", allocSize, buf.Len())
		}

		slice := buf.Slice()
		testPattern := []byte{0xef, 0xbe, 0xad, 0xde}
		copy(slice[:len(testPattern)], testPattern)
		for i, b := range testPattern {
			if slice[i] != b {
				t.Errorf("mismatch at %d: expected 0x%x, got 0x%x", i, b, slice[i])
			}
		}

		if err := buf.Free(); err != nil {
			t.Fatalf("unexpected error on Free: %v", err)
		}
		if err := buf.Free(); !errors.Is(err, ErrGEMBufferClosed) {
			t.Errorf("expected ErrGEMBufferClosed on double free, got %v", err)
		}
	})

	t.Run("ZeroAllocationsPointerResolution", func(t *testing.T) {
		alloc := NewKFDGEM2MBAllocator("", "")
		buf, err := alloc.Allocate(2 * 1024 * 1024)
		if err != nil {
			t.Fatalf("allocation failed: %v", err)
		}
		defer func() { _ = alloc.Free(buf) }()

		allocs := testing.AllocsPerRun(1000, func() {
			_ = buf.UnsafePointer()
			_ = buf.IsZeroCopy()
			_ = buf.IsHugepageAligned()
			_ = buf.Len()
		})
		if allocs > 0 {
			t.Errorf("expected 0 allocations during pointer resolution, got %f", allocs)
		}
	})

	t.Run("ProbeDRMRenderNode", func(t *testing.T) {
		// Non-existent path should report unavailable with actionable recommendation
		res := ProbeDRMRenderNode("/dev/nonexistent_drm_node_probe_test")
		if res.Available {
			t.Errorf("expected non-existent path to be unavailable")
		}
		if !strings.Contains(res.Recommendation, "simulation mode") && !strings.Contains(res.Recommendation, "modprobe amdgpu") {
			t.Errorf("expected actionable recommendation in probe result, got %q", res.Recommendation)
		}
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		alloc := NewKFDGEM2MBAllocator("", "")
		const totalSize = 4 * 1024 * 1024
		buf, err := alloc.Allocate(totalSize)
		if err != nil {
			t.Fatalf("allocation failed: %v", err)
		}
		defer func() { _ = alloc.Free(buf) }()

		const workers = 16
		const chunkSize = totalSize / workers
		var wg sync.WaitGroup

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				start := workerID * chunkSize
				end := start + chunkSize
				slice := buf.Slice()
				if slice == nil {
					t.Errorf("worker %d: nil slice", workerID)
					return
				}

				val := byte(workerID & 0xff)
				for i := start; i < end; i++ {
					slice[i] = val
				}
				for i := start; i < end; i++ {
					if slice[i] != val {
						t.Errorf("worker %d: data corrupted at %d: got %x, want %x", workerID, i, slice[i], val)
					}
				}

				if !buf.IsZeroCopy() || !buf.IsHugepageAligned() || buf.Len() != totalSize {
					t.Errorf("worker %d: invariant failure during concurrent access", workerID)
				}
			}(w)
		}

		wg.Wait()
	})
}
