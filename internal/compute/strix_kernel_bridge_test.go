package compute

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/strix"
)

func TestStrixKernelBridgePublicInterface(t *testing.T) {
	// 1. Interface satisfaction test: compile-time and runtime check.
	var _ strix.AccelerationEngine = (*StrixKernelBridge)(nil)

	// Create bridge with default system hardware detection.
	bridge := NewStrixKernelBridge()
	var engine strix.AccelerationEngine = bridge

	if engine == nil {
		t.Fatal("expected non-nil strix.AccelerationEngine")
	}
	if engine.Name() != DefaultStrixBridgeName {
		t.Fatalf("expected Name() == %q, got %q", DefaultStrixBridgeName, engine.Name())
	}

	// 2. Test graceful fallback when GFX1151 hardware is not present.
	// We explicitly test fallback using an isolated empty mock sysfs directory.
	emptySysfs := t.TempDir()
	fallbackBridge := NewStrixKernelBridgeWithConfig(StrixKernelBridgeConfig{
		SysfsDRMRoot: emptySysfs,
	})
	var fallbackEngine strix.AccelerationEngine = fallbackBridge

	if fallbackEngine.IsSupported() {
		t.Fatal("expected IsSupported() == false on host/directory without GFX1151 hardware")
	}
	if fallbackEngine.ComputeUnits() != 0 {
		t.Fatalf("expected 0 compute units on fallback, got %d", fallbackEngine.ComputeUnits())
	}
	if fallbackEngine.WavefrontSize() != 0 {
		t.Fatalf("expected 0 wavefront size on fallback, got %d", fallbackEngine.WavefrontSize())
	}

	// Calling AllocUSWCGTT on unsupported device must return ErrUnsupportedDevice gracefully (no panic).
	addr, err := fallbackEngine.AllocUSWCGTT(1024)
	if err == nil {
		t.Fatal("expected ErrUnsupportedDevice error when allocating on unsupported engine, got nil")
	}
	if !errors.Is(err, ErrUnsupportedDevice) {
		t.Fatalf("expected ErrUnsupportedDevice, got: %v", err)
	}
	if addr != 0 {
		t.Fatalf("expected 0 address on allocation failure, got 0x%x", addr)
	}

	// 3. Test GFX1151 hardware detection when DRM device is present (vendor 0x1002, device 0x1586).
	mockSysfs := t.TempDir()
	card0Device := filepath.Join(mockSysfs, "card0", "device")
	if err := os.MkdirAll(card0Device, 0755); err != nil {
		t.Fatalf("failed to create mock sysfs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(card0Device, "vendor"), []byte("0x1002\n"), 0644); err != nil {
		t.Fatalf("failed to write mock vendor: %v", err)
	}
	if err := os.WriteFile(filepath.Join(card0Device, "device"), []byte("0x1586\n"), 0644); err != nil {
		t.Fatalf("failed to write mock device: %v", err)
	}

	supportedBridge := NewStrixKernelBridgeWithConfig(StrixKernelBridgeConfig{
		SysfsDRMRoot: mockSysfs,
	})
	var supportedEngine strix.AccelerationEngine = supportedBridge

	if !supportedEngine.IsSupported() {
		t.Fatal("expected IsSupported() == true when GFX1151 DRM card is present")
	}
	if supportedEngine.ComputeUnits() != 40 {
		t.Fatalf("expected 40 CUs on GFX1151, got %d", supportedEngine.ComputeUnits())
	}
	if supportedEngine.WavefrontSize() != 32 {
		t.Fatalf("expected Wave32 (32) wavefront size, got %d", supportedEngine.WavefrontSize())
	}

	profile := supportedBridge.Profile()
	if profile.Architecture != "gfx1151" || profile.CUCount != 40 || profile.Wavefront != 32 || profile.BusWidthBits != 256 {
		t.Fatalf("unexpected supported profile: %+v", profile)
	}

	// AllocUSWCGTT on supported engine
	allocAddr, err := supportedEngine.AllocUSWCGTT(4096)
	if err != nil {
		t.Fatalf("AllocUSWCGTT failed on supported engine: %v", err)
	}
	if allocAddr == 0 {
		t.Fatal("expected non-zero address for AllocUSWCGTT")
	}

	// Free buffer
	if err := supportedBridge.FreeUSWCGTT(allocAddr); err != nil {
		t.Fatalf("FreeUSWCGTT failed: %v", err)
	}
	// Double free returns error
	if err := supportedBridge.FreeUSWCGTT(allocAddr); err == nil {
		t.Fatal("expected error on double free")
	}

	// Allocation bounds checks
	if _, err := supportedEngine.AllocUSWCGTT(0); !errors.Is(err, ErrInvalidAllocationSize) {
		t.Fatalf("expected ErrInvalidAllocationSize for 0 bytes, got %v", err)
	}
	if _, err := supportedEngine.AllocUSWCGTT(-256); !errors.Is(err, ErrInvalidAllocationSize) {
		t.Fatalf("expected ErrInvalidAllocationSize for negative bytes, got %v", err)
	}
}

func TestStrixKernelBridgeDRMDetectionFiltering(t *testing.T) {
	// Case A: Non-AMD GPU (Intel 0x8086 / 0x1586)
	intelSysfs := t.TempDir()
	cardIntel := filepath.Join(intelSysfs, "card0", "device")
	_ = os.MkdirAll(cardIntel, 0755)
	_ = os.WriteFile(filepath.Join(cardIntel, "vendor"), []byte("0x8086\n"), 0644)
	_ = os.WriteFile(filepath.Join(cardIntel, "device"), []byte("0x1586\n"), 0644)

	bIntel := NewStrixKernelBridgeWithConfig(StrixKernelBridgeConfig{SysfsDRMRoot: intelSysfs})
	if bIntel.IsSupported() {
		t.Fatal("expected Intel device to be unsupported")
	}

	// Case B: Non-GFX1151 AMD GPU (e.g. Phoenix/Hawk Point 0x1002 / 0x150e)
	phoenixSysfs := t.TempDir()
	cardPhoenix := filepath.Join(phoenixSysfs, "card0", "device")
	_ = os.MkdirAll(cardPhoenix, 0755)
	_ = os.WriteFile(filepath.Join(cardPhoenix, "vendor"), []byte("0x1002\n"), 0644)
	_ = os.WriteFile(filepath.Join(cardPhoenix, "device"), []byte("0x150e\n"), 0644)

	bPhoenix := NewStrixKernelBridgeWithConfig(StrixKernelBridgeConfig{SysfsDRMRoot: phoenixSysfs})
	if bPhoenix.IsSupported() {
		t.Fatal("expected AMD Phoenix device (0x150e) to be unsupported")
	}

	// Case C: Multi-card system where card0 is dGPU/integrated and card1 is GFX1151
	multiSysfs := t.TempDir()
	card0 := filepath.Join(multiSysfs, "card0", "device")
	card1 := filepath.Join(multiSysfs, "card1", "device")
	_ = os.MkdirAll(card0, 0755)
	_ = os.MkdirAll(card1, 0755)
	_ = os.WriteFile(filepath.Join(card0, "vendor"), []byte("0x8086\n"), 0644)
	_ = os.WriteFile(filepath.Join(card0, "device"), []byte("0x9999\n"), 0644)
	_ = os.WriteFile(filepath.Join(card1, "vendor"), []byte("0x1002\n"), 0644)
	_ = os.WriteFile(filepath.Join(card1, "device"), []byte("0x1586\n"), 0644)

	bMulti := NewStrixKernelBridgeWithConfig(StrixKernelBridgeConfig{SysfsDRMRoot: multiSysfs})
	if !bMulti.IsSupported() {
		t.Fatal("expected multi-card system to detect GFX1151 on card1")
	}
}

func TestStrixKernelBridgeCustomHooks(t *testing.T) {
	customAllocRan := false
	customBridge := NewStrixKernelBridgeWithConfig(StrixKernelBridgeConfig{
		DetectFn: func() (bool, string, error) {
			return true, "/mock/card", nil
		},
		Allocator: func(bytes int64) (uintptr, error) {
			customAllocRan = true
			return 0xCAFE0000, nil
		},
	})

	if !customBridge.IsSupported() {
		t.Fatal("expected custom bridge to be supported")
	}
	addr, err := customBridge.AllocUSWCGTT(512)
	if err != nil || addr != 0xCAFE0000 {
		t.Fatalf("unexpected custom alloc result: addr=0x%x, err=%v", addr, err)
	}
	if !customAllocRan {
		t.Fatal("expected custom allocator hook to execute")
	}
}
