package strix

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectGFX1151_EnvOverride(t *testing.T) {
	// Clear env after test
	defer os.Unsetenv("FAK_STRIX_GFX1151_OVERRIDE")

	// 1. Force enable via "1"
	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "1")
	detected, name, err := DetectGFX1151("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !detected {
		t.Errorf("expected detected=true with override=1")
	}
	if name == "" {
		t.Errorf("expected non-empty device name with override=1")
	}

	// 2. Force enable via "true"
	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "true")
	detected, _, err = DetectGFX1151("")
	if err != nil || !detected {
		t.Errorf("expected detected=true with override=true, err=%v", err)
	}

	// 3. Force disable via "0"
	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "0")
	detected, _, err = DetectGFX1151("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if detected {
		t.Errorf("expected detected=false with override=0")
	}

	// 4. Force disable via "false"
	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "false")
	detected, _, err = DetectGFX1151("")
	if err != nil || detected {
		t.Errorf("expected detected=false with override=false, err=%v", err)
	}
}

func TestDetectGFX1151_DeviceName(t *testing.T) {
	os.Unsetenv("FAK_STRIX_GFX1151_OVERRIDE")
	tmpDir := t.TempDir()

	cases := []struct {
		devName string
		want    bool
	}{
		{"AMD Radeon 8060S Graphics (gfx1151)", true},
		{"AMD Radeon 8050S Graphics", true},
		{"Ryzen AI Max+ 395 silicon", true},
		{"strix halo APU", true},
		{"NVIDIA GeForce RTX 4090", false},
		{"Apple M3 Max", false},
		{"", false},
	}

	for _, tc := range cases {
		detected, name, err := DetectGFX1151WithDeviceName(tmpDir, tc.devName)
		if err != nil {
			t.Errorf("unexpected error for %q: %v", tc.devName, err)
		}
		if detected != tc.want {
			t.Errorf("devName %q: got detected=%v, want %v", tc.devName, detected, tc.want)
		}
		if detected && name != tc.devName {
			t.Errorf("devName %q: got name %q", tc.devName, name)
		}
	}
}

func TestDetectGFX1151_SysfsDRM(t *testing.T) {
	os.Unsetenv("FAK_STRIX_GFX1151_OVERRIDE")

	tmpDir := t.TempDir()

	// 1. Non-existent sysfs path returns false without error
	detected, _, err := DetectGFX1151(filepath.Join(tmpDir, "nonexistent"))
	if err != nil {
		t.Fatalf("unexpected error on missing sysfs: %v", err)
	}
	if detected {
		t.Errorf("expected detected=false on missing sysfs")
	}

	// 2. Mock card0 with matching AMD vendor (0x1002) and GFX1151 device (0x1586)
	card0Device := filepath.Join(tmpDir, "card0", "device")
	if err := os.MkdirAll(card0Device, 0755); err != nil {
		t.Fatalf("failed to create mock sysfs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(card0Device, "vendor"), []byte("0x1002\n"), 0644); err != nil {
		t.Fatalf("failed to write mock vendor: %v", err)
	}
	if err := os.WriteFile(filepath.Join(card0Device, "device"), []byte("0x1586\n"), 0644); err != nil {
		t.Fatalf("failed to write mock device: %v", err)
	}

	detected, name, err := DetectGFX1151(tmpDir)
	if err != nil {
		t.Fatalf("unexpected error on mock sysfs: %v", err)
	}
	if !detected {
		t.Errorf("expected detected=true on matching mock sysfs")
	}
	if name != CanonicalDeviceNameStrixHalo {
		t.Errorf("got name %q, want %q", name, CanonicalDeviceNameStrixHalo)
	}

	// 3. Mock Intel device (vendor 0x8086) should not match
	tmpIntel := t.TempDir()
	intelDev := filepath.Join(tmpIntel, "card0", "device")
	_ = os.MkdirAll(intelDev, 0755)
	_ = os.WriteFile(filepath.Join(intelDev, "vendor"), []byte("0x8086\n"), 0644)
	_ = os.WriteFile(filepath.Join(intelDev, "device"), []byte("0x1586\n"), 0644)

	detected, _, err = DetectGFX1151(tmpIntel)
	if err != nil || detected {
		t.Errorf("expected detected=false for Intel vendor, got detected=%v, err=%v", detected, err)
	}
}

func TestIsStrixHaloArch(t *testing.T) {
	if !IsStrixHaloArch("gfx1151") {
		t.Errorf("expected true for gfx1151")
	}
	if !IsStrixHaloArch("AMD Ryzen AI Max+ 395") {
		t.Errorf("expected true for Ryzen AI Max+ 395")
	}
	if IsStrixHaloArch("gfx1100") {
		t.Errorf("expected false for gfx1100 (Navi 31)")
	}
	if IsStrixHaloArch("") {
		t.Errorf("expected false for empty")
	}
}
