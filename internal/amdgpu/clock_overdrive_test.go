package amdgpu

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// overdriveRecorder is a minimal injected file writer that records the exact
// (path, payload) write sequence an overdrive operation performs.
type overdriveRecorder struct {
	writes []string
}

func (r *overdriveRecorder) write(path string, data []byte, _ os.FileMode) error {
	r.writes = append(r.writes, path+"\x00"+string(data))
	return nil
}

// newOverdriveFixture builds a fake DRM sysfs root with card1/device and an
// optional pp_od_clk_voltage node, returning the root and the OD file path.
func newOverdriveFixture(t *testing.T, withOD bool) (root, odFile string) {
	t.Helper()
	root = t.TempDir()
	devDir := filepath.Join(root, "card1", "device")
	if err := os.MkdirAll(devDir, 0755); err != nil {
		t.Fatalf("mkdir device dir: %v", err)
	}
	odFile = filepath.Join(devDir, SysfsPPODClkVoltage)
	if withOD {
		if err := os.WriteFile(odFile, []byte("OD_SCLK: 600Mhz\nOD_RANGE: 600Mhz 2900Mhz\n"), 0644); err != nil {
			t.Fatalf("seed od file: %v", err)
		}
	}
	return root, odFile
}

// TestOverdriveCapApplyRestoreUnsupported is the table-driven acceptance for the
// issue: apply, restore, unsupported-device, and non-interference with DPM/sclk.
func TestOverdriveCapApplyRestoreUnsupported(t *testing.T) {
	t.Run("ApplyWritesRangeMaxThenCommit", func(t *testing.T) {
		root, odFile := newOverdriveFixture(t, true)
		rec := &overdriveRecorder{}
		g := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(root),
			WithClockGovernorFileWriter(rec.write),
		)
		res, err := g.CapOverdriveMaxCoreClock(2100)
		if err != nil {
			t.Fatalf("cap: unexpected error: %v", err)
		}
		if !res.Applied || res.MaxCoreMHz != 2100 {
			t.Fatalf("cap: want applied max=2100, got %+v", res)
		}
		want := []string{odFile + "\x00s 1 2100\n", odFile + "\x00c"}
		if len(rec.writes) != len(want) {
			t.Fatalf("cap: write sequence %v, want %v", rec.writes, want)
		}
		for i := range want {
			if rec.writes[i] != want[i] {
				t.Fatalf("cap: write[%d]=%q, want %q", i, rec.writes[i], want[i])
			}
		}
	})

	t.Run("RestoreWritesResetThenCommit", func(t *testing.T) {
		root, odFile := newOverdriveFixture(t, true)
		rec := &overdriveRecorder{}
		g := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(root),
			WithClockGovernorFileWriter(rec.write),
		)
		res, err := g.RestoreOverdrive()
		if err != nil {
			t.Fatalf("restore: unexpected error: %v", err)
		}
		if !res.Applied {
			t.Fatalf("restore: want applied, got %+v", res)
		}
		want := []string{odFile + "\x00r", odFile + "\x00c"}
		if len(rec.writes) != len(want) {
			t.Fatalf("restore: write sequence %v, want %v", rec.writes, want)
		}
		for i := range want {
			if rec.writes[i] != want[i] {
				t.Fatalf("restore: write[%d]=%q, want %q", i, rec.writes[i], want[i])
			}
		}
	})

	t.Run("UnsupportedDeviceRefusesAndNeverWrites", func(t *testing.T) {
		root, _ := newOverdriveFixture(t, false) // no pp_od_clk_voltage
		rec := &overdriveRecorder{}
		g := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(root),
			WithClockGovernorFileWriter(rec.write),
		)
		if _, err := g.CapOverdriveMaxCoreClock(2100); !errors.Is(err, ErrOverdriveUnsupported) {
			t.Fatalf("cap on unsupported device: want ErrOverdriveUnsupported, got %v", err)
		}
		if len(rec.writes) != 0 {
			t.Fatalf("cap on unsupported device must not write, got %v", rec.writes)
		}
		// Restore on an unsupported device is a benign no-op, still no write.
		res, err := g.RestoreOverdrive()
		if err != nil {
			t.Fatalf("restore on unsupported device: unexpected error: %v", err)
		}
		if res.Applied || !res.Unchanged {
			t.Fatalf("restore on unsupported device: want unchanged no-op, got %+v", res)
		}
		if len(rec.writes) != 0 {
			t.Fatalf("restore on unsupported device must not write, got %v", rec.writes)
		}
	})

	t.Run("AutoCapIsNoOpAndNeverWrites", func(t *testing.T) {
		root, _ := newOverdriveFixture(t, true)
		rec := &overdriveRecorder{}
		g := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(root),
			WithClockGovernorFileWriter(rec.write),
		)
		res, err := g.CapOverdriveMaxCoreClock(0)
		if err != nil {
			t.Fatalf("auto cap: unexpected error: %v", err)
		}
		if !res.Unchanged || res.Applied {
			t.Fatalf("auto cap: want unchanged no-op, got %+v", res)
		}
		if len(rec.writes) != 0 {
			t.Fatalf("auto cap must not write, got %v", rec.writes)
		}
	})

	t.Run("CapDoesNotMutateDPMOrSCLK", func(t *testing.T) {
		root, _ := newOverdriveFixture(t, true)
		devDir := filepath.Join(root, "card1", "device")
		dpmFile := filepath.Join(devDir, SysfsDPMForcePerformanceLevel)
		sclkFile := filepath.Join(devDir, SysfsPPDpmSclk)
		if err := os.WriteFile(dpmFile, []byte("auto\n"), 0644); err != nil {
			t.Fatalf("seed dpm: %v", err)
		}
		if err := os.WriteFile(sclkFile, []byte("0: 600Mhz\n1: 2200Mhz *\n"), 0644); err != nil {
			t.Fatalf("seed sclk: %v", err)
		}
		g := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(root),
		)
		if _, err := g.CapOverdriveMaxCoreClock(2100); err != nil {
			t.Fatalf("cap: unexpected error: %v", err)
		}
		if data, _ := os.ReadFile(dpmFile); string(data) != "auto\n" {
			t.Fatalf("cap must not mutate %s, got %q", SysfsDPMForcePerformanceLevel, string(data))
		}
		if data, _ := os.ReadFile(sclkFile); string(data) != "0: 600Mhz\n1: 2200Mhz *\n" {
			t.Fatalf("cap must not mutate %s, got %q", SysfsPPDpmSclk, string(data))
		}
	})
}
