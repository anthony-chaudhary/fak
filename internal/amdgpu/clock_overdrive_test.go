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

// strixODClkVoltage is the smu_v14 APU pp_od_clk_voltage layout witnessed on
// Strix Halo after DPM level "high" pinned both OD_SCLK points at peak.
const strixODClkVoltage = "OD_SCLK:\n0: 2900Mhz\n1: 2900Mhz\nOD_RANGE:\nSCLK:         600Mhz       2900Mhz\n"

// newOverdriveFixture builds a fake DRM sysfs root with card1/device and an
// optional pp_od_clk_voltage node, returning the root and the OD file path.
func newOverdriveFixture(t *testing.T, withOD bool) (root, odFile string) {
	return newOverdriveFixtureContent(t, withOD, strixODClkVoltage)
}

func newOverdriveFixtureContent(t *testing.T, withOD bool, content string) (root, odFile string) {
	t.Helper()
	root = t.TempDir()
	devDir := filepath.Join(root, "card1", "device")
	if err := os.MkdirAll(devDir, 0755); err != nil {
		t.Fatalf("mkdir device dir: %v", err)
	}
	odFile = filepath.Join(devDir, SysfsPPODClkVoltage)
	if withOD {
		if err := os.WriteFile(odFile, []byte(content), 0644); err != nil {
			t.Fatalf("seed od file: %v", err)
		}
	}
	return root, odFile
}

// TestOverdriveCapApplyRestoreUnsupported is the table-driven acceptance for the
// issue: apply, restore, unsupported-device, and non-interference with DPM/sclk.
func TestOverdriveCapApplyRestoreUnsupported(t *testing.T) {
	t.Run("ApplyWritesFloorThenMaxThenCommit", func(t *testing.T) {
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
		want := []string{odFile + "\x00s 0 600\n", odFile + "\x00s 1 2100\n", odFile + "\x00c"}
		assertOverdriveWrites(t, rec.writes, want)
	})

	t.Run("FloorClampsToMaxBelowRange", func(t *testing.T) {
		root, odFile := newOverdriveFixture(t, true)
		rec := &overdriveRecorder{}
		g := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(root),
			WithClockGovernorFileWriter(rec.write),
		)
		if _, err := g.CapOverdriveMaxCoreClock(500); err != nil {
			t.Fatalf("cap: unexpected error: %v", err)
		}
		want := []string{odFile + "\x00s 0 500\n", odFile + "\x00s 1 500\n", odFile + "\x00c"}
		assertOverdriveWrites(t, rec.writes, want)
	})

	t.Run("UnparseableRangeSkipsFloorWrite", func(t *testing.T) {
		root, odFile := newOverdriveFixtureContent(t, true, "OD_SCLK:\n0: 2900Mhz\n")
		rec := &overdriveRecorder{}
		g := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(root),
			WithClockGovernorFileWriter(rec.write),
		)
		if _, err := g.CapOverdriveMaxCoreClock(2100); err != nil {
			t.Fatalf("cap: unexpected error: %v", err)
		}
		want := []string{odFile + "\x00s 1 2100\n", odFile + "\x00c"}
		assertOverdriveWrites(t, rec.writes, want)
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

func assertOverdriveWrites(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("write sequence %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("write[%d]=%q, want %q", i, got[i], want[i])
		}
	}
}

// TestOverdriveCapFloorFromODRange pins which pp_od_clk_voltage layouts yield a
// floor write ("s 0 <OD_RANGE SCLK min>") ahead of the cap, and which skip it.
func TestOverdriveCapFloorFromODRange(t *testing.T) {
	cases := []struct {
		name    string
		content string
		floor   string // "" means the floor write is skipped
	}{
		{"KernelMultiLine", strixODClkVoltage, "s 0 600\n"},
		{"FlattenedSingleLine", "OD_SCLK: 0: 2900Mhz 1: 2900Mhz OD_RANGE: SCLK: 600Mhz 2900Mhz", "s 0 600\n"},
		{"UpperCaseUnit", "OD_RANGE:\nSCLK: 800MHz 2600MHz\n", "s 0 800\n"},
		{"NoRangeSection", "OD_SCLK:\n0: 600Mhz\n1: 2900Mhz\n", ""},
		{"SCLKBeforeRangeIgnored", "SCLK: 100Mhz 200Mhz\n", ""},
		{"InvertedBounds", "OD_RANGE:\nSCLK: 2900Mhz 600Mhz\n", ""},
		{"MissingUpper", "OD_RANGE:\nSCLK: 600Mhz\n", ""},
		{"Empty", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root, odFile := newOverdriveFixtureContent(t, true, tc.content)
			rec := &overdriveRecorder{}
			g := NewHardwareClockGovernor(
				WithClockGovernorPlatform(PlatformAMD),
				WithClockGovernorSysfsDRMRoot(root),
				WithClockGovernorFileWriter(rec.write),
			)
			if _, err := g.CapOverdriveMaxCoreClock(2100); err != nil {
				t.Fatalf("cap: unexpected error: %v", err)
			}
			var want []string
			if tc.floor != "" {
				want = append(want, odFile+"\x00"+tc.floor)
			}
			want = append(want, odFile+"\x00s 1 2100\n", odFile+"\x00c")
			assertOverdriveWrites(t, rec.writes, want)
		})
	}
}
