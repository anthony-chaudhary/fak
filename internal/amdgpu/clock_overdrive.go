// iGPU overdrive maximum-clock cap. Mechanism only; presence does not establish
// serving invocation, a tuned optimum, or physical qualification.
//
// Method note (clean-room, INSPIRE-ONLY):
//
//	Source pin: davidcanar/vllm-strix-halo @ 84da545d03 (Apache-2.0), a measured
//	one-lever iGPU SCLK cap via pp_od_clk_voltage swept over 1500-2900 MHz.
//	Only the sysfs write *mechanism* is captured; the optimum MHz is NOT ported —
//	it is rig- and workload-specific and must be re-swept locally. No upstream
//	number is claimed as an appliance prediction.
package amdgpu

import (
	"errors"
	"fmt"
	"path/filepath"
)

// SysfsPPODClkVoltage is the sysfs node exposing the (non-persistent) iGPU
// overdrive clock range. It is absent on devices that do not offer the OD range.
const SysfsPPODClkVoltage = "pp_od_clk_voltage"

// ErrOverdriveUnsupported is the typed, fail-closed result for a device that
// does not expose a writable overdrive clock range. It is non-fatal: a caller
// may treat it as "this lever is unavailable here" and leave the device as-is.
var ErrOverdriveUnsupported = errors.New("amdgpu: iGPU overdrive clock cap unsupported on this device")

// OverdriveCapResult records the outcome of an overdrive-cap operation.
type OverdriveCapResult struct {
	// Applied reports whether the device was actually mutated.
	Applied bool `json:"applied"`
	// MaxCoreMHz is the applied overdrive maximum in MHz (0 when untouched).
	MaxCoreMHz int `json:"max_core_mhz"`
	// Device is the DRM card directory the operation targeted ("" when untouched).
	Device string `json:"device,omitempty"`
	// Unchanged reports that the call was a deliberate no-op (auto/unset cap).
	Unchanged bool `json:"unchanged,omitempty"`
}

// CapOverdriveMaxCoreClock caps the iGPU overdrive maximum core clock to maxMHz.
//
// The operation is explicitly non-persistent (an OS reboot restores stock) and
// reversible in-process via RestoreOverdrive. A maxMHz <= 0 is an "auto"/unset
// no-op: the device is left untouched. A device without a readable
// pp_od_clk_voltage fails closed with ErrOverdriveUnsupported and performs no
// write, leaving the device exactly as it was found.
func (g *HardwareClockGovernor) CapOverdriveMaxCoreClock(maxMHz int) (OverdriveCapResult, error) {
	if maxMHz <= 0 {
		return OverdriveCapResult{Unchanged: true}, nil
	}
	g.effectsMu.Lock()
	defer g.effectsMu.Unlock()
	return g.applyOverdriveCap(maxMHz)
}

// RestoreOverdrive reverses CapOverdriveMaxCoreClock by asking the firmware to
// restore the stock overdrive range, then committing the request. It is a no-op
// (Unchanged) on a device that does not expose pp_od_clk_voltage.
func (g *HardwareClockGovernor) RestoreOverdrive() (OverdriveCapResult, error) {
	g.effectsMu.Lock()
	defer g.effectsMu.Unlock()
	return g.restoreOverdrive()
}

// applyOverdriveCap requires effectsMu.
func (g *HardwareClockGovernor) applyOverdriveCap(maxMHz int) (OverdriveCapResult, error) {
	devDir, odFile, err := g.resolveOverdriveFile()
	if err != nil {
		return OverdriveCapResult{}, err
	}

	// Bounded write: set the OD range maximum, then commit with a single byte.
	if err := g.writeOverdrive(odFile, fmt.Sprintf("s 1 %d\n", maxMHz)); err != nil {
		return OverdriveCapResult{}, fmt.Errorf("amdgpu: overdrive cap write failed at %s: %w", odFile, err)
	}
	if err := g.writeOverdrive(odFile, "c"); err != nil {
		return OverdriveCapResult{}, fmt.Errorf("amdgpu: overdrive cap commit failed at %s: %w", odFile, err)
	}
	return OverdriveCapResult{
		Applied:    true,
		MaxCoreMHz: maxMHz,
		Device:     filepath.Base(filepath.Dir(devDir)),
	}, nil
}

// restoreOverdrive requires effectsMu.
func (g *HardwareClockGovernor) restoreOverdrive() (OverdriveCapResult, error) {
	_, odFile, err := g.resolveOverdriveFile()
	if err != nil {
		if errors.Is(err, ErrOverdriveUnsupported) {
			// A device that never exposed the range cannot need a restore.
			return OverdriveCapResult{Unchanged: true}, nil
		}
		return OverdriveCapResult{}, err
	}
	if err := g.writeOverdrive(odFile, "r"); err != nil {
		return OverdriveCapResult{}, fmt.Errorf("amdgpu: overdrive restore failed at %s: %w", odFile, err)
	}
	if err := g.writeOverdrive(odFile, "c"); err != nil {
		return OverdriveCapResult{}, fmt.Errorf("amdgpu: overdrive restore commit failed at %s: %w", odFile, err)
	}
	return OverdriveCapResult{Applied: true}, nil
}

// resolveOverdriveFile locates the device dir and confirms pp_od_clk_voltage is
// present and readable BEFORE any write. A missing or unreadable node fails
// closed with ErrOverdriveUnsupported so the caller never performs a partial
// write against an unsupported device.
func (g *HardwareClockGovernor) resolveOverdriveFile() (devDir, odFile string, err error) {
	devDir = g.resolveAMDDeviceDir()
	if g.fileReader == nil {
		return devDir, "", fmt.Errorf("%s: %w", devDir, ErrOverdriveUnsupported)
	}
	odFile = filepath.Join(devDir, SysfsPPODClkVoltage)
	if _, rerr := g.fileReader(odFile); rerr != nil {
		return devDir, odFile, fmt.Errorf("%s: %w", odFile, ErrOverdriveUnsupported)
	}
	return devDir, odFile, nil
}

// writeOverdrive performs one injected write. Sharing the governor's fileWriter
// keeps the seam consistent with the DPM lock/unlock paths.
func (g *HardwareClockGovernor) writeOverdrive(path, payload string) error {
	if g.fileWriter == nil {
		return fmt.Errorf("%s: no file writer configured", path)
	}
	return g.fileWriter(path, []byte(payload), 0644)
}
