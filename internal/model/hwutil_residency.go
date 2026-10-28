package model

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/pkg/hwutil"
)

// Tier-residency emission for the hardware-usage telemetry plane
// (fak.hwutil.v1). No offload/tiering engine exists yet, so these seams emit
// what the loader actually does: the packed f32 weight buffer resident on the
// host after load (one RAM band), and each first-time host-to-device weight
// upload as a ram->vram transfer that grows the device VRAM band. The layer
// range is the whole decoder span [0, NumLayers-1]: tensor names do not map
// one-to-one onto layer bands, so per-tensor attribution stays out of the
// claim rather than inventing it.

// hwutilModelID derives the wire model identity for tier rows: the exact
// checkpoint name when the source declared one, else the architecture family.
func hwutilModelID(m *Model) string {
	if m == nil {
		return "unnamed-model"
	}
	if name := strings.TrimSpace(m.Cfg.Name); name != "" {
		return name
	}
	if arch := strings.TrimSpace(m.Cfg.ModelType); arch != "" {
		return "arch:" + arch
	}
	return "unnamed-model"
}

// recordHostWeightResidency emits the loader's RAM placement row: every byte
// of the packed f32 manifest now resident on the host. Called once from
// newModel, the single funnel every weight loader (safetensors, GGUF,
// in-memory) passes through.
func recordHostWeightResidency(m *Model) {
	if m == nil || len(m.raw) == 0 || m.Cfg.NumLayers <= 0 {
		return
	}
	hwutil.Default().RecordPlacement(hwutilModelID(m), hwutil.TierRAM, "host", 0, m.Cfg.NumLayers-1, int64(len(m.raw)), 0)
}

// recordWeightUpload emits one load-time ram->vram movement plus the VRAM
// band growth for bytes deviceBytes newly staged on a device backend. Host
// backends (reference CPU: Upload is a host identity operation) and nil
// backends (legacy session path) are skipped entirely — a host-to-host copy
// is not a tier move and must not show up as vram residency. started marks
// when the staging began; the duration is the measured host-to-device
// movement, never an estimate.
func recordWeightUpload(m *Model, be compute.Backend, deviceBytes int64, started time.Time) {
	if m == nil || be == nil || !be.Caps().DeviceMemory || deviceBytes <= 0 {
		return
	}
	device := strings.TrimSpace(be.Name())
	if device == "" {
		device = "device"
	}
	rec := hwutil.Default()
	dur := time.Since(started)
	if dur < 0 {
		dur = 0
	}
	rec.RecordTransfer(hwutil.TierRAM, hwutil.TierVRAM, deviceBytes, dur, hwutil.TransferLoad)
	if m.Cfg.NumLayers > 0 {
		rec.RecordPlacement(hwutilModelID(m), hwutil.TierVRAM, device, 0, m.Cfg.NumLayers-1, deviceBytes, 0)
	}
}

// tensorBytesOf estimates a staged tensor's storage from its element count and
// dtype width. Exact for the f32 path; quantized layouts may carry block
// metadata beyond the element product, so the figure is a lower bound there.
func tensorBytesOf(t compute.Tensor) int64 {
	if t.Numel() <= 0 {
		return 0
	}
	return int64(t.Numel()) * int64(t.Dtype.Bytes())
}
