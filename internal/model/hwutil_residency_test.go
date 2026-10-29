package model

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/pkg/hwutil"
)

// The loader seams emit into the process-wide recorder, which every other
// model-package test shares inside this one binary (parallel HAL sessions
// stage real weights through the same upload funnel). These tests therefore
// assert on MARKER rows — a distinctive model id and byte count no other
// test uses — rather than on an empty drain. Exact-row contracts live in
// pkg/hwutil, where the recorder is hermetic.

const (
	tierTestModel = "tier-test-model"
	tierTestBytes = int64(987654321) // no other test stages this many bytes
	tierTestSmall = int64(1024)
)

// tierTestDeviceBackend stands in for a real accelerator: it embeds the
// reference backend (whose methods never run here) and advertises device
// memory, which is the exact capability the upload seam keys on.
type tierTestDeviceBackend struct{ compute.Backend }

func (b tierTestDeviceBackend) Name() string { return "fake-device" }
func (b tierTestDeviceBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}

var (
	fakeDeviceBackendForTest = tierTestDeviceBackend{compute.Default()}
	fakeHostBackendForTest   = compute.Default() // cpu-ref: DeviceMemory stays false
)

func findTier(tiers []hwutil.TierRow, modelID string, kind hwutil.TierKind) []hwutil.TierRow {
	var out []hwutil.TierRow
	for _, row := range tiers {
		if row.ModelID == modelID && row.Kind == kind {
			out = append(out, row)
		}
	}
	return out
}

func TestTierLoaderPlacementFromNewModel(t *testing.T) {
	rec := hwutil.Default()
	_, _ = rec.Drain() // clear residue from earlier tests in this binary

	m := &Model{Cfg: Config{Name: tierTestModel, NumLayers: 4}, raw: make([]byte, tierTestSmall)}
	recordHostWeightResidency(m)

	tiers, _ := rec.Drain()
	bands := findTier(tiers, tierTestModel, hwutil.TierRAM)
	if len(bands) != 1 {
		t.Fatalf("marker ram bands = %+v, want exactly one", bands)
	}
	row := bands[0]
	if row.Device != "host" || row.LayersStart != 0 || row.LayersEnd != 3 ||
		row.Bytes != tierTestSmall || row.HotnessRank != 0 {
		t.Fatalf("ram band = %+v", row)
	}
	if row.LastTransferAt == nil {
		t.Fatal("placement must stamp LastTransferAt")
	}

	// Degenerate loads record nothing under the marker.
	recordHostWeightResidency(nil)
	recordHostWeightResidency(&Model{Cfg: Config{Name: tierTestModel}})
	tiers, _ = rec.Drain()
	if got := findTier(tiers, tierTestModel, hwutil.TierRAM); len(got) != 0 {
		t.Fatalf("degenerate loads leaked marker rows: %+v", got)
	}
}

func TestTierWeightUploadEmitsTransferAndVRAMBand(t *testing.T) {
	rec := hwutil.Default()
	_, _ = rec.Drain()

	started := time.Now().Add(-50 * time.Millisecond)
	recordWeightUpload(
		&Model{Cfg: Config{Name: tierTestModel, NumLayers: 4}},
		fakeDeviceBackendForTest,
		tierTestBytes,
		started,
	)

	tiers, transfers := rec.Drain()
	var marker []hwutil.TransferRow
	for _, tr := range transfers {
		if tr.Bytes == tierTestBytes && tr.FromTier == "ram" && tr.ToTier == "vram" {
			marker = append(marker, tr)
		}
	}
	if len(marker) != 1 {
		t.Fatalf("marker transfers = %+v, want exactly one", marker)
	}
	tr := marker[0]
	if tr.Reason != hwutil.TransferLoad {
		t.Fatalf("transfer reason = %q, want load", tr.Reason)
	}
	if tr.DurationS < 0.04 || tr.DurationS > 0.5 {
		t.Fatalf("duration %.3fs not the measured staging window", tr.DurationS)
	}
	vramBands := findTier(tiers, tierTestModel, hwutil.TierVRAM)
	if len(vramBands) != 1 {
		t.Fatalf("marker vram bands = %+v, want exactly one", vramBands)
	}
	vram := vramBands[0]
	if vram.Device != "fake-device" || vram.LayersStart != 0 || vram.LayersEnd != 3 || vram.Bytes != tierTestBytes {
		t.Fatalf("vram band = %+v", vram)
	}

	// Non-device backends are not tier moves: no marker rows may appear.
	_, _ = rec.Drain()
	recordWeightUpload(&Model{Cfg: Config{Name: tierTestModel, NumLayers: 4}}, fakeHostBackendForTest, tierTestBytes, started)
	recordWeightUpload(&Model{Cfg: Config{Name: tierTestModel, NumLayers: 4}}, nil, tierTestBytes, started)
	recordWeightUpload(&Model{Cfg: Config{Name: tierTestModel, NumLayers: 4}}, fakeDeviceBackendForTest, 0, started)
	tiers, transfers = rec.Drain()
	if got := findTier(tiers, tierTestModel, hwutil.TierVRAM); len(got) != 0 {
		t.Fatalf("non-device uploads leaked marker vram rows: %+v", got)
	}
	for _, tr := range transfers {
		if tr.Bytes == tierTestBytes && tr.FromTier == "ram" && tr.ToTier == "vram" {
			t.Fatalf("non-device uploads leaked marker transfer: %+v", tr)
		}
	}
}

// TestTierTensorBytesOf pins the element-based sizing used for upload rows.
func TestTierTensorBytesOf(t *testing.T) {
	if got := tensorBytesOf(compute.Tensor{Shape: []int{2, 3}, Dtype: compute.F32}); got != 24 {
		t.Fatalf("f32 2x3 tensor bytes = %d, want 24", got)
	}
	if got := tensorBytesOf(compute.Tensor{Shape: []int{0}, Dtype: compute.F32}); got != 0 {
		t.Fatalf("zero-dim shape bytes = %d, want 0", got)
	}
}
