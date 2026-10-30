package main

import (
	"os"
	"path/filepath"
	"testing"

	computestrix "github.com/anthony-chaudhary/fak/internal/compute/strix"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// The --sizing ring report reads the same eligibility predicate the serve uses
// (serveExpertRingEligible): an ordinary discrete-CUDA serve on a non-Halo host reaches the ring
// projection, while an integrated tier without the device-only policy keeps the refusal. Reaching
// the projection is observed through the device capacity it records, not through reason prose.
//
// fak-test:runtime fast est=50ms lane=default
func TestServeSizingRingNonHaloCUDARingReportMatchesServe(t *testing.T) {
	if halo, _, err := computestrix.DetectPhysicalGFX1151("", ""); err != nil || halo {
		t.Skip("physical Halo host: the device-only arm applies instead")
	}
	const E, H, I = 2, 256, 256
	path := filepath.Join(t.TempDir(), "mixed-moe.gguf")
	if err := os.WriteFile(path, nonHaloRingMixedMoEGGUF(t, E, H, I), 0o600); err != nil {
		t.Fatal(err)
	}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	t.Setenv(FakExpertRingBytesEnv, "")
	t.Setenv("FAK_Q4K", "")

	report := func(be nonHaloCUDARingBackend) *serveSizingRing {
		fs, sf := newServeFlagSet()
		if err := fs.Parse([]string{"--plan-json", "--gguf", path}); err != nil {
			t.Fatal(err)
		}
		return buildServeSizingRingForFlags(ws, be, sf)
	}

	cuda := newNonHaloCUDARingBackend(nonHaloRingSizingBytes, nonHaloRingSizingBytes)
	if r := report(cuda); r.DeviceTotalBytes != nonHaloRingSizingBytes || r.DeviceFreeBytes != nonHaloRingSizingBytes {
		t.Fatalf("non-Halo discrete CUDA ring report did not reach the projection: fit=%s device_total=%d", r.Fit, r.DeviceTotalBytes)
	}

	integrated := newNonHaloCUDARingBackend(nonHaloRingSizingBytes, nonHaloRingSizingBytes)
	integrated.tier = "integrated:sm90"
	if r := report(integrated); r.DeviceTotalBytes != 0 || r.Fit != "UNKNOWN" {
		t.Fatalf("integrated tier without the device-only policy must keep the refusal: fit=%s device_total=%d", r.Fit, r.DeviceTotalBytes)
	}
}
