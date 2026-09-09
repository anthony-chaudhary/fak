package strix_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/strix"
)

type mockAccelerationEngine struct {
	name      string
	supported bool
}

func (m *mockAccelerationEngine) Name() string                              { return m.name }
func (m *mockAccelerationEngine) IsSupported() bool                         { return m.supported }
func (m *mockAccelerationEngine) WavefrontSize() int                        { return 32 }
func (m *mockAccelerationEngine) ComputeUnits() int                         { return 40 }
func (m *mockAccelerationEngine) AllocUSWCGTT(bytes int64) (uintptr, error) { return 0x2000, nil }

var _ strix.AccelerationEngine = (*mockAccelerationEngine)(nil)

func TestPublicInterfaceDefinitions(t *testing.T) {
	var engine strix.AccelerationEngine = &mockAccelerationEngine{
		name:      "mock-engine",
		supported: true,
	}

	if engine.Name() != "mock-engine" {
		t.Fatalf("expected mock-engine, got %s", engine.Name())
	}
	if !engine.IsSupported() {
		t.Fatalf("expected supported")
	}
	if engine.WavefrontSize() != 32 {
		t.Fatalf("expected wavefront 32, got %d", engine.WavefrontSize())
	}
	if engine.ComputeUnits() != 40 {
		t.Fatalf("expected 40 CUs, got %d", engine.ComputeUnits())
	}

	addr, err := engine.AllocUSWCGTT(1024)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if addr != 0x2000 {
		t.Fatalf("expected address 0x2000, got 0x%x", addr)
	}

	profile := strix.DefaultStrixHaloProfile()
	if profile.Architecture != "gfx1151" {
		t.Fatalf("expected gfx1151, got %s", profile.Architecture)
	}
	if profile.CUCount != 40 {
		t.Fatalf("expected 40 CUs, got %d", profile.CUCount)
	}
	if profile.Wavefront != 32 {
		t.Fatalf("expected 32 wavefront, got %d", profile.Wavefront)
	}
	if profile.BusWidthBits != 256 {
		t.Fatalf("expected 256-bit bus, got %d", profile.BusWidthBits)
	}
	if profile.PeakBandwidthGBps <= 0 {
		t.Fatalf("expected positive peak bandwidth, got %f", profile.PeakBandwidthGBps)
	}
}
