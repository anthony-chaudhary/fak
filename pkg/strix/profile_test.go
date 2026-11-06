package strix_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/strix"
)

// TestDefaultStrixHaloProfile pins the canonical GFX1151 silicon constants.
func TestDefaultStrixHaloProfile(t *testing.T) {
	p := strix.DefaultStrixHaloProfile()
	if p.Architecture != "gfx1151" {
		t.Fatalf("expected gfx1151, got %s", p.Architecture)
	}
	if p.CUCount != 40 {
		t.Fatalf("expected 40 CUs, got %d", p.CUCount)
	}
	if p.Wavefront != 32 {
		t.Fatalf("expected 32 wavefront, got %d", p.Wavefront)
	}
	if p.BusWidthBits != 256 {
		t.Fatalf("expected 256-bit bus, got %d", p.BusWidthBits)
	}
	if p.PeakBandwidthGBps != 273.056 {
		t.Fatalf("expected 273.056 GB/s, got %f", p.PeakBandwidthGBps)
	}
}
