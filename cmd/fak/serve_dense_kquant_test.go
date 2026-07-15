package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestServeDenseKQuantOptionsDisableRawResidencyForDeviceBackend(t *testing.T) {
	if got := serveDenseKQuantOptions(nil); len(got) != 0 {
		t.Fatalf("CPU options = %d, want 0", len(got))
	}
	if got := serveDenseKQuantOptions(compute.Default()); len(got) != 1 {
		t.Fatalf("device options = %d, want 1 dense-k-quant Q8-routing option", len(got))
	}
}
