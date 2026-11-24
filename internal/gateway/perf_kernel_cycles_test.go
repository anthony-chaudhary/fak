package gateway

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func kernelCyclesTotals() nativeperf.DecodeHostWorkTotals {
	return nativeperf.DecodeHostWorkTotals{
		Windows: 1, Wall: 20 * time.Millisecond, Steps: 4,
		Kernels: &compute.KernelProfile{Backend: "cuda", GPUNS: 1.6e7, WallNS: 2e7,
			Rows: []compute.KernelRow{{Kernel: "k_q4k_mmvq_gemv", GX: 11008, GY: 1, Calls: 8, NS: 1.6e7}}},
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestNativeKernelsCarriesSMClockCycles(t *testing.T) {
	t.Setenv(compute.SMClockEnv, "2500")
	k := nativeKernels(kernelCyclesTotals())
	if k == nil || k.ClockSource != compute.SMClockDeclared || k.SMClockMHz != 2500 || k.GPUMCyclesPerToken != 10 {
		t.Fatalf("kernels = %+v, want declared 2500 MHz and 10 Mcycles/token", k)
	}
	if r := k.SelectedKernels[0]; r.CyclesPerCall != 5e6 {
		t.Fatalf("cycles/call = %v, want 5e6 (2ms at 2.5 GHz)", r.CyclesPerCall)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestNativeKernelsOmitsCyclesWithoutClock(t *testing.T) {
	t.Setenv(compute.SMClockEnv, "")
	k := nativeKernels(kernelCyclesTotals())
	if k == nil || k.ClockSource != "" || k.GPUMCyclesPerToken != 0 || k.SelectedKernels[0].CyclesPerCall != 0 {
		t.Fatalf("kernels = %+v, want no cycle fields without a clock", k)
	}
}
