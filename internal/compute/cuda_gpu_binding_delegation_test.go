package compute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/gpubinding"
)

// fak-test:runtime fast est=5ms lane=default
// TestGPUBindingDelegatesToSeam pins the historical compute.* names to the
// pkg/gpubinding seam they now alias: identity of types and values, same
// verdicts from the legacy entry points, and the legacy orchestrator's
// unsupported-platform behavior. The doctor's full suite lives at the seam.
func TestGPUBindingDelegatesToSeam(t *testing.T) {
	var v GPUBindingVerdict = GPUReady
	if v != gpubinding.VerdictReady || string(v) != "GPU_READY" {
		t.Fatalf("verdict alias broken: %v (%q)", v, string(v))
	}
	if BlackwellDriverFloor != gpubinding.BlackwellFloor {
		t.Fatalf("floor alias mismatch: %+v vs %+v", BlackwellDriverFloor, gpubinding.BlackwellFloor)
	}
	if CUDASourceWindowsPnP != gpubinding.SourceWindowsPnP || CUDASourceNvidiaSMI != gpubinding.SourceNvidiaSMI {
		t.Fatalf("source labels drifted from the seam")
	}

	ready := GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerYes, DriverStarted: GPULayerYes, CUDAVisible: GPULayerYes}
	if got := ClassifyGPUBinding(ready); got != GPUReady {
		t.Fatalf("ClassifyGPUBinding ready row = %v, want GPUReady", got)
	}
	low := ready
	low.DriverVersion = "550.54.15"
	if got := ClassifyGPUBindingWithFloor(low, BlackwellDriverFloor); got != GPUDriverBelowFloor {
		t.Fatalf("floored row = %v, want GPUDriverBelowFloor", got)
	}

	root := t.TempDir()
	rep := NewCUDAPCIeTopologyDiscovery(root + "/absent").DiagnoseGPUBinding(GPUBindingOptions{ExpectGPUs: 1})
	if rep.Status != CUDADiscoveryError || len(rep.Devices) != 0 || rep.Ready {
		t.Fatalf("legacy diagnose = status %s devices %d ready %v, want error / none / not-ready", rep.Status, len(rep.Devices), rep.Ready)
	}
}
