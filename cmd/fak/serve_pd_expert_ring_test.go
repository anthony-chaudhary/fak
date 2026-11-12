package main

import (
	"os"
	"path/filepath"
	"testing"
)

// ticket s2-pd-load-arms-20261108: the P/D role's resident Q4_K arm must take the same
// activated-expert device-ring placement as the single-worker serve when the full device
// plan is FitTooBig on an eligible discrete CUDA backend, instead of refusing the load.

// fak-test:runtime fast est=5s lane=default
func TestServePDQ4KArmTakesExpertRingWhenFullPlanFitTooBig(t *testing.T) {
	v41PackedTestEnv(t)
	const E, H, I = 2, 256, 256
	path := filepath.Join(t.TempDir(), "mixed-moe.gguf")
	if err := os.WriteFile(path, nonHaloRingMixedMoEGGUF(t, E, H, I), 0o600); err != nil {
		t.Fatal(err)
	}
	be := newNonHaloCUDARingBackend(nonHaloRingLoadBytes, nonHaloRingLoadBytes)
	fit := serveFitBudget{Base: nonHaloRingLoadBytes, Headroom: serveGGUFDeviceHeadroom}
	// 64-token context budget: the ring plan admits while the full plan refuses
	// (the same proven constants as the single-worker load-selection witness).
	m, err := loadServePDQ4KModelPlaced(be, path, 64, &fit)
	if err != nil {
		t.Fatalf("pd q4k arm refused the ring route: %v", err)
	}
	defer m.CloseWeights()
	stats := m.ExpertCheckpointStats()
	if !stats.Enabled || stats.Tensors != 3*E {
		t.Fatalf("checkpoint tier must index the streamed expert ranges: enabled=%t tensors=%d want %d", stats.Enabled, stats.Tensors, 3*E)
	}
	if m.SharedExpertRingFor(be) == nil {
		t.Fatal("the admitted device ring must register the model-scoped shared ring")
	}
	report := m.ResidentReport()
	if report.KQuantTensors != 0 || report.KQuantBytes != 0 {
		t.Fatalf("stream-through experts must leave no packed expert residency: tensors=%d bytes=%d", report.KQuantTensors, report.KQuantBytes)
	}
}

// fak-test:runtime fast est=5s lane=default
func TestServePDQ4KArmFitsPlanLoadsRetained(t *testing.T) {
	v41PackedTestEnv(t)
	const E, H, I = 2, 256, 256
	path := filepath.Join(t.TempDir(), "mixed-moe.gguf")
	if err := os.WriteFile(path, nonHaloRingMixedMoEGGUF(t, E, H, I), 0o600); err != nil {
		t.Fatal(err)
	}
	// A budget that admits the FULL plan: the arm must NOT take the ring and must load the
	// complete checkpoint through the retained entry (experts materialize, nothing streams).
	be := newNonHaloCUDARingBackend(8<<20, 8<<20)
	fit := serveFitBudget{Base: 8 << 20, Headroom: serveGGUFDeviceHeadroom}
	m, err := loadServePDQ4KModelPlaced(be, path, 64, &fit)
	if err != nil {
		t.Fatalf("pd q4k arm refused a fitting plan: %v", err)
	}
	defer m.CloseWeights()
	stats := m.ExpertCheckpointStats()
	if stats.Enabled {
		t.Fatalf("a fitting plan must not stream the experts: %+v", stats)
	}
	report := m.ResidentReport()
	if report.KQuantTensors != 3*E || report.KQuantBytes <= 0 {
		t.Fatalf("retained load must materialize the packed expert slabs: tensors=%d bytes=%d", report.KQuantTensors, report.KQuantBytes)
	}
}
