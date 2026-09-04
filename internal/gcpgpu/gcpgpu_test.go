package gcpgpu

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
)

func TestAcceleratorDescriptors(t *testing.T) {
	cases := []struct {
		accel        AcceleratorType
		expectedMem  int64
		expectedArch string
		expectedCap  string
		expectedRank int
		expectedFam  string
	}{
		{L4, 24 * GiB, "Ada Lovelace", "8.9", 70, "NVIDIA_L4"},
		{A100_40GB, 40 * GiB, "Ampere", "8.0", 75, "NVIDIA_A100"},
		{A100_80GB, 80 * GiB, "Ampere", "8.0", 80, "NVIDIA_A100_80GB"},
		{H100_80GB, 80 * GiB, "Hopper", "9.0", 90, "NVIDIA_H100"},
		{T4, 16 * GiB, "Turing", "7.5", 50, "NVIDIA_T4"},
	}

	for _, tc := range cases {
		t.Run(string(tc.accel), func(t *testing.T) {
			if !tc.accel.IsValid() {
				t.Fatalf("expected %s to be valid", tc.accel)
			}
			if tc.accel.MemoryPerGPU() != tc.expectedMem {
				t.Errorf("memory: got %d, want %d", tc.accel.MemoryPerGPU(), tc.expectedMem)
			}
			if tc.accel.Architecture() != tc.expectedArch {
				t.Errorf("arch: got %s, want %s", tc.accel.Architecture(), tc.expectedArch)
			}
			if tc.accel.ComputeCapability() != tc.expectedCap {
				t.Errorf("compute cap: got %s, want %s", tc.accel.ComputeCapability(), tc.expectedCap)
			}
			if tc.accel.GenerationRank() != tc.expectedRank {
				t.Errorf("rank: got %d, want %d", tc.accel.GenerationRank(), tc.expectedRank)
			}
			if tc.accel.GPUFamily() != tc.expectedFam {
				t.Errorf("family: got %s, want %s", tc.accel.GPUFamily(), tc.expectedFam)
			}
		})
	}

	// Invalid accelerator
	invalid := AcceleratorType("amd-mi300")
	if invalid.IsValid() {
		t.Errorf("expected invalid accelerator to report false")
	}
	if invalid.MemoryPerGPU() != 0 {
		t.Errorf("expected 0 memory for invalid, got %d", invalid.MemoryPerGPU())
	}
	if invalid.GenerationRank() != 0 {
		t.Errorf("expected 0 rank for invalid, got %d", invalid.GenerationRank())
	}
}

func TestParseAccelerator(t *testing.T) {
	tests := []struct {
		input string
		want  AcceleratorType
	}{
		{"l4", L4},
		{"NVIDIA-L4", L4},
		{"a100-40gb", A100_40GB},
		{"nvidia-tesla-a100", A100_40GB},
		{"a100", A100_40GB},
		{"a100-80gb", A100_80GB},
		{"nvidia-a100-80gb", A100_80GB},
		{"H100", H100_80GB},
		{"nvidia-h100-80gb", H100_80GB},
		{"t4", T4},
		{"nvidia-tesla-t4", T4},
	}

	for _, tt := range tests {
		got, err := ParseAccelerator(tt.input)
		if err != nil {
			t.Errorf("ParseAccelerator(%q) unexpected error: %v", tt.input, err)
			continue
		}
		if got != tt.want {
			t.Errorf("ParseAccelerator(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}

	_, err := ParseAccelerator("unknown-gpu")
	if !errors.Is(err, ErrInvalidAccelerator) {
		t.Errorf("expected ErrInvalidAccelerator for unknown, got: %v", err)
	}
}

func TestInstanceSpecValidation(t *testing.T) {
	// Valid spec
	spec, err := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.VRAMBytesPerGPU != 24*GiB {
		t.Errorf("expected 24 GiB VRAM per GPU, got %d", spec.VRAMBytesPerGPU)
	}
	if spec.TotalVRAMBytes != 24*GiB {
		t.Errorf("expected 24 GiB total VRAM, got %d", spec.TotalVRAMBytes)
	}

	// Multi-GPU spec
	spec4, err := NewInstanceSpec("a3-highgpu-4g", H100_80GB, 4, 104, 940*GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec4.TotalVRAMBytes != 4*80*GiB {
		t.Errorf("expected 320 GiB total VRAM, got %d", spec4.TotalVRAMBytes)
	}

	// Validation failures
	badCases := []struct {
		name        string
		machineType string
		accel       AcceleratorType
		count       int
		vcpus       int
		mem         int64
		wantErr     error
	}{
		{"empty machine type", "", L4, 1, 8, 32 * GiB, ErrInvalidNodeSpec},
		{"invalid accel", "g2-standard-8", "fake-gpu", 1, 8, 32 * GiB, ErrInvalidAccelerator},
		{"zero gpu count", "g2-standard-8", L4, 0, 8, 32 * GiB, ErrInvalidNodeSpec},
		{"negative gpu count", "g2-standard-8", L4, -2, 8, 32 * GiB, ErrInvalidNodeSpec},
		{"zero vcpus", "g2-standard-8", L4, 1, 0, 32 * GiB, ErrInvalidNodeSpec},
		{"negative host memory", "g2-standard-8", L4, 1, 8, -100, ErrInvalidNodeSpec},
	}

	for _, tc := range badCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewInstanceSpec(tc.machineType, tc.accel, tc.count, tc.vcpus, tc.mem)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("expected %v, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestNodeRegistrationAndLifecycle(t *testing.T) {
	mgr := NewFleetManager()

	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	node := GPUNode{
		ID:       "node-l4-01",
		Name:     "L4 Worker 01",
		Project:  "test-project",
		Zone:     "us-central1-a",
		Spec:     spec,
		Endpoint: "10.128.0.10:4765",
		Labels:   map[string]string{"env": "dev"},
	}

	// Register
	if err := mgr.RegisterNode(node); err != nil {
		t.Fatalf("failed to register node: %v", err)
	}

	// Region should be auto-derived from zone
	stored, err := mgr.GetNode("node-l4-01")
	if err != nil {
		t.Fatalf("failed to get node: %v", err)
	}
	if stored.Region != "us-central1" {
		t.Errorf("expected derived region 'us-central1', got %q", stored.Region)
	}
	if stored.Status != Ready {
		t.Errorf("expected default status Ready, got %q", stored.Status)
	}
	if !stored.IsDispatchReady() {
		t.Errorf("expected node to be dispatch ready")
	}

	// Duplicate registration must fail
	if err := mgr.RegisterNode(node); !errors.Is(err, ErrNodeAlreadyExists) {
		t.Errorf("expected ErrNodeAlreadyExists, got %v", err)
	}

	// Empty ID must fail
	badNode := node
	badNode.ID = "  "
	if err := mgr.RegisterNode(badNode); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for empty ID, got %v", err)
	}

	// Update node status
	if err := mgr.UpdateNodeStatus("node-l4-01", Busy); err != nil {
		t.Fatalf("failed to update status: %v", err)
	}
	updated, _ := mgr.GetNode("node-l4-01")
	if updated.Status != Busy {
		t.Errorf("expected status Busy, got %q", updated.Status)
	}
	if updated.IsDispatchReady() {
		t.Errorf("busy node should not be dispatch ready")
	}

	// List nodes
	nodes := mgr.ListNodes()
	if len(nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(nodes))
	}

	// Unregister
	if err := mgr.UnregisterNode("node-l4-01"); err != nil {
		t.Fatalf("failed to unregister node: %v", err)
	}
	if _, err := mgr.GetNode("node-l4-01"); !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound after unregister, got %v", err)
	}
	if err := mgr.UnregisterNode("node-l4-01"); !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound on repeated unregister, got %v", err)
	}
}

func TestBestNodeForModel_VRAMMatching(t *testing.T) {
	mgr := NewFleetManager()

	// 1. Empty fleet
	_, err := mgr.BestNodeForModel(10 * GiB)
	if !errors.Is(err, ErrFleetEmpty) {
		t.Errorf("expected ErrFleetEmpty on empty fleet, got %v", err)
	}

	// Register 4 nodes with various VRAM capacities:
	// - T4: 16 GiB total VRAM
	// - L4: 24 GiB total VRAM
	// - A100_40GB: 40 GiB total VRAM
	// - H100_80GB: 80 GiB total VRAM
	t4Spec, _ := NewInstanceSpec("n1-standard-4", T4, 1, 4, 15*GiB)
	l4Spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	a100Spec, _ := NewInstanceSpec("a2-highgpu-1g", A100_40GB, 1, 12, 85*GiB)
	h100Spec, _ := NewInstanceSpec("a3-highgpu-1g", H100_80GB, 1, 26, 235*GiB)

	_ = mgr.RegisterNode(GPUNode{ID: "node-t4", Spec: t4Spec, Endpoint: "10.0.0.1:4765"})
	_ = mgr.RegisterNode(GPUNode{ID: "node-l4", Spec: l4Spec, Endpoint: "10.0.0.2:4765"})
	_ = mgr.RegisterNode(GPUNode{ID: "node-a100", Spec: a100Spec, Endpoint: "10.0.0.3:4765"})
	_ = mgr.RegisterNode(GPUNode{ID: "node-h100", Spec: h100Spec, Endpoint: "10.0.0.4:4765"})

	// Model needing 12 GiB VRAM:
	// All can fit (16, 24, 40, 80).
	// Best-fit should select T4 (16 - 12 = 4 GiB waste, minimal headroom waste).
	best, err := mgr.BestNodeForModel(12 * GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if best.ID != "node-t4" {
		t.Errorf("expected best-fit 'node-t4' for 12 GiB, got %q (accel: %s)", best.ID, best.Spec.Accelerator)
	}

	// Model needing 20 GiB VRAM:
	// T4 (16 GiB) cannot fit.
	// Best-fit should select L4 (24 - 20 = 4 GiB waste).
	best, err = mgr.BestNodeForModel(20 * GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if best.ID != "node-l4" {
		t.Errorf("expected best-fit 'node-l4' for 20 GiB, got %q", best.ID)
	}

	// Model needing 35 GiB VRAM:
	// T4 and L4 cannot fit.
	// Best-fit should select A100 (40 - 35 = 5 GiB waste).
	best, err = mgr.BestNodeForModel(35 * GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if best.ID != "node-a100" {
		t.Errorf("expected best-fit 'node-a100' for 35 GiB, got %q", best.ID)
	}

	// Model needing 70 GiB VRAM:
	// Only H100 (80 GiB) fits.
	best, err = mgr.BestNodeForModel(70 * GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if best.ID != "node-h100" {
		t.Errorf("expected 'node-h100' for 70 GiB, got %q", best.ID)
	}

	// Model needing 120 GiB VRAM:
	// Exceeds all single nodes: should return ErrInsufficientVRAM.
	_, err = mgr.BestNodeForModel(120 * GiB)
	if !errors.Is(err, ErrInsufficientVRAM) {
		t.Errorf("expected ErrInsufficientVRAM for 120 GiB, got %v", err)
	}
}

func TestBestNodeForModel_TieBreakers(t *testing.T) {
	mgr := NewFleetManager()

	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)

	// Two L4 nodes with identical capacity:
	// node-a has 50% utilization, node-b has 10% utilization.
	_ = mgr.RegisterNode(GPUNode{
		ID:       "node-a",
		Spec:     spec,
		Endpoint: "10.0.0.1:4765",
		Telemetry: TelemetryMetrics{
			GPUUtilizationPct: 50.0,
			ActiveJobs:        2,
		},
	})
	_ = mgr.RegisterNode(GPUNode{
		ID:       "node-b",
		Spec:     spec,
		Endpoint: "10.0.0.2:4765",
		Telemetry: TelemetryMetrics{
			GPUUtilizationPct: 10.0,
			ActiveJobs:        0,
		},
	})

	// BestNodeForModel should pick node-b due to lower utilization
	best, err := mgr.BestNodeForModel(10 * GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if best.ID != "node-b" {
		t.Errorf("expected node-b with lower utilization, got %q", best.ID)
	}

	// Equal utilization, but node-c has older silicon than node-d:
	mgr2 := NewFleetManager()
	a100_80, _ := NewInstanceSpec("a2-ultragpu-1g", A100_80GB, 1, 12, 170*GiB)
	h100_80, _ := NewInstanceSpec("a3-highgpu-1g", H100_80GB, 1, 26, 235*GiB)
	_ = mgr2.RegisterNode(GPUNode{ID: "node-a100-80", Spec: a100_80, Endpoint: "10.0.0.3:4765"})
	_ = mgr2.RegisterNode(GPUNode{ID: "node-h100-80", Spec: h100_80, Endpoint: "10.0.0.4:4765"})

	// Both have 80 GiB. For 60 GiB model, both have 20 GiB headroom.
	// H100 has generation rank 90 vs A100 rank 80, so H100 should be preferred!
	bestGen, err := mgr2.BestNodeForModel(60 * GiB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bestGen.ID != "node-h100-80" {
		t.Errorf("expected newer generation node-h100-80, got %q", bestGen.ID)
	}
}

func TestHealthProbing(t *testing.T) {
	mgr := NewFleetManager()
	ctx := context.Background()

	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	_ = mgr.RegisterNode(GPUNode{
		ID:       "healthy-node",
		Spec:     spec,
		Endpoint: "10.0.0.1:4765",
	})
	_ = mgr.RegisterNode(GPUNode{
		ID:       "missing-endpoint-node",
		Spec:     spec,
		Endpoint: "", // unconfigured endpoint
	})

	// Probe healthy node
	res, err := mgr.ProbeNode(ctx, "healthy-node")
	if err != nil {
		t.Fatalf("unexpected probe error: %v", err)
	}
	if !res.Healthy {
		t.Errorf("expected healthy probe result, got unhealthy with errors: %v", res.Errors)
	}
	if res.Status != Ready {
		t.Errorf("expected status Ready, got %s", res.Status)
	}
	if res.Latency < 0 {
		t.Errorf("expected non-negative latency measurement, got %v", res.Latency)
	}

	// Probe missing endpoint node: should be marked Degraded
	resBad, err := mgr.ProbeNode(ctx, "missing-endpoint-node")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resBad.Healthy {
		t.Errorf("expected unhealthy probe result for missing endpoint")
	}
	if resBad.Status != Degraded {
		t.Errorf("expected Degraded status, got %s", resBad.Status)
	}

	// Node status in manager should reflect probe
	nodeBad, _ := mgr.GetNode("missing-endpoint-node")
	if nodeBad.Status != Degraded {
		t.Errorf("expected node status Degraded in manager, got %s", nodeBad.Status)
	}

	// Probe non-existent node
	_, err = mgr.ProbeNode(ctx, "non-existent")
	if !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound, got %v", err)
	}

	// Context cancellation during probe
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	resCanceled, err := mgr.ProbeNode(canceledCtx, "healthy-node")
	if err == nil {
		t.Errorf("expected error on cancelled context")
	}
	if resCanceled.Healthy {
		t.Errorf("expected unhealthy result on cancelled context")
	}

	// ProbeAll
	allResults, err := mgr.ProbeAll(context.Background())
	if err != nil {
		t.Fatalf("ProbeAll failed: %v", err)
	}
	if len(allResults) != 2 {
		t.Errorf("expected 2 probe results, got %d", len(allResults))
	}
}

func TestDegradedStateHandlingAndRecovery(t *testing.T) {
	// Custom prober that toggles health
	var (
		healthyMu sync.Mutex
		isHealthy = false
	)

	mgr := NewFleetManager(WithProber(func(ctx context.Context, node *GPUNode) (ProbeResult, error) {
		healthyMu.Lock()
		defer healthyMu.Unlock()
		if !isHealthy {
			return ProbeResult{
				NodeID:  node.ID,
				Healthy: false,
				Status:  Degraded,
				Errors:  []string{"driver communication timeout"},
			}, nil
		}
		return ProbeResult{
			NodeID:        node.ID,
			Healthy:       true,
			Status:        Ready,
			AvailableVRAM: node.AvailableVRAM(),
		}, nil
	}))

	spec, _ := NewInstanceSpec("a3-highgpu-1g", H100_80GB, 1, 26, 235*GiB)
	_ = mgr.RegisterNode(GPUNode{
		ID:       "flaky-h100",
		Spec:     spec,
		Endpoint: "10.0.0.5:4765",
	})

	// Initial probe: fails and degrades node
	res, _ := mgr.ProbeNode(context.Background(), "flaky-h100")
	if res.Healthy {
		t.Fatalf("expected unhealthy initial probe")
	}

	node, _ := mgr.GetNode("flaky-h100")
	if node.Status != Degraded {
		t.Errorf("expected Degraded status, got %s", node.Status)
	}
	if node.IsDispatchReady() {
		t.Errorf("degraded node must not be dispatch ready")
	}

	// BestNodeForModel must NOT allocate on degraded node even if it has 80 GiB free
	_, err := mgr.BestNodeForModel(20 * GiB)
	if !errors.Is(err, ErrNoReadyNodes) {
		t.Errorf("expected ErrNoReadyNodes, got %v", err)
	}

	// Recovery: make prober healthy and re-probe
	healthyMu.Lock()
	isHealthy = true
	healthyMu.Unlock()

	res2, _ := mgr.ProbeNode(context.Background(), "flaky-h100")
	if !res2.Healthy {
		t.Fatalf("expected recovered healthy probe")
	}

	nodeRecovered, _ := mgr.GetNode("flaky-h100")
	if nodeRecovered.Status != Ready {
		t.Errorf("expected Ready status after recovery, got %s", nodeRecovered.Status)
	}
	if !nodeRecovered.IsDispatchReady() {
		t.Errorf("recovered node should be dispatch ready")
	}

	// Now BestNodeForModel succeeds!
	best, err := mgr.BestNodeForModel(20 * GiB)
	if err != nil {
		t.Fatalf("unexpected error after recovery: %v", err)
	}
	if best.ID != "flaky-h100" {
		t.Errorf("expected 'flaky-h100', got %q", best.ID)
	}
}

func TestTelemetryAndThermalThrottling(t *testing.T) {
	mgr := NewFleetManager()

	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	_ = mgr.RegisterNode(GPUNode{
		ID:       "node-thermal",
		Spec:     spec,
		Endpoint: "10.0.0.1:4765",
	})

	// Normal telemetry update
	err := mgr.UpdateTelemetry("node-thermal", TelemetryMetrics{
		AllocatedVRAMBytes: 10 * GiB,
		UsedVRAMBytes:      8 * GiB,
		GPUUtilizationPct:  45.0,
		TemperatureCelsius: 72.0,
		ActiveJobs:         1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	n, _ := mgr.GetNode("node-thermal")
	if n.AvailableVRAM() != 14*GiB {
		t.Errorf("expected 14 GiB available VRAM, got %d", n.AvailableVRAM())
	}
	if n.Status != Ready {
		t.Errorf("expected Ready status, got %s", n.Status)
	}

	// Saturated VRAM marks node as Busy
	_ = mgr.UpdateTelemetry("node-thermal", TelemetryMetrics{
		AllocatedVRAMBytes: 24 * GiB,
		TemperatureCelsius: 75.0,
	})
	nBusy, _ := mgr.GetNode("node-thermal")
	if nBusy.Status != Busy {
		t.Errorf("expected Busy status on VRAM saturation, got %s", nBusy.Status)
	}

	// Freeing VRAM restores Ready status
	_ = mgr.UpdateTelemetry("node-thermal", TelemetryMetrics{
		AllocatedVRAMBytes: 12 * GiB,
		TemperatureCelsius: 75.0,
	})
	nReady, _ := mgr.GetNode("node-thermal")
	if nReady.Status != Ready {
		t.Errorf("expected Ready status after freeing VRAM, got %s", nReady.Status)
	}

	// Critical temperature triggers Degraded state
	_ = mgr.UpdateTelemetry("node-thermal", TelemetryMetrics{
		TemperatureCelsius: 96.5,
	})
	nDegraded, _ := mgr.GetNode("node-thermal")
	if nDegraded.Status != Degraded {
		t.Errorf("expected Degraded status on critical thermal, got %s", nDegraded.Status)
	}
	if len(nDegraded.HealthErrors) == 0 {
		t.Errorf("expected thermal health error message")
	}

	// Non-existent node update
	if err := mgr.UpdateTelemetry("ghost", TelemetryMetrics{}); !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound, got %v", err)
	}
}

func TestVRAMAllocationAndRelease(t *testing.T) {
	mgr := NewFleetManager()

	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	_ = mgr.RegisterNode(GPUNode{
		ID:       "alloc-node",
		Spec:     spec,
		Endpoint: "10.0.0.1:4765",
	})

	// Allocate 10 GiB
	if err := mgr.AllocateVRAM("alloc-node", 10*GiB); err != nil {
		t.Fatalf("unexpected allocation error: %v", err)
	}
	node, _ := mgr.GetNode("alloc-node")
	if node.AvailableVRAM() != 14*GiB {
		t.Errorf("expected 14 GiB free, got %d", node.AvailableVRAM())
	}
	if node.Telemetry.ActiveJobs != 1 {
		t.Errorf("expected 1 active job, got %d", node.Telemetry.ActiveJobs)
	}

	// Over-allocation should fail
	if err := mgr.AllocateVRAM("alloc-node", 20*GiB); !errors.Is(err, ErrInsufficientVRAM) {
		t.Errorf("expected ErrInsufficientVRAM, got %v", err)
	}

	// Release 5 GiB
	if err := mgr.ReleaseVRAM("alloc-node", 5*GiB); err != nil {
		t.Fatalf("unexpected release error: %v", err)
	}
	node, _ = mgr.GetNode("alloc-node")
	if node.AvailableVRAM() != 19*GiB {
		t.Errorf("expected 19 GiB free, got %d", node.AvailableVRAM())
	}
	if node.Telemetry.ActiveJobs != 0 {
		t.Errorf("expected 0 active jobs, got %d", node.Telemetry.ActiveJobs)
	}
}

func TestQuotaTrackingAndEnforcement(t *testing.T) {
	mgr := NewFleetManager(WithQuotaEnforcement(true))

	// Set quota of 2 L4 GPUs in us-central1
	if err := mgr.SetQuota(L4, "us-central1", 2); err != nil {
		t.Fatalf("SetQuota failed: %v", err)
	}

	quota, ok := mgr.GetQuota(L4, "us-central1")
	if !ok || quota.Limit != 2 {
		t.Fatalf("expected quota limit 2, got %v", quota)
	}
	if quota.Available() != 2 {
		t.Errorf("expected 2 available quota units, got %d", quota.Available())
	}

	spec1, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	// Register first L4: succeeds
	if err := mgr.RegisterNode(GPUNode{
		ID:       "l4-01",
		Zone:     "us-central1-a",
		Spec:     spec1,
		Endpoint: "10.0.0.1:4765",
	}); err != nil {
		t.Fatalf("failed to register first node: %v", err)
	}

	// InUse should now be 1
	quota, _ = mgr.GetQuota(L4, "us-central1")
	if quota.InUse != 1 {
		t.Errorf("expected quota InUse 1, got %d", quota.InUse)
	}

	// Register second L4: succeeds (InUse = 2)
	if err := mgr.RegisterNode(GPUNode{
		ID:       "l4-02",
		Zone:     "us-central1-b",
		Spec:     spec1,
		Endpoint: "10.0.0.2:4765",
	}); err != nil {
		t.Fatalf("failed to register second node: %v", err)
	}

	// Register third L4: should be rejected by quota enforcement!
	err := mgr.RegisterNode(GPUNode{
		ID:       "l4-03",
		Zone:     "us-central1-a",
		Spec:     spec1,
		Endpoint: "10.0.0.3:4765",
	})
	if !errors.Is(err, ErrQuotaExceeded) {
		t.Errorf("expected ErrQuotaExceeded, got %v", err)
	}

	// Unregister one node: frees quota
	if err := mgr.UnregisterNode("l4-01"); err != nil {
		t.Fatalf("failed to unregister: %v", err)
	}

	// Now third node can register successfully
	if err := mgr.RegisterNode(GPUNode{
		ID:       "l4-03",
		Zone:     "us-central1-a",
		Spec:     spec1,
		Endpoint: "10.0.0.3:4765",
	}); err != nil {
		t.Fatalf("failed to register after quota freed: %v", err)
	}
}

func TestCapacityReport(t *testing.T) {
	mgr := NewFleetManager()

	_ = mgr.SetQuota(L4, "us-central1", 4)
	_ = mgr.SetQuota(H100_80GB, "us-east4", 8)

	l4Spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	h100Spec, _ := NewInstanceSpec("a3-highgpu-4g", H100_80GB, 4, 104, 940*GiB)

	// Node 1: Ready L4 in us-central1
	_ = mgr.RegisterNode(GPUNode{
		ID:       "node-1",
		Zone:     "us-central1-a",
		Spec:     l4Spec,
		Status:   Ready,
		Endpoint: "10.0.0.1:4765",
	})

	// Node 2: Busy L4 in us-central1 (24 GiB allocated)
	_ = mgr.RegisterNode(GPUNode{
		ID:       "node-2",
		Zone:     "us-central1-b",
		Spec:     l4Spec,
		Status:   Busy,
		Endpoint: "10.0.0.2:4765",
		Telemetry: TelemetryMetrics{
			AllocatedVRAMBytes: 24 * GiB,
		},
	})

	// Node 3: Ready 4xH100 in us-east4 (320 GiB VRAM, 80 GiB allocated)
	_ = mgr.RegisterNode(GPUNode{
		ID:       "node-3",
		Zone:     "us-east4-a",
		Spec:     h100Spec,
		Status:   Ready,
		Endpoint: "10.0.0.3:4765",
		Telemetry: TelemetryMetrics{
			AllocatedVRAMBytes: 80 * GiB,
		},
	})

	// Node 4: Degraded node
	_ = mgr.RegisterNode(GPUNode{
		ID:           "node-4",
		Zone:         "us-east4-b",
		Spec:         h100Spec,
		Status:       Degraded,
		Endpoint:     "10.0.0.4:4765",
		HealthErrors: []string{"ECC error detected"},
	})

	report := mgr.CapacityReport()

	if report.TotalNodes != 4 {
		t.Errorf("total nodes: got %d, want 4", report.TotalNodes)
	}
	if report.ReadyNodes != 2 {
		t.Errorf("ready nodes: got %d, want 2", report.ReadyNodes)
	}
	if report.BusyNodes != 1 {
		t.Errorf("busy nodes: got %d, want 1", report.BusyNodes)
	}
	if report.DegradedNodes != 1 {
		t.Errorf("degraded nodes: got %d, want 1", report.DegradedNodes)
	}
	if report.TotalGPUs != 10 { // 1 + 1 + 4 + 4
		t.Errorf("total GPUs: got %d, want 10", report.TotalGPUs)
	}
	if report.ReadyGPUs != 5 { // 1 + 4
		t.Errorf("ready GPUs: got %d, want 5", report.ReadyGPUs)
	}

	expectedTotalVRAM := int64(2*24*GiB + 2*320*GiB)
	if report.TotalVRAMBytes != expectedTotalVRAM {
		t.Errorf("total VRAM: got %d, want %d", report.TotalVRAMBytes, expectedTotalVRAM)
	}

	// Verify accelerator grouping
	l4Cap, ok := report.ByAccelerator[L4]
	if !ok || l4Cap.TotalNodes != 2 || l4Cap.ReadyNodes != 1 {
		t.Errorf("unexpected L4 capacity: %+v", l4Cap)
	}

	h100Cap, ok := report.ByAccelerator[H100_80GB]
	if !ok || h100Cap.TotalNodes != 2 || h100Cap.ReadyGPUs != 4 {
		t.Errorf("unexpected H100 capacity: %+v", h100Cap)
	}

	// Verify region grouping
	usCentral, ok := report.ByRegion["us-central1"]
	if !ok || usCentral.TotalNodes != 2 {
		t.Errorf("unexpected us-central1 capacity: %+v", usCentral)
	}

	// Dispatch ready nodes should contain node-1 and node-3 (node-2 is busy, node-4 is degraded)
	if len(report.DispatchReadyNodes) != 2 {
		t.Errorf("expected 2 dispatch ready nodes, got %v", report.DispatchReadyNodes)
	}
}

func TestConcurrency_ThreadSafety(t *testing.T) {
	mgr := NewFleetManager()
	ctx := context.Background()

	l4Spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)

	// Pre-register some initial nodes
	for i := 0; i < 10; i++ {
		_ = mgr.RegisterNode(GPUNode{
			ID:       fmt.Sprintf("init-node-%02d", i),
			Zone:     "us-central1-a",
			Spec:     l4Spec,
			Endpoint: fmt.Sprintf("10.0.0.%d:4765", i+1),
		})
	}

	const goroutines = 40
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				nodeID := fmt.Sprintf("dyn-node-%d-%d", workerID, i)

				switch (workerID + i) % 7 {
				case 0:
					// Register
					_ = mgr.RegisterNode(GPUNode{
						ID:       nodeID,
						Zone:     "us-central1-a",
						Spec:     l4Spec,
						Endpoint: "10.1.0.1:4765",
					})
				case 1:
					// Read BestNode
					_, _ = mgr.BestNodeForModel(10 * GiB)
				case 2:
					// Update telemetry
					_ = mgr.UpdateTelemetry(fmt.Sprintf("init-node-%02d", (workerID+i)%10), TelemetryMetrics{
						AllocatedVRAMBytes: int64(i%24) * GiB,
						GPUUtilizationPct:  float64(i % 100),
					})
				case 3:
					// Probe
					_, _ = mgr.ProbeNode(ctx, fmt.Sprintf("init-node-%02d", (workerID+i)%10))
				case 4:
					// Capacity report
					_ = mgr.CapacityReport()
				case 5:
					// Allocate / release
					targetID := fmt.Sprintf("init-node-%02d", (workerID+i)%10)
					if err := mgr.AllocateVRAM(targetID, 1*GiB); err == nil {
						_ = mgr.ReleaseVRAM(targetID, 1*GiB)
					}
				case 6:
					// Unregister dynamic node
					_ = mgr.UnregisterNode(nodeID)
				}
			}
		}(g)
	}

	wg.Wait()

	// Final verification: manager state should remain intact and queryable
	report := mgr.CapacityReport()
	if report.TotalNodes < 10 {
		t.Errorf("expected at least 10 nodes, got %d", report.TotalNodes)
	}
}
