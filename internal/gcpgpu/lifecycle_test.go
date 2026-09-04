package gcpgpu

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestFleetAllocationLifecycle tests end-to-end allocation, packing, and release.
func TestFleetAllocationLifecycle(t *testing.T) {
	mgr := NewFleetManager()

	l4Spec, err := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	if err != nil {
		t.Fatalf("failed to create L4 spec: %v", err)
	}
	t4Spec, err := NewInstanceSpec("n1-standard-4", T4, 1, 4, 15*GiB)
	if err != nil {
		t.Fatalf("failed to create T4 spec: %v", err)
	}
	a100Spec, err := NewInstanceSpec("a2-highgpu-1g", A100_40GB, 1, 12, 85*GiB)
	if err != nil {
		t.Fatalf("failed to create A100 spec: %v", err)
	}

	nodes := []GPUNode{
		{ID: "node-t4", Spec: t4Spec, Endpoint: "10.0.0.1:4765", Zone: "us-central1-a"},
		{ID: "node-l4", Spec: l4Spec, Endpoint: "10.0.0.2:4765", Zone: "us-central1-a"},
		{ID: "node-a100", Spec: a100Spec, Endpoint: "10.0.0.3:4765", Zone: "us-central1-a"},
	}

	for _, n := range nodes {
		if err := mgr.RegisterNode(n); err != nil {
			t.Fatalf("failed to register node %s: %v", n.ID, err)
		}
	}

	// Workload of 12 GiB should best-fit pack into node-t4 (16 GiB total, 4 GiB waste)
	best, err := mgr.BestNodeForModel(12 * GiB)
	if err != nil {
		t.Fatalf("BestNodeForModel error: %v", err)
	}
	if best.ID != "node-t4" {
		t.Fatalf("expected node-t4 for 12 GiB, got %s", best.ID)
	}

	// Allocate 12 GiB on node-t4
	if err := mgr.AllocateVRAM("node-t4", 12*GiB); err != nil {
		t.Fatalf("failed to allocate VRAM on node-t4: %v", err)
	}

	// Check updated state
	t4, err := mgr.GetNode("node-t4")
	if err != nil {
		t.Fatalf("failed to get node-t4: %v", err)
	}
	if t4.AvailableVRAM() != 4*GiB {
		t.Errorf("expected 4 GiB available, got %d", t4.AvailableVRAM())
	}
	if t4.Telemetry.ActiveJobs != 1 {
		t.Errorf("expected 1 active job, got %d", t4.Telemetry.ActiveJobs)
	}

	// Now another 12 GiB workload cannot fit on node-t4 (only 4 GiB left), so it should pick node-l4 (24 GiB total)
	best, err = mgr.BestNodeForModel(12 * GiB)
	if err != nil {
		t.Fatalf("BestNodeForModel error: %v", err)
	}
	if best.ID != "node-l4" {
		t.Fatalf("expected node-l4 for second 12 GiB, got %s", best.ID)
	}

	// Satiate node-t4 with remaining 4 GiB
	if err := mgr.AllocateVRAM("node-t4", 4*GiB); err != nil {
		t.Fatalf("failed to allocate remaining 4 GiB on node-t4: %v", err)
	}
	t4, _ = mgr.GetNode("node-t4")
	if t4.Status != Busy {
		t.Errorf("expected node-t4 to be Busy, got %s", t4.Status)
	}
	if t4.IsDispatchReady() {
		t.Errorf("expected saturated node-t4 not to be dispatch ready")
	}

	// Over-allocation attempt on node-t4 should fail
	if err := mgr.AllocateVRAM("node-t4", 1*GiB); !errors.Is(err, ErrNodeNotReady) {
		t.Errorf("expected ErrNodeNotReady on busy node, got: %v", err)
	}

	// Release 8 GiB from node-t4
	if err := mgr.ReleaseVRAM("node-t4", 8*GiB); err != nil {
		t.Fatalf("failed to release 8 GiB: %v", err)
	}
	t4, _ = mgr.GetNode("node-t4")
	if t4.Status != Ready {
		t.Errorf("expected node-t4 to return to Ready status, got %s", t4.Status)
	}
	if t4.AvailableVRAM() != 8*GiB {
		t.Errorf("expected 8 GiB available after release, got %d", t4.AvailableVRAM())
	}
	if t4.Telemetry.ActiveJobs != 1 {
		t.Errorf("expected 1 active job remaining, got %d", t4.Telemetry.ActiveJobs)
	}

	// Release excess VRAM beyond allocated to verify zero floor clamping
	if err := mgr.ReleaseVRAM("node-t4", 100*GiB); err != nil {
		t.Fatalf("release clamp error: %v", err)
	}
	t4, _ = mgr.GetNode("node-t4")
	if t4.Telemetry.AllocatedVRAMBytes != 0 {
		t.Errorf("expected 0 allocated VRAM after excess release, got %d", t4.Telemetry.AllocatedVRAMBytes)
	}
	if t4.Telemetry.ActiveJobs != 0 {
		t.Errorf("expected 0 active jobs after excess release, got %d", t4.Telemetry.ActiveJobs)
	}
}

// TestAllocationBoundsAndErrors verifies edge cases and parameter validation.
func TestAllocationBoundsAndErrors(t *testing.T) {
	mgr := NewFleetManager()

	// Empty fleet returns ErrFleetEmpty
	if _, err := mgr.BestNodeForModel(8 * GiB); !errors.Is(err, ErrFleetEmpty) {
		t.Errorf("expected ErrFleetEmpty on empty fleet, got: %v", err)
	}

	spec, err := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	if err != nil {
		t.Fatalf("spec error: %v", err)
	}
	if err := mgr.RegisterNode(GPUNode{ID: "node-1", Spec: spec, Endpoint: "10.0.0.1:4765"}); err != nil {
		t.Fatalf("register error: %v", err)
	}

	// Negative and zero VRAM requests
	if _, err := mgr.BestNodeForModel(0); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for 0 VRAM, got: %v", err)
	}
	if _, err := mgr.BestNodeForModel(-5 * GiB); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for negative VRAM, got: %v", err)
	}

	// Request exceeding capacity
	if _, err := mgr.BestNodeForModel(30 * GiB); !errors.Is(err, ErrInsufficientVRAM) {
		t.Errorf("expected ErrInsufficientVRAM for 30 GiB on 24 GiB node, got: %v", err)
	}

	// Allocate with invalid bounds
	if err := mgr.AllocateVRAM("node-1", 0); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for 0 alloc bytes, got: %v", err)
	}
	if err := mgr.AllocateVRAM("node-1", -10); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for negative alloc bytes, got: %v", err)
	}
	if err := mgr.AllocateVRAM("non-existent", 4*GiB); !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound for unknown node, got: %v", err)
	}

	// Release with invalid bounds
	if err := mgr.ReleaseVRAM("node-1", 0); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for 0 release bytes, got: %v", err)
	}
	if err := mgr.ReleaseVRAM("node-1", -10); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for negative release bytes, got: %v", err)
	}
	if err := mgr.ReleaseVRAM("non-existent", 4*GiB); !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound for unknown node, got: %v", err)
	}
}

// TestTieBreakingStrategies verifies all levels of BestNodeForModel selection.
func TestTieBreakingStrategies(t *testing.T) {
	mgr := NewFleetManager()

	specL4, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	specA100_40, _ := NewInstanceSpec("a2-highgpu-1g", A100_40GB, 1, 12, 85*GiB)

	// Two L4 nodes with identical specs and capacity
	nodeA := GPUNode{ID: "node-a", Spec: specL4, Endpoint: "10.0.0.1:4765"}
	nodeB := GPUNode{ID: "node-b", Spec: specL4, Endpoint: "10.0.0.2:4765"}

	_ = mgr.RegisterNode(nodeA)
	_ = mgr.RegisterNode(nodeB)

	// Level 1: Deterministic ID tie-breaker when all metrics are equal
	best, err := mgr.BestNodeForModel(10 * GiB)
	if err != nil {
		t.Fatalf("best node error: %v", err)
	}
	if best.ID != "node-a" {
		t.Errorf("expected node-a by lexicographical ID order, got %s", best.ID)
	}

	// Level 2: GPU utilization tie-breaker
	_ = mgr.UpdateTelemetry("node-a", TelemetryMetrics{GPUUtilizationPct: 45.0})
	_ = mgr.UpdateTelemetry("node-b", TelemetryMetrics{GPUUtilizationPct: 15.0})

	best, err = mgr.BestNodeForModel(10 * GiB)
	if err != nil {
		t.Fatalf("best node error: %v", err)
	}
	if best.ID != "node-b" {
		t.Errorf("expected node-b with lower GPU utilization (15%% vs 45%%), got %s", best.ID)
	}

	// Level 3: Active jobs tie-breaker when utilization matches
	_ = mgr.UpdateTelemetry("node-a", TelemetryMetrics{GPUUtilizationPct: 15.0, ActiveJobs: 1})
	_ = mgr.UpdateTelemetry("node-b", TelemetryMetrics{GPUUtilizationPct: 15.0, ActiveJobs: 3})

	best, err = mgr.BestNodeForModel(10 * GiB)
	if err != nil {
		t.Fatalf("best node error: %v", err)
	}
	if best.ID != "node-a" {
		t.Errorf("expected node-a with fewer active jobs (1 vs 3), got %s", best.ID)
	}

	// Level 4: Silicon generation rank tie-breaker
	// Register an A100 node and allocate memory so both have identical headroom waste
	nodeC := GPUNode{ID: "node-c", Spec: specA100_40, Endpoint: "10.0.0.3:4765"}
	_ = mgr.RegisterNode(nodeC)

	// Allocate 16 GiB on A100 so available is 24 GiB (identical to L4's 24 GiB)
	_ = mgr.AllocateVRAM("node-c", 16*GiB)
	_ = mgr.UpdateTelemetry("node-a", TelemetryMetrics{GPUUtilizationPct: 0, ActiveJobs: 0})
	_ = mgr.UpdateTelemetry("node-c", TelemetryMetrics{GPUUtilizationPct: 0, ActiveJobs: 0, AllocatedVRAMBytes: 16 * GiB})

	// For a 20 GiB model, both have 4 GiB headroom waste.
	// A100 (Ampere, rank 75) beats L4 (Ada, rank 70).
	best, err = mgr.BestNodeForModel(20 * GiB)
	if err != nil {
		t.Fatalf("best node error: %v", err)
	}
	if best.ID != "node-c" {
		t.Errorf("expected node-c (A100 rank 75 > L4 rank 70), got %s", best.ID)
	}
}

// TestQuotaEnforcementLifecycle tests quota checks, dynamic tracking, and rejection.
func TestQuotaEnforcementLifecycle(t *testing.T) {
	mgr := NewFleetManager(WithQuotaEnforcement(true))

	// Set regional quota: us-central1 for L4 allowed 2 GPUs
	if err := mgr.SetQuota(L4, "us-central1", 2); err != nil {
		t.Fatalf("failed to set quota: %v", err)
	}

	// Verify initial quota state
	q, ok := mgr.GetQuota(L4, "us-central1")
	if !ok {
		t.Fatalf("expected quota to be configured")
	}
	if q.Limit != 2 || q.InUse != 0 || q.Available() != 2 {
		t.Errorf("unexpected quota values: %+v", q)
	}

	spec1, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	spec2, _ := NewInstanceSpec("g2-standard-16", L4, 2, 16, 64*GiB)

	// Register 1st node (1 GPU) -> InUse becomes 1
	err := mgr.RegisterNode(GPUNode{ID: "node-1", Spec: spec1, Zone: "us-central1-a", Endpoint: "10.0.0.1:4765"})
	if err != nil {
		t.Fatalf("failed to register node-1: %v", err)
	}

	q, _ = mgr.GetQuota(L4, "us-central1")
	if q.InUse != 1 || q.Available() != 1 {
		t.Errorf("expected 1 in use, 1 available; got %+v", q)
	}

	// Attempt to register 2nd node with 2 GPUs (would exceed limit of 2: 1 + 2 = 3 > 2)
	err = mgr.RegisterNode(GPUNode{ID: "node-2", Spec: spec2, Zone: "us-central1-b", Endpoint: "10.0.0.2:4765"})
	if !errors.Is(err, ErrQuotaExceeded) {
		t.Fatalf("expected ErrQuotaExceeded, got: %v", err)
	}

	// Register a 1 GPU node -> fits exactly within limit (1 + 1 = 2)
	err = mgr.RegisterNode(GPUNode{ID: "node-3", Spec: spec1, Zone: "us-central1-c", Endpoint: "10.0.0.3:4765"})
	if err != nil {
		t.Fatalf("failed to register node-3: %v", err)
	}

	q, _ = mgr.GetQuota(L4, "us-central1")
	if q.InUse != 2 || q.Available() != 0 {
		t.Errorf("expected 2 in use, 0 available; got %+v", q)
	}

	// Unregister node-1 -> InUse drops to 1
	if err := mgr.UnregisterNode("node-1"); err != nil {
		t.Fatalf("failed to unregister node-1: %v", err)
	}

	q, _ = mgr.GetQuota(L4, "us-central1")
	if q.InUse != 1 || q.Available() != 1 {
		t.Errorf("expected 1 in use after unregister, got %+v", q)
	}

	// Test negative quota limit rejection
	if err := mgr.SetQuota(L4, "us-central1", -1); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for negative quota, got: %v", err)
	}

	// Test invalid accelerator in quota
	if err := mgr.SetQuota(AcceleratorType("amd-mi250"), "us-central1", 4); !errors.Is(err, ErrInvalidAccelerator) {
		t.Errorf("expected ErrInvalidAccelerator, got: %v", err)
	}

	// Test empty region
	if err := mgr.SetQuota(L4, "", 4); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec for empty region, got: %v", err)
	}
}

// TestHealthProbeAndDegradedLifecycle tests health evaluation and transitions.
func TestHealthProbeAndDegradedLifecycle(t *testing.T) {
	mgr := NewFleetManager()
	ctx := context.Background()

	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	node := GPUNode{
		ID:       "probe-node",
		Spec:     spec,
		Zone:     "us-central1-a",
		Endpoint: "10.0.0.1:4765",
	}
	if err := mgr.RegisterNode(node); err != nil {
		t.Fatalf("registration error: %v", err)
	}

	// Normal probe should succeed
	res, err := mgr.ProbeNode(ctx, "probe-node")
	if err != nil {
		t.Fatalf("probe error: %v", err)
	}
	if !res.Healthy || res.Status != Ready {
		t.Errorf("expected healthy and ready, got %+v", res)
	}

	// Simulate thermal critical temperature
	if err := mgr.UpdateTelemetry("probe-node", TelemetryMetrics{TemperatureCelsius: 96.5}); err != nil {
		t.Fatalf("telemetry error: %v", err)
	}
	updated, _ := mgr.GetNode("probe-node")
	if updated.Status != Degraded {
		t.Errorf("expected status Degraded after critical temperature, got %s", updated.Status)
	}
	if len(updated.HealthErrors) == 0 {
		t.Errorf("expected health errors to be populated")
	}

	// BestNodeForModel should skip Degraded nodes
	_, err = mgr.BestNodeForModel(4 * GiB)
	if !errors.Is(err, ErrNoReadyNodes) {
		t.Errorf("expected ErrNoReadyNodes while sole node is Degraded, got: %v", err)
	}

	// Probing degraded node should reflect Degraded status
	res, _ = mgr.ProbeNode(ctx, "probe-node")
	if res.Healthy {
		t.Errorf("expected probe result to report unhealthy for degraded node")
	}

	// Clear thermal issue and reset health
	_ = mgr.UpdateTelemetry("probe-node", TelemetryMetrics{TemperatureCelsius: 55.0})
	_ = mgr.UpdateNodeStatus("probe-node", Ready)

	// Inject a custom prober that simulates recovery
	mgrWithProber := NewFleetManager(WithProber(func(ctx context.Context, n *GPUNode) (ProbeResult, error) {
		return ProbeResult{
			NodeID:    n.ID,
			Healthy:   true,
			Status:    Ready,
			TotalVRAM: n.Spec.TotalVRAMBytes,
		}, nil
	}))
	_ = mgrWithProber.RegisterNode(node)
	_ = mgrWithProber.UpdateNodeStatus("probe-node", Degraded)

	// Probe should recover Degraded node back to Ready
	res, err = mgrWithProber.ProbeNode(ctx, "probe-node")
	if err != nil {
		t.Fatalf("prober error: %v", err)
	}
	if !res.Healthy {
		t.Errorf("expected custom prober to return healthy")
	}
	recovered, _ := mgrWithProber.GetNode("probe-node")
	if recovered.Status != Ready {
		t.Errorf("expected node to recover to Ready status, got %s", recovered.Status)
	}

	// Test context cancellation in prober
	cancellingCtx, cancel := context.WithCancel(context.Background())
	cancel()
	res, err = mgr.ProbeNode(cancellingCtx, "probe-node")
	if err == nil {
		t.Errorf("expected error on cancelled context probe")
	}
	if res.Healthy {
		t.Errorf("expected unhealthy probe result on cancelled context")
	}
}

// TestProbeAllConcurrency verifies parallel probe execution across fleet.
func TestProbeAllConcurrency(t *testing.T) {
	mgr := NewFleetManager()
	ctx := context.Background()

	const nodeCount = 10
	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	for i := 0; i < nodeCount; i++ {
		_ = mgr.RegisterNode(GPUNode{
			ID:       fmt.Sprintf("node-%02d", i),
			Spec:     spec,
			Endpoint: fmt.Sprintf("10.0.0.%d:4765", i+1),
		})
	}

	results, err := mgr.ProbeAll(ctx)
	if err != nil {
		t.Fatalf("ProbeAll error: %v", err)
	}
	if len(results) != nodeCount {
		t.Fatalf("expected %d probe results, got %d", nodeCount, len(results))
	}

	// Verify results are sorted by NodeID
	for i := 0; i < nodeCount-1; i++ {
		if results[i].NodeID >= results[i+1].NodeID {
			t.Errorf("results not sorted: %s >= %s", results[i].NodeID, results[i+1].NodeID)
		}
	}
}

// TestCapacityReportRollups verifies accurate rollups by accelerator and region.
func TestCapacityReportRollups(t *testing.T) {
	mgr := NewFleetManager()

	specL4, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	specH100, _ := NewInstanceSpec("a3-highgpu-1g", H100_80GB, 1, 26, 235*GiB)

	_ = mgr.RegisterNode(GPUNode{ID: "node-l4-1", Spec: specL4, Zone: "us-central1-a", Endpoint: "10.0.1.1:4765"})
	_ = mgr.RegisterNode(GPUNode{ID: "node-l4-2", Spec: specL4, Zone: "us-central1-b", Endpoint: "10.0.1.2:4765"})
	_ = mgr.RegisterNode(GPUNode{ID: "node-h100-1", Spec: specH100, Zone: "us-east4-a", Endpoint: "10.0.2.1:4765"})

	// Mark node-l4-2 as Degraded
	_ = mgr.UpdateNodeStatus("node-l4-2", Degraded)

	report := mgr.CapacityReport()
	if report.TotalNodes != 3 {
		t.Errorf("expected 3 total nodes, got %d", report.TotalNodes)
	}
	if report.ReadyNodes != 2 {
		t.Errorf("expected 2 ready nodes, got %d", report.ReadyNodes)
	}
	if report.DegradedNodes != 1 {
		t.Errorf("expected 1 degraded node, got %d", report.DegradedNodes)
	}
	if report.TotalGPUs != 3 {
		t.Errorf("expected 3 total GPUs, got %d", report.TotalGPUs)
	}

	// Verify L4 rollup
	l4Cap, exists := report.ByAccelerator[L4]
	if !exists {
		t.Fatalf("missing L4 capacity rollup")
	}
	if l4Cap.TotalNodes != 2 || l4Cap.ReadyNodes != 1 {
		t.Errorf("unexpected L4 counts: total=%d, ready=%d", l4Cap.TotalNodes, l4Cap.ReadyNodes)
	}

	// Verify H100 rollup
	h100Cap, exists := report.ByAccelerator[H100_80GB]
	if !exists {
		t.Fatalf("missing H100 capacity rollup")
	}
	if h100Cap.TotalNodes != 1 || h100Cap.ReadyNodes != 1 {
		t.Errorf("unexpected H100 counts: total=%d, ready=%d", h100Cap.TotalNodes, h100Cap.ReadyNodes)
	}

	// Verify Regional rollups
	usCentral, exists := report.ByRegion["us-central1"]
	if !exists {
		t.Fatalf("missing us-central1 region rollup")
	}
	if usCentral.TotalNodes != 2 || usCentral.ReadyNodes != 1 {
		t.Errorf("unexpected us-central1 counts: total=%d, ready=%d", usCentral.TotalNodes, usCentral.ReadyNodes)
	}

	usEast, exists := report.ByRegion["us-east4"]
	if !exists {
		t.Fatalf("missing us-east4 region rollup")
	}
	if usEast.TotalNodes != 1 || usEast.ReadyNodes != 1 {
		t.Errorf("unexpected us-east4 counts: total=%d, ready=%d", usEast.TotalNodes, usEast.ReadyNodes)
	}

	// Dispatch ready nodes should only contain ready nodes with valid endpoints
	if len(report.DispatchReadyNodes) != 2 {
		t.Errorf("expected 2 dispatch ready nodes, got %d", len(report.DispatchReadyNodes))
	}
}

// TestConcurrentAllocationSafety stresses concurrent allocations across goroutines.
func TestConcurrentAllocationSafety(t *testing.T) {
	mgr := NewFleetManager()

	spec, _ := NewInstanceSpec("a3-highgpu-8g", H100_80GB, 8, 208, 1880*GiB)
	// 8 * 80 GiB = 640 GiB
	node := GPUNode{ID: "mega-node", Spec: spec, Endpoint: "10.0.0.1:4765"}
	if err := mgr.RegisterNode(node); err != nil {
		t.Fatalf("register error: %v", err)
	}

	const routines = 20
	const cycles = 50
	var wg sync.WaitGroup
	wg.Add(routines)

	for r := 0; r < routines; r++ {
		go func() {
			defer wg.Done()
			for c := 0; c < cycles; c++ {
				// Allocate 4 GiB
				if err := mgr.AllocateVRAM("mega-node", 4*GiB); err == nil {
					// Release back
					_ = mgr.ReleaseVRAM("mega-node", 4*GiB)
				}
			}
		}()
	}

	wg.Wait()

	finalNode, err := mgr.GetNode("mega-node")
	if err != nil {
		t.Fatalf("get error: %v", err)
	}
	if finalNode.Telemetry.AllocatedVRAMBytes != 0 {
		t.Errorf("expected 0 allocated VRAM at rest, got %d", finalNode.Telemetry.AllocatedVRAMBytes)
	}
	if finalNode.Telemetry.ActiveJobs != 0 {
		t.Errorf("expected 0 active jobs at rest, got %d", finalNode.Telemetry.ActiveJobs)
	}
}

// TestFleetManagerDeterministicClock tests WithTimeSource clock injection.
func TestFleetManagerDeterministicClock(t *testing.T) {
	fixedTime := time.Date(2026, time.September, 4, 12, 0, 0, 0, time.UTC)
	mgr := NewFleetManager(WithTimeSource(func() time.Time {
		return fixedTime
	}))

	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	node := GPUNode{ID: "node-time", Spec: spec, Endpoint: "10.0.0.1:4765"}
	if err := mgr.RegisterNode(node); err != nil {
		t.Fatalf("register error: %v", err)
	}

	n, err := mgr.GetNode("node-time")
	if err != nil {
		t.Fatalf("get node error: %v", err)
	}
	if !n.RegisteredAt.Equal(fixedTime) {
		t.Errorf("expected RegisteredAt %v, got %v", fixedTime, n.RegisteredAt)
	}
	if !n.Telemetry.LastHeartbeat.Equal(fixedTime) {
		t.Errorf("expected LastHeartbeat %v, got %v", fixedTime, n.Telemetry.LastHeartbeat)
	}
}

// BenchmarkGCPGPU measures complete lifecycle operations on FleetManager.
func BenchmarkGCPGPU(b *testing.B) {
	mgr := NewFleetManager()
	specL4, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	specA100, _ := NewInstanceSpec("a2-highgpu-1g", A100_40GB, 1, 12, 85*GiB)

	_ = mgr.RegisterNode(GPUNode{ID: "bench-l4", Spec: specL4, Endpoint: "10.0.0.1:4765"})
	_ = mgr.RegisterNode(GPUNode{ID: "bench-a100", Spec: specA100, Endpoint: "10.0.0.2:4765"})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node, err := mgr.BestNodeForModel(16 * GiB)
		if err != nil {
			b.Fatalf("BestNodeForModel error: %v", err)
		}
		if err := mgr.AllocateVRAM(node.ID, 16*GiB); err != nil {
			b.Fatalf("AllocateVRAM error: %v", err)
		}
		_ = mgr.UpdateTelemetry(node.ID, TelemetryMetrics{GPUUtilizationPct: 50.0})
		if err := mgr.ReleaseVRAM(node.ID, 16*GiB); err != nil {
			b.Fatalf("ReleaseVRAM error: %v", err)
		}
	}
}

// BenchmarkGCPGPUBestNodeForModel benchmarks pure best-fit selection speed.
func BenchmarkGCPGPUBestNodeForModel(b *testing.B) {
	mgr := NewFleetManager()
	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)

	for i := 0; i < 50; i++ {
		_ = mgr.RegisterNode(GPUNode{
			ID:       fmt.Sprintf("bench-node-%02d", i),
			Spec:     spec,
			Endpoint: fmt.Sprintf("10.0.0.%d:4765", i+1),
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := mgr.BestNodeForModel(12 * GiB)
		if err != nil {
			b.Fatalf("BestNodeForModel error: %v", err)
		}
	}
}

// BenchmarkGCPGPUAllocationCycle benchmarks lock-synchronized allocate and release cycles.
func BenchmarkGCPGPUAllocationCycle(b *testing.B) {
	mgr := NewFleetManager()
	spec, _ := NewInstanceSpec("a3-highgpu-1g", H100_80GB, 1, 26, 235*GiB)
	_ = mgr.RegisterNode(GPUNode{ID: "bench-alloc", Spec: spec, Endpoint: "10.0.0.1:4765"})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mgr.AllocateVRAM("bench-alloc", 10*GiB)
		_ = mgr.ReleaseVRAM("bench-alloc", 10*GiB)
	}
}

// BenchmarkGCPGPUCapacityReport benchmarks fleet-wide capacity reporting.
func BenchmarkGCPGPUCapacityReport(b *testing.B) {
	mgr := NewFleetManager()
	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)

	for i := 0; i < 100; i++ {
		_ = mgr.RegisterNode(GPUNode{
			ID:       fmt.Sprintf("bench-rep-%03d", i),
			Spec:     spec,
			Zone:     "us-central1-a",
			Endpoint: fmt.Sprintf("10.0.1.%d:4765", i+1),
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mgr.CapacityReport()
	}
}
