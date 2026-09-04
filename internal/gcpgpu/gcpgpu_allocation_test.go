package gcpgpu

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestComplexAllocationScheduling(t *testing.T) {
	mgr := NewFleetManager()

	// Setup fleet with diverse accelerator types and topologies
	t4Spec, err := NewInstanceSpec("n1-standard-4", T4, 1, 4, 15*GiB)
	if err != nil {
		t.Fatalf("failed to create T4 spec: %v", err)
	}
	l4Spec, err := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	if err != nil {
		t.Fatalf("failed to create L4 spec: %v", err)
	}
	a100_40Spec, err := NewInstanceSpec("a2-highgpu-1g", A100_40GB, 1, 12, 85*GiB)
	if err != nil {
		t.Fatalf("failed to create A100-40GB spec: %v", err)
	}
	a100_80Spec, err := NewInstanceSpec("a2-ultragpu-1g", A100_80GB, 1, 12, 170*GiB)
	if err != nil {
		t.Fatalf("failed to create A100-80GB spec: %v", err)
	}
	h100Spec, err := NewInstanceSpec("a3-highgpu-1g", H100_80GB, 1, 26, 235*GiB)
	if err != nil {
		t.Fatalf("failed to create H100 spec: %v", err)
	}

	nodes := []GPUNode{
		{ID: "node-t4", Spec: t4Spec, Endpoint: "10.0.0.1:4765"},
		{ID: "node-l4", Spec: l4Spec, Endpoint: "10.0.0.2:4765"},
		{ID: "node-a100-40", Spec: a100_40Spec, Endpoint: "10.0.0.3:4765"},
		{ID: "node-a100-80", Spec: a100_80Spec, Endpoint: "10.0.0.4:4765"},
		{ID: "node-h100", Spec: h100Spec, Endpoint: "10.0.0.5:4765"},
	}

	for _, n := range nodes {
		if err := mgr.RegisterNode(n); err != nil {
			t.Fatalf("failed to register node %s: %v", n.ID, err)
		}
	}

	// 1. Initial 14 GiB workload should best-fit pack onto node-t4 (16 GiB total, 2 GiB waste)
	best, err := mgr.BestNodeForModel(14 * GiB)
	if err != nil {
		t.Fatalf("unexpected BestNodeForModel error: %v", err)
	}
	if best.ID != "node-t4" {
		t.Fatalf("expected node-t4 for 14 GiB, got %s", best.ID)
	}

	// Allocate 10 GiB on node-t4
	if err := mgr.AllocateVRAM("node-t4", 10*GiB); err != nil {
		t.Fatalf("failed to allocate 10 GiB on node-t4: %v", err)
	}
	t4Node, err := mgr.GetNode("node-t4")
	if err != nil {
		t.Fatalf("failed to get node-t4: %v", err)
	}
	if t4Node.AvailableVRAM() != 6*GiB {
		t.Errorf("expected 6 GiB free on node-t4, got %d", t4Node.AvailableVRAM())
	}

	// 2. Next 10 GiB workload cannot fit on node-t4 (6 GiB left), should pick node-l4 (24 GiB total, 14 GiB waste)
	best, err = mgr.BestNodeForModel(10 * GiB)
	if err != nil {
		t.Fatalf("unexpected BestNodeForModel error: %v", err)
	}
	if best.ID != "node-l4" {
		t.Fatalf("expected node-l4 for 10 GiB, got %s", best.ID)
	}

	// 3. Smaller 5 GiB workload should still pack into remaining 6 GiB on node-t4 (1 GiB waste vs 19 GiB waste on L4)
	best, err = mgr.BestNodeForModel(5 * GiB)
	if err != nil {
		t.Fatalf("unexpected BestNodeForModel error: %v", err)
	}
	if best.ID != "node-t4" {
		t.Fatalf("expected node-t4 for 5 GiB small workload, got %s", best.ID)
	}

	// Allocate the remaining 6 GiB on node-t4 to saturate it completely
	if err := mgr.AllocateVRAM("node-t4", 6*GiB); err != nil {
		t.Fatalf("failed to allocate 6 GiB on node-t4: %v", err)
	}
	t4Node, _ = mgr.GetNode("node-t4")
	if t4Node.Status != Busy {
		t.Errorf("expected node-t4 to be Busy on full saturation, got %s", t4Node.Status)
	}
	if t4Node.AvailableVRAM() != 0 {
		t.Errorf("expected 0 available VRAM on node-t4, got %d", t4Node.AvailableVRAM())
	}
	if t4Node.IsDispatchReady() {
		t.Errorf("expected saturated node-t4 not to be dispatch ready")
	}

	// 4. While node-t4 is Busy, BestNodeForModel for 5 GiB must skip it and choose node-l4
	best, err = mgr.BestNodeForModel(5 * GiB)
	if err != nil {
		t.Fatalf("unexpected error when node-t4 is Busy: %v", err)
	}
	if best.ID != "node-l4" {
		t.Fatalf("expected node-l4 while node-t4 is Busy, got %s", best.ID)
	}

	// 5. Release 10 GiB from node-t4, bringing status back to Ready
	if err := mgr.ReleaseVRAM("node-t4", 10*GiB); err != nil {
		t.Fatalf("failed to release 10 GiB on node-t4: %v", err)
	}
	t4Node, _ = mgr.GetNode("node-t4")
	if t4Node.Status != Ready {
		t.Errorf("expected node-t4 to return to Ready, got %s", t4Node.Status)
	}
	if t4Node.AvailableVRAM() != 10*GiB {
		t.Errorf("expected 10 GiB free on node-t4 after release, got %d", t4Node.AvailableVRAM())
	}

	// 6. Now 8 GiB workload should pack back onto node-t4
	best, err = mgr.BestNodeForModel(8 * GiB)
	if err != nil {
		t.Fatalf("unexpected BestNodeForModel error after release: %v", err)
	}
	if best.ID != "node-t4" {
		t.Fatalf("expected node-t4 after release for 8 GiB, got %s", best.ID)
	}

	// 7. Test tie-breaking when two nodes have identical available headroom (A100-80GB vs H100-80GB)
	// Both have 80 GiB free. For a 60 GiB workload, both have 20 GiB headroom waste.
	// Generation rank tie-breaker: H100 (rank 90) must beat A100-80 (rank 80).
	best, err = mgr.BestNodeForModel(60 * GiB)
	if err != nil {
		t.Fatalf("unexpected error for 60 GiB model: %v", err)
	}
	if best.ID != "node-h100" {
		t.Fatalf("expected node-h100 due to higher generation rank, got %s", best.ID)
	}

	// Allocate 30 GiB on H100 so it has 50 GiB left. Now for a 45 GiB model:
	// A100-40GB cannot fit (only 40 GiB total).
	// A100-80GB has 80 - 45 = 35 GiB waste.
	// H100 has 50 - 45 = 5 GiB waste. Best-fit selects H100!
	if err := mgr.AllocateVRAM("node-h100", 30*GiB); err != nil {
		t.Fatalf("failed to allocate on H100: %v", err)
	}
	best, err = mgr.BestNodeForModel(45 * GiB)
	if err != nil {
		t.Fatalf("unexpected error for 45 GiB model: %v", err)
	}
	if best.ID != "node-h100" {
		t.Fatalf("expected best-fit node-h100 (5 GiB waste vs 35 GiB), got %s", best.ID)
	}

	// 8. Request exceeding all nodes
	_, err = mgr.BestNodeForModel(100 * GiB)
	if !errors.Is(err, ErrInsufficientVRAM) {
		t.Fatalf("expected ErrInsufficientVRAM for 100 GiB, got %v", err)
	}
}

func TestSimulatedDeviceChurn(t *testing.T) {
	mgr := NewFleetManager()
	ctx := context.Background()

	l4Spec, err := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	if err != nil {
		t.Fatalf("failed to create spec: %v", err)
	}

	const initialNodeCount = 8
	for i := 0; i < initialNodeCount; i++ {
		id := fmt.Sprintf("churn-node-%02d", i)
		if err := mgr.RegisterNode(GPUNode{
			ID:       id,
			Zone:     "us-central1-a",
			Spec:     l4Spec,
			Endpoint: fmt.Sprintf("10.0.1.%d:4765", i+1),
		}); err != nil {
			t.Fatalf("failed initial registration of %s: %v", id, err)
		}
	}

	const workers = 16
	const iterations = 40
	var wg sync.WaitGroup
	wg.Add(workers)

	var (
		allocFailures   int64
		probeCount      int64
		telemetryErrors int64
	)

	for w := 0; w < workers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for it := 0; it < iterations; it++ {
				targetIdx := (workerID + it) % initialNodeCount
				targetID := fmt.Sprintf("churn-node-%02d", targetIdx)

				action := (workerID*3 + it) % 6
				switch action {
				case 0:
					// Allocate and immediate release
					if err := mgr.AllocateVRAM(targetID, 4*GiB); err == nil {
						_ = mgr.ReleaseVRAM(targetID, 4*GiB)
					} else {
						atomic.AddInt64(&allocFailures, 1)
					}
				case 1:
					// Telemetry update with oscillating thermal and utilization conditions
					temp := 60.0 + float64((workerID+it)%40) // 60.0 to 99.0
					if err := mgr.UpdateTelemetry(targetID, TelemetryMetrics{
						AllocatedVRAMBytes: int64((it%5)*4) * GiB,
						GPUUtilizationPct:  float64((workerID * 17) % 100),
						TemperatureCelsius: temp,
						ActiveJobs:         (workerID + it) % 3,
					}); err != nil {
						atomic.AddInt64(&telemetryErrors, 1)
					}
				case 2:
					// Run node health probe
					_, _ = mgr.ProbeNode(ctx, targetID)
					atomic.AddInt64(&probeCount, 1)
				case 3:
					// Query best node for random workload
					_, _ = mgr.BestNodeForModel(int64(4+(it%4)*4) * GiB)
				case 4:
					// Toggle status (Degraded / Ready)
					newStatus := Ready
					if (workerID+it)%5 == 0 {
						newStatus = Degraded
					}
					_ = mgr.UpdateNodeStatus(targetID, newStatus)
				case 5:
					// Capacity report snapshot
					_ = mgr.CapacityReport()
				}
			}
		}(w)
	}

	wg.Wait()

	// Post-churn invariants verification
	nodes := mgr.ListNodes()
	if len(nodes) != initialNodeCount {
		t.Fatalf("expected %d nodes remaining after churn, got %d", initialNodeCount, len(nodes))
	}

	for _, n := range nodes {
		// Invariant: allocated VRAM cannot exceed total VRAM
		if n.Telemetry.AllocatedVRAMBytes > n.Spec.TotalVRAMBytes {
			t.Errorf("node %s allocated VRAM %d exceeds total %d", n.ID, n.Telemetry.AllocatedVRAMBytes, n.Spec.TotalVRAMBytes)
		}
		// Invariant: available VRAM must be non-negative and <= total VRAM
		avail := n.AvailableVRAM()
		if avail < 0 || avail > n.Spec.TotalVRAMBytes {
			t.Errorf("node %s invalid available VRAM: %d (total: %d)", n.ID, avail, n.Spec.TotalVRAMBytes)
		}
		// Invariant: status must remain valid
		if !n.Status.IsValid() {
			t.Errorf("node %s holds invalid status %q", n.ID, n.Status)
		}
	}

	if atomic.LoadInt64(&probeCount) == 0 {
		t.Errorf("expected probes to execute during churn")
	}
}

func TestQuotaReclamationAndOvercommit(t *testing.T) {
	mgr := NewFleetManager(WithQuotaEnforcement(true))

	// Define quotas in two distinct regions
	if err := mgr.SetQuota(L4, "us-central1", 4); err != nil {
		t.Fatalf("failed to set L4 quota: %v", err)
	}
	if err := mgr.SetQuota(H100_80GB, "europe-west4", 8); err != nil {
		t.Fatalf("failed to set H100 quota: %v", err)
	}

	// Verify initial quota states
	qL4, ok := mgr.GetQuota(L4, "us-central1")
	if !ok || qL4.Available() != 4 || qL4.InUse != 0 {
		t.Fatalf("unexpected L4 quota: %+v", qL4)
	}
	qH100, ok := mgr.GetQuota(H100_80GB, "europe-west4")
	if !ok || qH100.Available() != 8 || qH100.InUse != 0 {
		t.Fatalf("unexpected H100 quota: %+v", qH100)
	}

	// 1. Multi-GPU node registration consuming quota
	// Register 4-GPU H100 node in europe-west4
	h100Spec4, err := NewInstanceSpec("a3-highgpu-4g", H100_80GB, 4, 104, 940*GiB)
	if err != nil {
		t.Fatalf("failed to create 4xH100 spec: %v", err)
	}
	if err := mgr.RegisterNode(GPUNode{
		ID:       "h100-quad-1",
		Zone:     "europe-west4-a",
		Spec:     h100Spec4,
		Endpoint: "10.1.0.1:4765",
	}); err != nil {
		t.Fatalf("failed to register quad H100: %v", err)
	}

	qH100, _ = mgr.GetQuota(H100_80GB, "europe-west4")
	if qH100.InUse != 4 || qH100.Available() != 4 {
		t.Errorf("expected 4 in use and 4 available for H100, got inUse=%d avail=%d", qH100.InUse, qH100.Available())
	}

	// Register second 4-GPU H100 node in europe-west4, fully exhausting quota
	if err := mgr.RegisterNode(GPUNode{
		ID:       "h100-quad-2",
		Zone:     "europe-west4-b",
		Spec:     h100Spec4,
		Endpoint: "10.1.0.2:4765",
	}); err != nil {
		t.Fatalf("failed to register second quad H100: %v", err)
	}

	qH100, _ = mgr.GetQuota(H100_80GB, "europe-west4")
	if qH100.InUse != 8 || qH100.Available() != 0 {
		t.Errorf("expected quota exhausted (inUse=8, avail=0), got inUse=%d avail=%d", qH100.InUse, qH100.Available())
	}

	// Registering a 1-GPU H100 in europe-west4 must fail with ErrQuotaExceeded
	h100Spec1, _ := NewInstanceSpec("a3-highgpu-1g", H100_80GB, 1, 26, 235*GiB)
	err = mgr.RegisterNode(GPUNode{
		ID:       "h100-single-exceeded",
		Zone:     "europe-west4-c",
		Spec:     h100Spec1,
		Endpoint: "10.1.0.3:4765",
	})
	if !errors.Is(err, ErrQuotaExceeded) {
		t.Fatalf("expected ErrQuotaExceeded, got %v", err)
	}

	// 2. Unregistering one node reclaims quota immediately
	if err := mgr.UnregisterNode("h100-quad-1"); err != nil {
		t.Fatalf("failed to unregister h100-quad-1: %v", err)
	}

	qH100, _ = mgr.GetQuota(H100_80GB, "europe-west4")
	if qH100.InUse != 4 || qH100.Available() != 4 {
		t.Errorf("expected quota reclaimed (inUse=4, avail=4), got inUse=%d avail=%d", qH100.InUse, qH100.Available())
	}

	// Now the single-GPU H100 registration succeeds
	if err := mgr.RegisterNode(GPUNode{
		ID:       "h100-single-now-valid",
		Zone:     "europe-west4-c",
		Spec:     h100Spec1,
		Endpoint: "10.1.0.3:4765",
	}); err != nil {
		t.Fatalf("failed to register single H100 after quota reclamation: %v", err)
	}

	// 3. Dynamic quota expansion via SetQuota
	// Currently L4 quota in us-central1 is 4. Register 4 single L4 nodes
	l4Spec1, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)
	for i := 1; i <= 4; i++ {
		if err := mgr.RegisterNode(GPUNode{
			ID:       fmt.Sprintf("l4-worker-%d", i),
			Zone:     "us-central1-a",
			Spec:     l4Spec1,
			Endpoint: fmt.Sprintf("10.2.0.%d:4765", i),
		}); err != nil {
			t.Fatalf("failed registering l4-worker-%d: %v", i, err)
		}
	}

	// 5th L4 registration fails
	err = mgr.RegisterNode(GPUNode{
		ID:       "l4-worker-5",
		Zone:     "us-central1-a",
		Spec:     l4Spec1,
		Endpoint: "10.2.0.5:4765",
	})
	if !errors.Is(err, ErrQuotaExceeded) {
		t.Fatalf("expected ErrQuotaExceeded for 5th L4, got %v", err)
	}

	// Expand quota to 6
	if err := mgr.SetQuota(L4, "us-central1", 6); err != nil {
		t.Fatalf("failed to expand quota: %v", err)
	}

	// Now 5th L4 registers cleanly
	if err := mgr.RegisterNode(GPUNode{
		ID:       "l4-worker-5",
		Zone:     "us-central1-a",
		Spec:     l4Spec1,
		Endpoint: "10.2.0.5:4765",
	}); err != nil {
		t.Fatalf("failed to register 5th L4 after quota expansion: %v", err)
	}

	// InUse should be 5 and available 1
	qL4, _ = mgr.GetQuota(L4, "us-central1")
	if qL4.InUse != 5 || qL4.Available() != 1 {
		t.Errorf("expected inUse=5 avail=1, got inUse=%d avail=%d", qL4.InUse, qL4.Available())
	}
}

func TestEdgeCaseSpecsAndValidation(t *testing.T) {
	mgr := NewFleetManager()

	// 1. Spec creation edge cases
	spec, err := NewInstanceSpec("  g2-standard-8  ", L4, 1, 8, 32*GiB)
	if err != nil {
		t.Fatalf("failed valid spec: %v", err)
	}
	if spec.MachineType != "g2-standard-8" {
		t.Errorf("expected trimmed machine type, got %q", spec.MachineType)
	}

	// 2. DeriveRegion edge cases
	cases := []struct {
		input string
		want  string
	}{
		{"us-central1-a", "us-central1"},
		{"us-west1-b", "us-west1"},
		{"europe-west4-c", "europe-west4"},
		{"asia-northeast1-a", "asia-northeast1"},
		{"northamerica-northeast2-b", "northamerica-northeast2"},
		{"us-central1", "us-central1"},
		{"europe-west1", "europe-west1"},
		{"custom-zone", "custom-zone"},
		{"", ""},
	}
	for _, tc := range cases {
		got := DeriveRegion(tc.input)
		if got != tc.want {
			t.Errorf("DeriveRegion(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}

	// 3. Register valid node
	node := GPUNode{
		ID:       "edge-node",
		Zone:     "us-central1-f",
		Spec:     spec,
		Endpoint: "10.0.0.99:4765",
		Labels:   map[string]string{"tier": "edge"},
	}
	if err := mgr.RegisterNode(node); err != nil {
		t.Fatalf("failed to register edge-node: %v", err)
	}

	// 4. AllocateVRAM invalid arguments
	if err := mgr.AllocateVRAM("edge-node", 0); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on 0 bytes allocate, got %v", err)
	}
	if err := mgr.AllocateVRAM("edge-node", -5*GiB); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on negative bytes allocate, got %v", err)
	}
	if err := mgr.AllocateVRAM("non-existent-node", 1*GiB); !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound on missing node allocate, got %v", err)
	}

	// 5. ReleaseVRAM invalid arguments
	if err := mgr.ReleaseVRAM("edge-node", 0); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on 0 bytes release, got %v", err)
	}
	if err := mgr.ReleaseVRAM("edge-node", -10*GiB); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on negative bytes release, got %v", err)
	}
	if err := mgr.ReleaseVRAM("non-existent-node", 1*GiB); !errors.Is(err, ErrNodeNotFound) {
		t.Errorf("expected ErrNodeNotFound on missing node release, got %v", err)
	}

	// 6. Release more than allocated should floor at 0 without panic or underflow
	if err := mgr.ReleaseVRAM("edge-node", 100*GiB); err != nil {
		t.Fatalf("unexpected error releasing on clean node: %v", err)
	}
	n, _ := mgr.GetNode("edge-node")
	if n.Telemetry.AllocatedVRAMBytes != 0 {
		t.Errorf("expected allocated VRAM to remain 0, got %d", n.Telemetry.AllocatedVRAMBytes)
	}

	// 7. Allocation on non-ready nodes
	_ = mgr.UpdateNodeStatus("edge-node", Degraded)
	if err := mgr.AllocateVRAM("edge-node", 1*GiB); !errors.Is(err, ErrNodeNotReady) {
		t.Errorf("expected ErrNodeNotReady on Degraded node allocate, got %v", err)
	}

	_ = mgr.UpdateNodeStatus("edge-node", Offline)
	if err := mgr.AllocateVRAM("edge-node", 1*GiB); !errors.Is(err, ErrNodeNotReady) {
		t.Errorf("expected ErrNodeNotReady on Offline node allocate, got %v", err)
	}

	// Restore Ready
	_ = mgr.UpdateNodeStatus("edge-node", Ready)

	// 8. BestNodeForModel invalid arguments
	if _, err := mgr.BestNodeForModel(0); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on BestNodeForModel(0), got %v", err)
	}
	if _, err := mgr.BestNodeForModel(-10); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on BestNodeForModel(-10), got %v", err)
	}

	// 9. When UsedVRAMBytes > AllocatedVRAMBytes (runtime memory spike), AvailableVRAM reflects Used
	if err := mgr.UpdateTelemetry("edge-node", TelemetryMetrics{
		AllocatedVRAMBytes: 4 * GiB,
		UsedVRAMBytes:      10 * GiB, // Runtime spike beyond planned allocation
	}); err != nil {
		t.Fatalf("failed to update telemetry: %v", err)
	}
	nSpike, _ := mgr.GetNode("edge-node")
	// 24 GiB total - 10 GiB used = 14 GiB available
	if nSpike.AvailableVRAM() != 14*GiB {
		t.Errorf("expected 14 GiB available during spike, got %d", nSpike.AvailableVRAM())
	}

	// 10. Quota spec validation
	if err := mgr.SetQuota("invalid-accel", "us-central1", 4); !errors.Is(err, ErrInvalidAccelerator) {
		t.Errorf("expected ErrInvalidAccelerator on SetQuota, got %v", err)
	}
	if err := mgr.SetQuota(L4, "", 4); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on empty region SetQuota, got %v", err)
	}
	if err := mgr.SetQuota(L4, "us-central1", -2); !errors.Is(err, ErrInvalidNodeSpec) {
		t.Errorf("expected ErrInvalidNodeSpec on negative limit SetQuota, got %v", err)
	}

	// 11. Deep-copy isolation test
	nodeCopy, _ := mgr.GetNode("edge-node")
	nodeCopy.Labels["tier"] = "mutated"
	nodeCopy.HealthErrors = append(nodeCopy.HealthErrors, "injected error")

	nodeOriginal, _ := mgr.GetNode("edge-node")
	if nodeOriginal.Labels["tier"] != "edge" {
		t.Errorf("mutating copy corrupted internal labels: got %q", nodeOriginal.Labels["tier"])
	}
	if len(nodeOriginal.HealthErrors) != 0 {
		t.Errorf("mutating copy corrupted internal health errors: got %v", nodeOriginal.HealthErrors)
	}
}

func TestCustomProberAndContextHandling(t *testing.T) {
	spec, _ := NewInstanceSpec("g2-standard-8", L4, 1, 8, 32*GiB)

	// Custom prober that delays and simulates context deadline
	slowProber := func(ctx context.Context, node *GPUNode) (ProbeResult, error) {
		select {
		case <-time.After(100 * time.Millisecond):
			return ProbeResult{
				NodeID:  node.ID,
				Healthy: true,
				Status:  Ready,
			}, nil
		case <-ctx.Done():
			return ProbeResult{
				NodeID:  node.ID,
				Healthy: false,
				Status:  Degraded,
				Errors:  []string{ctx.Err().Error()},
			}, ctx.Err()
		}
	}

	mgr := NewFleetManager(WithProber(slowProber))
	if err := mgr.RegisterNode(GPUNode{
		ID:       "slow-node",
		Spec:     spec,
		Endpoint: "10.0.0.80:4765",
	}); err != nil {
		t.Fatalf("failed to register slow-node: %v", err)
	}

	// 1. Timeout context triggers context error and marks node Degraded
	timeoutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	res, err := mgr.ProbeNode(timeoutCtx, "slow-node")
	if err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected DeadlineExceeded error, got %v", err)
	}
	if res.Healthy {
		t.Errorf("expected unhealthy result on probe timeout")
	}

	node, _ := mgr.GetNode("slow-node")
	if node.Status != Degraded {
		t.Errorf("expected node to transition to Degraded on probe timeout, got %s", node.Status)
	}

	// 2. Normal probe with sufficient deadline succeeds and restores Ready
	normalCtx, cancelNormal := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancelNormal()

	resOk, err := mgr.ProbeNode(normalCtx, "slow-node")
	if err != nil {
		t.Fatalf("unexpected error on normal probe: %v", err)
	}
	if !resOk.Healthy {
		t.Errorf("expected healthy probe result, got unhealthy")
	}

	nodeOk, _ := mgr.GetNode("slow-node")
	if nodeOk.Status != Ready {
		t.Errorf("expected node to restore to Ready, got %s", nodeOk.Status)
	}
}
