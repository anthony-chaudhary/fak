package compute

import (
	"math"
	"testing"
)

func TestAMDGPUDirect_RDMAQueuePairPool_MultipathStriping(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
		PreferXGMI:             false,
	})

	// Register remote node with 128 GiB VRAM and Large BAR (MI300X)
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		GPUID:          1,
		DeviceName:     "MI300X-Peer",
		Architecture:   "gfx942",
		PCIeBDF:        "0000:60:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 128 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  128 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})

	// Export 1 MiB VRAM to DMA-BUF and register for direct RDMA
	transferSize := uint64(1024 * 1024) // 1 MiB
	dmabuf, err := hal.ExportVRAMToDMABUF(1, 0x80000000, transferSize)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF failed: %v", err)
	}

	mr, err := hal.RegisterDMABUFForRDMA(dmabuf.FD, transferSize)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA failed: %v", err)
	}

	// Initialize 8-rail 400G QP pool
	cfg := RDMAQueuePairPoolConfig{
		HCADevices:           DiscoverHCADevices(), // 8 HCAs, 400 Gbps each
		DefaultMTU:           4096,
		StripeThresholdBytes: 64 * 1024,
		InitialWindow:        32,
		MinWindow:            2,
		MaxWindow:            64,
		AIMDAlpha:            1,
		AIMDBeta:             0.5,
		CNPThreshold:         5,
		PFCThresholdNs:       1000,
	}

	pool, err := NewRDMAQueuePairPool(cfg, hal)
	if err != nil {
		t.Fatalf("NewRDMAQueuePairPool failed: %v", err)
	}

	// 1. Verify parallel striping across 8 simulated 400G HCAs
	req := &StripedWorkRequest{
		RequestID:     5001,
		OpCode:        RDMAOpWrite,
		LocalVRAMAddr: 0x10000000,
		RemoteAddr:    mr.IOVA,
		Length:        transferSize,
		RKey:          mr.RKey,
		LKey:          mr.LKey,
		NodeID:        0,
	}

	comp, err := pool.ExecuteStripedTransfer(req, hal)
	if err != nil {
		t.Fatalf("ExecuteStripedTransfer failed: %v", err)
	}

	expectedChunks := int(transferSize / 4096) // 256 chunks
	if comp.ChunkCount != expectedChunks {
		t.Errorf("ChunkCount = %d, want %d", comp.ChunkCount, expectedChunks)
	}
	if len(comp.RailsUsed) != 8 {
		t.Errorf("RailsUsed count = %d, want 8 (all 8 rails)", len(comp.RailsUsed))
	}

	// 2. Verify zero out-of-order completions and full data integrity
	if comp.OutOfOrderCount != 0 {
		t.Errorf("OutOfOrderCount = %d, want 0", comp.OutOfOrderCount)
	}
	if comp.Status != WCSuccess {
		t.Errorf("Status = %s, want WCSuccess", comp.Status)
	}
	if comp.TotalBytes != transferSize {
		t.Errorf("TotalBytes = %d, want %d", comp.TotalBytes, transferSize)
	}

	// 3. Verify zero CPU staging copies (StagingCopyCount == 0 invariant)
	if comp.StagingCopyCount() != 0 {
		t.Errorf("comp.StagingCopyCount = %d, want 0", comp.StagingCopyCount())
	}
	if req.StagingCopyCount() != 0 {
		t.Errorf("req.StagingCopyCount = %d, want 0", req.StagingCopyCount())
	}
	if pool.StagingCopyCount() != 0 {
		t.Errorf("pool.StagingCopyCount = %d, want 0", pool.StagingCopyCount())
	}

	// 4. Verify active ECN rate-throttling (AIMD window reduction) under simulated congestion
	rail0, err := pool.GetRail(0)
	if err != nil {
		t.Fatalf("GetRail(0) failed: %v", err)
	}
	initialWin0 := rail0.RateLimiter.CurrentWindow()

	// Inject 20 CNP marks and 2500ns PFC pause duration on rail 0
	if err := pool.InjectCongestion(0, 20, 2500); err != nil {
		t.Fatalf("InjectCongestion failed: %v", err)
	}

	// Post subsequent transfer under congestion
	compCongested, err := pool.ExecuteStripedTransfer(req, hal)
	if err != nil {
		t.Fatalf("ExecuteStripedTransfer under congestion failed: %v", err)
	}

	stats0 := rail0.RateLimiter.Stats()
	if stats0.WindowReductions == 0 {
		t.Errorf("expected WindowReductions > 0 on rail 0 under congestion")
	}
	if stats0.CurrentWindow >= initialWin0 {
		t.Errorf("rail 0 window = %d, want < %d after ECN/PFC congestion backoff", stats0.CurrentWindow, initialWin0)
	}
	if compCongested.ThroughputGBps >= comp.ThroughputGBps {
		t.Errorf("congested throughput = %.2f GB/s, want < baseline %.2f GB/s", compCongested.ThroughputGBps, comp.ThroughputGBps)
	}

	// 5. Verify aggregate throughput scaling linearly with QP rail count
	// Test 1, 2, 4, 8 rails
	allRails := pool.GetRails()
	for _, r := range allRails {
		_ = pool.SetRailActive(r.Device.RailID, false)
	}

	// 1 Rail active
	_ = pool.SetRailActive(0, true)
	bw1 := pool.AggregateBandwidthGbps()
	if math.Abs(bw1-400.0) > 1e-6 {
		t.Errorf("1-rail bandwidth = %.2f Gbps, want 400.0", bw1)
	}

	// 2 Rails active
	_ = pool.SetRailActive(1, true)
	bw2 := pool.AggregateBandwidthGbps()
	if math.Abs(bw2-800.0) > 1e-6 {
		t.Errorf("2-rail bandwidth = %.2f Gbps, want 800.0 (2x)", bw2)
	}

	// 4 Rails active
	_ = pool.SetRailActive(2, true)
	_ = pool.SetRailActive(3, true)
	bw4 := pool.AggregateBandwidthGbps()
	if math.Abs(bw4-1600.0) > 1e-6 {
		t.Errorf("4-rail bandwidth = %.2f Gbps, want 1600.0 (4x)", bw4)
	}

	// 8 Rails active
	for i := 4; i < 8; i++ {
		_ = pool.SetRailActive(i, true)
	}
	bw8 := pool.AggregateBandwidthGbps()
	if math.Abs(bw8-3200.0) > 1e-6 {
		t.Errorf("8-rail bandwidth = %.2f Gbps, want 3200.0 (8x)", bw8)
	}

	// Ratio scaling check
	if math.Abs((bw8/bw1)-8.0) > 1e-6 {
		t.Errorf("bw8 / bw1 = %.2f, want 8.0 linear scaling", bw8/bw1)
	}
	if math.Abs((bw4/bw1)-4.0) > 1e-6 {
		t.Errorf("bw4 / bw1 = %.2f, want 4.0 linear scaling", bw4/bw1)
	}
	if math.Abs((bw2/bw1)-2.0) > 1e-6 {
		t.Errorf("bw2 / bw1 = %.2f, want 2.0 linear scaling", bw2/bw1)
	}
}

func TestAMDGPUDirect_RDMAQueuePairPool_PoolAllocation(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	devices := DiscoverHCADevices()
	if len(devices) != 8 {
		t.Fatalf("expected 8 discovered HCAs, got %d", len(devices))
	}

	for i, dev := range devices {
		if dev.RailID != i {
			t.Errorf("HCA[%d] RailID = %d, want %d", i, dev.RailID, i)
		}
		if dev.SpeedGbps != 400.0 {
			t.Errorf("HCA[%d] SpeedGbps = %.1f, want 400.0", i, dev.SpeedGbps)
		}
		if dev.MTU != 4096 {
			t.Errorf("HCA[%d] MTU = %d, want 4096", i, dev.MTU)
		}
		if !dev.Active {
			t.Errorf("HCA[%d] should be active", i)
		}
	}

	pool, err := NewRDMAQueuePairPool(RDMAQueuePairPoolConfig{
		HCADevices: devices,
		QPsPerRail: 4,
	}, hal)
	if err != nil {
		t.Fatalf("NewRDMAQueuePairPool failed: %v", err)
	}

	stats := pool.Stats()
	if stats.TotalRails != 8 || stats.ActiveRails != 8 {
		t.Errorf("stats: TotalRails=%d, ActiveRails=%d, want 8/8", stats.TotalRails, stats.ActiveRails)
	}
	if stats.TotalQPs != 32 { // 8 rails * 4 QPs
		t.Errorf("stats: TotalQPs=%d, want 32", stats.TotalQPs)
	}

	// Test NUMA-local binding
	gpuNode0 := AMDDeviceNode{
		NodeID:   0,
		NUMANode: 0,
	}
	localRails := pool.GetRailsForNode(gpuNode0)
	if len(localRails) != 8 {
		t.Fatalf("expected 8 rails, got %d", len(localRails))
	}
	// First 2 rails must be NUMA 0 local
	if localRails[0].Device.NUMANode != 0 || localRails[1].Device.NUMANode != 0 {
		t.Errorf("expected first 2 rails to be NUMA 0 local, got %d, %d",
			localRails[0].Device.NUMANode, localRails[1].Device.NUMANode)
	}

	// Test GetNUMALocalRail
	numaRail, err := pool.GetNUMALocalRail(2)
	if err != nil {
		t.Fatalf("GetNUMALocalRail(2) failed: %v", err)
	}
	if numaRail.Device.NUMANode != 2 {
		t.Errorf("GetNUMALocalRail(2) NUMA = %d, want 2", numaRail.Device.NUMANode)
	}

	// Test AllocateQP with auto NUMA binding
	pqp, err := pool.AllocateQP(gpuNode0, -1)
	if err != nil {
		t.Fatalf("AllocateQP failed: %v", err)
	}
	if pqp.BoundNUMA != 0 {
		t.Errorf("BoundNUMA = %d, want 0", pqp.BoundNUMA)
	}
}

func TestAMDGPUDirect_RDMAQueuePairPool_RailBalancing(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		TotalVRAMBytes: 16 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  16 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})

	dmabuf, err := hal.ExportVRAMToDMABUF(1, 0x70000000, 512*1024)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF failed: %v", err)
	}
	mr, err := hal.RegisterDMABUFForRDMA(dmabuf.FD, 512*1024)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA failed: %v", err)
	}

	pool, err := NewRDMAQueuePairPool(RDMAQueuePairPoolConfig{
		DefaultMTU: 4096,
	}, hal)
	if err != nil {
		t.Fatalf("NewRDMAQueuePairPool failed: %v", err)
	}

	req := &StripedWorkRequest{
		RequestID:     6001,
		OpCode:        RDMAOpWrite,
		LocalVRAMAddr: 0x20000000,
		RemoteAddr:    mr.IOVA,
		Length:        512 * 1024, // 128 chunks
		RKey:          mr.RKey,
		LKey:          mr.LKey,
		NodeID:        0,
	}

	comp, err := pool.ExecuteStripedTransfer(req, hal)
	if err != nil {
		t.Fatalf("ExecuteStripedTransfer failed: %v", err)
	}

	if comp.ChunkCount != 128 {
		t.Fatalf("ChunkCount = %d, want 128", comp.ChunkCount)
	}

	// With 128 chunks and 8 rails, each rail must receive exactly 16 chunks
	for _, rail := range pool.GetRails() {
		_, _, pkts, bytes := rail.Telemetry()
		if pkts != 16 {
			t.Errorf("rail %d sent %d packets, want exactly 16", rail.Device.RailID, pkts)
		}
		if bytes != 16*4096 {
			t.Errorf("rail %d sent %d bytes, want %d", rail.Device.RailID, bytes, 16*4096)
		}
	}
}

func TestAMDGPUDirect_RDMAQueuePairPool_Failover(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		TotalVRAMBytes: 16 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  16 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})

	dmabuf, _ := hal.ExportVRAMToDMABUF(1, 0x60000000, 256*1024)
	mr, _ := hal.RegisterDMABUFForRDMA(dmabuf.FD, 256*1024)

	pool, _ := NewRDMAQueuePairPool(RDMAQueuePairPoolConfig{}, hal)

	// Simulate link down / hardware fault on rail 3
	if err := pool.SetRailActive(3, false); err != nil {
		t.Fatalf("SetRailActive(3, false) failed: %v", err)
	}

	if pool.ActiveRailCount() != 7 {
		t.Fatalf("ActiveRailCount = %d, want 7", pool.ActiveRailCount())
	}

	req := &StripedWorkRequest{
		RequestID:     7001,
		OpCode:        RDMAOpWrite,
		LocalVRAMAddr: 0x30000000,
		RemoteAddr:    mr.IOVA,
		Length:        256 * 1024, // 64 chunks
		RKey:          mr.RKey,
		LKey:          mr.LKey,
		NodeID:        0,
	}

	comp, err := pool.ExecuteStripedTransfer(req, hal)
	if err != nil {
		t.Fatalf("ExecuteStripedTransfer during failover failed: %v", err)
	}

	if comp.ChunkCount != 64 {
		t.Errorf("ChunkCount = %d, want 64", comp.ChunkCount)
	}

	// Verify rail 3 was completely excluded from used rails
	for _, rid := range comp.RailsUsed {
		if rid == 3 {
			t.Errorf("failed rail 3 found in RailsUsed")
		}
	}

	rail3, _ := pool.GetRail(3)
	_, _, pkts3, _ := rail3.Telemetry()
	if pkts3 != 0 {
		t.Errorf("failed rail 3 sent %d packets, want 0", pkts3)
	}

	// Restore rail 3 and verify it rejoins the pool
	if err := pool.SetRailActive(3, true); err != nil {
		t.Fatalf("SetRailActive(3, true) failed: %v", err)
	}
	if pool.ActiveRailCount() != 8 {
		t.Errorf("ActiveRailCount = %d, want 8 after restore", pool.ActiveRailCount())
	}
}

func TestAMDGPUDirect_RDMAQueuePairPool_RateLimiting_AIMD(t *testing.T) {
	limiter := NewAIMDRateLimiter(2, 64, 32, 1, 0.5, 5, 1000)

	if limiter.CurrentWindow() != 32 {
		t.Fatalf("initial window = %d, want 32", limiter.CurrentWindow())
	}

	// 1. Congestion threshold exceeded via CNP -> multiplicative decrease (halving)
	throttled := limiter.RecordCongestion(10, 0)
	if !throttled {
		t.Errorf("expected throttled = true")
	}
	if limiter.CurrentWindow() != 16 {
		t.Errorf("window after 1st CNP = %d, want 16", limiter.CurrentWindow())
	}

	// 2. Second decrease
	throttled = limiter.RecordCongestion(6, 0)
	if !throttled {
		t.Errorf("expected throttled = true")
	}
	if limiter.CurrentWindow() != 8 {
		t.Errorf("window after 2nd CNP = %d, want 8", limiter.CurrentWindow())
	}

	// 3. Drop to minWindow floor (2)
	limiter.RecordCongestion(10, 0) // 8 -> 4
	limiter.RecordCongestion(10, 0) // 4 -> 2
	limiter.RecordCongestion(10, 0) // 2 -> 2 (floor)
	if limiter.CurrentWindow() != 2 {
		t.Errorf("window after repeated congestion = %d, want minWindow 2", limiter.CurrentWindow())
	}

	// 4. Additive increase (AI) on successful epochs
	for i := 0; i < 10; i++ {
		limiter.OnTransferSuccess()
	}
	if limiter.CurrentWindow() != 12 { // 2 + 10 = 12
		t.Errorf("window after 10 successes = %d, want 12", limiter.CurrentWindow())
	}

	// 5. PFC pause duration threshold (>= 1000 ns) triggers multiplicative decrease
	throttled = limiter.RecordCongestion(0, 1500)
	if !throttled {
		t.Errorf("expected throttled = true on PFC pause")
	}
	if limiter.CurrentWindow() != 6 { // 12 * 0.5 = 6
		t.Errorf("window after PFC pause = %d, want 6", limiter.CurrentWindow())
	}

	stats := limiter.Stats()
	if stats.WindowReductions < 5 {
		t.Errorf("WindowReductions = %d, want >= 5", stats.WindowReductions)
	}
	if stats.WindowIncreases != 10 {
		t.Errorf("WindowIncreases = %d, want 10", stats.WindowIncreases)
	}
}
