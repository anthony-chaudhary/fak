package compute

import (
	"strings"
	"testing"
	"time"
)

func TestAMDGPUDirectHAL_TopologyAndReBAR(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
		DefaultPageSize:        4096,
	})

	// Register MI300X Node 0 (192 GB VRAM, 192 GB Large BAR)
	err := engine.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Instinct MI300X (Node 0)",
		Architecture:   "gfx942",
		PCIeBDF:        "0000:41:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024, // Full 192GB Large BAR
		ACSEnabled:     false,
		ACSRedirect:    false,
		KeepVRAMMapped: true,
		DMABUFCapable:  true,
		Peers: []PeerLink{
			{
				TargetNodeID:     1,
				Fabric:           FabricXGMI,
				BandwidthGBps:    896.0,
				LatencyNanos:     210,
				DirectP2PCapable: true,
				Coherent:         true,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error registering node 0: %v", err)
	}

	// Register MI300X Node 1 (192 GB VRAM, 192 GB Large BAR)
	err = engine.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		GPUID:          1,
		DeviceName:     "AMD Instinct MI300X (Node 1)",
		Architecture:   "gfx942",
		PCIeBDF:        "0000:42:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		ACSEnabled:     false,
		ACSRedirect:    false,
		KeepVRAMMapped: true,
		DMABUFCapable:  true,
		Peers: []PeerLink{
			{
				TargetNodeID:     0,
				Fabric:           FabricXGMI,
				BandwidthGBps:    896.0,
				LatencyNanos:     210,
				DirectP2PCapable: true,
				Coherent:         true,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error registering node 1: %v", err)
	}

	nodes := engine.DiscoverTopology()
	if len(nodes) != 2 {
		t.Fatalf("expected 2 topology nodes, got %d", len(nodes))
	}

	for _, n := range nodes {
		if !n.IsLargeBAR {
			t.Errorf("expected node %d to have IsLargeBAR=true", n.NodeID)
		}
	}

	// Validate P2P route between Node 0 and Node 1 (direct xGMI)
	ok, fabric, reason := engine.ValidateP2PRoute(0, 1)
	if !ok {
		t.Fatalf("expected route 0 -> 1 to be valid, reason: %s", reason)
	}
	if fabric != FabricXGMI {
		t.Errorf("expected FabricXGMI, got %s", fabric)
	}

	// Execute P2P transfer
	fb, bw, err := engine.TransferP2P(0, 1, 64*1024*1024)
	if err != nil {
		t.Fatalf("P2P transfer failed: %v", err)
	}
	if fb != FabricXGMI || bw != 896.0 {
		t.Errorf("expected FabricXGMI / 896 GBps, got %s / %.1f GBps", fb, bw)
	}

	audit := engine.Audit()
	if !audit.Healthy {
		t.Errorf("expected healthy audit report, got unhealthy: %+v", audit)
	}
	if audit.NodesWithLargeBAR != 2 || audit.NodesWithSmallBAR != 0 {
		t.Errorf("expected 2 Large BAR nodes and 0 Small BAR nodes, got %+v", audit)
	}
}

func TestAMDGPUDirectHAL_SmallBARWarningAndACSConflict(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
	})

	// Node 0 with Small BAR (256MB window for 24GB card)
	err := engine.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Radeon RX 7900 XTX (Small BAR)",
		Architecture:   "gfx1100",
		PCIeBDF:        "0000:03:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 24 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  256 * 1024 * 1024, // 256MB Small BAR!
		ACSEnabled:     false,
		ACSRedirect:    false,
		KeepVRAMMapped: true,
		DMABUFCapable:  true,
		Peers: []PeerLink{
			{
				TargetNodeID:     1,
				Fabric:           FabricPCIeSwitch,
				BandwidthGBps:    64.0,
				LatencyNanos:     450,
				DirectP2PCapable: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}

	// Node 1 with ACS Request Redirect enabled on PCIe bridge (P2P killer)
	err = engine.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		GPUID:          1,
		DeviceName:     "AMD Radeon RX 7900 XTX (ACS Conflict)",
		Architecture:   "gfx1100",
		PCIeBDF:        "0000:04:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 24 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  24 * 1024 * 1024 * 1024,
		ACSEnabled:     true,
		ACSRedirect:    true, // ACS Request Redirect active!
		KeepVRAMMapped: true,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}

	// Validate route should fail due to ACS Request Redirect over PCIe
	ok, _, reason := engine.ValidateP2PRoute(0, 1)
	if ok {
		t.Fatalf("expected route 0 -> 1 to fail due to ACS Request Redirect, but succeeded")
	}
	if !strings.Contains(reason, "Access Control Services (ACS) Request Redirect") {
		t.Errorf("expected ACS reason in error, got: %s", reason)
	}

	audit := engine.Audit()
	if audit.Healthy {
		t.Errorf("expected audit report to flag unhealthy status due to ACS conflict")
	}
	if !audit.ACSConflictDetected {
		t.Errorf("expected ACSConflictDetected=true")
	}
	if audit.NodesWithSmallBAR != 1 {
		t.Errorf("expected 1 Small BAR node, got %d", audit.NodesWithSmallBAR)
	}

	// Verify JSON output
	data, err := audit.JSON()
	if err != nil || len(data) == 0 {
		t.Fatalf("failed to serialize audit JSON: %v", err)
	}
}

func TestAMDGPUDirectHAL_TopologyMatrix(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
	})

	err := engine.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Instinct MI300X (Node 0)",
		Architecture:   "gfx942",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		Peers: []PeerLink{
			{
				TargetNodeID:     1,
				Fabric:           FabricXGMI,
				BandwidthGBps:    896.0,
				LatencyNanos:     210,
				DirectP2PCapable: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to register node 0: %v", err)
	}

	err = engine.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		GPUID:          1,
		DeviceName:     "AMD Instinct MI300X (Node 1)",
		Architecture:   "gfx942",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		Peers: []PeerLink{
			{
				TargetNodeID:     0,
				Fabric:           FabricXGMI,
				BandwidthGBps:    896.0,
				LatencyNanos:     210,
				DirectP2PCapable: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to register node 1: %v", err)
	}

	tm := engine.TopologyMatrix()
	if len(tm.NodeIDs) != 2 {
		t.Fatalf("expected 2 nodes in TopologyMatrix, got %d", len(tm.NodeIDs))
	}

	// 0 -> 1 route check
	r01, ok := tm.Routes[0][1]
	if !ok || !r01.DirectP2PCapable || r01.Fabric != FabricXGMI || r01.BandwidthGBps != 896.0 {
		t.Errorf("unexpected route 0 -> 1: %+v", r01)
	}

	// 1 -> 0 route check
	r10, ok := tm.Routes[1][0]
	if !ok || !r10.DirectP2PCapable || r10.Fabric != FabricXGMI || r10.BandwidthGBps != 896.0 {
		t.Errorf("unexpected route 1 -> 0: %+v", r10)
	}

	// Local loopback checks
	r00, ok := tm.Routes[0][0]
	if !ok || !r00.DirectP2PCapable || r00.Fabric != FabricXGMI {
		t.Errorf("unexpected loopback route 0 -> 0: %+v", r00)
	}

	data, err := tm.JSON()
	if err != nil || len(data) == 0 {
		t.Fatalf("failed to serialize TopologyMatrix JSON: %v", err)
	}
}

func TestAMDGPUDirectHAL_DMABUFAndRDMARegistration(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		DefaultPageSize: 4096,
	})

	err := engine.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Instinct MI300X",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("register node failed: %v", err)
	}

	// Export 64 MiB VRAM buffer into DMA-BUF
	vramAddr := uintptr(0x7f0000000000)
	size := uint64(64 * 1024 * 1024)
	dmabuf, err := engine.ExportVRAMToDMABUF(0, vramAddr, size)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF failed: %v", err)
	}
	if dmabuf.FD <= 0 {
		t.Errorf("invalid dmabuf fd %d", dmabuf.FD)
	}
	if dmabuf.VRAMAddress != vramAddr {
		t.Errorf("vram address mismatch: got 0x%x, want 0x%x", dmabuf.VRAMAddress, vramAddr)
	}

	// Register DMA-BUF with RDMA subsystem (matching ibv_reg_dmabuf_mr)
	rdmaRegion, err := engine.RegisterDMABUFForRDMA(dmabuf.FD, size)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA failed: %v", err)
	}

	// Crucial invariant: zero host CPU bounce/staging copies!
	if rdmaRegion.StagingCopyCount() != 0 {
		t.Fatalf("expected 0 staging copies, got %d", rdmaRegion.StagingCopyCount())
	}
	if rdmaRegion.RKey == 0 || rdmaRegion.LKey == 0 {
		t.Errorf("invalid rdma keys: rkey=0x%x, lkey=0x%x", rdmaRegion.RKey, rdmaRegion.LKey)
	}
	if len(rdmaRegion.SGEs) != 1 {
		t.Fatalf("expected 1 ScatterGatherElement, got %d", len(rdmaRegion.SGEs))
	}
	if rdmaRegion.SGEs[0].Address != vramAddr {
		t.Errorf("SGE address mismatch: got 0x%x, want 0x%x", rdmaRegion.SGEs[0].Address, vramAddr)
	}

	// Verify retrieval
	gotBuf := engine.GetDMABUF(dmabuf.FD)
	if gotBuf == nil || gotBuf.FD != dmabuf.FD {
		t.Errorf("GetDMABUF failed: got %+v", gotBuf)
	}
	gotMR := engine.GetRDMARegion(rdmaRegion.RKey)
	if gotMR == nil || gotMR.RKey != rdmaRegion.RKey {
		t.Errorf("GetRDMARegion failed: got %+v", gotMR)
	}

	// Deregister RDMA
	err = engine.DeregisterRDMARegion(rdmaRegion.RKey)
	if err != nil {
		t.Fatalf("DeregisterRDMARegion failed: %v", err)
	}
	if engine.GetRDMARegion(rdmaRegion.RKey) != nil {
		t.Errorf("expected GetRDMARegion to return nil after deregistration")
	}

	// Close DMA-BUF
	err = engine.CloseDMABUF(dmabuf.FD)
	if err != nil {
		t.Fatalf("CloseDMABUF failed: %v", err)
	}
	if engine.GetDMABUF(dmabuf.FD) != nil {
		t.Errorf("expected GetDMABUF to return nil after close")
	}
}

func TestAMDGPUDirectHAL_DMABUFNotCapableRefusal(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})

	err := engine.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "Legacy AMD GPU (No DMABUF)",
		TotalVRAMBytes: 16 * 1024 * 1024 * 1024,
		DMABUFCapable:  false, // Not capable
	})
	if err != nil {
		t.Fatalf("register node failed: %v", err)
	}

	_, err = engine.ExportVRAMToDMABUF(0, uintptr(0x7f0000000000), 4096)
	if err == nil {
		t.Fatalf("expected error exporting DMA-BUF on non-capable node, got nil")
	}
	if !strings.Contains(err.Error(), "does not support kernel DMA-BUF export") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestAMDGPUDirectHAL_MultiGigabyteSGEChunking(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		DefaultPageSize: 4096,
	})

	err := engine.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Instinct MI300X",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("register node failed: %v", err)
	}

	// Register a 6 GiB VRAM buffer (> math.MaxUint32 = 4 GiB - 1)
	// Must produce 2 SGE chunks without integer truncation
	vramAddr := uintptr(0x7f1000000000)
	size := uint64(6 * 1024 * 1024 * 1024) // 6 GiB
	dmabuf, err := engine.ExportVRAMToDMABUF(0, vramAddr, size)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF failed: %v", err)
	}

	rdmaRegion, err := engine.RegisterDMABUFForRDMA(dmabuf.FD, size)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA failed: %v", err)
	}

	if len(rdmaRegion.SGEs) != 2 {
		t.Fatalf("expected 2 SGE chunks for 6 GiB, got %d", len(rdmaRegion.SGEs))
	}
	if rdmaRegion.SGEs[0].Length != MaxSGELengthBytes {
		t.Errorf("expected chunk 0 length %d, got %d", MaxSGELengthBytes, rdmaRegion.SGEs[0].Length)
	}
	expectedRem := uint32(size - MaxSGELengthBytes)
	if rdmaRegion.SGEs[1].Length != expectedRem {
		t.Errorf("expected chunk 1 length %d, got %d", expectedRem, rdmaRegion.SGEs[1].Length)
	}
}

func TestAMDGPUDirectHAL_NVMeDirectStorageP2P(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})

	// Synchronous NVMe Read
	cmdRead := &NVMeP2PCommand{
		CommandID:      1,
		Opcode:         NVMeOpcodeRead, // 0x02 = Read
		NamespaceID:    1,
		StartingLBA:    1024,
		BlockCount:     8,
		TargetVRAMAddr: uintptr(0x7f8000000000),
		ByteLength:     32 * 1024, // 32 KiB
	}

	err := engine.ExecuteNVMeP2PTransfer(cmdRead)
	if err != nil {
		t.Fatalf("ExecuteNVMeP2PTransfer read failed: %v", err)
	}

	if !cmdRead.Completed || cmdRead.Status != 0 {
		t.Errorf("expected command completed with status 0, got completed=%v status=%d", cmdRead.Completed, cmdRead.Status)
	}
	if cmdRead.StagingCopyCount() != 0 {
		t.Fatalf("expected zero staging copies, got %d", cmdRead.StagingCopyCount())
	}

	// Asynchronous NVMe Write
	cmdWrite := &NVMeP2PCommand{
		CommandID:      2,
		Opcode:         NVMeOpcodeWrite, // 0x01 = Write
		NamespaceID:    1,
		StartingLBA:    2048,
		BlockCount:     16,
		TargetVRAMAddr: uintptr(0x7f8000010000),
		ByteLength:     64 * 1024, // 64 KiB
	}

	done, err := engine.ExecuteNVMeP2PTransferAsync(cmdWrite)
	if err != nil {
		t.Fatalf("ExecuteNVMeP2PTransferAsync write failed: %v", err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("async write returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for async NVMe write transfer")
	}

	if !cmdWrite.Completed || cmdWrite.Status != 0 {
		t.Errorf("expected async write completed with status 0, got completed=%v status=%d", cmdWrite.Completed, cmdWrite.Status)
	}
	if cmdWrite.StagingCopyCount() != 0 {
		t.Fatalf("expected zero staging copies on async write, got %d", cmdWrite.StagingCopyCount())
	}

	// Invalid opcode test
	cmdInvalid := &NVMeP2PCommand{
		CommandID:      3,
		Opcode:         0x99,
		TargetVRAMAddr: uintptr(0x7f8000020000),
		ByteLength:     4096,
	}
	err = engine.ExecuteNVMeP2PTransfer(cmdInvalid)
	if err == nil {
		t.Errorf("expected error for invalid opcode 0x99, got nil")
	}

	audit := engine.Audit()
	if audit.TotalTransfers != 2 || audit.TotalBytesMoved != 96*1024 {
		t.Errorf("audit stats mismatch: transfers=%d bytes=%d", audit.TotalTransfers, audit.TotalBytesMoved)
	}
}

func TestAMDGPUDirectHAL_HSAMemorySignal(t *testing.T) {
	sig := NewHSAMemorySignal("sig_001", 0, uintptr(0xf4001000))
	if sig.LoadRelaxed() != 0 {
		t.Errorf("expected initial signal value 0, got %d", sig.LoadRelaxed())
	}

	go func() {
		time.Sleep(5 * time.Millisecond)
		sig.StoreRelease(42)
	}()

	ok, err := sig.WaitRelaxed(42, 500*time.Millisecond)
	if err != nil || !ok {
		t.Fatalf("WaitRelaxed failed: ok=%v, err=%v", ok, err)
	}

	// Test timeout
	ok, err = sig.WaitRelaxed(999, 10*time.Millisecond)
	if err == nil || ok {
		t.Errorf("expected timeout error, got ok=%v, err=%v", ok, err)
	}

	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	engine.RegisterSignal(sig)
	retrieved := engine.GetSignal("sig_001")
	if retrieved == nil || retrieved.LoadRelaxed() != 42 {
		t.Errorf("failed to retrieve registered signal: %+v", retrieved)
	}
}

func TestAMDGPUDirectHAL_HSADoorbellSynchronization(t *testing.T) {
	doorbell := NewHSADoorbell("db_queue_0", uintptr(0xf5002000), 1)
	if doorbell.ReadRelaxed() != 0 {
		t.Errorf("expected initial doorbell packet index 0, got %d", doorbell.ReadRelaxed())
	}

	go func() {
		time.Sleep(5 * time.Millisecond)
		doorbell.Ring(128)
	}()

	ok, err := doorbell.WaitPacket(128, 500*time.Millisecond)
	if err != nil || !ok {
		t.Fatalf("WaitPacket failed: ok=%v, err=%v", ok, err)
	}
	if doorbell.ReadRelaxed() != 128 {
		t.Errorf("expected doorbell value 128, got %d", doorbell.ReadRelaxed())
	}

	// Test timeout
	ok, err = doorbell.WaitPacket(99999, 10*time.Millisecond)
	if err == nil || ok {
		t.Errorf("expected timeout error for unreachable packet, got ok=%v, err=%v", ok, err)
	}

	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	engine.RegisterDoorbell(doorbell)
	retrieved := engine.GetDoorbell("db_queue_0")
	if retrieved == nil || retrieved.ReadRelaxed() != 128 {
		t.Errorf("failed to retrieve registered doorbell: %+v", retrieved)
	}
}

func TestAMDGPUDirectHAL_LocalHostRadeonTopology(t *testing.T) {
	engine := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
	})

	// Local Host Discrete GPU: AMD Radeon RX 7600
	// Hardware facts verified from Win32_VideoController / Vulkan on this physical machine:
	// PCIe BDF: 0000:03:00.0, VRAM: 8 GiB, ReBAR enabled (Host-visible heap: 8 GiB)
	err := engine.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Radeon RX 7600 (Local Discrete GPU)",
		Architecture:   "gfx1102",
		PCIeBDF:        "0000:03:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 8573157376, // 7.98 GiB actual Vulkan device-local heap
		BAR1SizeBytes:  8573157376, // ReBAR enabled: full 8GB host-visible aperture
		ACSEnabled:     false,
		ACSRedirect:    false,
		KeepVRAMMapped: true,
		DMABUFCapable:  true,
		Peers: []PeerLink{
			{
				TargetNodeID:     1,
				Fabric:           FabricPCIeHostBridge,
				BandwidthGBps:    32.0, // PCIe Gen4 x8/x16 to root complex
				LatencyNanos:     650,
				DirectP2PCapable: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("register local RX 7600 failed: %v", err)
	}

	// Local Host Integrated GPU: AMD Radeon(TM) Graphics (APU)
	// PCIe BDF: 0000:7b:00.0, NUMA: 0
	err = engine.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		GPUID:          1,
		DeviceName:     "AMD Radeon(TM) Graphics (Local APU)",
		Architecture:   "gfx1103",
		PCIeBDF:        "0000:7b:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 2 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  2 * 1024 * 1024 * 1024,
		ACSEnabled:     false,
		ACSRedirect:    false,
		KeepVRAMMapped: true,
		DMABUFCapable:  true,
		Peers: []PeerLink{
			{
				TargetNodeID:     0,
				Fabric:           FabricPCIeHostBridge,
				BandwidthGBps:    32.0,
				LatencyNanos:     650,
				DirectP2PCapable: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("register local APU failed: %v", err)
	}

	nodes := engine.DiscoverTopology()
	if len(nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(nodes))
	}

	for _, n := range nodes {
		if !n.IsLargeBAR {
			t.Errorf("expected node %s to have IsLargeBAR=true under active ReBAR", n.DeviceName)
		}
	}

	ok, fabric, reason := engine.ValidateP2PRoute(0, 1)
	if !ok || fabric != FabricPCIeHostBridge {
		t.Fatalf("expected route 0 -> 1 to succeed over PCIe Host Bridge, got ok=%v fabric=%s reason=%s", ok, fabric, reason)
	}

	audit := engine.Audit()
	if !audit.Healthy || audit.NodesWithLargeBAR != 2 || audit.NodesWithSmallBAR != 0 {
		t.Errorf("expected healthy audit with 2 Large BAR nodes, got: %+v", audit)
	}
}
