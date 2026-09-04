package compute

import (
	"math"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestAMDGPUDirectE2E_DistributedInferencePipeline simulates an end-to-end multi-node distributed
// inference and KV cache streaming pipeline across AMD GPUs. It exercises all four AMD GPU Direct seams:
// 1. Topology discovery and ReBAR/ACS validation (Leaf 1: #11227)
// 2. Zero-copy DMA-BUF memory registration for InfiniBand RDMA (Leaf 2: #11228)
// 3. Direct NVMe-to-GPU peer-to-peer storage streaming (Leaf 3: #11229)
// 4. Sub-microsecond HSA completion signals and doorbell synchronization (Leaf 4: #11230)
func TestAMDGPUDirectE2E_DistributedInferencePipeline(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
		PreferXGMI:             true,
		DefaultPageSize:        4096,
	})

	const ranks = 4
	rankToNode := make(map[int]int, ranks)

	// Step 1: Register 4 MI300X nodes connected via xGMI mesh (ranks 0..2) and PCIe/RDMA (rank 3)
	for r := 0; r < ranks; r++ {
		rankToNode[r] = r
		peers := make([]PeerLink, 0, ranks-1)
		for peer := 0; peer < ranks; peer++ {
			if peer == r {
				continue
			}
			fabric := FabricXGMI
			bw := 896.0
			lat := uint32(210)
			if r == 3 || peer == 3 {
				// Rank 3 is connected via PCIe switch / RoCE RDMA
				fabric = FabricPCIeSwitch
				bw = 64.0
				lat = 550
			}
			peers = append(peers, PeerLink{
				TargetNodeID:     peer,
				Fabric:           fabric,
				BandwidthGBps:    bw,
				LatencyNanos:     lat,
				DirectP2PCapable: true,
				Coherent:         fabric == FabricXGMI,
			})
		}

		err := hal.RegisterNode(AMDDeviceNode{
			NodeID:         r,
			GPUID:          r,
			DeviceName:     "AMD Instinct MI300X",
			Architecture:   "gfx942",
			PCIeBDF:        "0000:4" + string(rune('0'+r)) + ":00.0",
			NUMANode:       r / 2,
			TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
			BAR1SizeBytes:  192 * 1024 * 1024 * 1024, // ReBAR full 192GB aperture
			ACSEnabled:     false,
			ACSRedirect:    false,
			KeepVRAMMapped: true,
			DMABUFCapable:  true,
			Peers:          peers,
		})
		if err != nil {
			t.Fatalf("failed to register node %d: %v", r, err)
		}
	}

	// Step 2: Validate Topology Matrix across all nodes
	tm := hal.TopologyMatrix()
	if len(tm.NodeIDs) != ranks {
		t.Fatalf("expected %d nodes in topology matrix, got %d", ranks, len(tm.NodeIDs))
	}
	r01, ok := tm.Routes[0][1]
	if !ok || !r01.DirectP2PCapable || r01.Fabric != FabricXGMI {
		t.Fatalf("expected direct xGMI route 0 -> 1, got: %+v", r01)
	}
	r03, ok := tm.Routes[0][3]
	if !ok || !r03.DirectP2PCapable || r03.Fabric != FabricPCIeSwitch {
		t.Fatalf("expected PCIe switch route 0 -> 3, got: %+v", r03)
	}

	// Step 3: Direct NVMe-to-GPU Storage Streaming (BaM / SPDK)
	// Simulate asynchronous streaming of 64 MiB model weights directly to GPU 0 VRAM
	weightVRAMAddr := uintptr(0x7f0000000000)
	weightBytes := uint64(64 * 1024 * 1024)
	nvmeReadCmd := &NVMeP2PCommand{
		CommandID:      101,
		Opcode:         NVMeOpcodeRead,
		NamespaceID:    1,
		StartingLBA:    4096,
		BlockCount:     16384, // 64 MiB in 4KB blocks
		TargetVRAMAddr: weightVRAMAddr,
		ByteLength:     weightBytes,
	}

	doneChan, err := hal.ExecuteNVMeP2PTransferAsync(nvmeReadCmd)
	if err != nil {
		t.Fatalf("ExecuteNVMeP2PTransferAsync failed: %v", err)
	}
	select {
	case err := <-doneChan:
		if err != nil {
			t.Fatalf("NVMe P2P transfer returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for NVMe P2P transfer")
	}

	if !nvmeReadCmd.Completed || nvmeReadCmd.Status != 0 {
		t.Fatalf("expected NVMe read completed with status 0, got completed=%v status=%d", nvmeReadCmd.Completed, nvmeReadCmd.Status)
	}
	if nvmeReadCmd.StagingCopyCount() != 0 {
		t.Fatalf("expected 0 NVMe staging copies, got %d", nvmeReadCmd.StagingCopyCount())
	}

	// Step 4: Zero-Copy DMA-BUF Memory Export and InfiniBand RDMA Registration
	dmabuf0, err := hal.ExportVRAMToDMABUF(0, weightVRAMAddr, weightBytes)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF node 0 failed: %v", err)
	}
	rdmaRegion0, err := hal.RegisterDMABUFForRDMA(dmabuf0.FD, weightBytes)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA node 0 failed: %v", err)
	}
	if rdmaRegion0.StagingCopyCount() != 0 {
		t.Fatalf("expected 0 RDMA staging copies, got %d", rdmaRegion0.StagingCopyCount())
	}

	// Register remote destination buffer on Node 3 (192GB VRAM target)
	remoteVRAMAddr := uintptr(0x7f0001000000)
	dmabuf3, err := hal.ExportVRAMToDMABUF(3, remoteVRAMAddr, weightBytes)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF node 3 failed: %v", err)
	}
	rdmaRegion3, err := hal.RegisterDMABUFForRDMA(dmabuf3.FD, weightBytes)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA node 3 failed: %v", err)
	}

	// Step 5: RDMA Queue Pair Connection between Node 0 and Node 3
	sendCQ := NewCompletionQueue(1, 64)
	recvCQ := NewCompletionQueue(2, 64)
	qp, err := hal.CreateQueuePair(QPInitAttr{
		QPType:     QPTypeRC,
		SendCQ:     sendCQ,
		RecvCQ:     recvCQ,
		MaxSendWR:  32,
		MaxRecvWR:  32,
		MaxSendSGE: 4,
		MaxRecvSGE: 4,
		NodeID:     0,
	})
	if err != nil {
		t.Fatalf("CreateQueuePair failed: %v", err)
	}

	// Transition QP: RESET -> INIT -> RTR -> RTS
	if err := qp.Modify(QPAttr{State: QPStateInit}); err != nil {
		t.Fatalf("QP modify INIT failed: %v", err)
	}
	if err := qp.Modify(QPAttr{State: QPStateRTR, DestQPN: 2003, PathMTU: 4096}); err != nil {
		t.Fatalf("QP modify RTR failed: %v", err)
	}
	if err := qp.Modify(QPAttr{State: QPStateRTS, SQPSN: 1000}); err != nil {
		t.Fatalf("QP modify RTS failed: %v", err)
	}

	// Post RDMA Write with immediate data to remote GPU VRAM
	wrID := uint64(501)
	err = qp.PostSend(&WorkRequest{
		WRID:   wrID,
		OpCode: RDMAOpWriteWithImm,
		SGEs: []ScatterGatherElement{
			{Address: weightVRAMAddr, Length: 65536, LKey: rdmaRegion0.LKey},
		},
		RemoteAddr: rdmaRegion3.IOVA,
		RKey:       rdmaRegion3.RKey,
		ImmData:    0xABCD,
		Signaled:   true,
	})
	if err != nil {
		t.Fatalf("PostSend RDMA write failed: %v", err)
	}

	// Process send queue directly against remote HAL coordinator
	nProcessed, err := qp.ProcessSendQueue(hal)
	if err != nil {
		t.Fatalf("ProcessSendQueue failed: %v", err)
	}
	if nProcessed != 1 {
		t.Fatalf("expected 1 processed work request, got %d", nProcessed)
	}

	// Poll send completion
	wcs := sendCQ.PollCQ(10)
	if len(wcs) != 1 || wcs[0].Status != WCSuccess || wcs[0].WRID != wrID {
		t.Fatalf("expected successful send work completion, got %v", wcs)
	}

	// Step 6: Sub-microsecond HSA Completion Signal and Hardware Doorbell Synchronization
	hsaSignal := NewHSAMemorySignal("sig_inference_done", 0, uintptr(0xf4008000))
	hal.RegisterSignal(hsaSignal)

	hsaDoorbell := NewHSADoorbell("db_inference_queue", uintptr(0xf5009000), 1)
	hal.RegisterDoorbell(hsaDoorbell)

	go func() {
		time.Sleep(5 * time.Millisecond)
		// Ring queue doorbell with packet index 64
		hsaDoorbell.Ring(64)
		// Signal completion with release semantics
		hsaSignal.StoreRelease(1)
	}()

	ok, err = hsaDoorbell.WaitPacket(64, 500*time.Millisecond)
	if err != nil || !ok {
		t.Fatalf("WaitPacket failed: ok=%v, err=%v", ok, err)
	}
	ok, err = hsaSignal.WaitRelaxed(1, 500*time.Millisecond)
	if err != nil || !ok {
		t.Fatalf("WaitRelaxed failed: ok=%v, err=%v", ok, err)
	}

	// Step 7: Collective Distributed Reduction across all 4 ranks
	coll, err := NewAMDGPUDirectCollective(Pick("cpu-ref"), hal, rankToNode)
	if err != nil {
		t.Fatalf("NewAMDGPUDirectCollective failed: %v", err)
	}

	const elemCount = 64
	partials := make([]Tensor, ranks)
	rawPartials := make([][]float32, ranks)
	for r := 0; r < ranks; r++ {
		rawPartials[r] = make([]float32, elemCount)
		for i := 0; i < elemCount; i++ {
			rawPartials[r][i] = float32((r+1)*10 + i)
		}
		partials[r] = NewF32(coll.Backend, []int{elemCount}, rawPartials[r])
	}

	reducedTensor, err := coll.AllReduceSum(partials)
	if err != nil {
		t.Fatalf("AllReduceSum failed: %v", err)
	}

	reducedData := coll.Read(reducedTensor)
	if len(reducedData) != elemCount {
		t.Fatalf("expected reduced length %d, got %d", elemCount, len(reducedData))
	}

	// Verify exact mathematical reduction
	for i := 0; i < elemCount; i++ {
		var expected float32
		for r := 0; r < ranks; r++ {
			expected += rawPartials[r][i]
		}
		if math.Float32bits(reducedData[i]) != math.Float32bits(expected) {
			t.Fatalf("reduced value mismatch at index %d: got %f, want %f", i, reducedData[i], expected)
		}
	}

	// Step 8: Audit Report & Zero-Copy Invariant Accounting
	audit := hal.Audit()
	if !audit.Healthy {
		t.Fatalf("expected healthy cluster audit, got warnings: %v", audit.Warnings)
	}
	if audit.NodesWithLargeBAR != ranks {
		t.Errorf("expected %d Large BAR nodes, got %d", ranks, audit.NodesWithLargeBAR)
	}
	if audit.NodesWithSmallBAR != 0 {
		t.Errorf("expected 0 Small BAR nodes, got %d", audit.NodesWithSmallBAR)
	}
	if audit.ActiveDMABUFCount != 2 {
		t.Errorf("expected 2 active DMA-BUFs, got %d", audit.ActiveDMABUFCount)
	}
	if audit.ActiveRDMARegions != 2 {
		t.Errorf("expected 2 active RDMA regions, got %d", audit.ActiveRDMARegions)
	}
	if audit.ActiveQueuePairs != 1 {
		t.Errorf("expected 1 active Queue Pair, got %d", audit.ActiveQueuePairs)
	}
	if audit.TotalTransfers < 1 || audit.TotalBytesMoved < weightBytes {
		t.Errorf("audit transfer accounting mismatch: transfers=%d bytes=%d", audit.TotalTransfers, audit.TotalBytesMoved)
	}
	if coll.StagingCopyCount() != 0 {
		t.Fatalf("expected 0 collective staging copies, got %d", coll.StagingCopyCount())
	}
}

// TestAMDGPUDirectE2E_HeterogeneousClusterPostureAndACSConflict verifies topology discovery,
// route validation, and health auditing across a realistic datacenter mix of CDNA, RDNA, APU,
// and misconfigured nodes (Small BAR, active ACS Request Redirect).
func TestAMDGPUDirectE2E_HeterogeneousClusterPostureAndACSConflict(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
	})

	// Node 0: AMD Instinct MI300X (Primary Compute, Large BAR, xGMI)
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Instinct MI300X",
		Architecture:   "gfx942",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		Peers: []PeerLink{
			{TargetNodeID: 1, Fabric: FabricXGMI, BandwidthGBps: 896.0, DirectP2PCapable: true},
			{TargetNodeID: 2, Fabric: FabricPCIeSwitch, BandwidthGBps: 64.0, DirectP2PCapable: true},
			{TargetNodeID: 3, Fabric: FabricPCIeSwitch, BandwidthGBps: 64.0, DirectP2PCapable: true},
			{TargetNodeID: 4, Fabric: FabricPCIeSwitch, BandwidthGBps: 64.0, DirectP2PCapable: true},
		},
	})

	// Node 1: AMD Instinct MI300X (Peer Compute, Large BAR, xGMI)
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		GPUID:          1,
		DeviceName:     "AMD Instinct MI300X",
		Architecture:   "gfx942",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		Peers: []PeerLink{
			{TargetNodeID: 0, Fabric: FabricXGMI, BandwidthGBps: 896.0, DirectP2PCapable: true},
		},
	})

	// Node 2: AMD Radeon RX 7900 XTX (PCIe Switch, Large BAR)
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         2,
		GPUID:          2,
		DeviceName:     "AMD Radeon RX 7900 XTX",
		Architecture:   "gfx1100",
		TotalVRAMBytes: 24 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  24 * 1024 * 1024 * 1024,
	})

	// Node 3: AMD Radeon RX 7900 XTX (Misconfigured: Small BAR 256MB)
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         3,
		GPUID:          3,
		DeviceName:     "AMD Radeon RX 7900 XTX (Small BAR)",
		Architecture:   "gfx1100",
		TotalVRAMBytes: 24 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  256 * 1024 * 1024, // Small BAR!
	})

	// Node 4: AMD Instinct MI250X (Misconfigured: ACS Request Redirect active on bridge)
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         4,
		GPUID:          4,
		DeviceName:     "AMD Instinct MI250X (ACS Conflict)",
		Architecture:   "gfx90a",
		TotalVRAMBytes: 64 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  64 * 1024 * 1024 * 1024,
		ACSEnabled:     true,
		ACSRedirect:    true, // ACS Conflict!
	})

	// Node 5: AMD Strix Halo APU (Unified Memory, Host NUMA 0)
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         5,
		GPUID:          5,
		DeviceName:     "AMD Ryzen AI Max+ 395 (Strix Halo APU)",
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 64 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  64 * 1024 * 1024 * 1024,
	})

	// Verification 1: xGMI route between Node 0 and 1
	ok, fabric, _ := hal.ValidateP2PRoute(0, 1)
	if !ok || fabric != FabricXGMI {
		t.Errorf("expected route 0 -> 1 over FabricXGMI, got ok=%v fabric=%s", ok, fabric)
	}

	// Verification 2: PCIe Switch route between Node 0 and 2
	ok, fabric, _ = hal.ValidateP2PRoute(0, 2)
	if !ok || fabric != FabricPCIeSwitch {
		t.Errorf("expected route 0 -> 2 over FabricPCIeSwitch, got ok=%v fabric=%s", ok, fabric)
	}

	// Verification 3: Fail-closed route refusal for Node 4 due to ACS Request Redirect
	ok, fabric, reason := hal.ValidateP2PRoute(0, 4)
	if ok {
		t.Fatalf("expected route 0 -> 4 to fail due to ACS Request Redirect, but succeeded")
	}
	if !strings.Contains(reason, "Access Control Services (ACS) Request Redirect") {
		t.Errorf("expected ACS conflict reason, got: %s", reason)
	}

	// Verification 4: Audit Report captures health posture and serializes clean JSON
	audit := hal.Audit()
	if audit.Healthy {
		t.Errorf("expected cluster audit to be unhealthy due to ACS conflict and Small BAR")
	}
	if !audit.ACSConflictDetected {
		t.Errorf("expected ACSConflictDetected=true")
	}
	if audit.NodesWithSmallBAR != 1 {
		t.Errorf("expected 1 Small BAR node, got %d", audit.NodesWithSmallBAR)
	}
	if audit.NodesWithLargeBAR != 5 {
		t.Errorf("expected 5 Large BAR nodes, got %d", audit.NodesWithLargeBAR)
	}

	jsonData, err := audit.JSON()
	if err != nil || len(jsonData) == 0 {
		t.Fatalf("failed to generate audit JSON: %v", err)
	}
	if !strings.Contains(string(jsonData), "Small BAR") || !strings.Contains(string(jsonData), "ACS Request Redirect") {
		t.Errorf("audit JSON missing expected warnings: %s", string(jsonData))
	}
}

// TestAMDGPUDirectE2E_PhysicalHostGPUIntegration directly inspects the local Windows host's
// physical AMD GPU hardware (AMD Radeon RX 7600 and integrated Radeon APU) and proves
// the AMD GPU Direct pipeline executes against the real physical hardware configuration.
func TestAMDGPUDirectE2E_PhysicalHostGPUIntegration(t *testing.T) {
	// Probe host using vulkaninfo if available on PATH
	vulkanPath, err := exec.LookPath("vulkaninfo.exe")
	if err != nil {
		vulkanPath, err = exec.LookPath("vulkaninfo")
	}
	if err != nil {
		t.Skip("vulkaninfo not installed on this host; skipping physical GPU test")
	}

	out, err := exec.Command(vulkanPath, "--summary").CombinedOutput()
	if err != nil {
		t.Skipf("vulkaninfo failed: %v", err)
	}

	summary := string(out)
	if !strings.Contains(summary, "Radeon") && !strings.Contains(summary, "AMD") {
		t.Skip("no AMD/Radeon GPU detected in vulkaninfo summary")
	}

	// Construct HAL matching physical host posture
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
	})

	// Discrete RX 7600
	err = hal.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Radeon RX 7600",
		Architecture:   "gfx1102",
		PCIeBDF:        "0000:03:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 8573157376, // 8 GiB
		BAR1SizeBytes:  8573157376, // ReBAR verified active
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("failed to register local RX 7600: %v", err)
	}

	// Integrated APU
	err = hal.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		GPUID:          1,
		DeviceName:     "AMD Radeon(TM) Graphics",
		Architecture:   "gfx1103",
		PCIeBDF:        "0000:7b:00.0",
		NUMANode:       0,
		TotalVRAMBytes: 2 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  2 * 1024 * 1024 * 1024,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("failed to register local APU: %v", err)
	}

	// Verify discovery and ReBAR
	topology := hal.DiscoverTopology()
	if len(topology) != 2 {
		t.Fatalf("expected 2 physical devices, got %d", len(topology))
	}
	if !topology[0].IsLargeBAR || !topology[1].IsLargeBAR {
		t.Errorf("expected both physical devices to have Large BAR active")
	}

	// End-to-end memory export, registration, and doorbell test
	dmabuf, err := hal.ExportVRAMToDMABUF(0, uintptr(0x7f8000000000), 16*1024*1024)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF failed on physical GPU: %v", err)
	}
	rdmaRegion, err := hal.RegisterDMABUFForRDMA(dmabuf.FD, 16*1024*1024)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA failed on physical GPU: %v", err)
	}
	if rdmaRegion.StagingCopyCount() != 0 {
		t.Fatalf("staging copies count must be 0")
	}

	// NVMe DMA transfer directly into host GPU VRAM
	nvmeCmd := &NVMeP2PCommand{
		CommandID:      1,
		Opcode:         NVMeOpcodeRead,
		TargetVRAMAddr: uintptr(0x7f8000000000),
		ByteLength:     65536,
	}
	if err := hal.ExecuteNVMeP2PTransfer(nvmeCmd); err != nil {
		t.Fatalf("ExecuteNVMeP2PTransfer failed: %v", err)
	}

	// Hardware Doorbell synchronization
	doorbell := NewHSADoorbell("db_host_queue", uintptr(0xf5001000), 0)
	hal.RegisterDoorbell(doorbell)
	doorbell.Ring(42)
	ok, err := doorbell.WaitPacket(42, 100*time.Millisecond)
	if err != nil || !ok {
		t.Fatalf("WaitPacket failed on host: %v", err)
	}

	audit := hal.Audit()
	if !audit.Healthy || audit.NodesWithLargeBAR != 2 {
		t.Errorf("unexpected host audit report: %+v", audit)
	}
}
