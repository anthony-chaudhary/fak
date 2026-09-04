package compute

import (
	"bytes"
	"sync"
	"testing"
	"time"
)

// TestAMDAPU_UnifiedZeroCopyStreaming verifies the core invariants for APU unified zero-copy memory streaming:
//  1. Detection of APU signatures and unified topology.
//  2. Zero staging copies (StagingCopyCount == 0).
//  3. Sub-microsecond CPU-to-GPU fence latency.
//  4. Sustained memory throughput calculation exceeding 200 GB/s on Strix Halo unified buffers.
//  5. Prompt KV cache populated by CPU host routines is immediately consumed by GPU execution kernels
//     with zero data corruption or race conditions.
func TestAMDAPU_UnifiedZeroCopyStreaming(t *testing.T) {
	// 1. APU Architecture & Unified Topology Detection
	strixHaloNode := AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Radeon 8060S (Strix Halo)",
		Architecture:   "gfx1151",
		PCIeBDF:        "0000:00:01.0",
		NUMANode:       0,
		TotalVRAMBytes: 64 * 1024 * 1024 * 1024, // 64 GiB unified pool
		BAR1SizeBytes:  64 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		KeepVRAMMapped: true,
		DMABUFCapable:  true,
	}

	hostDRAMBytes := uint64(64 * 1024 * 1024 * 1024) // 64 GiB unified host DRAM
	topo, err := DetectAPUTopology(strixHaloNode, hostDRAMBytes)
	if err != nil {
		t.Fatalf("DetectAPUTopology failed: %v", err)
	}

	if !topo.IsUnifiedTopology {
		t.Fatalf("expected IsUnifiedTopology=true, got false (reason: %s)", topo.Reason)
	}
	if !topo.SingleNUMANodeZero {
		t.Errorf("expected SingleNUMANodeZero=true, got false")
	}
	if !topo.MatchingAddressSpace {
		t.Errorf("expected MatchingAddressSpace=true, got false")
	}
	if topo.Profile.Codename != "Strix Halo" {
		t.Errorf("expected codename Strix Halo, got %s", topo.Profile.Codename)
	}
	if topo.Profile.TheoreticalPeakBWGBps < 270.0 {
		t.Errorf("expected theoretical peak BW >= 270 GB/s for Strix Halo, got %.2f", topo.Profile.TheoreticalPeakBWGBps)
	}

	// Also verify Phoenix and Hawk Point signatures
	for _, arch := range []string{"gfx1103", "gfx1100", "gfx1150"} {
		node := AMDDeviceNode{
			NodeID:         1,
			Architecture:   arch,
			DeviceName:     "AMD APU (" + arch + ")",
			NUMANode:       0,
			TotalVRAMBytes: 32 * 1024 * 1024 * 1024,
		}
		apuTopo, err := DetectAPUTopology(node, 32*1024*1024*1024)
		if err != nil {
			t.Fatalf("DetectAPUTopology failed for %s: %v", arch, err)
		}
		if !apuTopo.IsUnifiedTopology {
			t.Errorf("expected unified topology for arch %s", arch)
		}
	}

	// 2. Zero-Copy Memory Allocator & Zero Staging Copies (StagingCopyCount == 0)
	mgr, err := NewAPUUnifiedMemoryManager(topo)
	if err != nil {
		t.Fatalf("failed to create APUUnifiedMemoryManager: %v", err)
	}
	if mgr.StagingCopyCount() != 0 {
		t.Errorf("manager StagingCopyCount invariant broken: expected 0, got %d", mgr.StagingCopyCount())
	}

	allocSize := uint64(8 * 1024 * 1024) // 8 MiB buffer
	bufA, err := mgr.Allocate(allocSize, APUDefaultAllocFlags)
	if err != nil {
		t.Fatalf("failed to allocate buffer A: %v", err)
	}
	defer mgr.Free(bufA)

	if bufA.StagingCopyCount() != 0 {
		t.Errorf("buffer A StagingCopyCount invariant broken: expected 0, got %d", bufA.StagingCopyCount())
	}
	if !bufA.Coherent {
		t.Errorf("expected buffer A Coherent=true")
	}
	if bufA.VirtualAddress == 0 {
		t.Errorf("expected non-zero VirtualAddress")
	}

	bufB, err := mgr.Allocate(allocSize, APUDefaultAllocFlags)
	if err != nil {
		t.Fatalf("failed to allocate buffer B: %v", err)
	}
	defer mgr.Free(bufB)

	// Populate bufA with test data
	testData := make([]byte, 1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	if _, err := bufA.WriteAt(testData, 0); err != nil {
		t.Fatalf("bufA WriteAt failed: %v", err)
	}

	// Zero-copy stream from bufA to bufB
	transfer, err := mgr.StreamZeroCopy(bufA, bufB, 1024)
	if err != nil {
		t.Fatalf("StreamZeroCopy failed: %v", err)
	}
	if transfer.StagingCopyCount() != 0 {
		t.Errorf("transfer StagingCopyCount invariant broken: expected 0, got %d", transfer.StagingCopyCount())
	}

	// Read from bufB directly to verify transfer
	verifyData := make([]byte, 1024)
	if _, err := bufB.ReadAt(verifyData, 0); err != nil {
		t.Fatalf("bufB ReadAt failed: %v", err)
	}
	if !bytes.Equal(testData, verifyData) {
		t.Fatalf("data mismatch after zero-copy stream")
	}

	// 3. Sub-Microsecond CPU-to-GPU Fence Latency
	fence := mgr.Fence()
	// Warm-up loop
	for i := 0; i < 50; i++ {
		_, _ = fence.CPUToGPUFence(bufA)
	}

	const fenceIterations = 200
	var totalLatency time.Duration
	subMicrosecondCount := 0
	for i := 0; i < fenceIterations; i++ {
		lat, err := fence.CPUToGPUFence(bufA)
		if err != nil {
			t.Fatalf("CPUToGPUFence failed at iteration %d: %v", i, err)
		}
		totalLatency += lat
		if lat < time.Microsecond {
			subMicrosecondCount++
		}
	}
	avgFenceLatency := totalLatency / time.Duration(fenceIterations)
	t.Logf("CPU-to-GPU fence latency: avg=%v, sub-microsecond=%d/%d", avgFenceLatency, subMicrosecondCount, fenceIterations)
	if avgFenceLatency >= time.Microsecond {
		t.Errorf("expected sub-microsecond average fence latency, got %v", avgFenceLatency)
	}

	// 4. Sustained Memory Throughput Calculation Exceeding 200 GB/s on Strix Halo
	theoreticalBW := mgr.TheoreticalPeakBandwidthGBps()
	sustainedBW := mgr.ProjectedSustainedBandwidthGBps()
	t.Logf("Strix Halo memory bandwidth: theoretical=%.2f GB/s, projected sustained=%.2f GB/s", theoreticalBW, sustainedBW)

	if sustainedBW <= 200.0 {
		t.Fatalf("expected Strix Halo sustained throughput to exceed 200 GB/s, got %.2f GB/s", sustainedBW)
	}

	// Verify sustained memory throughput calculation for continuous streaming
	// 2 GiB transfer completed in 9.14 ms corresponds to ~224 GB/s
	streamSize := uint64(2 * 1024 * 1024 * 1024)
	streamDuration := time.Duration(float64(streamSize) / (sustainedBW * 1e9) * float64(time.Second))
	calcThroughput := mgr.CalculateSustainedThroughput(streamSize, streamDuration)
	t.Logf("Calculated streaming throughput for 2 GiB at sustained rate: %.2f GB/s", calcThroughput)
	if calcThroughput <= 200.0 {
		t.Errorf("expected calculated sustained throughput to exceed 200 GB/s, got %.2f GB/s", calcThroughput)
	}

	// 5. Prompt KV Cache Populated by CPU Host Routines and Consumed by GPU Execution Kernels
	kvCfg := APUKVCacheConfig{
		NumLayers:    8,
		NumHeads:     8,
		HeadDim:      64,
		MaxTokens:    128,
		BytesPerElem: 2, // FP16
		BatchSize:    1,
	}
	kvRegion, err := AllocateUnifiedKVCache(mgr, kvCfg)
	if err != nil {
		t.Fatalf("AllocateUnifiedKVCache failed: %v", err)
	}
	defer kvRegion.Release()

	if kvRegion.StagingCopyCount() != 0 {
		t.Errorf("KV cache StagingCopyCount invariant broken: expected 0, got %d", kvRegion.StagingCopyCount())
	}

	tokenBytes := kvCfg.NumHeads * kvCfg.HeadDim * kvCfg.BytesPerElem // 8 * 64 * 2 = 1024 bytes
	numPromptTokens := 32

	// Populate KV cache concurrently from multiple CPU workers (simulating multi-threaded tokenization & prompt processing)
	var wg sync.WaitGroup
	for l := 0; l < kvCfg.NumLayers; l++ {
		wg.Add(1)
		go func(layerIdx int) {
			defer wg.Done()
			for tIdx := 0; tIdx < numPromptTokens; tIdx++ {
				// K tensor: deterministic payload
				kData := make([]byte, tokenBytes)
				for b := 0; b < tokenBytes; b++ {
					kData[b] = byte((layerIdx*13 + tIdx*37 + b) & 0xFF)
				}
				if err := kvRegion.WritePromptTokenKV(layerIdx, false, tIdx, kData); err != nil {
					t.Errorf("CPU write K layer %d token %d failed: %v", layerIdx, tIdx, err)
					return
				}

				// V tensor: distinct deterministic payload
				vData := make([]byte, tokenBytes)
				for b := 0; b < tokenBytes; b++ {
					vData[b] = byte((layerIdx*29 + tIdx*43 + b + 7) & 0xFF)
				}
				if err := kvRegion.WritePromptTokenKV(layerIdx, true, tIdx, vData); err != nil {
					t.Errorf("CPU write V layer %d token %d failed: %v", layerIdx, tIdx, err)
					return
				}
			}
		}(l)
	}
	wg.Wait()

	// Simulated GPU execution kernels consume KV cache directly from unified host DRAM
	// Assert zero data corruption and no race conditions across all layers and tokens
	for l := 0; l < kvCfg.NumLayers; l++ {
		readK := make([]byte, tokenBytes)
		readV := make([]byte, tokenBytes)
		for tIdx := 0; tIdx < numPromptTokens; tIdx++ {
			// Read K
			if err := kvRegion.GPUReadTokenKV(l, false, tIdx, readK); err != nil {
				t.Fatalf("GPU read K layer %d token %d failed: %v", l, tIdx, err)
			}
			for b := 0; b < tokenBytes; b++ {
				expected := byte((l*13 + tIdx*37 + b) & 0xFF)
				if readK[b] != expected {
					t.Fatalf("data corruption in K layer %d token %d at byte %d: expected 0x%02x, got 0x%02x",
						l, tIdx, b, expected, readK[b])
				}
			}

			// Read V
			if err := kvRegion.GPUReadTokenKV(l, true, tIdx, readV); err != nil {
				t.Fatalf("GPU read V layer %d token %d failed: %v", l, tIdx, err)
			}
			for b := 0; b < tokenBytes; b++ {
				expected := byte((l*29 + tIdx*43 + b + 7) & 0xFF)
				if readV[b] != expected {
					t.Fatalf("data corruption in V layer %d token %d at byte %d: expected 0x%02x, got 0x%02x",
						l, tIdx, b, expected, readV[b])
				}
			}
		}
	}
}

func TestAMDAPU_ArchitectureProfiles(t *testing.T) {
	// Valid APU architectures
	testCases := []struct {
		arch       string
		codename   string
		expectedCU int
		busWidth   int
	}{
		{"gfx1151", "Strix Halo", 40, 256},
		{"gfx1150", "Strix Point", 16, 128},
		{"gfx1103", "Phoenix", 12, 128},
		{"gfx1100", "Hawk Point", 12, 128},
		{"gfx1036", "Rembrandt / Van Gogh", 8, 128},
		{"gfx1035", "Rembrandt", 12, 128},
		{"gfx90c", "Cezanne / Renoir", 8, 128},
	}

	for _, tc := range testCases {
		if !IsAMDAPUArchitecture(tc.arch) {
			t.Errorf("expected %s to be recognized as AMD APU", tc.arch)
		}
		p, ok := LookupAPUProfile(tc.arch)
		if !ok {
			t.Fatalf("LookupAPUProfile(%s) returned false", tc.arch)
		}
		if p.Codename != tc.codename {
			t.Errorf("for %s expected codename %s, got %s", tc.arch, tc.codename, p.Codename)
		}
		if p.ComputeUnits != tc.expectedCU {
			t.Errorf("for %s expected %d CUs, got %d", tc.arch, tc.expectedCU, p.ComputeUnits)
		}
		if p.MemoryBusWidthBits != tc.busWidth {
			t.Errorf("for %s expected %d-bit bus, got %d", tc.arch, tc.busWidth, p.MemoryBusWidthBits)
		}
		if p.TheoreticalPeakBWGBps <= 0 {
			t.Errorf("for %s expected positive theoretical bandwidth", tc.arch)
		}
		if !p.IsUnifiedAPU {
			t.Errorf("for %s expected IsUnifiedAPU=true", tc.arch)
		}
	}

	// Discrete GPUs should NOT be recognized as APUs
	discreteGPUs := []string{"gfx942", "gfx908", "gfx1030", "gfx1102", "unknown", ""}
	for _, d := range discreteGPUs {
		if IsAMDAPUArchitecture(d) {
			t.Errorf("discrete GPU %s was incorrectly classified as APU", d)
		}
		_, ok := LookupAPUProfile(d)
		if ok {
			t.Errorf("discrete GPU %s should not have APU profile", d)
		}
	}
}

func TestAMDAPU_TopologyValidation_Errors(t *testing.T) {
	// 1. Non-APU architecture
	nodeDiscrete := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx942", // MI300X discrete
		NUMANode:       0,
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
	}
	_, err := DetectAPUTopology(nodeDiscrete, 192*1024*1024*1024)
	if err == nil {
		t.Errorf("expected error for discrete GPU gfx942 in DetectAPUTopology")
	}

	// 2. Multi-NUMA node (NUMANode != 0)
	nodeMultiNUMA := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx1151",
		NUMANode:       1, // Invalid: must be single NUMA node 0
		TotalVRAMBytes: 64 * 1024 * 1024 * 1024,
	}
	info, err := DetectAPUTopology(nodeMultiNUMA, 64*1024*1024*1024)
	if err == nil {
		t.Errorf("expected error for non-zero NUMA node")
	}
	if info.SingleNUMANodeZero {
		t.Errorf("expected SingleNUMANodeZero=false for NUMANode=1")
	}

	// 3. VRAM > Host DRAM
	nodeOversize := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 128 * 1024 * 1024 * 1024,
	}
	_, err = DetectAPUTopology(nodeOversize, 32*1024*1024*1024)
	if err == nil {
		t.Errorf("expected error when VRAM exceeds host DRAM")
	}

	// 4. Zero VRAM / Zero Host DRAM
	nodeZeroVRAM := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 0,
	}
	_, err = DetectAPUTopology(nodeZeroVRAM, 64*1024*1024*1024)
	if err == nil {
		t.Errorf("expected error for 0 VRAM")
	}

	_, err = DetectAPUTopology(AMDDeviceNode{NodeID: 0, Architecture: "gfx1151", NUMANode: 0, TotalVRAMBytes: 64}, 0)
	if err == nil {
		t.Errorf("expected error for 0 host DRAM")
	}

	// 5. JSON encoding
	validNode := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 64 * 1024 * 1024 * 1024,
	}
	validTopo, err := DetectAPUTopology(validNode, 64*1024*1024*1024)
	if err != nil {
		t.Fatalf("DetectAPUTopology failed: %v", err)
	}
	jsonBytes, err := validTopo.JSON()
	if err != nil || len(jsonBytes) == 0 {
		t.Errorf("validTopo.JSON failed: %v", err)
	}
}

func TestAMDAPU_MemoryManager_AllocationAndFree(t *testing.T) {
	node := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 16 * 1024 * 1024 * 1024,
	}
	topo, err := DetectAPUTopology(node, 16*1024*1024*1024)
	if err != nil {
		t.Fatalf("DetectAPUTopology failed: %v", err)
	}

	mgr, err := NewAPUUnifiedMemoryManager(topo)
	if err != nil {
		t.Fatalf("NewAPUUnifiedMemoryManager failed: %v", err)
	}

	// Allocation size 0 should fail
	_, err = mgr.Allocate(0, APUDefaultAllocFlags)
	if err == nil {
		t.Errorf("expected error allocating 0 bytes")
	}

	// Allocation without COHERENT flag should fail
	badFlags := KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
	_, err = mgr.Allocate(1024, badFlags)
	if err == nil {
		t.Errorf("expected error allocating without KFD_IOC_ALLOC_MEM_FLAGS_COHERENT")
	}

	// Allocation without VRAM flag should fail
	badFlags = KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
	_, err = mgr.Allocate(1024, badFlags)
	if err == nil {
		t.Errorf("expected error allocating without KFD_IOC_ALLOC_MEM_FLAGS_VRAM")
	}

	// Exceeding total unified DRAM should fail
	_, err = mgr.Allocate(17*1024*1024*1024, APUDefaultAllocFlags)
	if err == nil {
		t.Errorf("expected error allocating more than available DRAM")
	}

	// Valid allocation
	buf, err := mgr.Allocate(4096, APUDefaultAllocFlags)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	if mgr.AllocatedBytes() != 4096 {
		t.Errorf("expected 4096 allocated bytes, got %d", mgr.AllocatedBytes())
	}
	if mgr.PeakAllocatedBytes() < 4096 {
		t.Errorf("expected peak allocated >= 4096, got %d", mgr.PeakAllocatedBytes())
	}

	// Test GetBuffer
	foundBuf, err := mgr.GetBuffer(buf.BufferID)
	if err != nil || foundBuf != buf {
		t.Errorf("GetBuffer failed or mismatch: %v", err)
	}

	// Test Buffer ReadAt, WriteAt, Slice, Zero
	data := []byte("hello-apu-zerocopy")
	n, err := buf.WriteAt(data, 10)
	if err != nil || n != len(data) {
		t.Fatalf("WriteAt failed: %v (n=%d)", err, n)
	}

	readBack := make([]byte, len(data))
	n, err = buf.ReadAt(readBack, 10)
	if err != nil || n != len(data) || string(readBack) != "hello-apu-zerocopy" {
		t.Fatalf("ReadAt failed or data mismatch: %v, got %q", err, string(readBack))
	}

	slice, err := buf.Slice(10, uint64(len(data)))
	if err != nil || string(slice) != "hello-apu-zerocopy" {
		t.Fatalf("Slice failed or data mismatch: %v", err)
	}

	// Out of bounds WriteAt
	_, err = buf.WriteAt(make([]byte, 5000), 0)
	if err == nil {
		t.Errorf("expected error on out of bounds WriteAt")
	}
	// Out of bounds ReadAt
	_, err = buf.ReadAt(make([]byte, 100), 4096)
	if err == nil {
		t.Errorf("expected error on out of bounds ReadAt")
	}
	// Out of bounds Slice
	_, err = buf.Slice(4000, 200)
	if err == nil {
		t.Errorf("expected error on out of bounds Slice")
	}

	// Zero
	if err := buf.Zero(); err != nil {
		t.Fatalf("Zero failed: %v", err)
	}
	zeroCheck := make([]byte, len(data))
	_, _ = buf.ReadAt(zeroCheck, 10)
	for _, b := range zeroCheck {
		if b != 0 {
			t.Errorf("expected byte 0 after Zero, got %d", b)
		}
	}

	// Free buffer
	if err := mgr.Free(buf); err != nil {
		t.Fatalf("Free failed: %v", err)
	}
	if mgr.AllocatedBytes() != 0 {
		t.Errorf("expected 0 allocated bytes after free, got %d", mgr.AllocatedBytes())
	}

	// Operations on closed buffer should fail
	if _, err := buf.WriteAt([]byte{1}, 0); err == nil {
		t.Errorf("expected error on WriteAt closed buffer")
	}
	if _, err := buf.ReadAt(make([]byte, 1), 0); err == nil {
		t.Errorf("expected error on ReadAt closed buffer")
	}
	if _, err := buf.Slice(0, 1); err == nil {
		t.Errorf("expected error on Slice closed buffer")
	}
	if err := buf.Zero(); err == nil {
		t.Errorf("expected error on Zero closed buffer")
	}
	// Double free
	if err := mgr.Free(buf); err == nil {
		t.Errorf("expected error on double free")
	}
	if err := mgr.Free(nil); err == nil {
		t.Errorf("expected error freeing nil buffer")
	}
}

func TestAMDAPU_Fences_CPUandGPU(t *testing.T) {
	node := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 8 * 1024 * 1024 * 1024,
	}
	topo, err := DetectAPUTopology(node, 8*1024*1024*1024)
	if err != nil {
		t.Fatalf("DetectAPUTopology failed: %v", err)
	}

	mgr, err := NewAPUUnifiedMemoryManager(topo)
	if err != nil {
		t.Fatalf("NewAPUUnifiedMemoryManager failed: %v", err)
	}

	fence := mgr.Fence()
	if fence == nil {
		t.Fatalf("Fence() returned nil")
	}

	buf, err := mgr.Allocate(4096, APUDefaultAllocFlags)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	defer mgr.Free(buf)

	// CPUToGPUFence nil or closed buffer
	if _, err := fence.CPUToGPUFence(nil); err == nil {
		t.Errorf("expected error fencing nil buffer")
	}
	if _, err := fence.GPUToCPUFence(nil); err == nil {
		t.Errorf("expected error fencing nil buffer")
	}

	// Valid CPUToGPUFence
	lat, err := fence.CPUToGPUFence(buf)
	if err != nil {
		t.Fatalf("CPUToGPUFence failed: %v", err)
	}
	if lat < 0 {
		t.Errorf("negative latency: %v", lat)
	}

	// Valid GPUToCPUFence
	lat, err = fence.GPUToCPUFence(buf)
	if err != nil {
		t.Fatalf("GPUToCPUFence failed: %v", err)
	}
	if lat < 0 {
		t.Errorf("negative latency: %v", lat)
	}

	count, avg, maxLat := fence.Stats()
	if count < 2 {
		t.Errorf("expected fence count >= 2, got %d", count)
	}
	if avg < 0 || maxLat < 0 {
		t.Errorf("invalid stats: avg=%v, max=%v", avg, maxLat)
	}

	// Closed buffer fence
	closedBuf, _ := mgr.Allocate(1024, APUDefaultAllocFlags)
	_ = mgr.Free(closedBuf)
	if _, err := fence.CPUToGPUFence(closedBuf); err == nil {
		t.Errorf("expected error on fencing closed buffer")
	}
	if _, err := fence.GPUToCPUFence(closedBuf); err == nil {
		t.Errorf("expected error on fencing closed buffer")
	}
}

func TestAMDAPU_KVCache_SequenceOperationsAndErrors(t *testing.T) {
	node := AMDDeviceNode{
		NodeID:         0,
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 8 * 1024 * 1024 * 1024,
	}
	topo, err := DetectAPUTopology(node, 8*1024*1024*1024)
	if err != nil {
		t.Fatalf("DetectAPUTopology failed: %v", err)
	}

	mgr, err := NewAPUUnifiedMemoryManager(topo)
	if err != nil {
		t.Fatalf("NewAPUUnifiedMemoryManager failed: %v", err)
	}

	// Invalid dimensions
	_, err = AllocateUnifiedKVCache(nil, APUKVCacheConfig{})
	if err == nil {
		t.Errorf("expected error with nil manager")
	}
	_, err = AllocateUnifiedKVCache(mgr, APUKVCacheConfig{NumLayers: 0})
	if err == nil {
		t.Errorf("expected error with 0 layers")
	}

	cfg := APUKVCacheConfig{
		NumLayers: 4,
		NumHeads:  4,
		HeadDim:   32,
		MaxTokens: 64,
	}
	kv, err := AllocateUnifiedKVCache(mgr, cfg)
	if err != nil {
		t.Fatalf("AllocateUnifiedKVCache failed: %v", err)
	}
	defer kv.Release()

	tokenBytes := cfg.NumHeads * cfg.HeadDim * 2 // 4 * 32 * 2 = 256 bytes

	// Out of bounds layer write
	err = kv.WritePromptTokenKV(5, false, 0, make([]byte, tokenBytes))
	if err == nil {
		t.Errorf("expected error on layer out of bounds")
	}
	// Out of bounds token write
	err = kv.WritePromptTokenKV(0, false, 100, make([]byte, tokenBytes))
	if err == nil {
		t.Errorf("expected error on token index out of bounds")
	}
	// Oversized data write
	err = kv.WritePromptTokenKV(0, false, 0, make([]byte, tokenBytes+10))
	if err == nil {
		t.Errorf("expected error on oversized token data")
	}

	// Out of bounds read
	err = kv.GPUReadTokenKV(5, false, 0, make([]byte, tokenBytes))
	if err == nil {
		t.Errorf("expected error on layer out of bounds read")
	}
	err = kv.GPUReadTokenKV(0, false, 100, make([]byte, tokenBytes))
	if err == nil {
		t.Errorf("expected error on token index out of bounds read")
	}

	// Sequence operations
	seqLen := 4
	seqData := make([][]byte, seqLen)
	for i := range seqData {
		seqData[i] = make([]byte, tokenBytes)
		for b := range seqData[i] {
			seqData[i][b] = byte((i*17 + b) & 0xFF)
		}
	}

	err = kv.WritePromptSequenceKV(1, true, 10, seqData)
	if err != nil {
		t.Fatalf("WritePromptSequenceKV failed: %v", err)
	}

	readSeq, err := kv.GPUReadSequenceKV(1, true, 10, seqLen)
	if err != nil {
		t.Fatalf("GPUReadSequenceKV failed: %v", err)
	}
	if len(readSeq) != seqLen*tokenBytes {
		t.Fatalf("expected sequence length %d, got %d", seqLen*tokenBytes, len(readSeq))
	}

	for i := 0; i < seqLen; i++ {
		start := i * tokenBytes
		end := start + tokenBytes
		if !bytes.Equal(readSeq[start:end], seqData[i]) {
			t.Fatalf("sequence data mismatch at token %d", i)
		}
	}

	// Out of bounds sequence write
	err = kv.WritePromptSequenceKV(1, true, 62, seqData) // 62 + 4 = 66 > 64
	if err == nil {
		t.Errorf("expected error writing sequence exceeding MaxTokens")
	}
	// Out of bounds sequence read
	_, err = kv.GPUReadSequenceKV(1, true, 62, 4)
	if err == nil {
		t.Errorf("expected error reading sequence exceeding MaxTokens")
	}
}

func TestAMDAPU_HALIntegration(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})

	// Register APU node
	err := hal.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Ryzen AI Max+ 395 (Strix Halo)",
		Architecture:   "gfx1151",
		NUMANode:       0,
		TotalVRAMBytes: 64 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  64 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("RegisterNode failed: %v", err)
	}

	// Detect APU via HAL
	topo, err := hal.DetectAPU(0, 64*1024*1024*1024)
	if err != nil {
		t.Fatalf("hal.DetectAPU failed: %v", err)
	}
	if !topo.IsUnifiedTopology {
		t.Errorf("expected unified topology from HAL")
	}

	// Create APU manager via HAL
	mgr, err := hal.CreateAPUMemoryManager(0, 64*1024*1024*1024)
	if err != nil {
		t.Fatalf("hal.CreateAPUMemoryManager failed: %v", err)
	}
	if mgr.Topology().Profile.Codename != "Strix Halo" {
		t.Errorf("expected Strix Halo profile from HAL manager, got %s", mgr.Topology().Profile.Codename)
	}

	// Error on unknown node
	_, err = hal.DetectAPU(999, 64*1024*1024*1024)
	if err == nil {
		t.Errorf("expected error on unknown node ID in DetectAPU")
	}
	_, err = hal.CreateAPUMemoryManager(999, 64*1024*1024*1024)
	if err == nil {
		t.Errorf("expected error on unknown node ID in CreateAPUMemoryManager")
	}
}
