package strix

import (
	"fmt"
	"sync"
	"testing"
)

func TestInvalidationProtocol_TargetArchGate(t *testing.T) {
	// Valid target: GFX1151
	cfg := DefaultDynamicReTilerConfig()
	retiler, err := NewDynamicReTiler(cfg, nil)
	if err != nil {
		t.Fatalf("expected valid init for gfx1151, got: %v", err)
	}
	if retiler.CurrentState() != StateIdle {
		t.Fatalf("expected initial state %s, got %s", StateIdle, retiler.CurrentState())
	}

	// Invalid target: gfx1030
	badCfg := cfg
	badCfg.TargetArch = "gfx1030"
	_, err = NewDynamicReTiler(badCfg, nil)
	if err == nil {
		t.Fatalf("expected error for invalid target arch gfx1030, got nil")
	}
}

func TestInvalidationProtocol_PacketSynthesis(t *testing.T) {
	retiler, err := NewDynamicReTiler(DefaultDynamicReTilerConfig(), nil)
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	const baseAddr uint64 = 0x10000000
	const sizeBytes int64 = 4 * 1024 * 1024 // 4 MiB -> 2 packets of 2 MiB each

	packets, err := retiler.SynthesizeBufferWBL2Packets(baseAddr, sizeBytes)
	if err != nil {
		t.Fatalf("SynthesizeBufferWBL2Packets failed: %v", err)
	}

	if len(packets) != 2 {
		t.Fatalf("expected 2 chunks for 4 MiB, got %d", len(packets))
	}

	for i, p := range packets {
		if p.Opcode != OpcodeBufferWBL2 {
			t.Errorf("packet %d: expected opcode %s, got %s", i, OpcodeBufferWBL2, p.Opcode)
		}
		if p.GLC != 1 || p.SLC != 1 {
			t.Errorf("packet %d: expected GLC=1, SLC=1, got GLC=%d, SLC=%d", i, p.GLC, p.SLC)
		}
		if !p.FenceConfirmed {
			t.Errorf("packet %d: expected FenceConfirmed=true", i)
		}
		expectedLines := int(p.RangeBytes / int64(MALLLineSizeBytes))
		if p.CacheLineCount != expectedLines {
			t.Errorf("packet %d: expected %d lines, got %d", i, expectedLines, p.CacheLineCount)
		}

		// Verify binary encoding contains GFX11 buffer_wbl2 opcode
		expectedDword0 := MUBUFOpcodePrefix | MUBUFOpcodeBufferWBL2 | MUBUFOffenBit | MUBUFGLCBit
		if p.RawInstructionDword0 != expectedDword0 {
			t.Errorf("packet %d: instruction Dword0 mismatch: 0x%08x != 0x%08x", i, p.RawInstructionDword0, expectedDword0)
		}
		if p.RawInstructionDword1 != MUBUFSLCBit {
			t.Errorf("packet %d: instruction Dword1 mismatch: 0x%08x != 0x%08x", i, p.RawInstructionDword1, MUBUFSLCBit)
		}
	}
}

func TestInvalidationProtocol_SelectiveFlushingWithCacheModel(t *testing.T) {
	cacheModel := NewMALLCacheModel()

	const prefixAddr uint64 = 0x20000000
	const prefixSize int64 = 2 * 1024 * 1024 // 2 MiB shared prefix
	const tenant1PrivateAddr uint64 = 0x30000000
	const tenant1PrivateSize int64 = 1 * 1024 * 1024 // 1 MiB private

	const tenant2PrivateAddr uint64 = 0x40000000
	const tenant2PrivateSize int64 = 1 * 1024 * 1024 // 1 MiB private

	// Pre-populate MALL cache model with Tenant 1 prefix and private KV lines
	flags := CacheModifierFlags{SLC: 0, GLC: 0, DLC: 0}
	cacheModel.Access(prefixAddr, int(prefixSize), flags, true)
	cacheModel.Access(tenant1PrivateAddr, int(tenant1PrivateSize), flags, true)

	initialPinned := cacheModel.PinnedKVResidencyPct()
	if initialPinned <= 0 {
		t.Fatalf("expected positive initial pinned residency, got %.2f", initialPinned)
	}

	retiler, err := NewDynamicReTiler(DefaultDynamicReTilerConfig(), cacheModel)
	if err != nil {
		t.Fatalf("NewDynamicReTiler failed: %v", err)
	}

	tenant1 := TenantKVSegment{
		TenantID:          "tenant-agent-A",
		PrefixFingerprint: "sha256-system-prompt-hash-001",
		PrefixBaseAddr:    prefixAddr,
		PrefixSizeBytes:   prefixSize,
		PrivateBaseAddr:   tenant1PrivateAddr,
		PrivateSizeBytes:  tenant1PrivateSize,
	}

	tenant2 := TenantKVSegment{
		TenantID:          "tenant-agent-B",
		PrefixFingerprint: "sha256-system-prompt-hash-001", // Matching shared prefix!
		PrefixBaseAddr:    prefixAddr,
		PrefixSizeBytes:   prefixSize,
		PrivateBaseAddr:   tenant2PrivateAddr,
		PrivateSizeBytes:  tenant2PrivateSize,
	}

	// Execute selective transition
	receipt, err := retiler.ExecuteTransition(TransitionRequest{
		PreviousTenant: tenant1,
		IncomingTenant: tenant2,
		AblationArm:    Arm1SelectiveInvalidation,
	})
	if err != nil {
		t.Fatalf("ExecuteTransition failed: %v", err)
	}

	if !receipt.PrefixPreserved {
		t.Fatalf("expected PrefixPreserved=true for shared prefix")
	}
	if receipt.PreservedPrefixBytes != prefixSize {
		t.Errorf("expected %d preserved prefix bytes, got %d", prefixSize, receipt.PreservedPrefixBytes)
	}
	if receipt.InvalidatedPrivateBytes != tenant1PrivateSize {
		t.Errorf("expected %d invalidated private bytes, got %d", tenant1PrivateSize, receipt.InvalidatedPrivateBytes)
	}
	if receipt.FallbackTriggered {
		t.Errorf("expected FallbackTriggered=false for clean selective transition")
	}

	// Verify current state is active tenant 2
	if retiler.CurrentTenant().TenantID != "tenant-agent-B" {
		t.Errorf("expected active tenant tenant-agent-B, got %s", retiler.CurrentTenant().TenantID)
	}
	if retiler.CurrentEpoch() != receipt.NewEpoch {
		t.Errorf("expected epoch %d, got %d", receipt.NewEpoch, retiler.CurrentEpoch())
	}

	// Verify that shared prefix remains warm (hit: true) in MALL cache model
	prefixHit, _ := cacheModel.Access(prefixAddr, int(MALLLineSizeBytes), flags, false)
	if !prefixHit {
		t.Errorf("expected cache hit for preserved prefix line, got miss")
	}

	// Verify that tenant 1 private line was evicted/invalidated (hit: false)
	privateHit, _ := cacheModel.Access(tenant1PrivateAddr, int(MALLLineSizeBytes), flags, false)
	if privateHit {
		t.Errorf("expected cache miss for invalidated tenant 1 private line, got hit")
	}
}

func TestInvalidationProtocol_GlobalCoarseFlush(t *testing.T) {
	cacheModel := NewMALLCacheModel()

	const prefixAddr uint64 = 0x20000000
	const prefixSize int64 = 1 * 1024 * 1024
	const privateAddr uint64 = 0x30000000
	const privateSize int64 = 1 * 1024 * 1024

	flags := CacheModifierFlags{SLC: 0, GLC: 0, DLC: 0}
	cacheModel.Access(prefixAddr, int(prefixSize), flags, true)
	cacheModel.Access(privateAddr, int(privateSize), flags, true)

	retiler, err := NewDynamicReTiler(DefaultDynamicReTilerConfig(), cacheModel)
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	t1 := TenantKVSegment{
		TenantID:          "tenant-1",
		PrefixFingerprint: "hash-001",
		PrefixBaseAddr:    prefixAddr,
		PrefixSizeBytes:   prefixSize,
		PrivateBaseAddr:   privateAddr,
		PrivateSizeBytes:  privateSize,
	}
	t2 := TenantKVSegment{
		TenantID:          "tenant-2",
		PrefixFingerprint: "hash-001", // matching, but Arm2 forces global flush
		PrefixBaseAddr:    prefixAddr,
		PrefixSizeBytes:   prefixSize,
		PrivateBaseAddr:   0x40000000,
		PrivateSizeBytes:  privateSize,
	}

	receipt, err := retiler.ExecuteTransition(TransitionRequest{
		PreviousTenant: t1,
		IncomingTenant: t2,
		AblationArm:    Arm2GlobalCoarseFlush,
	})
	if err != nil {
		t.Fatalf("ExecuteTransition failed: %v", err)
	}

	if receipt.PrefixPreserved {
		t.Errorf("expected PrefixPreserved=false on Arm2 Global Flush")
	}
	if !receipt.FallbackTriggered {
		t.Errorf("expected FallbackTriggered=true for global coarse flush")
	}
	expectedInvalidated := prefixSize + privateSize
	if receipt.InvalidatedPrivateBytes != expectedInvalidated {
		t.Errorf("expected %d total invalidated bytes, got %d", expectedInvalidated, receipt.InvalidatedPrivateBytes)
	}

	// Both prefix and private lines must now miss
	hitPrefix, _ := cacheModel.Access(prefixAddr, int(MALLLineSizeBytes), flags, false)
	if hitPrefix {
		t.Errorf("expected prefix line to be purged, got hit")
	}
	hitPrivate, _ := cacheModel.Access(privateAddr, int(MALLLineSizeBytes), flags, false)
	if hitPrivate {
		t.Errorf("expected private line to be purged, got hit")
	}
}

func TestInvalidationProtocol_QuarantinedFallback(t *testing.T) {
	retiler, err := NewDynamicReTiler(DefaultDynamicReTilerConfig(), nil)
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	t1 := TenantKVSegment{
		TenantID:          "tenant-A",
		PrefixFingerprint: "hash-A",
		PrefixBaseAddr:    0x10000000,
		PrefixSizeBytes:   1024 * 1024,
		PrivateBaseAddr:   0x20000000,
		PrivateSizeBytes:  1024 * 1024,
	}
	t2 := TenantKVSegment{
		TenantID:          "tenant-B",
		PrefixFingerprint: "hash-A",
		PrefixBaseAddr:    0x10000000,
		PrefixSizeBytes:   1024 * 1024,
		PrivateBaseAddr:   0x30000000,
		PrivateSizeBytes:  1024 * 1024,
	}

	// Force fallback
	receipt, err := retiler.ExecuteTransition(TransitionRequest{
		PreviousTenant: t1,
		IncomingTenant: t2,
		ForceFallback:  true,
	})
	if err != nil {
		t.Fatalf("ExecuteTransition failed: %v", err)
	}

	if !receipt.FallbackTriggered {
		t.Errorf("expected FallbackTriggered=true")
	}
	if receipt.FallbackReason == "" {
		t.Errorf("expected non-empty FallbackReason")
	}
	if receipt.PrefixPreserved {
		t.Errorf("expected PrefixPreserved=false on fallback")
	}

	tel := retiler.Telemetry()
	if tel.QuarantinedFallbacks != 1 {
		t.Errorf("expected 1 QuarantinedFallback in telemetry, got %d", tel.QuarantinedFallbacks)
	}
}

func TestInvalidationProtocol_LazyLRUBaseline(t *testing.T) {
	retiler, err := NewDynamicReTiler(DefaultDynamicReTilerConfig(), nil)
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	t1 := TenantKVSegment{
		TenantID:          "tenant-A",
		PrefixFingerprint: "hash-A",
		PrefixBaseAddr:    0x10000000,
		PrefixSizeBytes:   1024 * 1024,
		PrivateBaseAddr:   0x20000000,
		PrivateSizeBytes:  1024 * 1024,
	}
	t2 := TenantKVSegment{
		TenantID:          "tenant-B",
		PrefixFingerprint: "hash-A",
		PrefixBaseAddr:    0x10000000,
		PrefixSizeBytes:   1024 * 1024,
		PrivateBaseAddr:   0x30000000,
		PrivateSizeBytes:  1024 * 1024,
	}

	receipt, err := retiler.ExecuteTransition(TransitionRequest{
		PreviousTenant: t1,
		IncomingTenant: t2,
		AblationArm:    Arm3LazyLRUBaseline,
	})
	if err != nil {
		t.Fatalf("ExecuteTransition failed: %v", err)
	}

	if receipt.PacketsEmitted != 0 {
		t.Errorf("expected 0 packets emitted for lazy LRU, got %d", receipt.PacketsEmitted)
	}
	if receipt.InvalidatedPrivateBytes != 0 {
		t.Errorf("expected 0 invalidated bytes for lazy LRU, got %d", receipt.InvalidatedPrivateBytes)
	}
}

func TestInvalidationProtocol_CrossTenantIsolationEnforcement(t *testing.T) {
	retiler, err := NewDynamicReTiler(DefaultDynamicReTilerConfig(), nil)
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	t1 := TenantKVSegment{
		TenantID:          "tenant-A",
		PrefixFingerprint: "hash-root",
		PrefixBaseAddr:    0x10000000,
		PrefixSizeBytes:   512 * 1024,
		PrivateBaseAddr:   0x20000000,
		PrivateSizeBytes:  512 * 1024,
	}
	t2 := TenantKVSegment{
		TenantID:          "tenant-B",
		PrefixFingerprint: "hash-root",
		PrefixBaseAddr:    0x10000000,
		PrefixSizeBytes:   512 * 1024,
		PrivateBaseAddr:   0x30000000,
		PrivateSizeBytes:  512 * 1024,
	}

	_, err = retiler.ExecuteTransition(TransitionRequest{
		PreviousTenant: t1,
		IncomingTenant: t2,
		AblationArm:    Arm1SelectiveInvalidation,
	})
	if err != nil {
		t.Fatalf("ExecuteTransition failed: %v", err)
	}

	// 1. Current tenant (tenant-B) accessing its own private range: must SUCCEED
	allowed, err := retiler.CheckCrossTenantIsolation("tenant-B", 0x30000000, 64)
	if !allowed || err != nil {
		t.Fatalf("expected valid access for tenant-B, got allowed=%v, err=%v", allowed, err)
	}

	// 2. Foreign tenant (tenant-C) attempting access: must FAIL
	allowed, err = retiler.CheckCrossTenantIsolation("tenant-C", 0x30000000, 64)
	if allowed || err == nil {
		t.Fatalf("expected rejection for inactive tenant-C, got allowed=%v", allowed)
	}

	// 3. Current tenant (tenant-B) attempting to read old tenant-A private range: must FAIL
	allowed, err = retiler.CheckCrossTenantIsolation("tenant-B", 0x20000000, 64)
	if allowed || err == nil {
		t.Fatalf("expected rejection for reading stale tenant-A range, got allowed=%v", allowed)
	}

	tel := retiler.Telemetry()
	if tel.CrossTenantLeaksPrevented != 1 {
		t.Errorf("expected 1 CrossTenantLeaksPrevented, got %d", tel.CrossTenantLeaksPrevented)
	}
	if tel.StaleHitsPrevented != 1 {
		t.Errorf("expected 1 StaleHitsPrevented, got %d", tel.StaleHitsPrevented)
	}
}

func TestInvalidationProtocol_LatencyAndSub50usBenchmark(t *testing.T) {
	cfg := DefaultDynamicReTilerConfig()
	retiler, err := NewDynamicReTiler(cfg, nil)
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	const iterations = 500
	var totalDurationNs int64
	var sub50usCount int

	for i := 0; i < iterations; i++ {
		tPrev := TenantKVSegment{
			TenantID:          fmt.Sprintf("tenant-%d", i),
			PrefixFingerprint: "shared-hash-fixed",
			PrefixBaseAddr:    0x10000000,
			PrefixSizeBytes:   2 * 1024 * 1024,
			PrivateBaseAddr:   0x20000000,
			PrivateSizeBytes:  512 * 1024,
		}
		tNext := TenantKVSegment{
			TenantID:          fmt.Sprintf("tenant-%d", i+1),
			PrefixFingerprint: "shared-hash-fixed",
			PrefixBaseAddr:    0x10000000,
			PrefixSizeBytes:   2 * 1024 * 1024,
			PrivateBaseAddr:   0x25000000,
			PrivateSizeBytes:  512 * 1024,
		}

		receipt, err := retiler.ExecuteTransition(TransitionRequest{
			PreviousTenant: tPrev,
			IncomingTenant: tNext,
			AblationArm:    Arm1SelectiveInvalidation,
		})
		if err != nil {
			t.Fatalf("iteration %d failed: %v", i, err)
		}

		totalDurationNs += receipt.DurationNs
		if receipt.Sub50usMet {
			sub50usCount++
		}
	}

	avgDurationNs := totalDurationNs / iterations
	avgDurationUs := float64(avgDurationNs) / 1000.0

	t.Logf("Executed %d transitions: Average Latency = %.2f µs (%d ns), Sub-50µs rate = %.1f%%",
		iterations, avgDurationUs, avgDurationNs, float64(sub50usCount)/float64(iterations)*100.0)

	// In memory and native Go, re-tiling operations execute in << 50µs
	if avgDurationUs > 50.0 {
		t.Errorf("expected average transition latency <= 50µs, got %.2f µs", avgDurationUs)
	}
}

func TestInvalidationProtocol_ConcurrentTransitions(t *testing.T) {
	retiler, err := NewDynamicReTiler(DefaultDynamicReTilerConfig(), nil)
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	const workerCount = 10
	const opsPerWorker = 20
	var wg sync.WaitGroup

	for w := 0; w < workerCount; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < opsPerWorker; i++ {
				tPrev := TenantKVSegment{
					TenantID:          fmt.Sprintf("w%d-t%d", workerID, i),
					PrefixFingerprint: "shared-mesh-key",
					PrefixBaseAddr:    0x10000000,
					PrefixSizeBytes:   1024 * 1024,
					PrivateBaseAddr:   0x20000000,
					PrivateSizeBytes:  256 * 1024,
				}
				tNext := TenantKVSegment{
					TenantID:          fmt.Sprintf("w%d-t%d", workerID, i+1),
					PrefixFingerprint: "shared-mesh-key",
					PrefixBaseAddr:    0x10000000,
					PrefixSizeBytes:   1024 * 1024,
					PrivateBaseAddr:   0x22000000,
					PrivateSizeBytes:  256 * 1024,
				}

				receipt, err := retiler.ExecuteTransition(TransitionRequest{
					PreviousTenant: tPrev,
					IncomingTenant: tNext,
					AblationArm:    Arm1SelectiveInvalidation,
				})
				if err != nil {
					t.Errorf("worker %d step %d failed: %v", workerID, i, err)
					return
				}
				if receipt.IncomingTenantID != tNext.TenantID {
					t.Errorf("worker %d mismatch tenant ID", workerID)
				}

				// Check isolation query
				_, _ = retiler.CheckCrossTenantIsolation(tNext.TenantID, 0x22000000, 64)
			}
		}(w)
	}

	wg.Wait()

	tel := retiler.Telemetry()
	expectedTotal := uint64(workerCount * opsPerWorker)
	if tel.TotalTransitions != expectedTotal {
		t.Errorf("expected %d total transitions in telemetry, got %d", expectedTotal, tel.TotalTransitions)
	}
}
