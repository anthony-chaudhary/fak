package compute

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func setupTestNodes(t *testing.T) (*AMDGPUDirectHAL, *AMDGPUDirectHAL) {
	t.Helper()

	prefillHAL := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
		PreferXGMI:             true,
	})
	decodeHAL := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
		PreferXGMI:             true,
	})

	err := prefillHAL.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		DeviceName:     "AMD Instinct MI300X Prefill",
		Architecture:   "gfx942",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("failed to register prefill node: %v", err)
	}

	err = decodeHAL.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		DeviceName:     "AMD Instinct MI300X Decode",
		Architecture:   "gfx942",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("failed to register decode node: %v", err)
	}

	return prefillHAL, decodeHAL
}

// TestAMDGPUDirect_DisaggregatedPrefillDecodeKVTransfer verifies:
//  1. Multi-node prefill-to-decode transfer simulation over AMD GPU Direct RDMA.
//  2. Zero CPU staging copies verified on both endpoints: StagingCopyCount == 0.
//  3. Arrival notification verified: Decode HSAMemorySignal transitions to ready state in <1 microsecond
//     following RDMA transfer completion, triggering instant decode execution.
func TestAMDGPUDirect_DisaggregatedPrefillDecodeKVTransfer(t *testing.T) {
	prefillHAL, decodeHAL := setupTestNodes(t)

	coord, err := NewPrefillDecodeKVTransferCoordinator(prefillHAL, decodeHAL, PrefillDecodeTransferConfig{
		DefaultTTL:             5 * time.Second,
		DefaultBlockSize:       64 * 1024,
		EnforceZeroCopy:        true,
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
	})
	if err != nil {
		t.Fatalf("NewPrefillDecodeKVTransferCoordinator failed: %v", err)
	}

	transferID := "disagg-transfer-batch-77"
	transferBytes := uint64(4 * 1024 * 1024) // 4 MiB KV cache
	expectedImm := PackImmData(77, 64)

	// Step 1: Negotiate disaggregated KV transfer lease (handshake)
	lease, err := coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID:      transferID,
		PrefillNodeID:   0,
		DecodeNodeID:    1,
		PrefillVRAMAddr: 0x40000000,
		DecodeVRAMAddr:  0x80000000,
		ByteLength:      transferBytes,
		BlockSize:       64 * 1024,
		TTL:             10 * time.Second,
		ImmData:         expectedImm,
	})
	if err != nil {
		t.Fatalf("NegotiateLease failed: %v", err)
	}

	if lease.GetState() != LeaseStateActive {
		t.Fatalf("expected lease state ACTIVE, got %s", lease.GetState())
	}
	if lease.NumBlocks != 64 {
		t.Fatalf("expected 64 blocks, got %d", lease.NumBlocks)
	}

	// Step 2: Verify zero CPU staging copies invariant on both endpoints
	if lease.PrefillStagingCopies != 0 {
		t.Errorf("expected 0 prefill staging copies, got %d", lease.PrefillStagingCopies)
	}
	if lease.DecodeStagingCopies != 0 {
		t.Errorf("expected 0 decode staging copies, got %d", lease.DecodeStagingCopies)
	}
	if lease.StagingCopyCount() != 0 {
		t.Errorf("expected 0 total staging copies, got %d", lease.StagingCopyCount())
	}
	if coord.StagingCopyCount() != 0 {
		t.Errorf("expected coordinator staging copy count 0, got %d", coord.StagingCopyCount())
	}

	// Step 3: Launch decode GPU wavefront polling on the HSAMemorySignal
	decodeExecuted := atomic.Bool{}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	wavefrontCh, err := coord.StartDecodeWavefront(ctx, lease.LeaseID, func(l *KVTransferLease) error {
		// Sub-microsecond wavefront wakeup triggers instant autoregressive decode execution
		// without waiting for an OS context switch or interrupt.
		decodeExecuted.Store(true)
		return nil
	})
	if err != nil {
		t.Fatalf("StartDecodeWavefront failed: %v", err)
	}

	// Ensure the wavefront polling goroutine has started spinning
	time.Sleep(10 * time.Millisecond)

	// Step 4: Prefill node executes one-sided RDMA Write with Immediate (RDMAOpWriteWithImm)
	result, err := coord.ExecuteTransfer(lease.LeaseID)
	if err != nil {
		t.Fatalf("ExecuteTransfer failed: %v", err)
	}

	// Multi-node prefill-to-decode transfer simulation checks
	if !result.Success {
		t.Errorf("expected transfer result success=true")
	}
	if result.PrefillNodeID != 0 {
		t.Errorf("expected prefill node 0, got %d", result.PrefillNodeID)
	}
	if result.DecodeNodeID != 1 {
		t.Errorf("expected decode node 1, got %d", result.DecodeNodeID)
	}
	if result.BytesTransferred != transferBytes {
		t.Errorf("expected %d bytes transferred, got %d", transferBytes, result.BytesTransferred)
	}

	// Step 5: Zero CPU staging copies verified on both endpoints
	if result.PrefillStagingCopies != 0 {
		t.Errorf("result prefill staging copies = %d, want 0", result.PrefillStagingCopies)
	}
	if result.DecodeStagingCopies != 0 {
		t.Errorf("result decode staging copies = %d, want 0", result.DecodeStagingCopies)
	}
	if result.StagingCopyCount() != 0 {
		t.Errorf("result StagingCopyCount() = %d, want 0", result.StagingCopyCount())
	}
	if lease.StagingCopyCount() != 0 {
		t.Errorf("lease StagingCopyCount() = %d, want 0", lease.StagingCopyCount())
	}

	// Step 6: Arrival notification verified: Decode HSAMemorySignal transitions to ready state in <1 microsecond
	if result.SignalLatency >= time.Microsecond {
		t.Errorf("Decode HSAMemorySignal transition latency = %v, want < 1 microsecond", result.SignalLatency)
	}
	t.Logf("Decode HSAMemorySignal transitioned to ready state in %v (< 1 microsecond)", result.SignalLatency)

	// Step 7: Wavefront wake up and decode execution verification
	select {
	case wf := <-wavefrontCh:
		if wf.Error != nil {
			t.Fatalf("wavefront execution returned error: %v", wf.Error)
		}
		if !wf.Completed {
			t.Errorf("wavefront Completed = false")
		}
		if wf.ReceivedImmData != expectedImm {
			t.Errorf("wavefront ImmData = 0x%08x, want 0x%08x", wf.ReceivedImmData, expectedImm)
		}
		if wf.DecodedTokens != 1 {
			t.Errorf("wavefront DecodedTokens = %d, want 1", wf.DecodedTokens)
		}
		if wf.StagingCopyCount != 0 {
			t.Errorf("wavefront StagingCopyCount = %d, want 0", wf.StagingCopyCount)
		}
		if !decodeExecuted.Load() {
			t.Errorf("autoregressive decode execution callback was not invoked")
		}
		t.Logf("Decode wavefront woke up in %v and completed autoregressive decode step", wf.WakeupLatency)

	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for decode wavefront to wake up and execute")
	}

	// Step 8: Release lease
	if err := coord.ReleaseLease(lease.LeaseID); err != nil {
		t.Fatalf("ReleaseLease failed: %v", err)
	}
	if lease.GetState() != LeaseStateReleased {
		t.Errorf("expected lease state RELEASED, got %s", lease.GetState())
	}
}

func TestDisaggregatedKV_LeaseNegotiation(t *testing.T) {
	prefillHAL, decodeHAL := setupTestNodes(t)
	coord, err := NewPrefillDecodeKVTransferCoordinator(prefillHAL, decodeHAL, PrefillDecodeTransferConfig{})
	if err != nil {
		t.Fatalf("NewPrefillDecodeKVTransferCoordinator failed: %v", err)
	}

	// Test successful lease creation
	lease, err := coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID:      "transfer-alpha",
		PrefillNodeID:   0,
		DecodeNodeID:    1,
		PrefillVRAMAddr: 0x10000000,
		DecodeVRAMAddr:  0x20000000,
		ByteLength:      1024 * 1024,
		BlockSize:       128 * 1024,
	})
	if err != nil {
		t.Fatalf("NegotiateLease failed: %v", err)
	}

	if lease.NumBlocks != 8 {
		t.Errorf("expected 8 blocks, got %d", lease.NumBlocks)
	}
	if lease.RemoteRKey == 0 {
		t.Errorf("expected non-zero RemoteRKey")
	}
	if lease.LocalLKey == 0 {
		t.Errorf("expected non-zero LocalLKey")
	}

	// Retrieve by LeaseID
	fetched, err := coord.GetLease(lease.LeaseID)
	if err != nil {
		t.Fatalf("GetLease failed: %v", err)
	}
	if fetched.TransferID != "transfer-alpha" {
		t.Errorf("fetched transfer = %s, want transfer-alpha", fetched.TransferID)
	}

	// Retrieve by TransferID
	fetchedTransfer, err := coord.GetLeaseByID("transfer-alpha")
	if err != nil {
		t.Fatalf("GetLeaseByID failed: %v", err)
	}
	if fetchedTransfer.LeaseID != lease.LeaseID {
		t.Errorf("fetched lease ID = %s, want %s", fetchedTransfer.LeaseID, lease.LeaseID)
	}

	// Reject empty transfer ID
	_, err = coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID: "",
		ByteLength: 1024,
	})
	if err == nil {
		t.Errorf("expected error on empty transfer ID")
	}

	// Reject zero byte length
	_, err = coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID: "transfer-zero",
		ByteLength: 0,
	})
	if !errors.Is(err, ErrInvalidByteLength) {
		t.Errorf("expected ErrInvalidByteLength, got %v", err)
	}

	// Reject non-existent node
	_, err = coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID:    "transfer-badnode",
		PrefillNodeID: 999,
		DecodeNodeID:  1,
		ByteLength:    1024,
	})
	if err == nil {
		t.Errorf("expected error on non-existent prefill node ID")
	}

	// Release lease
	if err := coord.ReleaseLease(lease.LeaseID); err != nil {
		t.Errorf("ReleaseLease failed: %v", err)
	}
	if lease.GetState() != LeaseStateReleased {
		t.Errorf("expected state RELEASED, got %s", lease.GetState())
	}
}

func TestDisaggregatedKV_TimeoutHandling(t *testing.T) {
	prefillHAL, decodeHAL := setupTestNodes(t)
	coord, err := NewPrefillDecodeKVTransferCoordinator(prefillHAL, decodeHAL, PrefillDecodeTransferConfig{})
	if err != nil {
		t.Fatalf("NewPrefillDecodeKVTransferCoordinator failed: %v", err)
	}

	// Create lease with very short TTL
	lease, err := coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID:      "transfer-timeout",
		PrefillNodeID:   0,
		DecodeNodeID:    1,
		PrefillVRAMAddr: 0x30000000,
		DecodeVRAMAddr:  0x70000000,
		ByteLength:      64 * 1024,
		TTL:             10 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NegotiateLease failed: %v", err)
	}

	// Wait for TTL to expire
	time.Sleep(20 * time.Millisecond)

	// ExecuteTransfer should refuse with ErrLeaseExpired
	_, err = coord.ExecuteTransfer(lease.LeaseID)
	if !errors.Is(err, ErrLeaseExpired) {
		t.Errorf("expected ErrLeaseExpired, got %v", err)
	}
	if lease.GetState() != LeaseStateExpired {
		t.Errorf("expected lease state EXPIRED, got %s", lease.GetState())
	}

	// ExpireStaleLeases audit check
	expired := coord.ExpireStaleLeases(time.Now())
	t.Logf("ExpireStaleLeases found %d expired leases", expired)

	// Test WaitForDecodeReady timeout
	lease2, err := coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID:      "transfer-signal-timeout",
		PrefillNodeID:   0,
		DecodeNodeID:    1,
		PrefillVRAMAddr: 0x31000000,
		DecodeVRAMAddr:  0x71000000,
		ByteLength:      64 * 1024,
		TTL:             5 * time.Second,
	})
	if err != nil {
		t.Fatalf("NegotiateLease 2 failed: %v", err)
	}

	// Poll without executing transfer -> should time out
	_, err = coord.WaitForDecodeReady(lease2.LeaseID, 20*time.Millisecond)
	if !errors.Is(err, ErrSignalTimeout) {
		t.Errorf("expected ErrSignalTimeout, got %v", err)
	}
}

func TestDisaggregatedKV_CorruptSignalDetection(t *testing.T) {
	prefillHAL, decodeHAL := setupTestNodes(t)
	coord, err := NewPrefillDecodeKVTransferCoordinator(prefillHAL, decodeHAL, PrefillDecodeTransferConfig{})
	if err != nil {
		t.Fatalf("NewPrefillDecodeKVTransferCoordinator failed: %v", err)
	}

	expectedImm := PackImmData(99, 16)
	lease, err := coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID:      "transfer-corrupt",
		PrefillNodeID:   0,
		DecodeNodeID:    1,
		PrefillVRAMAddr: 0x50000000,
		DecodeVRAMAddr:  0x60000000,
		ByteLength:      128 * 1024,
		TTL:             5 * time.Second,
		ImmData:         expectedImm,
	})
	if err != nil {
		t.Fatalf("NegotiateLease failed: %v", err)
	}

	decodeRan := atomic.Bool{}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	wavefrontCh, err := coord.StartDecodeWavefront(ctx, lease.LeaseID, func(l *KVTransferLease) error {
		decodeRan.Store(true)
		return nil
	})
	if err != nil {
		t.Fatalf("StartDecodeWavefront failed: %v", err)
	}

	time.Sleep(10 * time.Millisecond)

	// Inject corrupted signal value
	if err := coord.InjectSignalCorruption(lease.LeaseID, CorruptSignalValue); err != nil {
		t.Fatalf("InjectSignalCorruption failed: %v", err)
	}

	select {
	case wf := <-wavefrontCh:
		if !errors.Is(wf.Error, ErrCorruptSignal) {
			t.Errorf("expected ErrCorruptSignal, got %v", wf.Error)
		}
		if wf.ReceivedImmData != CorruptSignalValue {
			t.Errorf("expected corrupt imm data 0x%08x, got 0x%08x", CorruptSignalValue, wf.ReceivedImmData)
		}
		if decodeRan.Load() {
			t.Errorf("decode callback must NOT run when signal is corrupted")
		}
		if lease.GetState() != LeaseStateCorrupted {
			t.Errorf("expected lease state CORRUPTED, got %s", lease.GetState())
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timed out waiting for corrupt signal detection")
	}

	stats := coord.Stats()
	if stats.CorruptSignals == 0 {
		t.Errorf("expected stats.CorruptSignals > 0, got %d", stats.CorruptSignals)
	}
}

func TestDisaggregatedKV_ZeroCopyInvariant(t *testing.T) {
	prefillHAL, decodeHAL := setupTestNodes(t)
	coord, err := NewPrefillDecodeKVTransferCoordinator(prefillHAL, decodeHAL, PrefillDecodeTransferConfig{
		EnforceZeroCopy: true,
	})
	if err != nil {
		t.Fatalf("NewPrefillDecodeKVTransferCoordinator failed: %v", err)
	}

	lease, err := coord.NegotiateLease(LeaseNegotiationRequest{
		TransferID:      "transfer-zerocopy",
		PrefillNodeID:   0,
		DecodeNodeID:    1,
		PrefillVRAMAddr: 0x20000000,
		DecodeVRAMAddr:  0x30000000,
		ByteLength:      256 * 1024,
	})
	if err != nil {
		t.Fatalf("NegotiateLease failed: %v", err)
	}

	if lease.StagingCopyCount() != 0 {
		t.Errorf("lease.StagingCopyCount() = %d, want 0", lease.StagingCopyCount())
	}
	if coord.StagingCopyCount() != 0 {
		t.Errorf("coord.StagingCopyCount() = %d, want 0", coord.StagingCopyCount())
	}

	res, err := coord.ExecuteTransfer(lease.LeaseID)
	if err != nil {
		t.Fatalf("ExecuteTransfer failed: %v", err)
	}
	if res.StagingCopyCount() != 0 {
		t.Errorf("res.StagingCopyCount() = %d, want 0", res.StagingCopyCount())
	}
	if lease.StagingCopyCount() != 0 {
		t.Errorf("post-transfer lease.StagingCopyCount() = %d, want 0", lease.StagingCopyCount())
	}
}

func TestDisaggregatedKV_ImmediateDataPacking(t *testing.T) {
	tag := uint16(0x1234)
	tokens := uint16(2048)

	imm := PackImmData(tag, tokens)
	gotTag, gotTokens, ready := UnpackImmData(imm)

	if !ready {
		t.Errorf("expected ready=true")
	}
	if gotTag != tag {
		t.Errorf("got tag 0x%04x, want 0x%04x", gotTag, tag)
	}
	if gotTokens != tokens {
		t.Errorf("got tokens %d, want %d", gotTokens, tokens)
	}
}
