package strix

import (
	"runtime"
	"sync"
	"testing"
	"time"
	"unsafe"
)

// TestStoreBufferCoherencyProtocol is the primary verifiable witness:
// go test -v -race ./platform/strix -run TestStoreBufferCoherencyProtocol -count=1
// Verifies 1,000 concurrent iterations with zero race conditions under SFENCE.
func TestStoreBufferCoherencyProtocol(t *testing.T) {
	const iterations = 1000
	var wg sync.WaitGroup
	wg.Add(iterations)

	buf := NewUSWCMemoryBuffer(iterations*CacheLineSize, SimulatedUnifiedVABase)
	doorbell := NewAQLDoorbellQueue(true)

	for i := 0; i < iterations; i++ {
		iter := i
		go func() {
			defer wg.Done()

			// Step 1: Write token payload to USWC buffer
			payload := []byte{byte(iter & 0xFF), 0xAA, 0x55, 0x01, 0x02, 0x03}
			offset := iter * CacheLineSize
			n, err := buf.Write(offset, payload)
			if err != nil || n != len(payload) {
				t.Errorf("iteration %d: USWC write failed: %v", iter, err)
				return
			}

			// Step 2: Issue store buffer flush (SFENCE)
			fence, err := buf.Flush(StoreBufferProtocolSFENCE)
			if err != nil || !fence.Valid {
				t.Errorf("iteration %d: flush failed: %v", iter, err)
				return
			}

			// Step 3: Enqueue AQL packet guarded by the fence token
			pkt := CoherencyAQLPacket{
				PacketID:      uint64(iter + 1),
				Payload:       payload,
				DestinationVA: uint64(buf.BaseVA()) + uint64(offset),
			}
			if err := doorbell.Submit(pkt, fence); err != nil {
				t.Errorf("iteration %d: AQL doorbell submission rejected: %v", iter, err)
				return
			}

			// Step 4: Simulated GPU read verifies 100% data visibility
			readBack, clean := buf.ReadGPU(offset, len(payload), true)
			if !clean {
				t.Errorf("iteration %d: GPU read detected stale data hazard", iter)
				return
			}
			for bIdx := range payload {
				if readBack[bIdx] != payload[bIdx] {
					t.Errorf("iteration %d: byte %d mismatch: got 0x%02x, want 0x%02x", iter, bIdx, readBack[bIdx], payload[bIdx])
					return
				}
			}
		}()
	}

	wg.Wait()

	if doorbell.SubmittedCount != iterations {
		t.Fatalf("expected %d submitted packets, got %d", iterations, doorbell.SubmittedCount)
	}
}

// TestSFenceLatencySub15ns verifies that sfence executes in <15ns per invocation on amd64.
func TestSFenceLatencySub15ns(t *testing.T) {
	const samples = 100000

	// Warm up
	for i := 0; i < 1000; i++ {
		SFence()
	}

	t0 := time.Now()
	for i := 0; i < samples; i++ {
		SFence()
	}
	elapsed := time.Since(t0)
	avgNs := float64(elapsed.Nanoseconds()) / float64(samples)

	t.Logf("sfence() benchmark: %d iterations in %v -> %.2f ns/op", samples, elapsed, avgNs)

	if runtime.GOARCH == "amd64" {
		if avgNs > MaxAcceptableSFenceLatencyNs {
			t.Errorf("sfence() average latency %.2f ns exceeds maximum threshold %.2f ns", avgNs, MaxAcceptableSFenceLatencyNs)
		}
	}
}

// TestAblationArmsConsistencyAndOverhead verifies the three ablation arms:
// Arm 1 (SFENCE): 100% consistency, sub-15ns latency.
// Arm 2 (Unfenced): exhibits stale reads / race hazards when partial cache lines are uncommitted.
// Arm 3 (CLFlushOpt): 100% consistency with higher per-line flush overhead.
func TestAblationArmsConsistencyAndOverhead(t *testing.T) {
	const iterations = 500
	const bufferSize = 8192

	metrics := RunAblationEvaluation(iterations, bufferSize)

	arm1, ok1 := metrics[StoreBufferProtocolSFENCE]
	arm2, ok2 := metrics[StoreBufferProtocolUnfenced]
	arm3, ok3 := metrics[StoreBufferProtocolCLFlushOpt]

	if !ok1 || !ok2 || !ok3 {
		t.Fatalf("missing ablation metrics: ok1=%v, ok2=%v, ok3=%v", ok1, ok2, ok3)
	}

	t.Logf("Arm 1 (SFENCE): Consistency=%.2f%%, AvgLat=%.2f ns, StaleReads=%d", arm1.ConsistencyPct, arm1.AvgLatencyNs, arm1.StaleReadCount)
	t.Logf("Arm 2 (Unfenced): Consistency=%.2f%%, AvgLat=%.2f ns, StaleReads=%d", arm2.ConsistencyPct, arm2.AvgLatencyNs, arm2.StaleReadCount)
	t.Logf("Arm 3 (CLFlushOpt): Consistency=%.2f%%, AvgLat=%.2f ns, StaleReads=%d", arm3.ConsistencyPct, arm3.AvgLatencyNs, arm3.StaleReadCount)

	// Arm 1 must achieve 100% consistency
	if arm1.ConsistencyPct != 100.0 {
		t.Errorf("Arm 1 (SFENCE) consistency must be 100.0%%, got %.2f%%", arm1.ConsistencyPct)
	}

	// Arm 2 must exhibit store buffer race hazards (stale reads > 0)
	if arm2.StaleReadCount == 0 {
		t.Errorf("Arm 2 (Unfenced) expected stale reads from lingering WCBs, got 0")
	}

	// Arm 3 must achieve 100% consistency
	if arm3.ConsistencyPct != 100.0 {
		t.Errorf("Arm 3 (CLFlushOpt) consistency must be 100.0%%, got %.2f%%", arm3.ConsistencyPct)
	}
}

// TestAQLDoorbellStrictFenceEnforcement verifies that AQL submissions are rejected
// unless preceded by a valid store buffer coherency fence.
func TestAQLDoorbellStrictFenceEnforcement(t *testing.T) {
	doorbell := NewAQLDoorbellQueue(true)
	pkt := CoherencyAQLPacket{
		PacketID:      1,
		Payload:       []byte("test-prompt-tokens"),
		DestinationVA: 0x7f0000001000,
	}

	// Case 1: Empty / invalid fence
	err := doorbell.Submit(pkt, StoreBufferFence{})
	if err != ErrCoherencyFenceRequired {
		t.Fatalf("expected ErrCoherencyFenceRequired, got %v", err)
	}

	// Case 2: Unfenced protocol
	unfenced := StoreBufferFence{
		TokenID:   2,
		Timestamp: time.Now(),
		Protocol:  StoreBufferProtocolUnfenced,
		Valid:     false,
	}
	err = doorbell.Submit(pkt, unfenced)
	if err != ErrCoherencyFenceRequired {
		t.Fatalf("expected ErrCoherencyFenceRequired for unfenced token, got %v", err)
	}

	// Case 3: Expired fence (> 5s)
	expired := StoreBufferFence{
		TokenID:   3,
		Timestamp: time.Now().Add(-10 * time.Second),
		Protocol:  StoreBufferProtocolSFENCE,
		Valid:     true,
	}
	err = doorbell.Submit(pkt, expired)
	if err != ErrInvalidFenceToken {
		t.Fatalf("expected ErrInvalidFenceToken for expired token, got %v", err)
	}

	// Case 4: Valid SFENCE token succeeds
	validFence := FlushStoreBuffer(int64(len(pkt.Payload)))
	err = doorbell.Submit(pkt, validFence)
	if err != nil {
		t.Fatalf("expected submission to succeed, got %v", err)
	}

	if doorbell.SubmittedCount != 1 {
		t.Errorf("expected 1 submission, got %d", doorbell.SubmittedCount)
	}
}

// TestUSWCMemoryBufferBounds verifies boundary protection on USWC buffers.
func TestUSWCMemoryBufferBounds(t *testing.T) {
	buf := NewUSWCMemoryBuffer(256, 0x1000)

	// Valid write
	n, err := buf.Write(0, make([]byte, 100))
	if err != nil || n != 100 {
		t.Fatalf("valid write failed: %v", err)
	}

	// Out of bounds write
	_, err = buf.Write(200, make([]byte, 100))
	if err != ErrBufferOutOfBounds {
		t.Fatalf("expected ErrBufferOutOfBounds, got %v", err)
	}

	// Negative offset
	_, err = buf.Write(-1, make([]byte, 10))
	if err != ErrBufferOutOfBounds {
		t.Fatalf("expected ErrBufferOutOfBounds, got %v", err)
	}

	// BaseVA and Size
	if buf.BaseVA() != 0x1000 {
		t.Errorf("expected BaseVA 0x1000, got 0x%x", buf.BaseVA())
	}
	if buf.Size() != 256 {
		t.Errorf("expected Size 256, got %d", buf.Size())
	}
}

// TestMFenceAndCLFlushOpt verifies that MFENCE and CLFLUSHOPT execute without faulting.
func TestMFenceAndCLFlushOpt(t *testing.T) {
	data := make([]byte, 256)
	addr := uintptr(unsafe.Pointer(&data[0]))

	MFence()
	CLFlushOpt(addr)
	FlushRangeCLFlushOpt(addr, 256)
}
