// Package strix implements direct userspace AQL queue management,
// cacheline-aligned 64-byte packet formatting, and low-overhead MMIO hardware
// doorbell signaling for sub-50µs tool-yield preemption and resumption on AMD
// Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

// TestAQLDoorbellResumption verifies:
// 1. Queue initialization and 64-byte alignment
// 2. AQL packet formatting and ring buffer wrapping
// 3. Sub-50µs doorbell write benchmark (< 5µs mock MMIO write latency)
// 4. Barrier flag handling and concurrent multi-session safety
func TestAQLDoorbellResumption(t *testing.T) {
	t.Run("QueueInitializationAnd64ByteAlignment", func(t *testing.T) {
		const size = 1024
		q, err := NewAQLQueue("test-worker-align", size)
		if err != nil {
			t.Fatalf("failed to create AQLQueue: %v", err)
		}

		if q.QueueSize() != size {
			t.Fatalf("expected queue size %d, got %d", size, q.QueueSize())
		}
		if q.QueueMask() != size-1 {
			t.Fatalf("expected queue mask %d, got %d", size-1, q.QueueMask())
		}

		// Compile-time & runtime size checks: packets must strictly equal 64 bytes
		if sz := unsafe.Sizeof(AQLKernelDispatchPacket{}); sz != AQLPacketSizeBytes {
			t.Fatalf("AQLKernelDispatchPacket size must be exactly 64 bytes, got %d", sz)
		}
		if sz := unsafe.Sizeof(AQLBarrierPacket{}); sz != AQLPacketSizeBytes {
			t.Fatalf("AQLBarrierPacket size must be exactly 64 bytes, got %d", sz)
		}
		if sz := unsafe.Sizeof(q.ringBuffer[0]); sz != AQLPacketSizeBytes {
			t.Fatalf("ring buffer element size must be exactly 64 bytes, got %d", sz)
		}

		// 64-byte alignment assertion: ring buffer base address
		base := q.RingBufferBase()
		if base == 0 {
			t.Fatalf("ring buffer base pointer is null")
		}
		if base%64 != 0 {
			t.Fatalf("ring buffer base 0x%x is not 64-byte aligned (offset=%d)", base, base%64)
		}
		if !q.Is64ByteAligned() {
			t.Fatalf("Is64ByteAligned reported false")
		}

		// Verify every slot in the ring buffer is 64-byte aligned
		for i := 0; i < int(size); i++ {
			slotAddr := uintptr(unsafe.Pointer(&q.ringBuffer[i]))
			if slotAddr%64 != 0 {
				t.Fatalf("slot %d at 0x%x is not 64-byte aligned", i, slotAddr)
			}
		}

		// Invalid queue size assertions: non-powers-of-two must be rejected
		invalidSizes := []uint32{0, 1, 2, 3, 5, 7, 100, 500, 1000, 65537}
		for _, inv := range invalidSizes {
			_, err := NewAQLQueue("invalid", inv)
			if err == nil {
				t.Errorf("expected error for invalid queue size %d, got nil", inv)
			}
		}
	})

	t.Run("AQLPacketFormattingAndRingBufferWrapping", func(t *testing.T) {
		q, err := NewAQLQueue("test-worker-format", 4)
		if err != nil {
			t.Fatalf("failed to create AQLQueue: %v", err)
		}

		// 1. Format and submit AQLKernelDispatchPacket
		hdr := NewPacketHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeSystem, AQLFenceScopeSystem)
		if hdr.Type() != AQLPacketTypeKernelDispatch {
			t.Fatalf("expected packet type KERNEL_DISPATCH, got %s", hdr.Type())
		}
		if hdr.Barrier() {
			t.Fatalf("expected barrier false")
		}
		if hdr.AcquireScope() != AQLFenceScopeSystem {
			t.Fatalf("expected acquire scope SYSTEM, got %s", hdr.AcquireScope())
		}
		if hdr.ReleaseScope() != AQLFenceScopeSystem {
			t.Fatalf("expected release scope SYSTEM, got %s", hdr.ReleaseScope())
		}

		kernelPkt := AQLKernelDispatchPacket{
			Header:             hdr,
			Dimensions:         3,
			WorkgroupSizeX:     256,
			WorkgroupSizeY:     1,
			WorkgroupSizeZ:     1,
			GridSizeX:          1024,
			GridSizeY:          1,
			GridSizeZ:          1,
			PrivateSegmentSize: 0,
			GroupSegmentSize:   32768,
			KernelObject:       0x7FFF10000000,
			KernargAddress:     0x7FFF20000000,
			CompletionSignal:   0xCAFE0001,
		}

		nextPtr, latNs, err := q.SubmitKernelDispatch(kernelPkt)
		if err != nil {
			t.Fatalf("SubmitKernelDispatch failed: %v", err)
		}
		if nextPtr != 1 {
			t.Fatalf("expected next write pointer 1, got %d", nextPtr)
		}
		if latNs <= 0 {
			t.Fatalf("expected positive doorbell latency, got %d", latNs)
		}

		// Read back from slot 0 and verify byte-for-byte fidelity
		decoded, err := q.GetKernelDispatchPacket(0)
		if err != nil {
			t.Fatalf("GetKernelDispatchPacket failed: %v", err)
		}
		if decoded.Dimensions != 3 || decoded.WorkgroupSizeX != 256 || decoded.KernelObject != 0x7FFF10000000 {
			t.Fatalf("decoded kernel packet fields mismatch: %+v", decoded)
		}
		if decoded.CompletionSignal != 0xCAFE0001 {
			t.Fatalf("expected completion signal 0xCAFE0001, got 0x%x", decoded.CompletionSignal)
		}

		// 2. Fill the remaining 3 slots (queue capacity = 4)
		for i := 2; i <= 4; i++ {
			pkt := kernelPkt
			pkt.CompletionSignal = uint64(0xCAFE0000 + i)
			ptr, _, err := q.SubmitPacket(pkt)
			if err != nil {
				t.Fatalf("submission %d failed: %v", i, err)
			}
			if ptr != uint64(i) {
				t.Fatalf("expected write pointer %d, got %d", i, ptr)
			}
		}

		// Queue is now full (4 pending packets in capacity 4)
		if !q.IsFull() {
			t.Fatalf("expected queue to be full")
		}
		if q.PendingCount() != 4 {
			t.Fatalf("expected 4 pending packets, got %d", q.PendingCount())
		}

		// 5th submission must be rejected with ErrQueueFull
		_, _, err = q.SubmitPacket(kernelPkt)
		if err != ErrQueueFull {
			t.Fatalf("expected ErrQueueFull, got %v", err)
		}

		// 3. Consume 2 packets by advancing read pointer
		q.AdvanceReadPtr(2)
		if q.PendingCount() != 2 {
			t.Fatalf("expected 2 pending packets after read advance, got %d", q.PendingCount())
		}

		// 4. Submit 2 more packets to trigger circular wrapping (slot index 4&3=0, 5&3=1)
		wrapPkt1 := kernelPkt
		wrapPkt1.KernelObject = 0xAAAA1111
		ptr1, _, err := q.SubmitPacket(wrapPkt1)
		if err != nil {
			t.Fatalf("wrapped submission 1 failed: %v", err)
		}
		if ptr1 != 5 {
			t.Fatalf("expected write pointer 5, got %d", ptr1)
		}

		wrapPkt2 := kernelPkt
		wrapPkt2.KernelObject = 0xBBBB2222
		ptr2, _, err := q.SubmitPacket(wrapPkt2)
		if err != nil {
			t.Fatalf("wrapped submission 2 failed: %v", err)
		}
		if ptr2 != 6 {
			t.Fatalf("expected write pointer 6, got %d", ptr2)
		}

		// Verify wrapped slots contain the newly overwritten packets
		decWrap1, err := q.GetKernelDispatchPacket(q.SlotIndex(4))
		if err != nil {
			t.Fatalf("failed reading wrapped slot 0: %v", err)
		}
		if decWrap1.KernelObject != 0xAAAA1111 {
			t.Fatalf("expected wrapped packet 1 kernel object 0xAAAA1111, got 0x%x", decWrap1.KernelObject)
		}

		decWrap2, err := q.GetKernelDispatchPacket(q.SlotIndex(5))
		if err != nil {
			t.Fatalf("failed reading wrapped slot 1: %v", err)
		}
		if decWrap2.KernelObject != 0xBBBB2222 {
			t.Fatalf("expected wrapped packet 2 kernel object 0xBBBB2222, got 0x%x", decWrap2.KernelObject)
		}
	})

	t.Run("Sub50MicrosecondDoorbellWriteBenchmark", func(t *testing.T) {
		q, err := NewAQLQueue("test-worker-bench", 1024)
		if err != nil {
			t.Fatalf("failed to create AQLQueue: %v", err)
		}

		// Allocate simulated MMIO hardware aperture (64-byte aligned)
		mockMMIORaw := make([]byte, 64+64)
		base := uintptr(unsafe.Pointer(&mockMMIORaw[0]))
		alignedOffset := (64 - (base % 64)) % 64
		mockMMIOPtr := unsafe.Pointer(&mockMMIORaw[alignedOffset])

		// 1. Benchmark 32-bit MMIO doorbell write latency (< 5µs mock MMIO ceiling)
		q.SetMMIOPointer(mockMMIOPtr)
		q.SetDoorbellWidth(DoorbellWidth32)

		const iterations = 1000
		latencies := make([]int64, iterations)

		for i := 0; i < iterations; i++ {
			writePtr := uint64(i + 1)
			lat, err := q.RingDoorbell(writePtr)
			if err != nil {
				t.Fatalf("iteration %d: RingDoorbell failed: %v", i, err)
			}
			latencies[i] = lat

			// Verify atomic hardware register store
			actualMMIO := atomic.LoadUint32((*uint32)(mockMMIOPtr))
			if actualMMIO != uint32(writePtr) {
				t.Fatalf("iteration %d: MMIO register contains 0x%x, expected 0x%x",
					i, actualMMIO, uint32(writePtr))
			}
		}

		// Verify doorbell telemetry
		if q.RingCount() != iterations {
			t.Fatalf("expected %d doorbell rings, got %d", iterations, q.RingCount())
		}
		if q.FallbackCount() != 0 {
			t.Fatalf("expected 0 fallback rings when MMIO pointer mapped, got %d", q.FallbackCount())
		}

		// Calculate latency percentiles
		sorted := make([]int64, iterations)
		copy(sorted, latencies)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

		p50Ns := sorted[iterations*50/100]
		p90Ns := sorted[iterations*90/100]
		p99Ns := sorted[iterations*99/100]

		p50Us := float64(p50Ns) / 1000.0
		p90Us := float64(p90Ns) / 1000.0
		p99Us := float64(p99Ns) / 1000.0

		t.Logf("32-bit MMIO Doorbell Latency (N=%d): P50=%.2fµs (%dns), P90=%.2fµs (%dns), P99=%.2fµs (%dns)",
			iterations, p50Us, p50Ns, p90Us, p90Ns, p99Us, p99Ns)

		// Assert: Mock MMIO write latency ceiling (< 5µs) and P99 preemption SLA (< 50µs)
		if p50Ns > MockMMIOWriteLatencyCeilingNs {
			t.Errorf("mock MMIO write P50 latency violation: %.2fµs > 5.0µs", p50Us)
		}
		if p99Ns > TargetDoorbellLatencyP99Ns {
			t.Errorf("P99 doorbell preemption SLA violation: %.2fµs > 50.0µs", p99Us)
		}

		// 2. Benchmark 64-bit MMIO doorbell write latency
		q.SetDoorbellWidth(DoorbellWidth64)
		lat64, err := q.RingDoorbell(0x1122334455667788)
		if err != nil {
			t.Fatalf("64-bit RingDoorbell failed: %v", err)
		}
		actualMMIO64 := atomic.LoadUint64((*uint64)(mockMMIOPtr))
		if actualMMIO64 != 0x1122334455667788 {
			t.Fatalf("expected 64-bit MMIO value 0x1122334455667788, got 0x%x", actualMMIO64)
		}
		if lat64 > TargetDoorbellLatencyP99Ns {
			t.Errorf("64-bit MMIO write latency %dns exceeded 50µs SLA", lat64)
		}

		// 3. Fallback mechanism verification: when MMIO pointer is nil/unmapped
		q.SetMMIOPointer(nil)
		q.SetDoorbellWidth(DoorbellWidth32)

		fbLat, err := q.RingDoorbell(9999)
		if err != nil {
			t.Fatalf("fallback RingDoorbell failed: %v", err)
		}
		if q.FallbackCount() != 1 {
			t.Fatalf("expected 1 fallback ring, got %d", q.FallbackCount())
		}
		if q.LoadSimulatedDoorbell32() != 9999 {
			t.Fatalf("expected simulated 32-bit doorbell 9999, got %d", q.LoadSimulatedDoorbell32())
		}
		if q.LoadSimulatedDoorbell64() != 9999 {
			t.Fatalf("expected simulated 64-bit doorbell 9999, got %d", q.LoadSimulatedDoorbell64())
		}
		if fbLat > TargetDoorbellLatencyP99Ns {
			t.Errorf("fallback in-memory simulation latency %dns exceeded 50µs SLA", fbLat)
		}
	})

	t.Run("BarrierFlagHandling", func(t *testing.T) {
		q, err := NewAQLQueue("test-worker-barrier", 16)
		if err != nil {
			t.Fatalf("failed to create AQLQueue: %v", err)
		}

		// 1. Submit regular dispatch packet (barrier = false)
		normHdr := NewPacketHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeComponent, AQLFenceScopeSystem)
		normPkt := AQLKernelDispatchPacket{
			Header:       normHdr,
			Dimensions:   1,
			KernelObject: 0x1000,
		}
		_, _, err = q.SubmitPacket(normPkt)
		if err != nil {
			t.Fatalf("SubmitPacket normPkt failed: %v", err)
		}
		if q.BarrierCount() != 0 {
			t.Fatalf("expected 0 barriers after normal packet, got %d", q.BarrierCount())
		}

		// 2. Submit AQLBarrierPacket (BARRIER_AND with 5 dependency signals)
		barrHdr := NewPacketHeader(AQLPacketTypeBarrierAnd, true, AQLFenceScopeSystem, AQLFenceScopeSystem)
		barrPkt := AQLBarrierPacket{
			Header:           barrHdr,
			DepSignals:       [5]uint64{101, 102, 103, 104, 105},
			CompletionSignal: 999,
		}

		if !barrHdr.Barrier() {
			t.Fatalf("expected barrier flag true on barrier header")
		}
		if barrHdr.Type() != AQLPacketTypeBarrierAnd {
			t.Fatalf("expected packet type BARRIER_AND, got %s", barrHdr.Type())
		}

		nextPtr, lat, err := q.SubmitBarrier(barrPkt)
		if err != nil {
			t.Fatalf("SubmitBarrier failed: %v", err)
		}
		if nextPtr != 2 {
			t.Fatalf("expected next write pointer 2, got %d", nextPtr)
		}
		if lat <= 0 {
			t.Fatalf("expected positive doorbell latency, got %d", lat)
		}
		if q.BarrierCount() != 1 {
			t.Fatalf("expected barrier count 1, got %d", q.BarrierCount())
		}

		// Read back barrier packet from slot 1
		decodedBarr, err := q.GetBarrierPacket(1)
		if err != nil {
			t.Fatalf("GetBarrierPacket failed: %v", err)
		}
		if decodedBarr.Header.Type() != AQLPacketTypeBarrierAnd {
			t.Fatalf("expected BARRIER_AND, got %s", decodedBarr.Header.Type())
		}
		expectedDeps := [5]uint64{101, 102, 103, 104, 105}
		if decodedBarr.DepSignals != expectedDeps {
			t.Fatalf("expected dep signals %v, got %v", expectedDeps, decodedBarr.DepSignals)
		}
		if decodedBarr.CompletionSignal != 999 {
			t.Fatalf("expected completion signal 999, got %d", decodedBarr.CompletionSignal)
		}

		// 3. Submit kernel dispatch with barrier bit set (in-order synchronization)
		syncKernelHdr := NewPacketHeader(AQLPacketTypeKernelDispatch, true, AQLFenceScopeSystem, AQLFenceScopeSystem)
		syncKernelPkt := AQLKernelDispatchPacket{
			Header:       syncKernelHdr,
			Dimensions:   1,
			KernelObject: 0x2000,
		}
		_, _, err = q.SubmitPacket(syncKernelPkt)
		if err != nil {
			t.Fatalf("SubmitPacket syncKernelPkt failed: %v", err)
		}
		if q.BarrierCount() != 2 {
			t.Fatalf("expected barrier count 2 after sync kernel packet, got %d", q.BarrierCount())
		}
	})

	t.Run("ConcurrentMultiSessionSafety", func(t *testing.T) {
		registry := NewAQLQueueRegistry()
		const numSessions = 8
		const turnsPerSession = 50

		var wg sync.WaitGroup
		errs := make(chan error, numSessions*turnsPerSession)
		var latMu sync.Mutex
		var resumeLatencies []int64

		// Launch concurrent worker sessions, each with its own isolated AQL queue
		for s := 0; s < numSessions; s++ {
			sessID := fmt.Sprintf("strix-agent-session-%d", s)
			queue, err := registry.Register(sessID, 256)
			if err != nil {
				t.Fatalf("failed registering session %s: %v", sessID, err)
			}

			wg.Add(1)
			go func(session string, q *AQLQueue) {
				defer wg.Done()

				for turn := 0; turn < turnsPerSession; turn++ {
					// 1. Tool yield: agent enters YIELDED_IO
					if err := q.PreemptForTool(); err != nil {
						errs <- fmt.Errorf("%s turn %d preempt failed: %w", session, turn, err)
						return
					}
					if !q.IsYielded() {
						errs <- fmt.Errorf("%s turn %d expected IsYielded true", session, turn)
						return
					}

					// 2. Tool execution simulated (sub-millisecond external work)
					time.Sleep(50 * time.Microsecond)

					// 3. Lockless resumption: child exit code 0, ringing doorbell
					resumeLat, err := q.ResumeToolYield(0)
					if err != nil {
						errs <- fmt.Errorf("%s turn %d ResumeToolYield failed: %w", session, turn, err)
						return
					}
					latMu.Lock()
					resumeLatencies = append(resumeLatencies, resumeLat)
					latMu.Unlock()
					if q.IsYielded() {
						errs <- fmt.Errorf("%s turn %d expected IsYielded false after resume", session, turn)
						return
					}

					// 4. Submit active decoding packet
					pkt := AQLKernelDispatchPacket{
						Header:       NewPacketHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeComponent, AQLFenceScopeSystem),
						KernelObject: uint64(turn + 1),
					}
					_, _, err = q.SubmitPacket(pkt)
					if err != nil {
						errs <- fmt.Errorf("%s turn %d SubmitPacket failed: %w", session, turn, err)
						return
					}
				}
			}(sessID, queue)
		}

		wg.Wait()
		close(errs)

		for err := range errs {
			t.Errorf("concurrent worker error: %v", err)
		}

		if len(resumeLatencies) > 0 {
			sort.Slice(resumeLatencies, func(i, j int) bool { return resumeLatencies[i] < resumeLatencies[j] })
			p50Ns := resumeLatencies[len(resumeLatencies)*50/100]
			p99Ns := resumeLatencies[len(resumeLatencies)*99/100]
			t.Logf("Concurrent Multi-Session Resumption (N=%d): P50=%.2fµs, P99=%.2fµs",
				len(resumeLatencies), float64(p50Ns)/1000.0, float64(p99Ns)/1000.0)
			if p99Ns > TargetDoorbellLatencyP99Ns {
				t.Errorf("concurrent multi-session P99 resume latency %.2fµs exceeded 50µs SLA", float64(p99Ns)/1000.0)
			}
		}

		if registry.Count() != numSessions {
			t.Errorf("expected %d registered queues, got %d", numSessions, registry.Count())
		}

		// Also test concurrent submissions on a single shared queue
		sharedQueue, err := NewAQLQueue("shared-queue", 1024)
		if err != nil {
			t.Fatalf("failed to create shared queue: %v", err)
		}

		const sharedWorkers = 8
		const packetsPerWorker = 50
		var sharedWg sync.WaitGroup

		for w := 0; w < sharedWorkers; w++ {
			sharedWg.Add(1)
			go func(workerID int) {
				defer sharedWg.Done()
				for p := 0; p < packetsPerWorker; p++ {
					pkt := AQLKernelDispatchPacket{
						Header:       NewPacketHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeNone, AQLFenceScopeNone),
						KernelObject: uint64(workerID*1000 + p),
					}
					_, _, err := sharedQueue.SubmitPacket(pkt)
					if err != nil {
						t.Errorf("shared worker %d submission %d failed: %v", workerID, p, err)
					}
				}
			}(w)
		}

		sharedWg.Wait()

		expectedTotal := uint64(sharedWorkers * packetsPerWorker)
		if sharedQueue.LoadWritePtr() != expectedTotal {
			t.Fatalf("expected shared write pointer %d, got %d", expectedTotal, sharedQueue.LoadWritePtr())
		}
		if sharedQueue.RingCount() != expectedTotal {
			t.Fatalf("expected %d doorbell rings on shared queue, got %d", expectedTotal, sharedQueue.RingCount())
		}
	})

	t.Run("ToolYieldPreemptionAndResumptionLifecycle", func(t *testing.T) {
		q, err := NewAQLQueue("test-worker-yield", 64)
		if err != nil {
			t.Fatalf("failed to create AQLQueue: %v", err)
		}

		// Initial state must be ACTIVE
		if q.State() != QueueStateActive {
			t.Fatalf("expected initial state ACTIVE, got %s", q.State())
		}
		if q.IsYielded() {
			t.Fatalf("expected initial IsYielded false")
		}

		// Preempt for tool
		if err := q.PreemptForTool(); err != nil {
			t.Fatalf("PreemptForTool failed: %v", err)
		}
		if q.State() != QueueStateYielded {
			t.Fatalf("expected state YIELDED, got %s", q.State())
		}
		if !q.IsYielded() {
			t.Fatalf("expected IsYielded true")
		}

		// Resume without packet
		resumeLat, err := q.ResumeToolYield(0)
		if err != nil {
			t.Fatalf("ResumeToolYield failed: %v", err)
		}
		if q.State() != QueueStateActive {
			t.Fatalf("expected state ACTIVE after resume, got %s", q.State())
		}
		if q.LastExitCode() != 0 {
			t.Fatalf("expected exit code 0, got %d", q.LastExitCode())
		}
		if resumeLat > TargetDoorbellLatencyP99Ns {
			t.Errorf("resumption latency %dns exceeded 50µs SLA", resumeLat)
		}

		// Preempt and resume with non-zero exit code and explicit resumption packet
		_ = q.PreemptForTool()
		resumptionPkt := AQLKernelDispatchPacket{
			Header:       NewPacketHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeSystem, AQLFenceScopeSystem),
			KernelObject: 0x99990000,
		}

		latWithPkt, err := q.ResumeToolYield(42, resumptionPkt)
		if err != nil {
			t.Fatalf("ResumeToolYield with packet failed: %v", err)
		}
		if q.LastExitCode() != 42 {
			t.Fatalf("expected exit code 42, got %d", q.LastExitCode())
		}
		if latWithPkt > TargetDoorbellLatencyP99Ns {
			t.Errorf("resumption with packet latency %dns exceeded 50µs SLA", latWithPkt)
		}

		// Verify resumption packet landed in ring buffer
		slotIdx := q.SlotIndex(q.LoadWritePtr() - 1)
		readPkt, err := q.GetKernelDispatchPacket(slotIdx)
		if err != nil {
			t.Fatalf("failed reading resumption packet: %v", err)
		}
		if readPkt.KernelObject != 0x99990000 {
			t.Fatalf("expected kernel object 0x99990000, got 0x%x", readPkt.KernelObject)
		}
	})
}

// TestAQLPacketHeaderAndTypes tests packet header string formatting, bit manipulation,
// and packet type / scope enumeration strings.
func TestAQLPacketHeaderAndTypes(t *testing.T) {
	// Header constructors and bit manipulation
	hdr := NewPacketHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeNone, AQLFenceScopeNone)
	if hdr.Raw() != 2 {
		t.Fatalf("expected raw 2, got %d", hdr.Raw())
	}

	hdrBarrier := hdr.WithBarrier(true)
	if !hdrBarrier.Barrier() {
		t.Fatalf("expected barrier true")
	}
	if hdrBarrier.Type() != AQLPacketTypeKernelDispatch {
		t.Fatalf("expected type KERNEL_DISPATCH, got %s", hdrBarrier.Type())
	}

	hdrNoBarrier := hdrBarrier.WithBarrier(false)
	if hdrNoBarrier.Barrier() {
		t.Fatalf("expected barrier false")
	}

	hdrFences := hdr.WithFenceScope(AQLFenceScopeSystem, AQLFenceScopeComponent)
	if hdrFences.AcquireScope() != AQLFenceScopeSystem {
		t.Fatalf("expected acquire SYSTEM, got %s", hdrFences.AcquireScope())
	}
	if hdrFences.ReleaseScope() != AQLFenceScopeComponent {
		t.Fatalf("expected release COMPONENT, got %s", hdrFences.ReleaseScope())
	}

	// String representation
	hdrStr := hdrFences.String()
	if hdrStr == "" {
		t.Fatalf("expected non-empty header string")
	}

	// Packet types string coverage
	types := []AQLPacketType{
		AQLPacketTypeVendorSpecific,
		AQLPacketTypeInvalid,
		AQLPacketTypeKernelDispatch,
		AQLPacketTypeBarrierAnd,
		AQLPacketTypeAgentDispatch,
		AQLPacketTypeBarrierOr,
		AQLPacketType(99),
	}
	for _, pt := range types {
		if pt.String() == "" {
			t.Fatalf("expected non-empty string for packet type %d", pt)
		}
	}

	// Fence scopes string coverage
	scopes := []AQLFenceScope{
		AQLFenceScopeNone,
		AQLFenceScopeComponent,
		AQLFenceScopeSystem,
		AQLFenceScope(99),
	}
	for _, sc := range scopes {
		if sc.String() == "" {
			t.Fatalf("expected non-empty string for fence scope %d", sc)
		}
	}

	// DoorbellWidth string coverage
	widths := []DoorbellWidth{
		DoorbellWidth32,
		DoorbellWidth64,
		DoorbellWidth(128),
	}
	for _, w := range widths {
		if w.String() == "" {
			t.Fatalf("expected non-empty string for doorbell width %d", w)
		}
	}

	// QueueState string coverage
	states := []QueueState{
		QueueStateActive,
		QueueStateYielded,
		QueueStateResuming,
		QueueState(99),
	}
	for _, st := range states {
		if st.String() == "" {
			t.Fatalf("expected non-empty string for queue state %d", st)
		}
	}
}

// TestAQLQueueEdgeCasesAndRegistry tests submission validation, closing behavior,
// and registry lifecycle.
func TestAQLQueueEdgeCasesAndRegistry(t *testing.T) {
	var mockReg uint64
	q, err := NewDefaultAQLQueue("test-worker-edge",
		WithDoorbellWidth(DoorbellWidth64),
		WithQueueID(42),
		WithMMIOPointer(unsafe.Pointer(&mockReg)),
	)
	if err != nil {
		t.Fatalf("NewDefaultAQLQueue failed: %v", err)
	}

	if q.LoadMMIOPointer() != unsafe.Pointer(&mockReg) {
		t.Fatalf("expected MMIOPointer to match mockReg")
	}

	// Read pointer update
	q.SetReadPtr(1)
	if q.LoadReadPtr() != 1 {
		t.Fatalf("expected read pointer 1, got %d", q.LoadReadPtr())
	}
	q.SetReadPtr(0)

	if q.QueueID() != 42 {
		t.Fatalf("expected queue ID 42, got %d", q.QueueID())
	}
	if q.DoorbellWidth() != DoorbellWidth64 {
		t.Fatalf("expected doorbell width 64-bit, got %s", q.DoorbellWidth())
	}
	if q.SessionID() != "test-worker-edge" {
		t.Fatalf("expected session ID test-worker-edge, got %s", q.SessionID())
	}

	// PacketHeader interface verification
	kPkt := AQLKernelDispatchPacket{Header: NewPacketHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeNone, AQLFenceScopeNone)}
	if kPkt.PacketHeader() != kPkt.Header {
		t.Fatalf("AQLKernelDispatchPacket.PacketHeader mismatch")
	}
	bPkt := AQLBarrierPacket{Header: NewPacketHeader(AQLPacketTypeBarrierAnd, true, AQLFenceScopeNone, AQLFenceScopeNone)}
	if bPkt.PacketHeader() != bPkt.Header {
		t.Fatalf("AQLBarrierPacket.PacketHeader mismatch")
	}

	// Submission validation errors
	if _, _, err := q.SubmitPacket(nil); err != ErrNilPacket {
		t.Fatalf("expected ErrNilPacket, got %v", err)
	}
	if _, _, err := q.SubmitPacket((*AQLKernelDispatchPacket)(nil)); err != ErrNilPacket {
		t.Fatalf("expected ErrNilPacket for nil pointer, got %v", err)
	}
	if _, _, err := q.SubmitPacket((*AQLBarrierPacket)(nil)); err != ErrNilPacket {
		t.Fatalf("expected ErrNilPacket for nil barrier pointer, got %v", err)
	}
	if _, _, err := q.SubmitPacket((*[64]byte)(nil)); err != ErrNilPacket {
		t.Fatalf("expected ErrNilPacket for nil array pointer, got %v", err)
	}
	if _, _, err := q.SubmitPacket("unsupported-type"); err == nil {
		t.Fatalf("expected error for unsupported packet type, got nil")
	}
	if _, _, err := q.SubmitPacket([]byte{1, 2, 3}); err == nil {
		t.Fatalf("expected error for invalid byte slice length, got nil")
	}

	// Submit via raw [64]byte and []byte of length 64
	var raw64 [64]byte
	raw64[0] = uint8(AQLPacketTypeKernelDispatch)
	if _, _, err := q.SubmitPacket(raw64); err != nil {
		t.Fatalf("SubmitPacket [64]byte failed: %v", err)
	}
	if _, _, err := q.SubmitPacket(&raw64); err != nil {
		t.Fatalf("SubmitPacket *[64]byte failed: %v", err)
	}
	if _, _, err := q.SubmitPacket(raw64[:]); err != nil {
		t.Fatalf("SubmitPacket []byte failed: %v", err)
	}

	if q.LastRingNs() <= 0 {
		t.Fatalf("expected positive LastRingNs, got %d", q.LastRingNs())
	}

	// Empty queue base check
	emptyQ := &AQLQueue{}
	if emptyQ.RingBufferBase() != 0 {
		t.Fatalf("expected 0 for empty queue base")
	}

	// GetPacket and decoding mismatch checks
	// Slot 0 has KERNEL_DISPATCH, GetBarrierPacket must fail
	if _, err := q.GetBarrierPacket(0); err == nil {
		t.Fatalf("expected error from GetBarrierPacket on kernel dispatch slot, got nil")
	}

	// Slot with barrier, GetKernelDispatchPacket must fail
	var barrRaw [64]byte
	barrRaw[0] = uint8(AQLPacketTypeBarrierAnd)
	if _, _, err := q.SubmitPacket(barrRaw); err != nil {
		t.Fatalf("submit barrier raw failed: %v", err)
	}
	if _, err := q.GetKernelDispatchPacket(q.LoadWritePtr() - 1); err == nil {
		t.Fatalf("expected error from GetKernelDispatchPacket on barrier slot, got nil")
	}

	// Close queue and verify submissions are rejected
	if err := q.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	if _, _, err := q.SubmitPacket(raw64); err != ErrQueueClosed {
		t.Fatalf("expected ErrQueueClosed after Close(), got %v", err)
	}

	// Registry lifecycle
	reg := NewAQLQueueRegistry()
	if _, err := reg.Register("", 64); err == nil {
		t.Fatalf("expected error for empty session ID in register")
	}

	regQ, err := reg.Register("session-alpha", 64)
	if err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	if _, err := reg.Register("session-alpha", 64); err == nil {
		t.Fatalf("expected error registering duplicate session")
	}

	fetched, ok := reg.Get("session-alpha")
	if !ok || fetched != regQ {
		t.Fatalf("Get returned unexpected queue")
	}
	if _, ok := reg.Get("missing-session"); ok {
		t.Fatalf("Get returned ok true for missing session")
	}

	gotOrCreate, err := reg.GetOrCreate("session-alpha", 64)
	if err != nil || gotOrCreate != regQ {
		t.Fatalf("GetOrCreate failed on existing queue")
	}
	newCreated, err := reg.GetOrCreate("session-beta", 128)
	if err != nil || newCreated == nil {
		t.Fatalf("GetOrCreate failed on new queue: %v", err)
	}
	if reg.Count() != 2 {
		t.Fatalf("expected count 2, got %d", reg.Count())
	}

	if err := reg.Unregister("session-alpha"); err != nil {
		t.Fatalf("Unregister failed: %v", err)
	}
	if err := reg.Unregister("session-alpha"); err == nil {
		t.Fatalf("expected error unregistering already removed session")
	}
	if reg.Count() != 1 {
		t.Fatalf("expected count 1, got %d", reg.Count())
	}
}
