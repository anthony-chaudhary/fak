package compute

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestAMDGPUDirect_WatchdogAndQPAutoHealing(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})

	// 1. Setup AMD GPU device node with DMA-BUF & Large BAR capabilities
	err := hal.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "Instinct MI300X",
		Architecture:   "gfx942",
		PCIeBDF:        "0000:41:00.0",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})
	if err != nil {
		t.Fatalf("RegisterNode failed: %v", err)
	}

	t.Run("ActiveDetectionOfStalledHSASignalsAndQueueFlush", func(t *testing.T) {
		sendCQ := NewCompletionQueue(101, 64)
		recvCQ := NewCompletionQueue(102, 64)

		qp, err := hal.CreateQueuePair(QPInitAttr{
			QPType:    QPTypeRC,
			SendCQ:    sendCQ,
			RecvCQ:    recvCQ,
			MaxSendWR: 32,
			MaxRecvWR: 32,
			NodeID:    0,
		})
		if err != nil {
			t.Fatalf("CreateQueuePair failed: %v", err)
		}

		_ = qp.Modify(QPAttr{State: QPStateInit})
		_ = qp.Modify(QPAttr{State: QPStateRTR, DestQPN: 9999})
		_ = qp.Modify(QPAttr{State: QPStateRTS})

		// Post in-flight send and receive work requests
		sendWR := &WorkRequest{
			WRID:   1001,
			OpCode: RDMAOpSend,
			SGEs: []ScatterGatherElement{
				{Address: 0x10000000, Length: 4096, LKey: 1},
			},
		}
		recvWR := &WorkRequest{
			WRID:   2001,
			OpCode: RDMAOpReceive,
			SGEs: []ScatterGatherElement{
				{Address: 0x20000000, Length: 4096, LKey: 2},
			},
		}

		if err := qp.PostSend(sendWR); err != nil {
			t.Fatalf("PostSend failed: %v", err)
		}
		if err := qp.PostRecv(recvWR); err != nil {
			t.Fatalf("PostRecv failed: %v", err)
		}

		// Verify queues have pending entries before stall
		qp.mu.RLock()
		sendLen := len(qp.sendQueue)
		recvLen := len(qp.recvQueue)
		qp.mu.RUnlock()
		if sendLen != 1 || recvLen != 1 {
			t.Fatalf("expected 1 send and 1 recv pending, got send=%d, recv=%d", sendLen, recvLen)
		}

		// Setup watchdog and stalled HSA completion signal (target 1, remains 0)
		watchdog := NewLinkHealthWatchdog(WatchdogConfig{
			SignalSLADeadline: 15 * time.Millisecond,
		}, hal)

		sig := NewHSAMemorySignal("sig-stall-test", 0, 0x80001000)

		// Wait for signal - should detect stall past SLA deadline
		completed, pollErr := watchdog.PollHSASignalWithDeadline(sig, 1, 15*time.Millisecond, qp)
		if completed {
			t.Errorf("expected signal to not complete, but completed=true")
		}
		if !errors.Is(pollErr, ErrTransferStalled) {
			t.Fatalf("expected ErrTransferStalled, got: %v", pollErr)
		}

		// Verify QP transitioned to QPStateError
		qp.mu.RLock()
		state := qp.State
		sendLenAfter := len(qp.sendQueue)
		recvLenAfter := len(qp.recvQueue)
		qp.mu.RUnlock()

		if state != QPStateError {
			t.Errorf("expected QP state %s, got %s", QPStateError, state)
		}
		if sendLenAfter != 0 || recvLenAfter != 0 {
			t.Errorf("expected pending queues flushed, got send=%d, recv=%d", sendLenAfter, recvLenAfter)
		}

		// Verify Completion Queues received flushed entries with status WCWrFlushedErr
		sendCompletions := sendCQ.PollCQ(10)
		if len(sendCompletions) != 1 {
			t.Fatalf("expected 1 send completion, got %d", len(sendCompletions))
		}
		if sendCompletions[0].Status != WCWrFlushedErr {
			t.Errorf("expected status %s, got %s", WCWrFlushedErr, sendCompletions[0].Status)
		}
		if sendCompletions[0].WRID != 1001 {
			t.Errorf("expected WRID 1001, got %d", sendCompletions[0].WRID)
		}
		if sendCompletions[0].StagingCopyCount() != 0 {
			t.Errorf("expected 0 host staging copies, got %d", sendCompletions[0].StagingCopyCount())
		}

		recvCompletions := recvCQ.PollCQ(10)
		if len(recvCompletions) != 1 {
			t.Fatalf("expected 1 recv completion, got %d", len(recvCompletions))
		}
		if recvCompletions[0].Status != WCWrFlushedErr {
			t.Errorf("expected status %s, got %s", WCWrFlushedErr, recvCompletions[0].Status)
		}
		if recvCompletions[0].WRID != 2001 {
			t.Errorf("expected WRID 2001, got %d", recvCompletions[0].WRID)
		}
	})

	t.Run("AutomaticQueueFlushViaPollCQDeadline", func(t *testing.T) {
		sendCQ := NewCompletionQueue(201, 64)
		recvCQ := NewCompletionQueue(202, 64)

		qp, _ := hal.CreateQueuePair(QPInitAttr{
			QPType:    QPTypeRC,
			SendCQ:    sendCQ,
			RecvCQ:    recvCQ,
			MaxSendWR: 32,
			MaxRecvWR: 32,
			NodeID:    0,
		})
		_ = qp.Modify(QPAttr{State: QPStateInit})
		_ = qp.Modify(QPAttr{State: QPStateRTR, DestQPN: 8888})
		_ = qp.Modify(QPAttr{State: QPStateRTS})

		_ = qp.PostSend(&WorkRequest{
			WRID:   3001,
			OpCode: RDMAOpSend,
			SGEs:   []ScatterGatherElement{{Address: 0x1000, Length: 512, LKey: 1}},
		})

		watchdog := NewLinkHealthWatchdog(WatchdogConfig{
			CQSLADeadline: 10 * time.Millisecond,
		}, hal)

		// Poll CQ for WRID 3001 with deadline - will stall because send queue is not processed
		wc, err := watchdog.PollCQWithDeadline(sendCQ, 3001, 10*time.Millisecond, qp)
		if !errors.Is(err, ErrTransferStalled) {
			t.Fatalf("expected ErrTransferStalled, got %v", err)
		}
		if wc == nil {
			t.Fatal("expected flushed WorkCompletion entry, got nil")
		}
		if wc.Status != WCWrFlushedErr {
			t.Errorf("expected status %s, got %s", WCWrFlushedErr, wc.Status)
		}
		if qp.State != QPStateError {
			t.Errorf("expected QPStateError, got %s", qp.State)
		}
	})

	t.Run("SuccessfulQPReinitializationToRTS", func(t *testing.T) {
		sendCQ := NewCompletionQueue(301, 64)
		recvCQ := NewCompletionQueue(302, 64)

		qp, _ := hal.CreateQueuePair(QPInitAttr{
			QPType:    QPTypeRC,
			SendCQ:    sendCQ,
			RecvCQ:    recvCQ,
			MaxSendWR: 32,
			MaxRecvWR: 32,
			NodeID:    0,
		})

		// Force into ERROR state
		_ = qp.Modify(QPAttr{State: QPStateError})
		if qp.State != QPStateError {
			t.Fatalf("expected state ERROR, got %s", qp.State)
		}

		handshakeCalled := false
		healer := NewQPAutoHealer(AutoHealerConfig{
			InitialBackoff: 2 * time.Millisecond,
			MaxBackoff:     20 * time.Millisecond,
			BackoffFactor:  2.0,
			JitterFraction: 0.1,
			MaxRetries:     3,
			HandshakeFn: func(localQPN, destQPN uint32) (bool, error) {
				handshakeCalled = true
				return true, nil
			},
		})

		// Execute automated healing sequence: ERROR -> RESET -> INIT -> RTR -> RTS
		err := healer.HealQP(qp, 5555, 4096, 10, 20)
		if err != nil {
			t.Fatalf("HealQP failed: %v", err)
		}

		if !handshakeCalled {
			t.Errorf("expected peer handshake to be verified during auto-healing")
		}

		qp.mu.RLock()
		finalState := qp.State
		destQPN := qp.DestQPN
		qp.mu.RUnlock()

		if finalState != QPStateRTS {
			t.Fatalf("expected QP state %s, got %s", QPStateRTS, finalState)
		}
		if destQPN != 5555 {
			t.Errorf("expected DestQPN 5555, got %d", destQPN)
		}

		// Verify work requests can be posted and executed again after healing
		postErr := qp.PostSend(&WorkRequest{
			WRID:   4001,
			OpCode: RDMAOpSend,
			SGEs:   []ScatterGatherElement{{Address: 0x5000, Length: 1024, LKey: 1}},
		})
		if postErr != nil {
			t.Errorf("PostSend on healed QP failed: %v", postErr)
		}
	})

	t.Run("ZeroLeakedResourcesRuntimeAssertion", func(t *testing.T) {
		tm := NewTeardownManager(hal)

		// Allocate and track DMA-BUF
		dmabuf, err := hal.ExportVRAMToDMABUF(0, 0x60000000, 16*1024*1024)
		if err != nil {
			t.Fatalf("ExportVRAMToDMABUF failed: %v", err)
		}
		if err := tm.TrackDMABUF(dmabuf, nil); err != nil {
			t.Fatalf("TrackDMABUF failed: %v", err)
		}

		// Allocate and track RDMA registered region
		mr, err := hal.RegisterDMABUFForRDMA(dmabuf.FD, 16*1024*1024)
		if err != nil {
			t.Fatalf("RegisterDMABUFForRDMA failed: %v", err)
		}
		if err := tm.TrackRDMARegion(mr, nil); err != nil {
			t.Fatalf("TrackRDMARegion failed: %v", err)
		}

		// Allocate and track HSA Doorbell
		db := NewHSADoorbell("db-test-1", 0x70000000, 1)
		hal.RegisterDoorbell(db)
		if err := tm.TrackDoorbell(db, nil); err != nil {
			t.Fatalf("TrackDoorbell failed: %v", err)
		}

		// Track BAR1 VRAM aperture
		apertureUnmapped := false
		err = tm.TrackBAR1Aperture(0x80000000, 64*1024*1024, 0, func(addr uintptr, size uint64) error {
			apertureUnmapped = true
			return nil
		})
		if err != nil {
			t.Fatalf("TrackBAR1Aperture failed: %v", err)
		}

		// Before teardown: assert leak detection detects unclosed resources
		leakErr := tm.AssertZeroLeaks()
		if leakErr == nil {
			t.Errorf("expected AssertZeroLeaks to report active resources before teardown")
		}

		repBefore := tm.Report()
		if repBefore.ActiveDMABUFs != 1 || repBefore.ActiveRDMARegs != 1 || repBefore.ActiveDoorbells != 1 || repBefore.ActiveApertures != 1 {
			t.Errorf("unexpected pre-teardown report: %+v", repBefore)
		}

		// Execute teardown
		if err := tm.Teardown(); err != nil {
			t.Fatalf("Teardown failed: %v", err)
		}

		// After teardown: assert zero leaks
		if err := tm.AssertZeroLeaks(); err != nil {
			t.Fatalf("expected 0 leaks after teardown, got: %v", err)
		}

		repAfter := tm.Report()
		if repAfter.ActiveDMABUFs != 0 || repAfter.ActiveRDMARegs != 0 || repAfter.ActiveDoorbells != 0 || repAfter.ActiveApertures != 0 {
			t.Errorf("unexpected post-teardown report: %+v", repAfter)
		}
		if !repAfter.Closed {
			t.Errorf("expected TeardownManager to be marked closed")
		}

		if !dmabuf.Closed {
			t.Errorf("expected DMA-BUF handle to be closed")
		}
		if mr.Active {
			t.Errorf("expected RDMA region to be inactive")
		}
		if !apertureUnmapped {
			t.Errorf("expected BAR1 aperture to be unmapped")
		}

		// Confirm coordinator audit shows zero lingering resources
		audit := hal.Audit()
		if audit.ActiveDMABUFCount != 0 {
			t.Errorf("expected 0 active DMA-BUFs in coordinator, got %d", audit.ActiveDMABUFCount)
		}
		if audit.ActiveRDMARegions != 0 {
			t.Errorf("expected 0 active RDMA regions in coordinator, got %d", audit.ActiveRDMARegions)
		}
	})

	t.Run("FaultInjectionSimulatingNetworkDisconnect", func(t *testing.T) {
		tm := NewTeardownManager(hal)

		// Create 3 active DMA-BUFs and RDMA registrations
		for i := 0; i < 3; i++ {
			buf, err := hal.ExportVRAMToDMABUF(0, uintptr(0x90000000+i*0x1000000), 4*1024*1024)
			if err != nil {
				t.Fatalf("ExportVRAMToDMABUF[%d] failed: %v", i, err)
			}
			_ = tm.TrackDMABUF(buf, nil)

			region, err := hal.RegisterDMABUFForRDMA(buf.FD, 4*1024*1024)
			if err != nil {
				t.Fatalf("RegisterDMABUFForRDMA[%d] failed: %v", i, err)
			}
			_ = tm.TrackRDMARegion(region, nil)
		}

		rep := tm.Report()
		if rep.ActiveDMABUFs != 3 || rep.ActiveRDMARegs != 3 {
			t.Fatalf("expected 3 dmabufs and 3 rdma regs, got: %+v", rep)
		}

		// Fault injection: sudden network socket disconnect (ECONNRESET)
		socketDisconnectErr := errors.New("read tcp 10.0.1.10:4791->10.0.1.20:4791: connection reset by peer (ECONNRESET)")
		handleErr := tm.HandleFault(socketDisconnectErr)
		if handleErr != nil {
			t.Fatalf("HandleFault returned error: %v", handleErr)
		}

		// Verify all handles were cleanly closed and zero memory leaks remain
		if err := tm.AssertZeroLeaks(); err != nil {
			t.Fatalf("leak detected following emergency fault recovery: %v", err)
		}

		audit := hal.Audit()
		if audit.ActiveDMABUFCount != 0 || audit.ActiveRDMARegions != 0 {
			t.Errorf("lingering resources in HAL after fault teardown: dmabufs=%d, rdma=%d",
				audit.ActiveDMABUFCount, audit.ActiveRDMARegions)
		}
	})
}

func TestAMDGPUDirect_AERThresholdEvaluation(t *testing.T) {
	t.Run("ParseAERCounters", func(t *testing.T) {
		rawSysfs := `
Receiver_Error 3
Bad_TLP 1
Bad_DLLP 0
RELAY_NUM_ROLLOVER 0
replay_timer_timeout 0
Advisory_Non-Fatal 2
Corrected_Internal_Error 0
Header_Log_Overflow 0
TOTAL_ERR_COR 6
`
		val := ParseAERCounters(rawSysfs)
		if val != 6 {
			t.Errorf("expected 6, got %d", val)
		}

		// Table without explicit total line
		rawSum := `
RxErr 4
BadTLP 5
`
		valSum := ParseAERCounters(rawSum)
		if valSum != 9 {
			t.Errorf("expected 9, got %d", valSum)
		}

		// Single integer
		valInt := ParseAERCounters("42\n")
		if valInt != 42 {
			t.Errorf("expected 42, got %d", valInt)
		}

		// Empty string
		valEmpty := ParseAERCounters("")
		if valEmpty != 0 {
			t.Errorf("expected 0, got %d", valEmpty)
		}
	})

	t.Run("EvaluateAERStatusThresholds", func(t *testing.T) {
		thresholds := AERThresholdConfig{
			MaxCorrectable: 10,
			MaxFatal:       0,
		}

		// Healthy: correctable within limits, zero fatal
		st, _ := EvaluateAERStatus(AERCounters{Correctable: 5, Fatal: 0}, thresholds)
		if st != LinkHealthHealthy {
			t.Errorf("expected %s, got %s", LinkHealthHealthy, st)
		}

		// Degraded: correctable exceeds limit, zero fatal
		st, reason := EvaluateAERStatus(AERCounters{Correctable: 15, Fatal: 0}, thresholds)
		if st != LinkHealthDegraded {
			t.Errorf("expected %s, got %s", LinkHealthDegraded, st)
		}
		if reason == "" {
			t.Errorf("expected non-empty degradation reason")
		}

		// Failed: fatal > 0
		st, _ = EvaluateAERStatus(AERCounters{Correctable: 2, Fatal: 1}, thresholds)
		if st != LinkHealthFailed {
			t.Errorf("expected %s, got %s", LinkHealthFailed, st)
		}
	})

	t.Run("WatchdogLinkHealthMonitoringWithCallback", func(t *testing.T) {
		hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
		watchdog := NewLinkHealthWatchdog(WatchdogConfig{
			DefaultAERLimits: AERThresholdConfig{MaxCorrectable: 5, MaxFatal: 0},
		}, hal)

		bdf := "0000:41:00.0"
		watchdog.RegisterMonitoredBDF(bdf, AERThresholdConfig{MaxCorrectable: 8, MaxFatal: 0})

		// Mock reader providing degraded counters
		simulatedCounters := AERCounters{Correctable: 12, Fatal: 0}
		watchdog.SetAERReader(func(devBDF string) (AERCounters, error) {
			if devBDF == bdf {
				return simulatedCounters, nil
			}
			return AERCounters{}, errors.New("unknown device")
		})

		callbackInvoked := false
		watchdog.SetDegradationCallback(func(targetBDF string, status LinkHealthStatus, reason string) {
			if targetBDF == bdf && status == LinkHealthDegraded {
				callbackInvoked = true
			}
		})

		status, counters, err := watchdog.CheckLinkHealth(bdf)
		if status != LinkHealthDegraded {
			t.Errorf("expected status %s, got %s", LinkHealthDegraded, status)
		}
		if counters.Correctable != 12 {
			t.Errorf("expected 12 correctable errors, got %d", counters.Correctable)
		}
		if !errors.Is(err, ErrLinkDegraded) {
			t.Errorf("expected ErrLinkDegraded, got %v", err)
		}
		if !callbackInvoked {
			t.Errorf("expected degradation callback to be invoked")
		}

		// Now simulate fatal link drop
		simulatedCounters = AERCounters{Correctable: 12, Fatal: 1}
		statusFatal, _, errFatal := watchdog.CheckLinkHealth(bdf)
		if statusFatal != LinkHealthFailed {
			t.Errorf("expected status %s, got %s", LinkHealthFailed, statusFatal)
		}
		if !errors.Is(errFatal, ErrLinkFatal) {
			t.Errorf("expected ErrLinkFatal, got %v", errFatal)
		}
	})
}

func TestAMDGPUDirect_AutoHealerExponentialBackoff(t *testing.T) {
	cfg := AutoHealerConfig{
		InitialBackoff: 10 * time.Millisecond,
		MaxBackoff:     100 * time.Millisecond,
		BackoffFactor:  2.0,
		JitterFraction: 0.15,
		MaxRetries:     5,
	}
	healer := NewQPAutoHealer(cfg)

	// Attempt 0: ~10ms +/- 15% (8.5ms .. 11.5ms)
	b0 := healer.ComputeBackoff(0)
	if b0 < 8*time.Millisecond || b0 > 13*time.Millisecond {
		t.Errorf("attempt 0 backoff out of expected range: %v", b0)
	}

	// Attempt 1: ~20ms +/- 15% (17ms .. 23ms)
	b1 := healer.ComputeBackoff(1)
	if b1 < 16*time.Millisecond || b1 > 25*time.Millisecond {
		t.Errorf("attempt 1 backoff out of expected range: %v", b1)
	}

	// Attempt 2: ~40ms +/- 15% (34ms .. 46ms)
	b2 := healer.ComputeBackoff(2)
	if b2 < 32*time.Millisecond || b2 > 48*time.Millisecond {
		t.Errorf("attempt 2 backoff out of expected range: %v", b2)
	}

	// Attempt 10: clamped to MaxBackoff (100ms) +/- 15% (85ms .. 115ms)
	b10 := healer.ComputeBackoff(10)
	if b10 < 80*time.Millisecond || b10 > 120*time.Millisecond {
		t.Errorf("attempt 10 backoff out of expected clamped range: %v", b10)
	}
}

func TestAMDGPUDirect_ConcurrentTeardownSafety(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "MI300X",
		TotalVRAMBytes: 64 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  64 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})

	tm := NewTeardownManager(hal)

	// Create and track resources
	for i := 0; i < 10; i++ {
		buf, err := hal.ExportVRAMToDMABUF(0, uintptr(0xA0000000+i*0x100000), 1024*1024)
		if err != nil {
			t.Fatalf("ExportVRAMToDMABUF failed: %v", err)
		}
		_ = tm.TrackDMABUF(buf, nil)

		mr, err := hal.RegisterDMABUFForRDMA(buf.FD, 1024*1024)
		if err != nil {
			t.Fatalf("RegisterDMABUFForRDMA failed: %v", err)
		}
		_ = tm.TrackRDMARegion(mr, nil)

		db := NewHSADoorbell(fmt.Sprintf("concurrent-db-%d", i), uintptr(0xB0000000+i*0x1000), uint32(i))
		_ = tm.TrackDoorbell(db, nil)
	}

	// Launch 30 goroutines all invoking Teardown concurrently
	const concurrency = 30
	var wg sync.WaitGroup
	wg.Add(concurrency)

	errCh := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()
			err := tm.Teardown()
			if err != nil {
				errCh <- err
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent Teardown returned unexpected error: %v", err)
	}

	// Verify zero leaks after concurrent teardown
	if err := tm.AssertZeroLeaks(); err != nil {
		t.Fatalf("AssertZeroLeaks failed after concurrent teardown: %v", err)
	}
}

func TestAMDGPUDirect_TeardownPanicRecovery(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	tm := NewTeardownManager(hal)

	// Register normal cleanup and one that panics
	normalExecuted := false
	tm.TrackCleanup(func() error {
		normalExecuted = true
		return nil
	})

	tm.TrackCleanup(func() error {
		panic("kernel driver segfault simulated")
	})

	// Execute with panic recovery
	err := tm.TeardownWithPanicRecovery()
	if err == nil {
		t.Fatal("expected error from recovered panic, got nil")
	}
	if !errors.Is(err, ErrTeardownPanic) {
		t.Fatalf("expected ErrTeardownPanic, got: %v", err)
	}

	if !normalExecuted {
		t.Errorf("expected normal cleanup to have executed")
	}

	rep := tm.Report()
	if !rep.Closed {
		t.Errorf("expected TeardownManager to be marked closed despite panic")
	}
}
