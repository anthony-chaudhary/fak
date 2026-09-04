package compute

import (
	"testing"
	"time"
)

func TestAMDGPUDirectRDMA_QPStateTransitions(t *testing.T) {
	sendCQ := NewCompletionQueue(1, 64)
	recvCQ := NewCompletionQueue(2, 64)

	qp, err := NewRDMAQueuePair(1001, QPInitAttr{
		QPType:    QPTypeRC,
		SendCQ:    sendCQ,
		RecvCQ:    recvCQ,
		MaxSendWR: 32,
		MaxRecvWR: 32,
		NodeID:    0,
	})
	if err != nil {
		t.Fatalf("failed to create QP: %v", err)
	}

	if qp.State != QPStateReset {
		t.Fatalf("expected state RESET, got %s", qp.State)
	}

	// RESET -> INIT
	if err := qp.Modify(QPAttr{State: QPStateInit}); err != nil {
		t.Fatalf("transition to INIT failed: %v", err)
	}
	if qp.State != QPStateInit {
		t.Errorf("expected state INIT, got %s", qp.State)
	}

	// INIT -> RTR (requires DestQPN)
	if err := qp.Modify(QPAttr{State: QPStateRTR, DestQPN: 2002, PathMTU: 4096}); err != nil {
		t.Fatalf("transition to RTR failed: %v", err)
	}
	if qp.State != QPStateRTR {
		t.Errorf("expected state RTR, got %s", qp.State)
	}

	// RTR -> RTS
	if err := qp.Modify(QPAttr{State: QPStateRTS, SQPSN: 100}); err != nil {
		t.Fatalf("transition to RTS failed: %v", err)
	}
	if qp.State != QPStateRTS {
		t.Errorf("expected state RTS, got %s", qp.State)
	}

	// RTS -> ERROR (flush)
	if err := qp.Modify(QPAttr{State: QPStateError}); err != nil {
		t.Fatalf("transition to ERROR failed: %v", err)
	}
	if qp.State != QPStateError {
		t.Errorf("expected state ERROR, got %s", qp.State)
	}

	// ERROR -> RESET
	if err := qp.Modify(QPAttr{State: QPStateReset}); err != nil {
		t.Fatalf("transition to RESET failed: %v", err)
	}
	if qp.State != QPStateReset {
		t.Errorf("expected state RESET, got %s", qp.State)
	}
}

func TestAMDGPUDirectRDMA_PostSendAndProcessQueue_Write(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})

	// Register remote node with DMA-BUF RDMA region
	_ = hal.RegisterNode(AMDDeviceNode{
		NodeID:         1,
		TotalVRAMBytes: 16 * 1024 * 1024 * 1024,
		BAR1SizeBytes:  16 * 1024 * 1024 * 1024,
		IsLargeBAR:     true,
		DMABUFCapable:  true,
	})

	dmabuf, err := hal.ExportVRAMToDMABUF(1, 0x50000000, 64*1024*1024)
	if err != nil {
		t.Fatalf("ExportVRAMToDMABUF failed: %v", err)
	}

	mr, err := hal.RegisterDMABUFForRDMA(dmabuf.FD, 64*1024*1024)
	if err != nil {
		t.Fatalf("RegisterDMABUFForRDMA failed: %v", err)
	}

	sendCQ := NewCompletionQueue(1, 64)
	recvCQ := NewCompletionQueue(2, 64)

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
	_ = qp.Modify(QPAttr{State: QPStateRTR, DestQPN: 2001})
	_ = qp.Modify(QPAttr{State: QPStateRTS})

	// Post RDMA Write targeting the remote DMA-BUF memory region
	wr := &WorkRequest{
		WRID:   101,
		OpCode: RDMAOpWrite,
		SGEs: []ScatterGatherElement{
			{
				Address: 0x10000000,  // local VRAM
				Length:  1024 * 1024, // 1 MiB
				LKey:    0x1001,
			},
		},
		RemoteAddr: mr.IOVA,
		RKey:       mr.RKey,
	}

	if err := qp.PostSend(wr); err != nil {
		t.Fatalf("PostSend failed: %v", err)
	}

	processed, err := qp.ProcessSendQueue(hal)
	if err != nil {
		t.Fatalf("ProcessSendQueue failed: %v", err)
	}
	if processed != 1 {
		t.Fatalf("expected 1 processed WR, got %d", processed)
	}

	// Poll SendCQ
	wcs := sendCQ.PollCQ(10)
	if len(wcs) != 1 {
		t.Fatalf("expected 1 completion, got %d", len(wcs))
	}
	wc := wcs[0]
	if wc.Status != WCSuccess {
		t.Errorf("completion status = %s, want WCSuccess", wc.Status)
	}
	if wc.ByteLen != 1024*1024 {
		t.Errorf("completion bytes = %d, want %d", wc.ByteLen, 1024*1024)
	}
	if wc.StagingCopyCount() != 0 {
		t.Errorf("expected 0 staging copies, got %d", wc.StagingCopyCount())
	}

	stats := qp.Stats()
	if stats.TotalSent != 1 || stats.BytesSent != 1024*1024 {
		t.Errorf("unexpected QP stats: %+v", stats)
	}
}

func TestAMDGPUDirectRDMA_PostSendAndProcessQueue_SendRecv(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})

	cq0Send := NewCompletionQueue(1, 64)
	cq0Recv := NewCompletionQueue(2, 64)
	qp0, _ := hal.CreateQueuePair(QPInitAttr{
		QPType: QPTypeRC,
		SendCQ: cq0Send,
		RecvCQ: cq0Recv,
		NodeID: 0,
	})

	cq1Send := NewCompletionQueue(3, 64)
	cq1Recv := NewCompletionQueue(4, 64)
	qp1, _ := hal.CreateQueuePair(QPInitAttr{
		QPType: QPTypeRC,
		SendCQ: cq1Send,
		RecvCQ: cq1Recv,
		NodeID: 1,
	})

	// Connect QP0 -> QP1
	_ = qp0.Modify(QPAttr{State: QPStateInit})
	_ = qp0.Modify(QPAttr{State: QPStateRTR, DestQPN: qp1.QPNum})
	_ = qp0.Modify(QPAttr{State: QPStateRTS})

	// Connect QP1 -> QP0
	_ = qp1.Modify(QPAttr{State: QPStateInit})
	_ = qp1.Modify(QPAttr{State: QPStateRTR, DestQPN: qp0.QPNum})
	_ = qp1.Modify(QPAttr{State: QPStateRTS})

	// Post receive on QP1
	recvWR := &WorkRequest{
		WRID: 201,
		SGEs: []ScatterGatherElement{
			{
				Address: 0x60000000,
				Length:  2048,
				LKey:    0x2001,
			},
		},
	}
	if err := qp1.PostRecv(recvWR); err != nil {
		t.Fatalf("PostRecv failed: %v", err)
	}

	// Post send on QP0 with immediate data
	sendWR := &WorkRequest{
		WRID:    102,
		OpCode:  RDMAOpSendWithImm,
		ImmData: 0xCAFEBABE,
		SGEs: []ScatterGatherElement{
			{
				Address: 0x10000000,
				Length:  1024,
				LKey:    0x1001,
			},
		},
	}
	if err := qp0.PostSend(sendWR); err != nil {
		t.Fatalf("PostSend failed: %v", err)
	}

	// Process send queue
	processed, err := qp0.ProcessSendQueue(hal)
	if err != nil {
		t.Fatalf("ProcessSendQueue failed: %v", err)
	}
	if processed != 1 {
		t.Fatalf("expected 1 processed WR, got %d", processed)
	}

	// Sender CQ check
	sendCompletions := cq0Send.PollCQ(10)
	if len(sendCompletions) != 1 || sendCompletions[0].Status != WCSuccess {
		t.Fatalf("expected success send completion, got: %+v", sendCompletions)
	}

	// Receiver CQ check
	recvCompletions := cq1Recv.PollCQ(10)
	if len(recvCompletions) != 1 || recvCompletions[0].Status != WCSuccess {
		t.Fatalf("expected success recv completion, got: %+v", recvCompletions)
	}
	if recvCompletions[0].ImmData != 0xCAFEBABE {
		t.Errorf("expected ImmData 0xCAFEBABE, got 0x%x", recvCompletions[0].ImmData)
	}
	if recvCompletions[0].ByteLen != 1024 {
		t.Errorf("expected 1024 bytes, got %d", recvCompletions[0].ByteLen)
	}
	if recvCompletions[0].StagingCopyCount() != 0 {
		t.Errorf("expected 0 staging copies, got %d", recvCompletions[0].StagingCopyCount())
	}
}

func TestAMDGPUDirectRDMA_ErrorHandling_InvalidRKey(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})

	sendCQ := NewCompletionQueue(1, 64)
	recvCQ := NewCompletionQueue(2, 64)
	qp, _ := hal.CreateQueuePair(QPInitAttr{
		QPType: QPTypeRC,
		SendCQ: sendCQ,
		RecvCQ: recvCQ,
		NodeID: 0,
	})
	_ = qp.Modify(QPAttr{State: QPStateInit})
	_ = qp.Modify(QPAttr{State: QPStateRTR, DestQPN: 9999})
	_ = qp.Modify(QPAttr{State: QPStateRTS})

	// RDMA Write with non-existent RKey
	wr := &WorkRequest{
		WRID:   103,
		OpCode: RDMAOpWrite,
		SGEs: []ScatterGatherElement{
			{
				Address: 0x10000000,
				Length:  4096,
				LKey:    0x1001,
			},
		},
		RemoteAddr: 0x50000000,
		RKey:       0xDEADBEEF, // Invalid!
	}
	_ = qp.PostSend(wr)

	_, _ = qp.ProcessSendQueue(hal)

	wcs := sendCQ.PollCQ(10)
	if len(wcs) != 1 {
		t.Fatalf("expected 1 completion, got %d", len(wcs))
	}
	if wcs[0].Status != WCRemAccErr {
		t.Errorf("expected status WCRemAccErr, got %s", wcs[0].Status)
	}
}

func TestAMDGPUDirectRDMA_CompletionQueueNotification(t *testing.T) {
	cq := NewCompletionQueue(1, 10)
	ch := cq.NotifyChannel()

	select {
	case <-ch:
		t.Fatalf("unexpected notification on empty CQ")
	default:
	}

	cq.Enqueue(WorkCompletion{
		WRID:   1,
		Status: WCSuccess,
	})

	select {
	case <-ch:
		// Expected notification
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("timed out waiting for CQ notification")
	}

	drained := cq.PollCQ(10)
	if len(drained) != 1 {
		t.Errorf("expected 1 drained entry, got %d", len(drained))
	}
}
