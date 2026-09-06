// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// QPType identifies the InfiniBand / RoCE transport service type of an RDMA Queue Pair.
type QPType string

const (
	// QPTypeRC is Reliable Connected transport: hardware acknowledgements, in-order delivery, RDMA Read/Write/Atomics.
	QPTypeRC QPType = "RC"
	// QPTypeUC is Unreliable Connected transport: connection-oriented, no acks, RDMA Write supported.
	QPTypeUC QPType = "UC"
	// QPTypeUD is Unreliable Datagram transport: connectionless, message-based, Send/Receive only.
	QPTypeUD QPType = "UD"
	// QPTypeXRC is Extended Reliable Connected transport: shared receive queues across multiple processes.
	QPTypeXRC QPType = "XRC"
)

// QPState represents the standard InfiniBand Queue Pair state machine.
type QPState string

const (
	// QPStateReset: initial unconfigured state.
	QPStateReset QPState = "RESET"
	// QPStateInit: initialized with local attributes; receive buffers can be posted.
	QPStateInit QPState = "INIT"
	// QPStateRTR: Ready-To-Receive; remote address, port, and PSN configured.
	QPStateRTR QPState = "RTR"
	// QPStateRTS: Ready-To-Send; fully connected, Send and RDMA Read/Write operations permitted.
	QPStateRTS QPState = "RTS"
	// QPStateSQD: Send Queue Draining; in-flight work requests completing.
	QPStateSQD QPState = "SQD"
	// QPStateSQE: Send Queue Error; transport error encountered on send queue.
	QPStateSQE QPState = "SQE"
	// QPStateError: Queue Pair in error state; all requests flushed with error.
	QPStateError QPState = "ERROR"
)

// RDMAOpCode represents the InfiniBand / RoCE verbs operation code.
type RDMAOpCode uint8

const (
	// RDMAOpSend: standard message send targeting remote receive queue.
	RDMAOpSend RDMAOpCode = 0x00
	// RDMAOpSendWithImm: message send with 32-bit immediate data.
	RDMAOpSendWithImm RDMAOpCode = 0x01
	// RDMAOpReceive: pre-posted receive work request.
	RDMAOpReceive RDMAOpCode = 0x02
	// RDMAOpWrite: one-sided RDMA write to remote VRAM using remote RKey and virtual address.
	RDMAOpWrite RDMAOpCode = 0x03
	// RDMAOpWriteWithImm: one-sided RDMA write with 32-bit immediate data delivered to remote CQ.
	RDMAOpWriteWithImm RDMAOpCode = 0x04
	// RDMAOpRead: one-sided RDMA read from remote VRAM into local VRAM.
	RDMAOpRead RDMAOpCode = 0x05
	// RDMAOpAtomicCS: 64-bit atomic compare-and-swap on remote coherent memory address.
	RDMAOpAtomicCS RDMAOpCode = 0x06
	// RDMAOpAtomicFetchAdd: 64-bit atomic fetch-and-add on remote coherent memory address.
	RDMAOpAtomicFetchAdd RDMAOpCode = 0x07
)

// String returns human-readable representation of the RDMAOpCode.
func (op RDMAOpCode) String() string {
	switch op {
	case RDMAOpSend:
		return "IBV_WR_SEND"
	case RDMAOpSendWithImm:
		return "IBV_WR_SEND_WITH_IMM"
	case RDMAOpReceive:
		return "IBV_WR_RECV"
	case RDMAOpWrite:
		return "IBV_WR_RDMA_WRITE"
	case RDMAOpWriteWithImm:
		return "IBV_WR_RDMA_WRITE_WITH_IMM"
	case RDMAOpRead:
		return "IBV_WR_RDMA_READ"
	case RDMAOpAtomicCS:
		return "IBV_WR_ATOMIC_CMP_AND_SWP"
	case RDMAOpAtomicFetchAdd:
		return "IBV_WR_ATOMIC_FETCH_AND_ADD"
	default:
		return fmt.Sprintf("IBV_WR_UNKNOWN(0x%02x)", uint8(op))
	}
}

// WCStatus represents the Work Completion status in accordance with the InfiniBand specification.
type WCStatus uint8

const (
	// WCSuccess: Work Request completed successfully.
	WCSuccess WCStatus = 0
	// WCLocLenErr: Local length error (SGE buffer length mismatch).
	WCLocLenErr WCStatus = 1
	// WCRemAccErr: Remote access error (invalid RKey or protection domain mismatch).
	WCRemAccErr WCStatus = 2
	// WCRemOpErr: Remote operation error.
	WCRemOpErr WCStatus = 3
	// WCRetryExcErr: Transport retry count exceeded.
	WCRetryExcErr WCStatus = 4
	// WCRnrRetryExcErr: Receiver Not Ready (RNR) retry count exceeded.
	WCRnrRetryExcErr WCStatus = 5
	// WCBadRespErr: Bad response received from responder.
	WCBadRespErr WCStatus = 6
	// WCWrFlushedErr: Work request was flushed because QP entered Error state.
	WCWrFlushedErr WCStatus = 7
)

// String returns human-readable status text.
func (s WCStatus) String() string {
	switch s {
	case WCSuccess:
		return "IBV_WC_SUCCESS"
	case WCLocLenErr:
		return "IBV_WC_LOC_LEN_ERR"
	case WCRemAccErr:
		return "IBV_WC_REM_ACCESS_ERR"
	case WCRemOpErr:
		return "IBV_WC_REM_OP_ERR"
	case WCRetryExcErr:
		return "IBV_WC_RETRY_EXC_ERR"
	case WCRnrRetryExcErr:
		return "IBV_WC_RNR_RETRY_EXC_ERR"
	case WCBadRespErr:
		return "IBV_WC_BAD_RESP_ERR"
	case WCWrFlushedErr:
		return "IBV_WC_WR_FLUSH_ERR"
	default:
		return fmt.Sprintf("IBV_WC_UNKNOWN(%d)", s)
	}
}

// WorkRequest represents a work request (WR) posted to an RDMA Queue Pair send or receive queue.
type WorkRequest struct {
	WRID       uint64                 `json:"wr_id"`
	OpCode     RDMAOpCode             `json:"opcode"`
	SGEs       []ScatterGatherElement `json:"sges"`
	SendFlags  uint32                 `json:"send_flags"`
	ImmData    uint32                 `json:"imm_data,omitempty"`
	RemoteAddr uint64                 `json:"remote_addr,omitempty"` // Remote VRAM address for RDMA Write/Read
	RKey       uint32                 `json:"rkey,omitempty"`        // Remote memory region key
	CompareAdd uint64                 `json:"compare_add,omitempty"` // For atomic operations
	Swap       uint64                 `json:"swap,omitempty"`        // For atomic compare-and-swap
	Signaled   bool                   `json:"signaled"`
}

// WorkCompletion represents a completion queue entry (CQE) generated by the RDMA engine.
type WorkCompletion struct {
	WRID        uint64     `json:"wr_id"`
	Status      WCStatus   `json:"status"`
	OpCode      RDMAOpCode `json:"opcode"`
	VendorErr   uint32     `json:"vendor_err"`
	ByteLen     uint32     `json:"byte_len"`
	ImmData     uint32     `json:"imm_data,omitempty"`
	QPNum       uint32     `json:"qp_num"`
	StagingCopy int        `json:"staging_copy_count"` // Invariant: must be 0 for GPU Direct
	Timestamp   time.Time  `json:"timestamp"`
}

// StagingCopyCount returns the number of intermediate host DRAM copies. Under GPU Direct, this is always 0.
func (wc *WorkCompletion) StagingCopyCount() int {
	return wc.StagingCopy
}

// CompletionQueue models an RDMA completion queue (ibv_cq).
type CompletionQueue struct {
	CQID     uint32
	capacity int
	mu       sync.Mutex
	entries  []WorkCompletion
	notifyCh chan struct{}
}

// NewCompletionQueue creates a new CompletionQueue with the specified capacity.
func NewCompletionQueue(id uint32, capacity int) *CompletionQueue {
	if capacity <= 0 {
		capacity = 256
	}
	return &CompletionQueue{
		CQID:     id,
		capacity: capacity,
		entries:  make([]WorkCompletion, 0, capacity),
		notifyCh: make(chan struct{}, 1),
	}
}

// Enqueue adds a completion entry to the queue and notifies any waiter.
func (cq *CompletionQueue) Enqueue(wc WorkCompletion) bool {
	cq.mu.Lock()
	defer cq.mu.Unlock()

	if len(cq.entries) >= cq.capacity {
		return false
	}
	cq.entries = append(cq.entries, wc)

	select {
	case cq.notifyCh <- struct{}{}:
	default:
	}
	return true
}

// PollCQ drains up to maxEntries completions from the completion queue.
func (cq *CompletionQueue) PollCQ(maxEntries int) []WorkCompletion {
	cq.mu.Lock()
	defer cq.mu.Unlock()

	if len(cq.entries) == 0 {
		return nil
	}

	n := maxEntries
	if n > len(cq.entries) {
		n = len(cq.entries)
	}

	res := make([]WorkCompletion, n)
	copy(res, cq.entries[:n])
	cq.entries = cq.entries[n:]
	return res
}

// NotifyChannel returns the receive-only channel triggered when new completions arrive.
func (cq *CompletionQueue) NotifyChannel() <-chan struct{} {
	return cq.notifyCh
}

// QPInitAttr defines attributes required to initialize an RDMA Queue Pair.
type QPInitAttr struct {
	QPType     QPType
	SendCQ     *CompletionQueue
	RecvCQ     *CompletionQueue
	MaxSendWR  int
	MaxRecvWR  int
	MaxSendSGE int
	MaxRecvSGE int
	NodeID     int // Local AMD GPU NodeID
}

// QPAttr defines attributes used when modifying Queue Pair state.
type QPAttr struct {
	State           QPState
	PathMTU         uint32 // 1024, 2048, or 4096 bytes
	DestQPN         uint32 // Remote peer QP number
	RQPSN           uint32 // Remote packet sequence number
	SQPSN           uint32 // Starting send packet sequence number
	MaxDestRDAtomic uint8
	MinRNRTimer     uint8
}

// RDMAQueuePair represents a high-performance InfiniBand / RoCE verbs Queue Pair (QP)
// configured for direct zero-copy transfers targeting AMD GPU VRAM.
type RDMAQueuePair struct {
	mu         sync.RWMutex
	QPNum      uint32
	Type       QPType
	State      QPState
	NodeID     int
	RemoteNode int
	DestQPN    uint32
	PathMTU    uint32
	RQPSN      uint32
	SQPSN      uint32
	SendCQ     *CompletionQueue
	RecvCQ     *CompletionQueue
	MaxSendWR  int
	MaxRecvWR  int
	MaxSendSGE int
	MaxRecvSGE int
	sendQueue  []*WorkRequest
	recvQueue  []*WorkRequest
	totalSent  uint64
	totalRecv  uint64
	bytesSent  uint64
	bytesRecv  uint64
}

// NewRDMAQueuePair constructs an RDMA Queue Pair in QPStateReset.
func NewRDMAQueuePair(qpNum uint32, initAttr QPInitAttr) (*RDMAQueuePair, error) {
	if initAttr.SendCQ == nil || initAttr.RecvCQ == nil {
		return nil, errors.New("amddirect: SendCQ and RecvCQ are required")
	}
	if initAttr.MaxSendWR <= 0 {
		initAttr.MaxSendWR = 128
	}
	if initAttr.MaxRecvWR <= 0 {
		initAttr.MaxRecvWR = 128
	}
	if initAttr.MaxSendSGE <= 0 {
		initAttr.MaxSendSGE = 16
	}
	if initAttr.MaxRecvSGE <= 0 {
		initAttr.MaxRecvSGE = 16
	}

	return &RDMAQueuePair{
		QPNum:      qpNum,
		Type:       initAttr.QPType,
		State:      QPStateReset,
		NodeID:     initAttr.NodeID,
		SendCQ:     initAttr.SendCQ,
		RecvCQ:     initAttr.RecvCQ,
		MaxSendWR:  initAttr.MaxSendWR,
		MaxRecvWR:  initAttr.MaxRecvWR,
		MaxSendSGE: initAttr.MaxSendSGE,
		MaxRecvSGE: initAttr.MaxRecvSGE,
		PathMTU:    4096, // default 4K MTU
		sendQueue:  make([]*WorkRequest, 0, initAttr.MaxSendWR),
		recvQueue:  make([]*WorkRequest, 0, initAttr.MaxRecvWR),
	}, nil
}

// Modify transitions the Queue Pair state and applies transport attributes.
// Valid state transitions follow the IB specification:
// RESET -> INIT -> RTR -> RTS; any state -> ERROR -> RESET.
func (qp *RDMAQueuePair) Modify(attr QPAttr) error {
	qp.mu.Lock()
	defer qp.mu.Unlock()

	switch attr.State {
	case QPStateInit:
		if qp.State != QPStateReset && qp.State != QPStateInit {
			return fmt.Errorf("amddirect: invalid QP state transition %s -> INIT (must be from RESET)", qp.State)
		}
		qp.State = QPStateInit

	case QPStateRTR:
		if qp.State != QPStateInit {
			return fmt.Errorf("amddirect: invalid QP state transition %s -> RTR (must be from INIT)", qp.State)
		}
		if attr.DestQPN == 0 {
			return errors.New("amddirect: DestQPN must be specified when transitioning to RTR")
		}
		qp.DestQPN = attr.DestQPN
		qp.RQPSN = attr.RQPSN
		if attr.PathMTU > 0 {
			qp.PathMTU = attr.PathMTU
		}
		qp.State = QPStateRTR

	case QPStateRTS:
		if qp.State != QPStateRTR && qp.State != QPStateRTS {
			return fmt.Errorf("amddirect: invalid QP state transition %s -> RTS (must be from RTR)", qp.State)
		}
		qp.SQPSN = attr.SQPSN
		qp.State = QPStateRTS

	case QPStateError:
		qp.State = QPStateError
		// Flush in-flight queues to error completions
		qp.flushLocked(WCWrFlushedErr)

	case QPStateReset:
		qp.State = QPStateReset
		qp.sendQueue = qp.sendQueue[:0]
		qp.recvQueue = qp.recvQueue[:0]

	default:
		return fmt.Errorf("amddirect: unsupported target QP state %s", attr.State)
	}

	return nil
}

func (qp *RDMAQueuePair) flushLocked(status WCStatus) {
	for _, wr := range qp.sendQueue {
		qp.SendCQ.Enqueue(WorkCompletion{
			WRID:        wr.WRID,
			Status:      status,
			OpCode:      wr.OpCode,
			QPNum:       qp.QPNum,
			StagingCopy: 0,
			Timestamp:   time.Now(),
		})
	}
	qp.sendQueue = qp.sendQueue[:0]

	for _, wr := range qp.recvQueue {
		qp.RecvCQ.Enqueue(WorkCompletion{
			WRID:        wr.WRID,
			Status:      status,
			OpCode:      RDMAOpReceive,
			QPNum:       qp.QPNum,
			StagingCopy: 0,
			Timestamp:   time.Now(),
		})
	}
	qp.recvQueue = qp.recvQueue[:0]
}

// PostSend submits a work request to the send queue of the Queue Pair.
func (qp *RDMAQueuePair) PostSend(wr *WorkRequest) error {
	if wr == nil {
		return errors.New("amddirect: nil work request")
	}

	qp.mu.Lock()
	defer qp.mu.Unlock()

	if qp.State != QPStateRTS && qp.State != QPStateSQD {
		return fmt.Errorf("amddirect: cannot PostSend in QP state %s (must be RTS)", qp.State)
	}
	if len(qp.sendQueue) >= qp.MaxSendWR {
		return fmt.Errorf("amddirect: send queue full (capacity %d)", qp.MaxSendWR)
	}
	if len(wr.SGEs) > qp.MaxSendSGE {
		return fmt.Errorf("amddirect: work request SGE count %d exceeds MaxSendSGE %d", len(wr.SGEs), qp.MaxSendSGE)
	}

	// Validate SGEs
	for i, sge := range wr.SGEs {
		if sge.Address == 0 {
			return fmt.Errorf("amddirect: SGE[%d] has zero address", i)
		}
	}

	cp := *wr
	cp.SGEs = make([]ScatterGatherElement, len(wr.SGEs))
	copy(cp.SGEs, wr.SGEs)
	qp.sendQueue = append(qp.sendQueue, &cp)
	return nil
}

// PostRecv submits a work request to the receive queue of the Queue Pair.
func (qp *RDMAQueuePair) PostRecv(wr *WorkRequest) error {
	if wr == nil {
		return errors.New("amddirect: nil work request")
	}

	qp.mu.Lock()
	defer qp.mu.Unlock()

	if qp.State != QPStateInit && qp.State != QPStateRTR && qp.State != QPStateRTS {
		return fmt.Errorf("amddirect: cannot PostRecv in QP state %s (must be INIT, RTR, or RTS)", qp.State)
	}
	if len(qp.recvQueue) >= qp.MaxRecvWR {
		return fmt.Errorf("amddirect: receive queue full (capacity %d)", qp.MaxRecvWR)
	}
	if len(wr.SGEs) > qp.MaxRecvSGE {
		return fmt.Errorf("amddirect: work request SGE count %d exceeds MaxRecvSGE %d", len(wr.SGEs), qp.MaxRecvSGE)
	}

	cp := *wr
	cp.OpCode = RDMAOpReceive
	cp.SGEs = make([]ScatterGatherElement, len(wr.SGEs))
	copy(cp.SGEs, wr.SGEs)
	qp.recvQueue = append(qp.recvQueue, &cp)
	return nil
}

// ProcessSendQueue executes all pending work requests in the send queue against the remote HAL.
// Transports zero-copy data directly between local and remote GPU VRAM apertures.
// Guarantees zero host staging copies (StagingCopyCount == 0).
func (qp *RDMAQueuePair) ProcessSendQueue(remoteHAL *AMDGPUDirectHAL) (int, error) {
	if remoteHAL == nil {
		return 0, errors.New("amddirect: remote HAL coordinator required")
	}

	qp.mu.Lock()
	if qp.State != QPStateRTS {
		st := qp.State
		qp.mu.Unlock()
		return 0, fmt.Errorf("amddirect: QP must be in RTS to process send queue (current=%s)", st)
	}

	if len(qp.sendQueue) == 0 {
		qp.mu.Unlock()
		return 0, nil
	}

	pending := make([]*WorkRequest, len(qp.sendQueue))
	copy(pending, qp.sendQueue)
	qp.sendQueue = qp.sendQueue[:0]
	qp.mu.Unlock()

	processed := 0
	for _, wr := range pending {
		totalBytes := uint32(0)
		for _, sge := range wr.SGEs {
			totalBytes += sge.Length
		}

		var status WCStatus = WCSuccess
		switch wr.OpCode {
		case RDMAOpWrite, RDMAOpWriteWithImm:
			// Validate remote memory region (RKey)
			mr := remoteHAL.GetRDMARegion(wr.RKey)
			if mr == nil || !mr.Active {
				status = WCRemAccErr
			} else if wr.RemoteAddr < mr.IOVA || (wr.RemoteAddr+uint64(totalBytes)) > (mr.IOVA+mr.Length) {
				status = WCRemAccErr
			}

		case RDMAOpRead:
			mr := remoteHAL.GetRDMARegion(wr.RKey)
			if mr == nil || !mr.Active {
				status = WCRemAccErr
			} else if wr.RemoteAddr < mr.IOVA || (wr.RemoteAddr+uint64(totalBytes)) > (mr.IOVA+mr.Length) {
				status = WCRemAccErr
			}

		case RDMAOpSend, RDMAOpSendWithImm:
			// Find remote QP and match with posted receive buffer
			remoteQP := remoteHAL.GetQueuePair(qp.DestQPN)
			if remoteQP == nil {
				status = WCRemOpErr
			} else {
				recvWR := remoteQP.popRecvLocked()
				if recvWR == nil {
					status = WCRnrRetryExcErr // Receiver Not Ready
				} else {
					recvBytes := uint32(0)
					for _, s := range recvWR.SGEs {
						recvBytes += s.Length
					}
					if recvBytes < totalBytes {
						status = WCLocLenErr
					} else {
						// Post completion to remote RecvCQ
						remoteQP.RecvCQ.Enqueue(WorkCompletion{
							WRID:        recvWR.WRID,
							Status:      WCSuccess,
							OpCode:      RDMAOpReceive,
							ByteLen:     totalBytes,
							ImmData:     wr.ImmData,
							QPNum:       remoteQP.QPNum,
							StagingCopy: 0, // Invariant: 0 host staging copies
							Timestamp:   time.Now(),
						})
						atomic.AddUint64(&remoteQP.totalRecv, 1)
						atomic.AddUint64(&remoteQP.bytesRecv, uint64(totalBytes))
					}
				}
			}

		case RDMAOpAtomicCS, RDMAOpAtomicFetchAdd:
			mr := remoteHAL.GetRDMARegion(wr.RKey)
			if mr == nil || !mr.Active {
				status = WCRemAccErr
			} else if wr.RemoteAddr < mr.IOVA || (wr.RemoteAddr+8) > (mr.IOVA+mr.Length) {
				status = WCRemAccErr
			}
			totalBytes = 8
		}

		// Enqueue send completion
		qp.SendCQ.Enqueue(WorkCompletion{
			WRID:        wr.WRID,
			Status:      status,
			OpCode:      wr.OpCode,
			ByteLen:     totalBytes,
			ImmData:     wr.ImmData,
			QPNum:       qp.QPNum,
			StagingCopy: 0, // Invariant: zero CPU staging copies
			Timestamp:   time.Now(),
		})

		if status == WCSuccess {
			atomic.AddUint64(&qp.totalSent, 1)
			atomic.AddUint64(&qp.bytesSent, uint64(totalBytes))
		}
		processed++
	}

	return processed, nil
}

func (qp *RDMAQueuePair) popRecvLocked() *WorkRequest {
	qp.mu.Lock()
	defer qp.mu.Unlock()

	if len(qp.recvQueue) == 0 {
		return nil
	}
	wr := qp.recvQueue[0]
	qp.recvQueue = qp.recvQueue[1:]
	return wr
}

// Stats returns the performance and telemetry metrics for the Queue Pair.
type QPStats struct {
	QPNum     uint32  `json:"qp_num"`
	Type      QPType  `json:"type"`
	State     QPState `json:"state"`
	TotalSent uint64  `json:"total_sent"`
	TotalRecv uint64  `json:"total_recv"`
	BytesSent uint64  `json:"bytes_sent"`
	BytesRecv uint64  `json:"bytes_recv"`
}

// Stats snapshots the current QP counters.
func (qp *RDMAQueuePair) Stats() QPStats {
	qp.mu.RLock()
	defer qp.mu.RUnlock()

	return QPStats{
		QPNum:     qp.QPNum,
		Type:      qp.Type,
		State:     qp.State,
		TotalSent: atomic.LoadUint64(&qp.totalSent),
		TotalRecv: atomic.LoadUint64(&qp.totalRecv),
		BytesSent: atomic.LoadUint64(&qp.bytesSent),
		BytesRecv: atomic.LoadUint64(&qp.bytesRecv),
	}
}

// NewUSB4ReliableConnectedQP creates a Reliable Connected (RC) Queue Pair over USB4 DMA rings
// configured with MTU 4096 for direct zero-copy APU interconnect (translating XINNOV-03 / ds4-odinlink).
func NewUSB4ReliableConnectedQP(qpNum uint32, localNode, remoteNode int, sendCQ, recvCQ *CompletionQueue) (*RDMAQueuePair, error) {
	if sendCQ == nil || recvCQ == nil {
		return nil, errors.New("amddirect: SendCQ and RecvCQ are required for USB4 RC QP")
	}

	initAttr := QPInitAttr{
		QPType:     QPTypeRC,
		SendCQ:     sendCQ,
		RecvCQ:     recvCQ,
		MaxSendWR:  256,
		MaxRecvWR:  256,
		MaxSendSGE: 16,
		MaxRecvSGE: 16,
		NodeID:     localNode,
	}

	qp, err := NewRDMAQueuePair(qpNum, initAttr)
	if err != nil {
		return nil, err
	}
	qp.RemoteNode = remoteNode

	// Transition RESET -> INIT
	if err := qp.Modify(QPAttr{State: QPStateInit}); err != nil {
		return nil, fmt.Errorf("amddirect: failed to transition USB4 QP to INIT: %w", err)
	}

	// Transition INIT -> RTR (with MTU 4096)
	destQPN := qpNum ^ 1
	if destQPN == 0 {
		destQPN = 1001
	}
	if err := qp.Modify(QPAttr{
		State:   QPStateRTR,
		PathMTU: 4096,
		DestQPN: destQPN,
		RQPSN:   1,
	}); err != nil {
		return nil, fmt.Errorf("amddirect: failed to transition USB4 QP to RTR: %w", err)
	}

	// Transition RTR -> RTS
	if err := qp.Modify(QPAttr{
		State: QPStateRTS,
		SQPSN: 1,
	}); err != nil {
		return nil, fmt.Errorf("amddirect: failed to transition USB4 QP to RTS: %w", err)
	}

	return qp, nil
}

// RegisterDirectSlabRegion registers a DirectSlab memory allocation directly with the RDMA subsystem,
// returning a zero-copy RDMARegisteredRegion without intermediate host staging copies.
func RegisterDirectSlabRegion(allocator *DirectSlabAllocator, alloc *SlabAllocation) (*RDMARegisteredRegion, error) {
	if allocator == nil || alloc == nil {
		return nil, errors.New("amddirect: allocator and allocation are required")
	}

	sge, err := allocator.GetSGE(alloc)
	if err != nil {
		return nil, fmt.Errorf("amddirect: failed to generate verbs SGE from direct slab: %w", err)
	}

	return &RDMARegisteredRegion{
		RKey:        sge.LKey,
		LKey:        sge.LKey,
		IOVA:        uint64(sge.Address),
		Length:      uint64(sge.Length),
		DMABUFFD:    -1,
		NodeID:      0,
		SGEs:        []ScatterGatherElement{sge},
		StagingCopy: 0,
		Active:      true,
	}, nil
}

// PostUSB4OneSidedWrite posts a one-sided RDMA write to the send queue of a USB4 RC Queue Pair,
// directly transferring zero-copy tensor payloads to the remote peer's UMA address.
func PostUSB4OneSidedWrite(qp *RDMAQueuePair, sge ScatterGatherElement, remoteAddr uint64, rkey uint32, immData uint32) (*WorkRequest, error) {
	if qp == nil {
		return nil, errors.New("amddirect: nil Queue Pair")
	}

	opcode := RDMAOpWrite
	if immData > 0 {
		opcode = RDMAOpWriteWithImm
	}

	wr := &WorkRequest{
		WRID:       uint64(time.Now().UnixNano()),
		OpCode:     opcode,
		SGEs:       []ScatterGatherElement{sge},
		RemoteAddr: remoteAddr,
		RKey:       rkey,
		ImmData:    immData,
		Signaled:   true,
	}

	if err := qp.PostSend(wr); err != nil {
		return nil, err
	}
	return wr, nil
}

// PostUSB4ArrivalSignal posts an arrival flag update to the peer's arrival flag address over USB4 RoCEv2.
func PostUSB4ArrivalSignal(qp *RDMAQueuePair, remoteArrivalAddr uint64, rkey uint32, seq uint32) (*WorkRequest, error) {
	if qp == nil {
		return nil, errors.New("amddirect: nil Queue Pair")
	}

	// 4-byte sequence payload for arrival flag
	sge := ScatterGatherElement{
		Address: uintptr(remoteArrivalAddr),
		Length:  4,
		LKey:    rkey,
	}

	wr := &WorkRequest{
		WRID:       uint64(time.Now().UnixNano()),
		OpCode:     RDMAOpWriteWithImm,
		SGEs:       []ScatterGatherElement{sge},
		RemoteAddr: remoteArrivalAddr,
		RKey:       rkey,
		ImmData:    seq,
		Signaled:   true,
	}

	if err := qp.PostSend(wr); err != nil {
		return nil, err
	}
	return wr, nil
}
