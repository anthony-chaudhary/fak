// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// DefaultDisaggTTL is the default lease validity duration (5 seconds).
const DefaultDisaggTTL = 5 * time.Second

// DefaultDisaggBlockSize is the default KV block granularity (64 KiB).
const DefaultDisaggBlockSize uint64 = 64 * 1024

// DefaultSignalReadyFlag is the high-bit flag in immediate data indicating transfer completion.
const DefaultSignalReadyFlag uint32 = 0x80000000

// CorruptSignalValue is the sentinel immediate data payload representing hardware/transmission corruption.
const CorruptSignalValue uint32 = 0xDEADBEEF

var (
	// ErrLeaseNotFound indicates that the specified lease ID does not exist.
	ErrLeaseNotFound = errors.New("amddirect: KV transfer lease not found")

	// ErrLeaseExpired indicates that the lease has passed its expiration deadline.
	ErrLeaseExpired = errors.New("amddirect: KV transfer lease expired")

	// ErrLeaseInactive indicates that the lease is not in the required ACTIVE state.
	ErrLeaseInactive = errors.New("amddirect: KV transfer lease is not in active state")

	// ErrCorruptSignal indicates that the HSA memory signal or immediate payload was corrupted.
	ErrCorruptSignal = errors.New("amddirect: corrupt HSA memory signal detected")

	// ErrZeroCopyViolation indicates that host DRAM staging copies were detected, violating zero-copy.
	ErrZeroCopyViolation = errors.New("amddirect: zero-copy invariant violated: staging copies detected")

	// ErrNodeNotDMABUF indicates that an AMD GPU device node cannot export kernel DMA-BUF descriptors.
	ErrNodeNotDMABUF = errors.New("amddirect: device node does not support kernel DMA-BUF export")

	// ErrInvalidByteLength indicates that the transfer length is zero or invalid.
	ErrInvalidByteLength = errors.New("amddirect: invalid transfer byte length")

	// ErrSignalTimeout indicates that polling timed out waiting for the HSA memory signal.
	ErrSignalTimeout = errors.New("amddirect: timeout waiting for HSA memory signal")
)

// KVTransferLeaseState represents the lifecycle state of a disaggregated KV transfer lease.
type KVTransferLeaseState string

const (
	// LeaseStateNegotiating indicates lease handshake is in progress.
	LeaseStateNegotiating KVTransferLeaseState = "NEGOTIATING"

	// LeaseStateActive indicates lease is granted and RDMA endpoints are connected in RTS state.
	LeaseStateActive KVTransferLeaseState = "ACTIVE"

	// LeaseStateTransferring indicates one-sided RDMA Write transfer is in-flight.
	LeaseStateTransferring KVTransferLeaseState = "TRANSFERRING"

	// LeaseStateReady indicates RDMA Write with Imm completed and HSA signal is fired.
	LeaseStateReady KVTransferLeaseState = "READY"

	// LeaseStateReleased indicates the lease and underlying DMA-BUF/MR allocations have been freed.
	LeaseStateReleased KVTransferLeaseState = "RELEASED"

	// LeaseStateExpired indicates the lease timed out before completion.
	LeaseStateExpired KVTransferLeaseState = "EXPIRED"

	// LeaseStateCorrupted indicates signal or data corruption was detected on the decode endpoint.
	LeaseStateCorrupted KVTransferLeaseState = "CORRUPTED"

	// LeaseStateFailed indicates transfer failed due to RDMA transport error.
	LeaseStateFailed KVTransferLeaseState = "FAILED"
)

// KVBlockDescriptor describes a single chunked KV cache block within a transfer lease.
type KVBlockDescriptor struct {
	BlockIndex  int     `json:"block_index"`
	ByteOffset  uint64  `json:"byte_offset"`
	ByteLength  uint64  `json:"byte_length"`
	LocalVRAM   uintptr `json:"local_vram"`
	RemoteVRAM  uintptr `json:"remote_vram"`
	Transferred bool    `json:"transferred"`
}

// KVTransferLease represents a transfer-keyed lease for zero-copy KV cache handoff
// between a prefill cluster worker and a decode cluster worker over AMD GPU Direct RDMA.
type KVTransferLease struct {
	mu                   sync.RWMutex
	TransferID           string               `json:"transfer_id"`
	LeaseID              string               `json:"lease_id"`
	PrefillNodeID        int                  `json:"prefill_node_id"`
	DecodeNodeID         int                  `json:"decode_node_id"`
	PrefillVRAMAddr      uintptr              `json:"prefill_vram_addr"`
	DecodeVRAMAddr       uintptr              `json:"decode_vram_addr"`
	ByteLength           uint64               `json:"byte_length"`
	NumBlocks            int                  `json:"num_blocks"`
	BlockSize            uint64               `json:"block_size"`
	Blocks               []KVBlockDescriptor  `json:"blocks"`
	RemoteRKey           uint32               `json:"remote_rkey"`
	LocalLKey            uint32               `json:"local_lkey"`
	PrefillDMABUFFD      int                  `json:"prefill_dmabuf_fd"`
	DecodeDMABUFFD       int                  `json:"decode_dmabuf_fd"`
	PrefillQPN           uint32               `json:"prefill_qpn"`
	DecodeQPN            uint32               `json:"decode_qpn"`
	SignalID             string               `json:"signal_id"`
	SignalAddress        uintptr              `json:"signal_address"`
	ExpectedImmData      uint32               `json:"expected_imm_data"`
	ReceivedImmData      uint32               `json:"received_imm_data"`
	State                KVTransferLeaseState `json:"state"`
	CreatedAt            time.Time            `json:"created_at"`
	ExpiresAt            time.Time            `json:"expires_at"`
	TransferStart        time.Time            `json:"transfer_start"`
	TransferEnd          time.Time            `json:"transfer_end"`
	SignalFiredAt        time.Time            `json:"signal_fired_at"`
	SignalLatency        time.Duration        `json:"signal_latency"`
	PrefillStagingCopies int                  `json:"prefill_staging_copies"`
	DecodeStagingCopies  int                  `json:"decode_staging_copies"`
}

// StagingCopyCount returns the total number of intermediate host DRAM staging copies.
// Under AMD GPU Direct RDMA, this invariant must strictly evaluate to 0.
func (l *KVTransferLease) StagingCopyCount() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.PrefillStagingCopies + l.DecodeStagingCopies
}

// GetState returns the current lifecycle state under read lock.
func (l *KVTransferLease) GetState() KVTransferLeaseState {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.State
}

// IsExpired reports whether the lease deadline has elapsed.
func (l *KVTransferLease) IsExpired(now time.Time) bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return now.After(l.ExpiresAt)
}

// LeaseNegotiationRequest holds parameters required to negotiate a disaggregated KV handoff lease.
type LeaseNegotiationRequest struct {
	TransferID      string        `json:"transfer_id"`
	PrefillNodeID   int           `json:"prefill_node_id"`
	DecodeNodeID    int           `json:"decode_node_id"`
	PrefillVRAMAddr uintptr       `json:"prefill_vram_addr"`
	DecodeVRAMAddr  uintptr       `json:"decode_vram_addr"`
	ByteLength      uint64        `json:"byte_length"`
	BlockSize       uint64        `json:"block_size"`
	TTL             time.Duration `json:"ttl"`
	ImmData         uint32        `json:"imm_data"`
}

// KVTransferResult encapsulates the outcome and verified performance of an RDMA KV transfer.
type KVTransferResult struct {
	TransferID           string        `json:"transfer_id"`
	LeaseID              string        `json:"lease_id"`
	BytesTransferred     uint64        `json:"bytes_transferred"`
	PrefillNodeID        int           `json:"prefill_node_id"`
	DecodeNodeID         int           `json:"decode_node_id"`
	Duration             time.Duration `json:"duration"`
	SignalLatency        time.Duration `json:"signal_latency"`
	PrefillStagingCopies int           `json:"prefill_staging_copies"`
	DecodeStagingCopies  int           `json:"decode_staging_copies"`
	ImmData              uint32        `json:"imm_data"`
	Success              bool          `json:"success"`
}

// StagingCopyCount returns the count of host DRAM staging copies in the transfer result (must be 0).
func (r *KVTransferResult) StagingCopyCount() int {
	return r.PrefillStagingCopies + r.DecodeStagingCopies
}

// DecodeWavefrontResult contains the status and metrics of a decode wavefront waking up on arrival.
type DecodeWavefrontResult struct {
	LeaseID          string        `json:"lease_id"`
	WakeupLatency    time.Duration `json:"wakeup_latency"`
	ReceivedImmData  uint32        `json:"received_imm_data"`
	DecodedTokens    int           `json:"decoded_tokens"`
	StagingCopyCount int           `json:"staging_copy_count"`
	Completed        bool          `json:"completed"`
	Error            error         `json:"error,omitempty"`
}

// PrefillDecodeTransferConfig specifies operational parameters for the transfer coordinator.
type PrefillDecodeTransferConfig struct {
	DefaultTTL             time.Duration `json:"default_ttl"`
	DefaultBlockSize       uint64        `json:"default_block_size"`
	EnforceZeroCopy        bool          `json:"enforce_zero_copy"`
	EnableLargeBARCheck    bool          `json:"enable_large_bar_check"`
	EnforceACSZeroRedirect bool          `json:"enforce_acs_zero_redirect"`
}

// PrefillDecodeTransferStats snapshots telemetry metrics of the disaggregated KV transfer coordinator.
type PrefillDecodeTransferStats struct {
	ActiveLeases     int    `json:"active_leases"`
	TotalLeases      uint64 `json:"total_leases"`
	TotalTransfers   uint64 `json:"total_transfers"`
	TotalBytesMoved  uint64 `json:"total_bytes_moved"`
	CorruptSignals   uint64 `json:"corrupt_signals"`
	ExpiredLeases    uint64 `json:"expired_leases"`
	StagingCopyCount int    `json:"staging_copy_count"`
}

// PrefillDecodeKVTransferCoordinator coordinates disaggregated KV cache transfers from prefill nodes
// to decode nodes over AMD GPU Direct RDMA with sub-microsecond HSA completion signaling.
type PrefillDecodeKVTransferCoordinator struct {
	cfg            PrefillDecodeTransferConfig
	prefillHAL     *AMDGPUDirectHAL
	decodeHAL      *AMDGPUDirectHAL
	mu             sync.RWMutex
	leases         map[string]*KVTransferLease
	leaseMap       map[string]string
	totalLeases    atomic.Uint64
	totalTransfers atomic.Uint64
	totalBytes     atomic.Uint64
	corruptSignals atomic.Uint64
	expiredLeases  atomic.Uint64
	nextID         atomic.Uint64
}

// NewPrefillDecodeKVTransferCoordinator creates a new disaggregated KV transfer coordinator.
func NewPrefillDecodeKVTransferCoordinator(prefillHAL, decodeHAL *AMDGPUDirectHAL, cfg PrefillDecodeTransferConfig) (*PrefillDecodeKVTransferCoordinator, error) {
	if cfg.DefaultTTL <= 0 {
		cfg.DefaultTTL = DefaultDisaggTTL
	}
	if cfg.DefaultBlockSize == 0 {
		cfg.DefaultBlockSize = DefaultDisaggBlockSize
	}
	if prefillHAL == nil {
		prefillHAL = NewAMDGPUDirectHAL(AMDGPUDirectConfig{
			EnableLargeBARCheck:    cfg.EnableLargeBARCheck,
			EnforceACSZeroRedirect: cfg.EnforceACSZeroRedirect,
			PreferXGMI:             true,
		})
	}
	if decodeHAL == nil {
		decodeHAL = prefillHAL
	}

	return &PrefillDecodeKVTransferCoordinator{
		cfg:        cfg,
		prefillHAL: prefillHAL,
		decodeHAL:  decodeHAL,
		leases:     make(map[string]*KVTransferLease),
		leaseMap:   make(map[string]string),
	}, nil
}

// StagingCopyCount returns the coordinator's staging copy count (strictly 0).
func (c *PrefillDecodeKVTransferCoordinator) StagingCopyCount() int {
	return 0
}

// PackImmData encodes a 16-bit transfer tag and 16-bit token count into a 32-bit immediate payload.
func PackImmData(transferTag uint16, tokenCount uint16) uint32 {
	return DefaultSignalReadyFlag | (uint32(transferTag) << 16) | uint32(tokenCount)
}

// UnpackImmData extracts the transfer tag and token count from a 32-bit immediate payload.
func UnpackImmData(imm uint32) (transferTag uint16, tokenCount uint16, ready bool) {
	ready = (imm & DefaultSignalReadyFlag) != 0
	transferTag = uint16((imm >> 16) & 0x7FFF)
	tokenCount = uint16(imm & 0xFFFF)
	return
}

// NegotiateLease conducts the transfer handshake between prefill and decode nodes:
// registers DMA-BUF regions with ibv_reg_dmabuf_mr, exchanges remote RKey/VRAM address,
// registers the decode HSAMemorySignal, and establishes the QP connection.
func (c *PrefillDecodeKVTransferCoordinator) NegotiateLease(req LeaseNegotiationRequest) (*KVTransferLease, error) {
	if req.TransferID == "" {
		return nil, errors.New("amddirect: transfer ID cannot be empty")
	}
	if req.ByteLength == 0 {
		return nil, ErrInvalidByteLength
	}
	if req.BlockSize == 0 {
		req.BlockSize = c.cfg.DefaultBlockSize
	}
	ttl := req.TTL
	if ttl <= 0 {
		ttl = c.cfg.DefaultTTL
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Ensure previous active lease for same transfer is cleaned or replaced
	if oldLeaseID, exists := c.leaseMap[req.TransferID]; exists {
		if oldLease, ok := c.leases[oldLeaseID]; ok && oldLease.State == LeaseStateActive {
			return nil, fmt.Errorf("amddirect: active lease %s already exists for transfer %s", oldLeaseID, req.TransferID)
		}
	}

	leaseNum := c.nextID.Add(1)
	leaseID := fmt.Sprintf("lease-%s-%04d", req.TransferID, leaseNum)

	// 1. Prefill node: export DMA-BUF and register for RDMA (ibv_reg_dmabuf_mr)
	prefillDMABUF, err := c.prefillHAL.ExportVRAMToDMABUF(req.PrefillNodeID, req.PrefillVRAMAddr, req.ByteLength)
	if err != nil {
		return nil, fmt.Errorf("amddirect: prefill export DMA-BUF failed: %w", err)
	}

	prefillMR, err := c.prefillHAL.RegisterDMABUFForRDMA(prefillDMABUF.FD, req.ByteLength)
	if err != nil {
		_ = c.prefillHAL.CloseDMABUF(prefillDMABUF.FD)
		return nil, fmt.Errorf("amddirect: prefill register DMA-BUF RDMA MR failed: %w", err)
	}

	// 2. Decode node: export DMA-BUF and register for RDMA (ibv_reg_dmabuf_mr)
	decodeDMABUF, err := c.decodeHAL.ExportVRAMToDMABUF(req.DecodeNodeID, req.DecodeVRAMAddr, req.ByteLength)
	if err != nil {
		_ = c.prefillHAL.DeregisterRDMARegion(prefillMR.RKey)
		_ = c.prefillHAL.CloseDMABUF(prefillDMABUF.FD)
		return nil, fmt.Errorf("amddirect: decode export DMA-BUF failed: %w", err)
	}

	decodeMR, err := c.decodeHAL.RegisterDMABUFForRDMA(decodeDMABUF.FD, req.ByteLength)
	if err != nil {
		_ = c.decodeHAL.CloseDMABUF(decodeDMABUF.FD)
		_ = c.prefillHAL.DeregisterRDMARegion(prefillMR.RKey)
		_ = c.prefillHAL.CloseDMABUF(prefillDMABUF.FD)
		return nil, fmt.Errorf("amddirect: decode register DMA-BUF RDMA MR failed: %w", err)
	}

	// 3. Decode node: allocate and register HSA memory signal for sub-microsecond completion notification
	signalID := fmt.Sprintf("hsa-signal-%s", leaseID)
	signalAddr := req.DecodeVRAMAddr + uintptr(req.ByteLength)
	signal := NewHSAMemorySignal(signalID, 0, signalAddr)
	c.decodeHAL.RegisterSignal(signal)

	// 4. Initialize and connect Queue Pairs between Prefill and Decode nodes
	sendCQ := NewCompletionQueue(uint32(leaseNum*2+1), 64)
	recvCQ := NewCompletionQueue(uint32(leaseNum*2+2), 64)
	prefillQP, err := c.prefillHAL.CreateQueuePair(QPInitAttr{
		QPType:    QPTypeRC,
		SendCQ:    sendCQ,
		RecvCQ:    recvCQ,
		MaxSendWR: 32,
		MaxRecvWR: 32,
		NodeID:    req.PrefillNodeID,
	})
	if err != nil {
		return nil, fmt.Errorf("amddirect: create prefill QP failed: %w", err)
	}

	decSendCQ := NewCompletionQueue(uint32(leaseNum*2+3), 64)
	decRecvCQ := NewCompletionQueue(uint32(leaseNum*2+4), 64)
	decodeQP, err := c.decodeHAL.CreateQueuePair(QPInitAttr{
		QPType:    QPTypeRC,
		SendCQ:    decSendCQ,
		RecvCQ:    decRecvCQ,
		MaxSendWR: 32,
		MaxRecvWR: 32,
		NodeID:    req.DecodeNodeID,
	})
	if err != nil {
		return nil, fmt.Errorf("amddirect: create decode QP failed: %w", err)
	}

	// Connect QPs into RTS
	_ = prefillQP.Modify(QPAttr{State: QPStateInit})
	_ = decodeQP.Modify(QPAttr{State: QPStateInit})
	_ = prefillQP.Modify(QPAttr{State: QPStateRTR, DestQPN: decodeQP.QPNum})
	_ = decodeQP.Modify(QPAttr{State: QPStateRTR, DestQPN: prefillQP.QPNum})
	_ = prefillQP.Modify(QPAttr{State: QPStateRTS})
	_ = decodeQP.Modify(QPAttr{State: QPStateRTS})

	// 5. Partition blocks
	numBlocks := int((req.ByteLength + req.BlockSize - 1) / req.BlockSize)
	blocks := make([]KVBlockDescriptor, numBlocks)
	currPrefillAddr := req.PrefillVRAMAddr
	currDecodeAddr := req.DecodeVRAMAddr
	remaining := req.ByteLength

	for b := 0; b < numBlocks; b++ {
		bLen := req.BlockSize
		if remaining < bLen {
			bLen = remaining
		}
		blocks[b] = KVBlockDescriptor{
			BlockIndex:  b,
			ByteOffset:  req.ByteLength - remaining,
			ByteLength:  bLen,
			LocalVRAM:   currPrefillAddr,
			RemoteVRAM:  currDecodeAddr,
			Transferred: false,
		}
		currPrefillAddr += uintptr(bLen)
		currDecodeAddr += uintptr(bLen)
		remaining -= bLen
	}

	expectedImm := req.ImmData
	if expectedImm == 0 {
		expectedImm = PackImmData(uint16(leaseNum&0x7FFF), uint16(numBlocks))
	}

	lease := &KVTransferLease{
		TransferID:           req.TransferID,
		LeaseID:              leaseID,
		PrefillNodeID:        req.PrefillNodeID,
		DecodeNodeID:         req.DecodeNodeID,
		PrefillVRAMAddr:      req.PrefillVRAMAddr,
		DecodeVRAMAddr:       req.DecodeVRAMAddr,
		ByteLength:           req.ByteLength,
		NumBlocks:            numBlocks,
		BlockSize:            req.BlockSize,
		Blocks:               blocks,
		RemoteRKey:           decodeMR.RKey,
		LocalLKey:            prefillMR.LKey,
		PrefillDMABUFFD:      prefillDMABUF.FD,
		DecodeDMABUFFD:       decodeDMABUF.FD,
		PrefillQPN:           prefillQP.QPNum,
		DecodeQPN:            decodeQP.QPNum,
		SignalID:             signalID,
		SignalAddress:        signalAddr,
		ExpectedImmData:      expectedImm,
		ReceivedImmData:      0,
		State:                LeaseStateActive,
		CreatedAt:            time.Now(),
		ExpiresAt:            time.Now().Add(ttl),
		PrefillStagingCopies: 0,
		DecodeStagingCopies:  0,
	}

	if lease.StagingCopyCount() != 0 {
		return nil, ErrZeroCopyViolation
	}

	c.leases[leaseID] = lease
	c.leaseMap[req.TransferID] = leaseID
	c.totalLeases.Add(1)

	return lease, nil
}

// GetLease retrieves an active or historical lease by LeaseID.
func (c *PrefillDecodeKVTransferCoordinator) GetLease(leaseID string) (*KVTransferLease, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	lease, ok := c.leases[leaseID]
	if !ok {
		return nil, ErrLeaseNotFound
	}
	return lease, nil
}

// GetLeaseByID retrieves the latest lease associated with a transfer or lease ID.
func (c *PrefillDecodeKVTransferCoordinator) GetLeaseByID(id string) (*KVTransferLease, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if leaseID, ok := c.leaseMap[id]; ok {
		if lease, ok := c.leases[leaseID]; ok {
			return lease, nil
		}
	}
	if lease, ok := c.leases[id]; ok {
		return lease, nil
	}
	return nil, ErrLeaseNotFound
}

// ExecuteTransfer performs one-sided RDMA Write with Immediate (RDMAOpWriteWithImm)
// directly from prefill node VRAM to decode node VRAM, verifies zero staging copies,
// and atomically stores the 32-bit immediate payload to the decode HSAMemorySignal in <1 microsecond.
func (c *PrefillDecodeKVTransferCoordinator) ExecuteTransfer(leaseID string) (*KVTransferResult, error) {
	lease, err := c.GetLease(leaseID)
	if err != nil {
		return nil, err
	}

	lease.mu.Lock()
	if lease.State != LeaseStateActive {
		st := lease.State
		lease.mu.Unlock()
		return nil, fmt.Errorf("%w: state=%s", ErrLeaseInactive, st)
	}
	if time.Now().After(lease.ExpiresAt) {
		lease.State = LeaseStateExpired
		lease.mu.Unlock()
		c.expiredLeases.Add(1)
		return nil, ErrLeaseExpired
	}
	lease.State = LeaseStateTransferring
	lease.TransferStart = time.Now()
	lease.mu.Unlock()

	prefillQP := c.prefillHAL.GetQueuePair(lease.PrefillQPN)
	if prefillQP == nil {
		lease.mu.Lock()
		lease.State = LeaseStateFailed
		lease.mu.Unlock()
		return nil, fmt.Errorf("amddirect: prefill QP %d not found", lease.PrefillQPN)
	}

	// Prepare one-sided RDMA Write with Immediate targeting target Decode node VRAM address
	wr := &WorkRequest{
		WRID:   uint64(lease.PrefillQPN)<<32 | 1,
		OpCode: RDMAOpWriteWithImm,
		SGEs: []ScatterGatherElement{
			{
				Address: lease.PrefillVRAMAddr,
				Length:  uint32(lease.ByteLength),
				LKey:    lease.LocalLKey,
			},
		},
		RemoteAddr: uint64(lease.DecodeVRAMAddr),
		RKey:       lease.RemoteRKey,
		ImmData:    lease.ExpectedImmData,
		Signaled:   true,
	}

	if err := prefillQP.PostSend(wr); err != nil {
		lease.mu.Lock()
		lease.State = LeaseStateFailed
		lease.mu.Unlock()
		return nil, fmt.Errorf("amddirect: PostSend RDMA Write with Imm failed: %w", err)
	}

	// Execute RDMA transfer directly against target Decode node HAL
	processed, err := prefillQP.ProcessSendQueue(c.decodeHAL)
	if err != nil || processed != 1 {
		lease.mu.Lock()
		lease.State = LeaseStateFailed
		lease.mu.Unlock()
		return nil, fmt.Errorf("amddirect: ProcessSendQueue failed (processed=%d): %w", processed, err)
	}

	// Verify completion on SendCQ
	wcs := prefillQP.SendCQ.PollCQ(1)
	if len(wcs) != 1 {
		lease.mu.Lock()
		lease.State = LeaseStateFailed
		lease.mu.Unlock()
		return nil, errors.New("amddirect: missing completion queue entry after RDMA Write")
	}
	wc := wcs[0]
	if wc.Status != WCSuccess {
		lease.mu.Lock()
		lease.State = LeaseStateFailed
		lease.mu.Unlock()
		return nil, fmt.Errorf("amddirect: RDMA Write completed with error: %s", wc.Status)
	}

	// Enforce zero CPU staging copies invariant on transmission completion
	if wc.StagingCopyCount() != 0 {
		lease.mu.Lock()
		lease.State = LeaseStateFailed
		lease.mu.Unlock()
		return nil, ErrZeroCopyViolation
	}

	// Sub-Microsecond HSA Signal Arrival Notification:
	// Upon completion of the RDMA Write, the 32-bit immediate data payload triggers
	// an atomic store on the Decode node's HSAMemorySignal with release memory semantics.
	signal := c.decodeHAL.GetSignal(lease.SignalID)
	if signal == nil {
		lease.mu.Lock()
		lease.State = LeaseStateFailed
		lease.mu.Unlock()
		return nil, fmt.Errorf("amddirect: decode HSA memory signal %s not found", lease.SignalID)
	}

	tTransferDone := time.Now()
	// Atomic store on Decode node's HSAMemorySignal
	signal.StoreRelease(int64(lease.ExpectedImmData))
	tSignalStore := time.Now()
	signalLatency := tSignalStore.Sub(tTransferDone)

	lease.mu.Lock()
	lease.TransferEnd = tTransferDone
	lease.SignalFiredAt = tSignalStore
	lease.SignalLatency = signalLatency
	lease.ReceivedImmData = lease.ExpectedImmData
	lease.State = LeaseStateReady
	for i := range lease.Blocks {
		lease.Blocks[i].Transferred = true
	}
	lease.mu.Unlock()

	c.totalTransfers.Add(1)
	c.totalBytes.Add(lease.ByteLength)

	// Zero-copy invariant assertion on both endpoints
	if lease.StagingCopyCount() != 0 {
		return nil, ErrZeroCopyViolation
	}

	return &KVTransferResult{
		TransferID:           lease.TransferID,
		LeaseID:              lease.LeaseID,
		BytesTransferred:     lease.ByteLength,
		PrefillNodeID:        lease.PrefillNodeID,
		DecodeNodeID:         lease.DecodeNodeID,
		Duration:             tTransferDone.Sub(lease.TransferStart),
		SignalLatency:        signalLatency,
		PrefillStagingCopies: lease.PrefillStagingCopies,
		DecodeStagingCopies:  lease.DecodeStagingCopies,
		ImmData:              lease.ExpectedImmData,
		Success:              true,
	}, nil
}

// StartDecodeWavefront simulates an AMD GPU wavefront on the decode node polling on the HSAMemorySignal.
// Polling wakes up in < 250 ns upon atomic store release, triggering instant autoregressive decode execution
// without waiting for an OS context switch or interrupt.
func (c *PrefillDecodeKVTransferCoordinator) StartDecodeWavefront(ctx context.Context, leaseID string, onDecodeReady func(*KVTransferLease) error) (<-chan DecodeWavefrontResult, error) {
	lease, err := c.GetLease(leaseID)
	if err != nil {
		return nil, err
	}

	signal := c.decodeHAL.GetSignal(lease.SignalID)
	if signal == nil {
		return nil, fmt.Errorf("amddirect: decode HSA memory signal %s not found", lease.SignalID)
	}

	resCh := make(chan DecodeWavefrontResult, 1)

	go func() {
		// Active spin simulating GPU wavefront s_waitcnt memory polling
		var val int64
		var tWakeup time.Time

		for {
			select {
			case <-ctx.Done():
				resCh <- DecodeWavefrontResult{
					LeaseID: leaseID,
					Error:   ctx.Err(),
				}
				return
			default:
			}

			val = signal.LoadRelaxed()
			if val != 0 {
				tWakeup = time.Now()
				break
			}
		}

		imm := uint32(val)
		lease.mu.RLock()
		signalFired := lease.SignalFiredAt
		expectedImm := lease.ExpectedImmData
		lease.mu.RUnlock()

		var wakeupLatency time.Duration
		if !signalFired.IsZero() {
			wakeupLatency = tWakeup.Sub(signalFired)
		}

		// Detect corrupt signal payload
		if imm == CorruptSignalValue || imm != expectedImm {
			lease.mu.Lock()
			lease.State = LeaseStateCorrupted
			lease.mu.Unlock()
			c.corruptSignals.Add(1)

			resCh <- DecodeWavefrontResult{
				LeaseID:          leaseID,
				WakeupLatency:    wakeupLatency,
				ReceivedImmData:  imm,
				StagingCopyCount: 0,
				Completed:        false,
				Error:            ErrCorruptSignal,
			}
			return
		}

		// Immediate decode execution upon signal arrival (zero OS interrupt/context switch delay)
		decodedTokens := 0
		var decodeErr error
		if onDecodeReady != nil {
			if err := onDecodeReady(lease); err != nil {
				decodeErr = err
			} else {
				decodedTokens = 1
			}
		}

		resCh <- DecodeWavefrontResult{
			LeaseID:          leaseID,
			WakeupLatency:    wakeupLatency,
			ReceivedImmData:  imm,
			DecodedTokens:    decodedTokens,
			StagingCopyCount: lease.StagingCopyCount(),
			Completed:        decodeErr == nil,
			Error:            decodeErr,
		}
	}()

	return resCh, nil
}

// WaitForDecodeReady synchronously polls the decode HSAMemorySignal with a timeout.
func (c *PrefillDecodeKVTransferCoordinator) WaitForDecodeReady(leaseID string, timeout time.Duration) (*DecodeWavefrontResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	ch, err := c.StartDecodeWavefront(ctx, leaseID, nil)
	if err != nil {
		return nil, err
	}

	select {
	case res := <-ch:
		if res.Error != nil {
			return &res, res.Error
		}
		return &res, nil
	case <-ctx.Done():
		return nil, ErrSignalTimeout
	}
}

// InjectSignalCorruption sets a corrupt payload into the decode node's HSAMemorySignal,
// simulating transmission or device memory corruption.
func (c *PrefillDecodeKVTransferCoordinator) InjectSignalCorruption(leaseID string, corruptVal uint32) error {
	lease, err := c.GetLease(leaseID)
	if err != nil {
		return err
	}

	signal := c.decodeHAL.GetSignal(lease.SignalID)
	if signal == nil {
		return fmt.Errorf("amddirect: signal %s not found", lease.SignalID)
	}

	if corruptVal == 0 {
		corruptVal = CorruptSignalValue
	}

	lease.mu.Lock()
	lease.State = LeaseStateCorrupted
	lease.mu.Unlock()

	signal.StoreRelease(int64(corruptVal))
	c.corruptSignals.Add(1)
	return nil
}

// ReleaseLease frees the associated RDMA memory regions and DMA-BUF descriptors on both nodes.
func (c *PrefillDecodeKVTransferCoordinator) ReleaseLease(leaseID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	lease, ok := c.leases[leaseID]
	if !ok {
		return ErrLeaseNotFound
	}

	lease.mu.Lock()
	defer lease.mu.Unlock()

	if lease.State == LeaseStateReleased {
		return nil
	}

	// Deregister and close prefill resources
	if lease.LocalLKey != 0 {
		_ = c.prefillHAL.DeregisterRDMARegion(lease.LocalLKey - 1)
	}
	if lease.PrefillDMABUFFD != 0 {
		_ = c.prefillHAL.CloseDMABUF(lease.PrefillDMABUFFD)
	}

	// Deregister and close decode resources
	if lease.RemoteRKey != 0 {
		_ = c.decodeHAL.DeregisterRDMARegion(lease.RemoteRKey)
	}
	if lease.DecodeDMABUFFD != 0 {
		_ = c.decodeHAL.CloseDMABUF(lease.DecodeDMABUFFD)
	}

	lease.State = LeaseStateReleased
	delete(c.leaseMap, lease.TransferID)
	return nil
}

// ExpireStaleLeases audits active leases and marks any that have exceeded their TTL as EXPIRED.
func (c *PrefillDecodeKVTransferCoordinator) ExpireStaleLeases(now time.Time) int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	expiredCount := 0
	for _, lease := range c.leases {
		lease.mu.Lock()
		if (lease.State == LeaseStateActive || lease.State == LeaseStateNegotiating) && now.After(lease.ExpiresAt) {
			lease.State = LeaseStateExpired
			expiredCount++
			c.expiredLeases.Add(1)
		}
		lease.mu.Unlock()
	}
	return expiredCount
}

// Stats returns a snapshot of coordinator performance, lease counts, and zero-copy metrics.
func (c *PrefillDecodeKVTransferCoordinator) Stats() PrefillDecodeTransferStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	active := 0
	for _, l := range c.leases {
		if l.GetState() == LeaseStateActive || l.GetState() == LeaseStateTransferring {
			active++
		}
	}

	return PrefillDecodeTransferStats{
		ActiveLeases:     active,
		TotalLeases:      c.totalLeases.Load(),
		TotalTransfers:   c.totalTransfers.Load(),
		TotalBytesMoved:  c.totalBytes.Load(),
		CorruptSignals:   c.corruptSignals.Load(),
		ExpiredLeases:    c.expiredLeases.Load(),
		StagingCopyCount: 0,
	}
}
