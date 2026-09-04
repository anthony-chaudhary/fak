package compute

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

var (
	// ErrConditionalUploadUnderCapture is returned when a caller attempts to conditionally
	// skip a constant upload while CUDA graph capture is active (#10716).
	ErrConditionalUploadUnderCapture = errors.New("conditional constant upload skipped during graph capture violates self-contained graph invariant")

	// ErrStaleConstantReplay is returned when a replayed graph detects that a constant buffer
	// is missing an unconditional capture node, leading to stale device state.
	ErrStaleConstantReplay = errors.New("stale constant buffer detected during graph replay")

	// ErrCaptureIncomplete is returned when EndCapture is called but declared constant slots were omitted.
	ErrCaptureIncomplete = errors.New("graph capture does not contain self-contained uploads for all required constant slots")

	// ErrSlotNotFound is returned when operating on an unregistered constant slot.
	ErrSlotNotFound = errors.New("constant slot not registered")

	// ErrNoActiveCapture is returned when capture-specific operations occur without an active capture.
	ErrNoActiveCapture = errors.New("no active graph capture in progress")
)

// ConstantSlotID identifies a specific constant buffer slot (e.g. "layer0_dequant_scale", "layer1_mult").
type ConstantSlotID string

// ConstantUploadReceipt records the outcome of a constant upload operation.
type ConstantUploadReceipt struct {
	Slot          ConstantSlotID `json:"slot"`
	Revision      uint64         `json:"revision"`
	BytesUploaded int            `json:"bytes_uploaded"`
	Captured      bool           `json:"captured"`
	Skipped       bool           `json:"skipped"` // True ONLY if eager caching skipped an identical upload outside capture
	Checksum      [32]byte       `json:"checksum"`
}

// ConstantSlot tracks the state, revision, and device buffer for a constant memory slot.
type ConstantSlot struct {
	mu                   sync.RWMutex
	ID                   ConstantSlotID
	ExpectedSize         int
	CurrentRevision      uint64
	LastUploadedRevision uint64
	DeviceBuffer         []byte
	CurrentChecksum      [32]byte
	TotalUploads         int64
	SkippedUploads       int64
}

// RecordedUploadNode represents a captured constant upload node inside an executable graph.
type RecordedUploadNode struct {
	SlotID   ConstantSlotID
	Revision uint64
	Data     []byte
	Checksum [32]byte
}

// CapturedConstantGraph represents an instantiated, replayable graph with self-contained constant nodes.
type CapturedConstantGraph struct {
	GraphID         string
	RecordedNodes   []RecordedUploadNode
	SlotRevisions   map[ConstantSlotID]uint64
	SelfContained   bool
	CapturedUploads int
}

// ConstantUploadManager manages constant memory uploads and enforces the self-contained
// invariant under CUDA graph capture (#10716, TurboQuant lesson).
//
// The TurboQuant failure pattern:
// During eager execution, skipping constant uploads when host-side data matches device data is
// a harmless micro-optimization. However, under CUDA graph capture, skipping an upload emits NO
// copy node into the captured stream. When alternating layers or configurations replay the graph,
// the kernel reads whatever stale constants happen to be left in device memory from earlier launches,
// causing silent data corruption.
//
// ConstantUploadManager enforces that:
// 1. All constant uploads under capture are UNCONDITIONAL and record an explicit upload node.
// 2. Constant buffer revisions are monotonically tracked.
// 3. Captured graphs verify that every declared slot was captured before completion.
// 4. Graph replays reproduce self-contained constant state without cross-layer contamination.
type ConstantUploadManager struct {
	mu             sync.RWMutex
	slots          map[ConstantSlotID]*ConstantSlot
	globalRev      uint64
	activeCapture  *activeCaptureSession
	capturedGraphs map[string]*CapturedConstantGraph
}

type activeCaptureSession struct {
	graphID       string
	requiredSlots map[ConstantSlotID]bool
	recordedNodes []RecordedUploadNode
	slotUploads   map[ConstantSlotID]int
	slotRevisions map[ConstantSlotID]uint64
}

// NewConstantUploadManager creates an initialized constant upload manager.
func NewConstantUploadManager() *ConstantUploadManager {
	return &ConstantUploadManager{
		slots:          make(map[ConstantSlotID]*ConstantSlot),
		capturedGraphs: make(map[string]*CapturedConstantGraph),
	}
}

// RegisterSlot registers a constant buffer slot with an expected byte size.
func (m *ConstantUploadManager) RegisterSlot(id ConstantSlotID, size int) error {
	if size < 0 {
		return fmt.Errorf("invalid slot size: %d", size)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.slots[id]; exists {
		return fmt.Errorf("constant slot %s already registered", id)
	}

	m.slots[id] = &ConstantSlot{
		ID:           id,
		ExpectedSize: size,
		DeviceBuffer: make([]byte, size),
	}
	return nil
}

// BeginCapture initiates a graph capture session with a set of required constant slots.
func (m *ConstantUploadManager) BeginCapture(graphID string, requiredSlots []ConstantSlotID) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.activeCapture != nil {
		return fmt.Errorf("capture session already active for graph %s", m.activeCapture.graphID)
	}

	reqMap := make(map[ConstantSlotID]bool, len(requiredSlots))
	for _, s := range requiredSlots {
		if _, exists := m.slots[s]; !exists {
			return fmt.Errorf("%w: required slot %s", ErrSlotNotFound, s)
		}
		reqMap[s] = true
	}

	m.activeCapture = &activeCaptureSession{
		graphID:       graphID,
		requiredSlots: reqMap,
		recordedNodes: make([]RecordedUploadNode, 0, len(requiredSlots)),
		slotUploads:   make(map[ConstantSlotID]int),
		slotRevisions: make(map[ConstantSlotID]uint64),
	}
	return nil
}

// IsCapturing reports whether graph capture is currently in progress.
func (m *ConstantUploadManager) IsCapturing() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeCapture != nil
}

// UploadConstant uploads constant data to the specified slot.
//
// Invariant (#10716):
// If graph capture is active, the upload is UNCONDITIONAL. Caching/skipping is strictly forbidden
// because every launch must emit an explicit graph copy node to prevent silent replay corruption.
// If not capturing, identical data may be skipped as a host-side eager optimization.
func (m *ConstantUploadManager) UploadConstant(slotID ConstantSlotID, data []byte) (*ConstantUploadReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	slot, exists := m.slots[slotID]
	if !exists {
		return nil, fmt.Errorf("%w: %s", ErrSlotNotFound, slotID)
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	csum := sha256.Sum256(data)
	isCapturing := m.activeCapture != nil

	// Eager-mode optimization: outside of graph capture, if data matches current device buffer, we may skip
	if !isCapturing && slot.CurrentChecksum == csum && bytes.Equal(slot.DeviceBuffer, data) {
		atomic.AddInt64(&slot.SkippedUploads, 1)
		return &ConstantUploadReceipt{
			Slot:          slotID,
			Revision:      slot.CurrentRevision,
			BytesUploaded: 0,
			Captured:      false,
			Skipped:       true,
			Checksum:      csum,
		}, nil
	}

	// Under capture: Invariant enforces UNCONDITIONAL upload.
	// Even if data is identical to what's on device, we MUST emit the upload node into the graph!
	m.globalRev++
	newRev := m.globalRev

	slot.CurrentRevision = newRev
	slot.LastUploadedRevision = newRev
	slot.CurrentChecksum = csum

	// Update device buffer representation
	if len(slot.DeviceBuffer) != len(data) {
		slot.DeviceBuffer = make([]byte, len(data))
	}
	copy(slot.DeviceBuffer, data)
	atomic.AddInt64(&slot.TotalUploads, 1)

	receipt := &ConstantUploadReceipt{
		Slot:          slotID,
		Revision:      newRev,
		BytesUploaded: len(data),
		Captured:      isCapturing,
		Skipped:       false,
		Checksum:      csum,
	}

	// If capturing, record the node unconditionally into the active graph session
	if isCapturing {
		dataCopy := append([]byte(nil), data...)
		m.activeCapture.recordedNodes = append(m.activeCapture.recordedNodes, RecordedUploadNode{
			SlotID:   slotID,
			Revision: newRev,
			Data:     dataCopy,
			Checksum: csum,
		})
		m.activeCapture.slotUploads[slotID]++
		m.activeCapture.slotRevisions[slotID] = newRev
	}

	return receipt, nil
}

// UploadConstantConditional attempts a conditional upload, but if graph capture is active,
// it rejects the attempt with ErrConditionalUploadUnderCapture to enforce self-contained graph invariants (#10716).
func (m *ConstantUploadManager) UploadConstantConditional(slotID ConstantSlotID, data []byte) (*ConstantUploadReceipt, error) {
	m.mu.RLock()
	isCapturing := m.activeCapture != nil
	m.mu.RUnlock()

	if isCapturing {
		return nil, ErrConditionalUploadUnderCapture
	}
	return m.UploadConstant(slotID, data)
}

// AbortCapture cancels the active graph capture session without recording a graph.
func (m *ConstantUploadManager) AbortCapture() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.activeCapture = nil
}

// EndCapture finalizes the graph capture and enforces that all required constant slots
// were unconditionally recorded.
func (m *ConstantUploadManager) EndCapture() (*CapturedConstantGraph, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.activeCapture == nil {
		return nil, ErrNoActiveCapture
	}

	session := m.activeCapture
	m.activeCapture = nil

	// Verify self-contained invariant: every required slot must have at least one upload node
	var missingSlots []ConstantSlotID
	for req := range session.requiredSlots {
		if count := session.slotUploads[req]; count == 0 {
			missingSlots = append(missingSlots, req)
		}
	}

	if len(missingSlots) > 0 {
		sort.Slice(missingSlots, func(i, j int) bool {
			return missingSlots[i] < missingSlots[j]
		})
		return nil, fmt.Errorf("%w: missing required uploads for slots %v", ErrCaptureIncomplete, missingSlots)
	}

	capturedGraph := &CapturedConstantGraph{
		GraphID:         session.graphID,
		RecordedNodes:   session.recordedNodes,
		SlotRevisions:   session.slotRevisions,
		SelfContained:   true,
		CapturedUploads: len(session.recordedNodes),
	}

	m.capturedGraphs[session.graphID] = capturedGraph
	return capturedGraph, nil
}

// ReplayGraph simulates executing the captured graph nodes, reproducing the constant uploads
// unconditionally into device memory and verifying zero cross-layer contamination (#10716).
func (m *ConstantUploadManager) ReplayGraph(graphID string) ([]*ConstantUploadReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	graph, exists := m.capturedGraphs[graphID]
	if !exists {
		return nil, fmt.Errorf("captured graph %s not found", graphID)
	}

	if !graph.SelfContained {
		return nil, ErrStaleConstantReplay
	}

	var receipts []*ConstantUploadReceipt
	for _, node := range graph.RecordedNodes {
		slot, ok := m.slots[node.SlotID]
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrSlotNotFound, node.SlotID)
		}

		slot.mu.Lock()
		// Replay node executes upload into device buffer
		if len(slot.DeviceBuffer) != len(node.Data) {
			slot.DeviceBuffer = make([]byte, len(node.Data))
		}
		copy(slot.DeviceBuffer, node.Data)
		slot.CurrentChecksum = node.Checksum
		slot.LastUploadedRevision = node.Revision
		slot.mu.Unlock()

		receipts = append(receipts, &ConstantUploadReceipt{
			Slot:          node.SlotID,
			Revision:      node.Revision,
			BytesUploaded: len(node.Data),
			Captured:      true,
			Skipped:       false,
			Checksum:      node.Checksum,
		})
	}

	return receipts, nil
}

// ReadDeviceConstant reads the current content of a constant slot's device buffer.
func (m *ConstantUploadManager) ReadDeviceConstant(slotID ConstantSlotID) ([]byte, uint64, error) {
	m.mu.RLock()
	slot, exists := m.slots[slotID]
	m.mu.RUnlock()

	if !exists {
		return nil, 0, ErrSlotNotFound
	}

	slot.mu.RLock()
	defer slot.mu.RUnlock()

	return append([]byte(nil), slot.DeviceBuffer...), slot.LastUploadedRevision, nil
}

// SlotRevision returns the current revision of a constant slot.
func (m *ConstantUploadManager) SlotRevision(slotID ConstantSlotID) (uint64, error) {
	m.mu.RLock()
	slot, exists := m.slots[slotID]
	m.mu.RUnlock()

	if !exists {
		return 0, ErrSlotNotFound
	}

	slot.mu.RLock()
	defer slot.mu.RUnlock()

	return slot.CurrentRevision, nil
}

// CapturedGraph retrieves a captured graph definition by ID.
func (m *ConstantUploadManager) CapturedGraph(graphID string) (*CapturedConstantGraph, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	g, ok := m.capturedGraphs[graphID]
	return g, ok
}

// ConstantSlotStats contains snapshot metrics for a constant slot.
type ConstantSlotStats struct {
	ID             ConstantSlotID `json:"id"`
	Revision       uint64         `json:"revision"`
	TotalUploads   int64          `json:"total_uploads"`
	SkippedUploads int64          `json:"skipped_uploads"`
	BufferSize     int            `json:"buffer_size"`
}

// SlotStats returns snapshot metrics for a given constant slot.
func (m *ConstantUploadManager) SlotStats(slotID ConstantSlotID) (ConstantSlotStats, error) {
	m.mu.RLock()
	slot, ok := m.slots[slotID]
	m.mu.RUnlock()

	if !ok {
		return ConstantSlotStats{}, ErrSlotNotFound
	}

	slot.mu.RLock()
	defer slot.mu.RUnlock()

	return ConstantSlotStats{
		ID:             slot.ID,
		Revision:       slot.CurrentRevision,
		TotalUploads:   atomic.LoadInt64(&slot.TotalUploads),
		SkippedUploads: atomic.LoadInt64(&slot.SkippedUploads),
		BufferSize:     len(slot.DeviceBuffer),
	}, nil
}
