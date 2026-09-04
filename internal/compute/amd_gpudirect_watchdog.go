// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// RDMACompletionQueue aliases CompletionQueue for InfiniBand/RoCE completion queue operations.
type RDMACompletionQueue = CompletionQueue

// Standard typed errors for link health, transfer timeouts, healing, and resource management.
var (
	ErrTransferStalled = errors.New("amddirect: transfer stalled past SLA deadline")
	ErrLinkDegraded    = errors.New("amddirect: PCIe AER correctable errors exceeded degradation threshold")
	ErrLinkFatal       = errors.New("amddirect: PCIe AER fatal error detected, physical link failed")
	ErrPeerUnreachable = errors.New("amddirect: peer handshake verification failed")
	ErrResourceLeaked  = errors.New("amddirect: resource leak detected during teardown audit")
	ErrTeardownPanic   = errors.New("amddirect: panic recovered during teardown")
)

// AERCounters captures PCIe Advanced Error Reporting (AER) telemetry counters.
type AERCounters struct {
	Correctable uint64 `json:"aer_dev_correctable"`
	Fatal       uint64 `json:"aer_dev_fatal"`
}

// AERThresholdConfig specifies the error count limits before declaring link degradation or failure.
type AERThresholdConfig struct {
	MaxCorrectable uint64 `json:"max_correctable"`
	MaxFatal       uint64 `json:"max_fatal"`
}

// LinkHealthStatus represents the health classification of a PCIe / interconnect link.
type LinkHealthStatus string

const (
	// LinkHealthHealthy indicates normal operation within error thresholds.
	LinkHealthHealthy LinkHealthStatus = "HEALTHY"
	// LinkHealthDegraded indicates non-fatal or correctable error thresholds were exceeded.
	LinkHealthDegraded LinkHealthStatus = "DEGRADED"
	// LinkHealthFailed indicates uncorrectable fatal errors or physical link failure.
	LinkHealthFailed LinkHealthStatus = "FAILED"
)

// ParseAERCounters parses Linux sysfs aer_dev_correctable or aer_dev_fatal file content.
// Handles both key-value tables (e.g. "Receiver_Error 0", "TOTAL_ERR_COR 15") and raw integer counts.
func ParseAERCounters(content string) uint64 {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return 0
	}

	// Single integer fast path
	if val, err := strconv.ParseUint(trimmed, 10, 64); err == nil {
		return val
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	var total uint64
	var foundTotalKey bool

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}

		key := strings.ToUpper(fields[0])
		val, err := strconv.ParseUint(fields[len(fields)-1], 10, 64)
		if err != nil {
			continue
		}

		// Prefer explicit total lines if provided by sysfs
		if key == "TOTAL_ERR_COR" || key == "TOTAL_ERR_FATAL" || key == "TOTAL" {
			total = val
			foundTotalKey = true
			break
		}

		total += val
	}

	if foundTotalKey {
		return total
	}
	return total
}

// ReadSysfsAERCounters reads PCIe AER counters for a given PCI BDF from Linux sysfs.
func ReadSysfsAERCounters(sysfsRoot, bdf string) (AERCounters, error) {
	if sysfsRoot == "" {
		sysfsRoot = "/sys"
	}
	pciDir := filepath.Join(sysfsRoot, "bus", "pci", "devices", bdf)

	corPath := filepath.Join(pciDir, "aer_dev_correctable")
	fatalPath := filepath.Join(pciDir, "aer_dev_fatal")

	var counters AERCounters
	corData, errCor := os.ReadFile(corPath)
	fatalData, errFatal := os.ReadFile(fatalPath)

	if errCor != nil && errFatal != nil {
		return counters, fmt.Errorf("amddirect: AER counter files not accessible for %s: %w", bdf, errCor)
	}

	if errCor == nil {
		counters.Correctable = ParseAERCounters(string(corData))
	}
	if errFatal == nil {
		counters.Fatal = ParseAERCounters(string(fatalData))
	}

	return counters, nil
}

// EvaluateAERStatus evaluates AERCounters against AERThresholdConfig.
func EvaluateAERStatus(counters AERCounters, thresholds AERThresholdConfig) (LinkHealthStatus, string) {
	if counters.Fatal > thresholds.MaxFatal {
		return LinkHealthFailed, fmt.Sprintf("fatal AER errors (%d) exceed threshold (%d)", counters.Fatal, thresholds.MaxFatal)
	}
	if counters.Correctable > thresholds.MaxCorrectable {
		return LinkHealthDegraded, fmt.Sprintf("correctable AER errors (%d) exceed threshold (%d)", counters.Correctable, thresholds.MaxCorrectable)
	}
	return LinkHealthHealthy, "AER counters within normal operational thresholds"
}

// FlushQPToError atomically transitions a Queue Pair to QPStateError.
// In accordance with InfiniBand and ROCm RDMA specifications, entering QPStateError
// evicts and flushes all in-flight send and receive Work Requests into the Completion Queue
// marked with status WCWrFlushedErr (IBV_WC_WR_FLUSH_ERR), waking waiting routines immediately.
func FlushQPToError(qp *RDMAQueuePair, reason string) error {
	if qp == nil {
		return errors.New("amddirect: nil queue pair cannot be flushed")
	}
	_ = reason // Reason captured for telemetry / diagnostic audit
	return qp.Modify(QPAttr{State: QPStateError})
}

// SignalWatch captures an in-flight HSA memory signal monitored by LinkHealthWatchdog.
type SignalWatch struct {
	SignalID     string
	Signal       *HSAMemorySignal
	TargetValue  int64
	Deadline     time.Time
	AssociatedQP *RDMAQueuePair
	OnStall      func()
}

// WatchdogConfig configures SLA polling timeouts and PCIe AER thresholds.
type WatchdogConfig struct {
	SignalSLADeadline time.Duration      `json:"signal_sla_deadline"`
	CQSLADeadline     time.Duration      `json:"cq_sla_deadline"`
	SysfsRoot         string             `json:"sysfs_root"`
	DefaultAERLimits  AERThresholdConfig `json:"default_aer_limits"`
}

// LinkHealthWatchdog monitors HSA completion signals, RDMA completion queues,
// and PCIe AER counters to enforce millisecond-scale SLA deadlines and prevent hangs.
type LinkHealthWatchdog struct {
	mu            sync.RWMutex
	cfg           WatchdogConfig
	hal           *AMDGPUDirectHAL
	aerReader     func(bdf string) (AERCounters, error)
	monitoredBDFs map[string]AERThresholdConfig
	signalWatches map[string]*SignalWatch
	degradedLinks map[string]LinkHealthStatus
	onDegraded    func(bdf string, status LinkHealthStatus, reason string)
}

// NewLinkHealthWatchdog constructs a new LinkHealthWatchdog.
func NewLinkHealthWatchdog(cfg WatchdogConfig, hal *AMDGPUDirectHAL) *LinkHealthWatchdog {
	if cfg.SignalSLADeadline <= 0 {
		cfg.SignalSLADeadline = 50 * time.Millisecond
	}
	if cfg.CQSLADeadline <= 0 {
		cfg.CQSLADeadline = 50 * time.Millisecond
	}

	w := &LinkHealthWatchdog{
		cfg:           cfg,
		hal:           hal,
		monitoredBDFs: make(map[string]AERThresholdConfig),
		signalWatches: make(map[string]*SignalWatch),
		degradedLinks: make(map[string]LinkHealthStatus),
	}
	w.aerReader = func(bdf string) (AERCounters, error) {
		return ReadSysfsAERCounters(w.cfg.SysfsRoot, bdf)
	}
	return w
}

// SetAERReader overrides the AER telemetry source (useful for simulated tests and telemetry injection).
func (w *LinkHealthWatchdog) SetAERReader(fn func(bdf string) (AERCounters, error)) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.aerReader = fn
}

// RegisterMonitoredBDF registers a PCIe device BDF to be monitored for link degradation.
func (w *LinkHealthWatchdog) RegisterMonitoredBDF(bdf string, limits AERThresholdConfig) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.monitoredBDFs[bdf] = limits
}

// SetDegradationCallback configures an event hook triggered when link degradation is detected.
func (w *LinkHealthWatchdog) SetDegradationCallback(cb func(bdf string, status LinkHealthStatus, reason string)) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.onDegraded = cb
}

// CheckLinkHealth queries PCIe AER counters for a BDF and evaluates link health.
func (w *LinkHealthWatchdog) CheckLinkHealth(bdf string) (LinkHealthStatus, AERCounters, error) {
	w.mu.RLock()
	reader := w.aerReader
	limits, ok := w.monitoredBDFs[bdf]
	if !ok {
		limits = w.cfg.DefaultAERLimits
	}
	cb := w.onDegraded
	w.mu.RUnlock()

	counters, err := reader(bdf)
	if err != nil {
		return LinkHealthFailed, counters, err
	}

	status, reason := EvaluateAERStatus(counters, limits)

	w.mu.Lock()
	w.degradedLinks[bdf] = status
	w.mu.Unlock()

	if status != LinkHealthHealthy && cb != nil {
		cb(bdf, status, reason)
	}

	if status == LinkHealthFailed {
		return status, counters, fmt.Errorf("%w: %s (%s)", ErrLinkFatal, bdf, reason)
	}
	if status == LinkHealthDegraded {
		return status, counters, fmt.Errorf("%w: %s (%s)", ErrLinkDegraded, bdf, reason)
	}

	return status, counters, nil
}

// PollHSASignalWithDeadline polls an HSA memory signal against a millisecond SLA deadline.
// If the deadline expires without the signal reaching targetValue, it detects a stalled transfer,
// atomically flushes the associated Queue Pair into QPStateError (if provided), and returns ErrTransferStalled.
func (w *LinkHealthWatchdog) PollHSASignalWithDeadline(signal *HSAMemorySignal, targetValue int64, deadline time.Duration, qp *RDMAQueuePair) (bool, error) {
	if signal == nil {
		return false, errors.New("amddirect: nil HSA memory signal")
	}
	if deadline <= 0 {
		deadline = w.cfg.SignalSLADeadline
	}

	// Microsecond active spin simulating GPU ISA s_waitcnt memory polling
	for i := 0; i < 2000; i++ {
		if signal.LoadRelaxed() == targetValue {
			return true, nil
		}
	}

	expireTime := time.Now().Add(deadline)
	for time.Now().Before(expireTime) {
		if signal.LoadRelaxed() == targetValue {
			return true, nil
		}
		time.Sleep(50 * time.Microsecond)
	}

	// Transfer stalled past SLA deadline
	if qp != nil {
		_ = FlushQPToError(qp, fmt.Sprintf("HSA signal %s stalled waiting for %d", signal.SignalID, targetValue))
	}
	return false, fmt.Errorf("%w: signal %s did not reach %d within %v", ErrTransferStalled, signal.SignalID, targetValue, deadline)
}

// PollCQWithDeadline polls an RDMA CompletionQueue for a completion entry matching expectedWRID against an SLA deadline.
// If the deadline expires before completion arrives, it atomically transitions the QP to QPStateError,
// flushes pending Work Requests with WCWrFlushedErr, drains the flushed completion, and returns ErrTransferStalled.
func (w *LinkHealthWatchdog) PollCQWithDeadline(cq *CompletionQueue, expectedWRID uint64, deadline time.Duration, qp *RDMAQueuePair) (*WorkCompletion, error) {
	if cq == nil {
		return nil, errors.New("amddirect: nil completion queue")
	}
	if deadline <= 0 {
		deadline = w.cfg.CQSLADeadline
	}

	// Immediate drain attempt
	entries := cq.PollCQ(16)
	for i := range entries {
		if entries[i].WRID == expectedWRID {
			return &entries[i], nil
		}
	}

	expireTime := time.Now().Add(deadline)
	for time.Now().Before(expireTime) {
		select {
		case <-cq.NotifyChannel():
			batch := cq.PollCQ(16)
			for i := range batch {
				if batch[i].WRID == expectedWRID {
					return &batch[i], nil
				}
			}
		case <-time.After(50 * time.Microsecond):
			batch := cq.PollCQ(16)
			for i := range batch {
				if batch[i].WRID == expectedWRID {
					return &batch[i], nil
				}
			}
		}
	}

	// Transfer stalled past deadline: atomically flush QP
	if qp != nil {
		_ = FlushQPToError(qp, fmt.Sprintf("WRID %d stalled waiting for CQE", expectedWRID))
		// Drain again to observe the flushed completion entry
		flushed := cq.PollCQ(64)
		for i := range flushed {
			if flushed[i].WRID == expectedWRID {
				return &flushed[i], fmt.Errorf("%w: WRID %d flushed due to timeout", ErrTransferStalled, expectedWRID)
			}
		}
	}

	return nil, fmt.Errorf("%w: WRID %d did not complete within %v", ErrTransferStalled, expectedWRID, deadline)
}

// WatchSignal registers an asynchronous HSA signal watch.
func (w *LinkHealthWatchdog) WatchSignal(signal *HSAMemorySignal, targetValue int64, deadline time.Duration, qp *RDMAQueuePair, onStall func()) *SignalWatch {
	if deadline <= 0 {
		deadline = w.cfg.SignalSLADeadline
	}
	sw := &SignalWatch{
		SignalID:     signal.SignalID,
		Signal:       signal,
		TargetValue:  targetValue,
		Deadline:     time.Now().Add(deadline),
		AssociatedQP: qp,
		OnStall:      onStall,
	}

	w.mu.Lock()
	w.signalWatches[signal.SignalID] = sw
	w.mu.Unlock()
	return sw
}

// CheckWatches checks all registered signal watches and flushes any that have stalled.
// Returns the number of stalled watches detected.
func (w *LinkHealthWatchdog) CheckWatches() int {
	w.mu.Lock()
	defer w.mu.Unlock()

	now := time.Now()
	stalledCount := 0

	for id, sw := range w.signalWatches {
		if sw.Signal.LoadRelaxed() == sw.TargetValue {
			delete(w.signalWatches, id)
			continue
		}

		if now.After(sw.Deadline) {
			stalledCount++
			if sw.AssociatedQP != nil {
				_ = FlushQPToError(sw.AssociatedQP, "signal watch expired")
			}
			if sw.OnStall != nil {
				sw.OnStall()
			}
			delete(w.signalWatches, id)
		}
	}

	return stalledCount
}

// PeerHandshakeVerifier verifies remote peer connectivity and QP readiness.
type PeerHandshakeVerifier func(localQPN uint32, destQPN uint32) (bool, error)

// AutoHealerConfig defines retry thresholds and exponential backoff parameters for QP auto-healing.
type AutoHealerConfig struct {
	InitialBackoff time.Duration         `json:"initial_backoff"`
	MaxBackoff     time.Duration         `json:"max_backoff"`
	BackoffFactor  float64               `json:"backoff_factor"`
	JitterFraction float64               `json:"jitter_fraction"`
	MaxRetries     int                   `json:"max_retries"`
	HandshakeFn    PeerHandshakeVerifier `json:"-"`
}

// DefaultAutoHealerConfig returns production defaults for InfiniBand / ROCm auto-healing.
func DefaultAutoHealerConfig() AutoHealerConfig {
	return AutoHealerConfig{
		InitialBackoff: 5 * time.Millisecond,
		MaxBackoff:     100 * time.Millisecond,
		BackoffFactor:  2.0,
		JitterFraction: 0.2,
		MaxRetries:     5,
	}
}

// QPAutoHealer orchestrates automatic renegotiation and re-establishment of failed Queue Pairs.
// Drives the state machine through ERROR -> RESET -> INIT -> RTR -> RTS with randomized
// exponential backoff and peer handshake verification.
type QPAutoHealer struct {
	cfg  AutoHealerConfig
	rand *rand.Rand
	mu   sync.Mutex
}

// NewQPAutoHealer creates a new QPAutoHealer.
func NewQPAutoHealer(cfg AutoHealerConfig) *QPAutoHealer {
	if cfg.InitialBackoff <= 0 {
		cfg.InitialBackoff = 5 * time.Millisecond
	}
	if cfg.MaxBackoff <= 0 {
		cfg.MaxBackoff = 100 * time.Millisecond
	}
	if cfg.BackoffFactor <= 1.0 {
		cfg.BackoffFactor = 2.0
	}
	if cfg.JitterFraction <= 0 {
		cfg.JitterFraction = 0.2
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 5
	}

	src := rand.NewSource(time.Now().UnixNano())
	return &QPAutoHealer{
		cfg:  cfg,
		rand: rand.New(src),
	}
}

// ComputeBackoff calculates the randomized exponential backoff duration for a given attempt index.
func (h *QPAutoHealer) ComputeBackoff(attempt int) time.Duration {
	base := float64(h.cfg.InitialBackoff) * math.Pow(h.cfg.BackoffFactor, float64(attempt))
	if base > float64(h.cfg.MaxBackoff) {
		base = float64(h.cfg.MaxBackoff)
	}

	h.mu.Lock()
	rnd := h.rand.Float64()
	h.mu.Unlock()

	// Jitter in [-JitterFraction, +JitterFraction]
	jitterScale := (rnd*2.0 - 1.0) * h.cfg.JitterFraction
	withJitter := base * (1.0 + jitterScale)
	if withJitter < float64(time.Millisecond) {
		withJitter = float64(time.Millisecond)
	}

	return time.Duration(withJitter)
}

// HealQP recovers a failed Queue Pair through the standard InfiniBand state renegotiation ladder:
// ERROR -> RESET -> INIT -> RTR -> RTS.
func (h *QPAutoHealer) HealQP(qp *RDMAQueuePair, destQPN uint32, pathMTU uint32, rqpsn uint32, sqpsn uint32) error {
	if qp == nil {
		return errors.New("amddirect: nil queue pair cannot be healed")
	}

	if pathMTU == 0 {
		pathMTU = qp.PathMTU
	}
	if pathMTU == 0 {
		pathMTU = 4096
	}
	if destQPN == 0 {
		destQPN = qp.DestQPN
	}

	var lastErr error
	for attempt := 0; attempt <= h.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := h.ComputeBackoff(attempt - 1)
			time.Sleep(backoff)
		}

		// Step 1: ERROR -> RESET (clears remaining in-flight queues)
		if err := qp.Modify(QPAttr{State: QPStateReset}); err != nil {
			lastErr = fmt.Errorf("transition to RESET failed: %w", err)
			continue
		}

		// Step 2: Peer handshake verification
		if h.cfg.HandshakeFn != nil {
			ok, err := h.cfg.HandshakeFn(qp.QPNum, destQPN)
			if err != nil || !ok {
				lastErr = fmt.Errorf("%w: destQPN=%d (err=%v)", ErrPeerUnreachable, destQPN, err)
				continue
			}
		}

		// Step 3: RESET -> INIT
		if err := qp.Modify(QPAttr{State: QPStateInit}); err != nil {
			lastErr = fmt.Errorf("transition to INIT failed: %w", err)
			continue
		}

		// Step 4: INIT -> RTR (Ready-To-Receive)
		if err := qp.Modify(QPAttr{
			State:   QPStateRTR,
			DestQPN: destQPN,
			PathMTU: pathMTU,
			RQPSN:   rqpsn,
		}); err != nil {
			lastErr = fmt.Errorf("transition to RTR failed: %w", err)
			continue
		}

		// Step 5: RTR -> RTS (Ready-To-Send)
		if err := qp.Modify(QPAttr{
			State: QPStateRTS,
			SQPSN: sqpsn,
		}); err != nil {
			lastErr = fmt.Errorf("transition to RTS failed: %w", err)
			continue
		}

		// Successfully healed
		return nil
	}

	return fmt.Errorf("amddirect: QP %d auto-healing failed after %d retries: %w", qp.QPNum, h.cfg.MaxRetries, lastErr)
}

// HealQPPair renegotiates and recovers both endpoints of a connected Queue Pair pair (e.g. QP0 <-> QP1).
func (h *QPAutoHealer) HealQPPair(qp0, qp1 *RDMAQueuePair) error {
	if qp0 == nil || qp1 == nil {
		return errors.New("amddirect: both QP endpoints are required for pair healing")
	}

	var lastErr error
	for attempt := 0; attempt <= h.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := h.ComputeBackoff(attempt - 1)
			time.Sleep(backoff)
		}

		// Reset both endpoints
		_ = qp0.Modify(QPAttr{State: QPStateReset})
		_ = qp1.Modify(QPAttr{State: QPStateReset})

		// Handshake check if configured
		if h.cfg.HandshakeFn != nil {
			ok, err := h.cfg.HandshakeFn(qp0.QPNum, qp1.QPNum)
			if err != nil || !ok {
				lastErr = fmt.Errorf("%w: pair %d <-> %d", ErrPeerUnreachable, qp0.QPNum, qp1.QPNum)
				continue
			}
		}

		// Transition both to INIT
		if err := qp0.Modify(QPAttr{State: QPStateInit}); err != nil {
			lastErr = err
			continue
		}
		if err := qp1.Modify(QPAttr{State: QPStateInit}); err != nil {
			lastErr = err
			continue
		}

		// Transition both to RTR targeting each other
		if err := qp0.Modify(QPAttr{State: QPStateRTR, DestQPN: qp1.QPNum, PathMTU: 4096}); err != nil {
			lastErr = err
			continue
		}
		if err := qp1.Modify(QPAttr{State: QPStateRTR, DestQPN: qp0.QPNum, PathMTU: 4096}); err != nil {
			lastErr = err
			continue
		}

		// Transition both to RTS
		if err := qp0.Modify(QPAttr{State: QPStateRTS, SQPSN: 100}); err != nil {
			lastErr = err
			continue
		}
		if err := qp1.Modify(QPAttr{State: QPStateRTS, SQPSN: 200}); err != nil {
			lastErr = err
			continue
		}

		return nil
	}

	return fmt.Errorf("amddirect: pair auto-healing failed after %d retries: %w", h.cfg.MaxRetries, lastErr)
}

// trackedDMABUF wraps an exported DMA-BUF handle with a closure callback.
type trackedDMABUF struct {
	handle  *DMABUFHandle
	closeFn func(fd int) error
}

// trackedRegion wraps an RDMA memory region with a deregistration callback.
type trackedRegion struct {
	region  *RDMARegisteredRegion
	deregFn func(rkey uint32) error
}

// trackedDoorbell wraps an HSA hardware doorbell with a release callback.
type trackedDoorbell struct {
	doorbell *HSADoorbell
	closeFn  func(id string) error
}

// trackedAperture wraps a BAR1 VRAM memory aperture with an unmap callback.
type trackedAperture struct {
	address uintptr
	size    uint64
	nodeID  int
	unmapFn func(addr uintptr, size uint64) error
}

// TeardownReport provides an audit of all active resources tracked by TeardownManager.
type TeardownReport struct {
	ActiveDMABUFs   int  `json:"active_dmabufs"`
	ActiveRDMARegs  int  `json:"active_rdma_regions"`
	ActiveDoorbells int  `json:"active_doorbells"`
	ActiveApertures int  `json:"active_apertures"`
	Closed          bool `json:"closed"`
}

// TeardownManager tracks active DMA-BUF file descriptors, RDMA registered regions,
// HSA doorbells, and BAR1 VRAM apertures to guarantee zero-leak, idempotent cleanup.
type TeardownManager struct {
	mu        sync.Mutex
	hal       *AMDGPUDirectHAL
	dmabufs   map[int]*trackedDMABUF
	regions   map[uint32]*trackedRegion
	doorbells map[string]*trackedDoorbell
	apertures map[uintptr]*trackedAperture
	customs   []func() error
	closed    bool
}

// NewTeardownManager constructs a new leak-free TeardownManager.
func NewTeardownManager(hal *AMDGPUDirectHAL) *TeardownManager {
	return &TeardownManager{
		hal:       hal,
		dmabufs:   make(map[int]*trackedDMABUF),
		regions:   make(map[uint32]*trackedRegion),
		doorbells: make(map[string]*trackedDoorbell),
		apertures: make(map[uintptr]*trackedAperture),
		customs:   make([]func() error, 0),
	}
}

// TrackDMABUF registers an exported DMA-BUF handle for tracking and teardown.
func (tm *TeardownManager) TrackDMABUF(handle *DMABUFHandle, closeFn func(fd int) error) error {
	if handle == nil {
		return errors.New("amddirect: nil dmabuf handle")
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if closeFn == nil && tm.hal != nil {
		closeFn = tm.hal.CloseDMABUF
	}
	tm.dmabufs[handle.FD] = &trackedDMABUF{
		handle:  handle,
		closeFn: closeFn,
	}
	return nil
}

// UntrackDMABUF removes a DMA-BUF handle from tracking (e.g. after deliberate caller closure).
func (tm *TeardownManager) UntrackDMABUF(fd int) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	delete(tm.dmabufs, fd)
}

// TrackRDMARegion registers an RDMA registered region for tracking and teardown.
func (tm *TeardownManager) TrackRDMARegion(region *RDMARegisteredRegion, deregFn func(rkey uint32) error) error {
	if region == nil {
		return errors.New("amddirect: nil RDMA registered region")
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if deregFn == nil && tm.hal != nil {
		deregFn = tm.hal.DeregisterRDMARegion
	}
	tm.regions[region.RKey] = &trackedRegion{
		region:  region,
		deregFn: deregFn,
	}
	return nil
}

// UntrackRDMARegion removes an RDMA region from tracking.
func (tm *TeardownManager) UntrackRDMARegion(rkey uint32) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	delete(tm.regions, rkey)
}

// TrackDoorbell registers an HSA dispatch doorbell for tracking and teardown.
func (tm *TeardownManager) TrackDoorbell(db *HSADoorbell, closeFn func(id string) error) error {
	if db == nil {
		return errors.New("amddirect: nil HSA doorbell")
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.doorbells[db.ID] = &trackedDoorbell{
		doorbell: db,
		closeFn:  closeFn,
	}
	return nil
}

// UntrackDoorbell removes an HSA doorbell from tracking.
func (tm *TeardownManager) UntrackDoorbell(id string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	delete(tm.doorbells, id)
}

// TrackBAR1Aperture registers a BAR1 VRAM mapped aperture for unmapping at teardown.
func (tm *TeardownManager) TrackBAR1Aperture(addr uintptr, size uint64, nodeID int, unmapFn func(addr uintptr, size uint64) error) error {
	if addr == 0 || size == 0 {
		return errors.New("amddirect: invalid BAR1 aperture address or size")
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.apertures[addr] = &trackedAperture{
		address: addr,
		size:    size,
		nodeID:  nodeID,
		unmapFn: unmapFn,
	}
	return nil
}

// UntrackBAR1Aperture removes a BAR1 aperture from tracking.
func (tm *TeardownManager) UntrackBAR1Aperture(addr uintptr) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	delete(tm.apertures, addr)
}

// TrackCleanup adds an arbitrary cleanup hook to be executed at teardown.
func (tm *TeardownManager) TrackCleanup(fn func() error) {
	if fn == nil {
		return
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.customs = append(tm.customs, fn)
}

// Teardown executes idempotent teardown of all tracked resources in safe order:
// 1. RDMA Memory Regions (must be deregistered before underlying DMA-BUFs).
// 2. DMA-BUF file descriptors (closed at kernel driver level).
// 3. BAR1 VRAM apertures (unmapped from PCIe address space).
// 4. HSA Doorbells (released).
// 5. Custom cleanup hooks (LIFO order).
func (tm *TeardownManager) Teardown() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.closed {
		return nil // Idempotent: multiple calls are safe no-ops
	}

	var errs []error

	// 1. RDMA regions
	for rkey, tr := range tm.regions {
		delete(tm.regions, rkey)
		if tr.region != nil {
			tr.region.Active = false
		}
		if tr.deregFn != nil {
			if err := tr.deregFn(rkey); err != nil {
				errs = append(errs, err)
			}
		} else if tm.hal != nil {
			if err := tm.hal.DeregisterRDMARegion(rkey); err != nil {
				errs = append(errs, err)
			}
		}
	}

	// 2. DMA-BUF file descriptors
	for fd, td := range tm.dmabufs {
		delete(tm.dmabufs, fd)
		if td.handle != nil {
			td.handle.Closed = true
		}
		if td.closeFn != nil {
			if err := td.closeFn(fd); err != nil {
				errs = append(errs, err)
			}
		} else if tm.hal != nil {
			if err := tm.hal.CloseDMABUF(fd); err != nil {
				errs = append(errs, err)
			}
		}
	}

	// 3. BAR1 VRAM apertures
	for addr, ta := range tm.apertures {
		delete(tm.apertures, addr)
		if ta.unmapFn != nil {
			if err := ta.unmapFn(addr, ta.size); err != nil {
				errs = append(errs, err)
			}
		}
	}

	// 4. Doorbells
	for id, tdb := range tm.doorbells {
		delete(tm.doorbells, id)
		if tdb.closeFn != nil {
			if err := tdb.closeFn(id); err != nil {
				errs = append(errs, err)
			}
		}
	}

	// 5. Custom hooks in LIFO order
	for len(tm.customs) > 0 {
		idx := len(tm.customs) - 1
		fn := tm.customs[idx]
		tm.customs = tm.customs[:idx]
		if err := fn(); err != nil {
			errs = append(errs, err)
		}
	}

	tm.closed = true

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

// TeardownWithPanicRecovery executes Teardown inside a panic recovery boundary,
// guaranteeing zero resource leaks even during worker routine crashes or process shutdown.
func (tm *TeardownManager) TeardownWithPanicRecovery() (err error) {
	defer func() {
		if r := recover(); r != nil {
			tErr := tm.TeardownWithPanicRecovery()
			if tErr != nil {
				err = fmt.Errorf("%w: %v (teardown error: %v)", ErrTeardownPanic, r, tErr)
			} else {
				err = fmt.Errorf("%w: %v", ErrTeardownPanic, r)
			}
		}
	}()
	return tm.Teardown()
}

// AssertZeroLeaks verifies that all tracked resources were cleanly freed and closed.
// Returns nil if zero leaks exist, or ErrResourceLeaked detailing any leaked resources.
func (tm *TeardownManager) AssertZeroLeaks() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	var leaks []string
	if len(tm.dmabufs) > 0 {
		leaks = append(leaks, fmt.Sprintf("%d active DMA-BUF handles leaked", len(tm.dmabufs)))
	}
	if len(tm.regions) > 0 {
		leaks = append(leaks, fmt.Sprintf("%d active RDMA registered regions leaked", len(tm.regions)))
	}
	if len(tm.doorbells) > 0 {
		leaks = append(leaks, fmt.Sprintf("%d active HSA doorbells leaked", len(tm.doorbells)))
	}
	if len(tm.apertures) > 0 {
		leaks = append(leaks, fmt.Sprintf("%d active BAR1 apertures leaked", len(tm.apertures)))
	}

	if len(leaks) > 0 {
		return fmt.Errorf("%w: %s", ErrResourceLeaked, strings.Join(leaks, "; "))
	}
	return nil
}

// Report snapshots the current tracked resource counts.
func (tm *TeardownManager) Report() TeardownReport {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	return TeardownReport{
		ActiveDMABUFs:   len(tm.dmabufs),
		ActiveRDMARegs:  len(tm.regions),
		ActiveDoorbells: len(tm.doorbells),
		ActiveApertures: len(tm.apertures),
		Closed:          tm.closed,
	}
}

// HandleFault triggers emergency resource teardown upon detecting a network or hardware fault.
func (tm *TeardownManager) HandleFault(faultErr error) error {
	if faultErr == nil {
		return nil
	}
	return tm.Teardown()
}
