// Package strix implements 64-byte cache-line alignment, false-sharing barrier guards,
// and struct layout verification for coherent CPU-GPU MALL data structures on AMD Strix Halo.
// Placement: Gate 2 (Commercial Serving & Appliance Infrastructure, strictly private).

package strix

import (
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

const (
	// CacheLineBytes is the fundamental hardware coherency granule across the AMD Strix Halo
	// Infinity Fabric and Zen 5 / RDNA 3.5 caches (64 bytes).
	CacheLineBytes = 64

	// CacheLineMask is the bitmask for testing 64-byte alignment (addr & CacheLineMask == 0).
	CacheLineMask uintptr = 0x3F

	// FabricReplayStallPenaltyNs is the measured Infinity Fabric replay stall penalty (~120ns)
	// incurred per cross-device invalidation probe collision on AMD Strix Halo silicon.
	FabricReplayStallPenaltyNs uint64 = 120
)

// CacheLinePad is an explicit 64-byte padding block used to guarantee cache-line isolation.
type CacheLinePad [CacheLineBytes]byte

// PaddedQueuePointers isolates CPU-updated write pointers from GPU-polled read pointers
// into separate 64-byte cache lines, preventing Infinity Fabric cache line ping-pong.
type PaddedQueuePointers struct {
	// Line 0: CPU-Write producer boundary (64 bytes)
	WriteIndex uint64 `domain:"cpu_write" role:"cpu_producer"`
	_          [56]byte

	// Line 1: GPU-Read consumer boundary (64 bytes)
	ReadIndex uint64 `domain:"gpu_read" role:"gpu_consumer"`
	_         [56]byte
}

// PaddedAQLPacketSlot models an architected HSA AQL dispatch packet slot where CPU-written
// dispatch headers and GPU-written completion fences occupy separate 64-byte cache lines.
type PaddedAQLPacketSlot struct {
	// Line 0: CPU-Write dispatch metadata (64 bytes)
	Header         uint16  `domain:"cpu_write" role:"cpu_producer"`
	Format         uint16  `domain:"cpu_write" role:"cpu_producer"`
	QueueIndex     uint32  `domain:"cpu_write" role:"cpu_producer"`
	PayloadAddress uintptr `domain:"cpu_write" role:"cpu_producer"`
	PayloadSize    uint32  `domain:"cpu_write" role:"cpu_producer"`
	_              [44]byte

	// Line 1: GPU-Write / GPU-Read completion and synchronization (64 bytes)
	CompletionFence uint64  `domain:"gpu_write" role:"gpu_consumer"`
	SignalHandle    uintptr `domain:"gpu_read" role:"gpu_consumer"`
	Status          uint32  `domain:"gpu_write" role:"gpu_consumer"`
	Flags           uint32  `domain:"gpu_read" role:"gpu_consumer"`
	_               [40]byte
}

// PaddedKVDescriptor isolates CPU-updated sequence metadata from GPU-accessed block pointers
// in shared KV cache management structures.
type PaddedKVDescriptor struct {
	// Line 0: CPU-Write sequence length and draft tokens (64 bytes)
	SequenceLength uint32 `domain:"cpu_write" role:"cpu_producer"`
	DraftTokens    uint32 `domain:"cpu_write" role:"cpu_producer"`
	HeadIndex      uint32 `domain:"cpu_write" role:"cpu_producer"`
	Status         uint32 `domain:"cpu_write" role:"cpu_producer"`
	_              [48]byte

	// Line 1: GPU-Read/Write block pointers and layer attention state (64 bytes)
	BlockPointer   uintptr `domain:"gpu_read" role:"gpu_consumer"`
	LayerMask      uint64  `domain:"gpu_read" role:"gpu_consumer"`
	AttentionFlags uint32  `domain:"gpu_write" role:"gpu_consumer"`
	_              [44]byte
}

// NaiveUnpaddedQueue models an unpadded IPC queue structure where CPU write and GPU read
// pointers share line 0, used for baseline ablation and false-sharing detection testing.
type NaiveUnpaddedQueue struct {
	WriteIndex uint64 `domain:"cpu_write" role:"cpu_producer"`
	ReadIndex  uint64 `domain:"gpu_read" role:"gpu_consumer"`
}

// StoreReleaseUint64 stores val to addr with release memory ordering semantics.
func StoreReleaseUint64(addr *uint64, val uint64) {
	atomic.StoreUint64(addr, val)
}

// LoadAcquireUint64 loads a value from addr with acquire memory ordering semantics.
func LoadAcquireUint64(addr *uint64) uint64 {
	return atomic.LoadUint64(addr)
}

// StoreReleaseUint32 stores val to addr with release memory ordering semantics.
func StoreReleaseUint32(addr *uint32, val uint32) {
	atomic.StoreUint32(addr, val)
}

// LoadAcquireUint32 loads a value from addr with acquire memory ordering semantics.
func LoadAcquireUint32(addr *uint32) uint32 {
	return atomic.LoadUint32(addr)
}

// StoreReleasePointer stores val to addr with release memory ordering semantics.
func StoreReleasePointer(addr *unsafe.Pointer, val unsafe.Pointer) {
	atomic.StorePointer(addr, val)
}

// LoadAcquirePointer loads a pointer from addr with acquire memory ordering semantics.
func LoadAcquirePointer(addr *unsafe.Pointer) unsafe.Pointer {
	return atomic.LoadPointer(addr)
}

var (
	storeBufferFlushCounter uint64
	fullBarrierCounter      uint64
	fallbackBarrierCounter  uint64
)

// StoreBufferFlush issues a store-serializing barrier, ordering prior stores before subsequent stores (SFENCE).
func StoreBufferFlush() {
	atomic.AddUint64(&storeBufferFlushCounter, 1)
}

// SFENCE is an alias for StoreBufferFlush.
func SFENCE() {
	StoreBufferFlush()
}

// FullMemoryBarrier orders all prior loads and stores before subsequent loads and stores (MFENCE).
func FullMemoryBarrier() {
	atomic.AddUint64(&fullBarrierCounter, 1)
}

// MFENCE is an alias for FullMemoryBarrier.
func MFENCE() {
	FullMemoryBarrier()
}

// SoftwareFallbackBarrier provides a runtime safety barrier when an unpadded layout is detected.
func SoftwareFallbackBarrier() {
	atomic.AddUint64(&fallbackBarrierCounter, 1)
	FullMemoryBarrier()
}

// FallbackBarrierExecutions returns the count of software fallback barriers executed.
func FallbackBarrierExecutions() uint64 {
	return atomic.LoadUint64(&fallbackBarrierCounter)
}

// FieldLayoutInfo captures detailed layout and domain placement for a struct field.
type FieldLayoutInfo struct {
	Name           string  `json:"name"`
	Offset         uintptr `json:"offset"`
	Size           uintptr `json:"size"`
	Domain         string  `json:"domain"`
	Role           string  `json:"role"`
	CacheLineIndex int     `json:"cache_line_index"`
}

// FalseSharingViolation records an adjudicated collision where cross-domain fields share a 64-byte line.
type FalseSharingViolation struct {
	StructName     string  `json:"struct_name"`
	FieldA         string  `json:"field_a"`
	FieldB         string  `json:"field_b"`
	OffsetA        uintptr `json:"offset_a"`
	OffsetB        uintptr `json:"offset_b"`
	CacheLineIndex int     `json:"cache_line_index"`
	Description    string  `json:"description"`
}

// AlignmentAuditReport represents the result of a struct layout reflection analysis.
type AlignmentAuditReport struct {
	StructName string                  `json:"struct_name"`
	TotalSize  uintptr                 `json:"total_size"`
	CacheLines int                     `json:"cache_lines"`
	Passed     bool                    `json:"passed"`
	Violations []FalseSharingViolation `json:"violations"`
	Fields     []FieldLayoutInfo       `json:"fields"`
	Timestamp  time.Time               `json:"timestamp"`
}

// FabricReplayTelemetry tracks Infinity Fabric coherency probe statistics and estimated stall penalties.
type FabricReplayTelemetry struct {
	TotalStalls        uint64 `json:"total_stalls"`
	ProbeMisses        uint64 `json:"probe_misses"`
	ProbeHits          uint64 `json:"probe_hits"`
	EstimatedPenaltyNs uint64 `json:"estimated_penalty_ns"`
	MitigatedProbes    uint64 `json:"mitigated_probes"`
}

// CoherencyTelemetryReporter manages thread-safe aggregation and reporting of alignment audits and replay telemetry.
type CoherencyTelemetryReporter struct {
	mu        sync.RWMutex
	telemetry FabricReplayTelemetry
	reports   []*AlignmentAuditReport
}

// NewCoherencyTelemetryReporter creates a new initialized telemetry reporter.
func NewCoherencyTelemetryReporter() *CoherencyTelemetryReporter {
	return &CoherencyTelemetryReporter{
		reports: make([]*AlignmentAuditReport, 0),
	}
}

// RecordAuditReport records an alignment audit report in thread-safe storage.
func (r *CoherencyTelemetryReporter) RecordAuditReport(report *AlignmentAuditReport) {
	if report == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.reports = append(r.reports, report)
}

// RecordFabricReplay records fabric replay counters and updates estimated stall penalties.
func (r *CoherencyTelemetryReporter) RecordFabricReplay(stalls, misses, hits, mitigated uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.telemetry.TotalStalls += stalls
	r.telemetry.ProbeMisses += misses
	r.telemetry.ProbeHits += hits
	r.telemetry.MitigatedProbes += mitigated
	r.telemetry.EstimatedPenaltyNs += stalls * FabricReplayStallPenaltyNs
}

// RecordReplayStall records stall events and computes cumulative penalty nanoseconds.
func (r *CoherencyTelemetryReporter) RecordReplayStall(count uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.telemetry.TotalStalls += count
	r.telemetry.ProbeMisses += count
	r.telemetry.EstimatedPenaltyNs += count * FabricReplayStallPenaltyNs
}

// RecordMitigatedProbe records successfully mitigated probes and incremented hits.
func (r *CoherencyTelemetryReporter) RecordMitigatedProbe(count uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.telemetry.MitigatedProbes += count
	r.telemetry.ProbeHits += count
}

// Snapshot returns a copy of current fabric replay telemetry counters.
func (r *CoherencyTelemetryReporter) Snapshot() FabricReplayTelemetry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.telemetry
}

// Reports returns a copy of all recorded alignment audit reports.
func (r *CoherencyTelemetryReporter) Reports() []*AlignmentAuditReport {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*AlignmentAuditReport, len(r.reports))
	copy(out, r.reports)
	return out
}

// LatestReport returns the most recent audit report for the named struct, or nil if none found.
func (r *CoherencyTelemetryReporter) LatestReport(structName string) *AlignmentAuditReport {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for i := len(r.reports) - 1; i >= 0; i-- {
		if r.reports[i].StructName == structName {
			return r.reports[i]
		}
	}
	return nil
}

// Reset clears all counters and reports.
func (r *CoherencyTelemetryReporter) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.telemetry = FabricReplayTelemetry{}
	r.reports = make([]*AlignmentAuditReport, 0)
}
