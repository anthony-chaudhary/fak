// agent_invariants.go — O(1) agent lifecycle invariants, thread quotas, and admission backpressure.
//
// This file formalizes the frozen ABI contracts for agent lifecycle management:
//   1. Thread Priority Hierarchy: P0 (System/Control), P1 (Interactive), P2 (Batch), P3 (Speculative).
//   2. Workload Ceilings & Queue Limits: MaxQueueCapacity (512), tier worker pool bounds.
//   3. Fail-Closed Admission Backpressure: Structured AdmissionError types with retry_after_ms.
//   4. Retention & Growth Bounds: Storage and memory byte ceilings for journals, scratch, and cache.

package abi

import "fmt"

// ThreadPriority defines the scheduling priority tier for agent execution threads and workers.
type ThreadPriority uint8

const (
	// ThreadPriorityP0System is the highest priority tier reserved for kernel/system control,
	// health checks, heartbeats, liveness probes, lease renewals, and emergency resource reclaim.
	ThreadPriorityP0System ThreadPriority = iota

	// ThreadPriorityP1Interactive is reserved for user-facing interactive queries, CLI sessions,
	// direct synchronous RPC handling, and interactive tool calls.
	ThreadPriorityP1Interactive

	// ThreadPriorityP2Batch is for autonomous background worker jobs, bulk sweeps, autonomous
	// subagent task loops, and scheduled maintenance tasks.
	ThreadPriorityP2Batch

	// ThreadPriorityP3Speculative is for opportunistic speculative execution, background cache warming,
	// branch pre-computation, and idle-time indexing. Strictly yieldable and preemptible under pressure.
	ThreadPriorityP3Speculative
)

// String returns the canonical name of the thread priority tier.
func (p ThreadPriority) String() string {
	switch p {
	case ThreadPriorityP0System:
		return "P0_SYSTEM"
	case ThreadPriorityP1Interactive:
		return "P1_INTERACTIVE"
	case ThreadPriorityP2Batch:
		return "P2_BATCH"
	case ThreadPriorityP3Speculative:
		return "P3_SPECULATIVE"
	default:
		return fmt.Sprintf("THREAD_PRIORITY_%d", p)
	}
}

// Rank returns the integer priority rank (0 is highest priority, 3 is lowest).
func (p ThreadPriority) Rank() int {
	return int(p)
}

// IsValid reports whether p is a declared priority tier.
func (p ThreadPriority) IsValid() bool {
	return p <= ThreadPriorityP3Speculative
}

// Standard capacity, queue, and worker pool bounds.
const (
	// MaxQueueCapacity is the hard upper bound on pending admission queue depth (512).
	// Enqueue attempts beyond this limit fail closed immediately with ErrQueueFull.
	MaxQueueCapacity = 512

	// Default worker pool ceilings (K) per priority tier.
	DefaultWorkerPoolP0System      = 4
	DefaultWorkerPoolP1Interactive = 16
	DefaultWorkerPoolP2Batch       = 32
	DefaultWorkerPoolP3Speculative = 8

	// DefaultMaxTotalWorkers is the global default worker ceiling across all tiers.
	DefaultMaxTotalWorkers = 64
)

// Retention & Growth Bounds — hard byte ceilings for journals, scratch storage, and in-memory caches.
const (
	// MaxJournalSizeBytes is the hard ceiling for a single rolling event/run journal (64 MiB).
	// When exceeded, segment rotation or compaction is triggered.
	MaxJournalSizeBytes int64 = 64 * 1024 * 1024 // 64 MiB

	// MaxScratchStorageBytes is the total workspace scratch storage ceiling (512 MiB).
	MaxScratchStorageBytes int64 = 512 * 1024 * 1024 // 512 MiB

	// MaxPerRunScratchBytes is the scratch storage ceiling for an individual agent run (50 MiB).
	MaxPerRunScratchBytes int64 = 50 * 1024 * 1024 // 50 MiB

	// MaxInMemoryCacheBytes is the ceiling for volatile in-memory cache structures (256 MiB).
	MaxInMemoryCacheBytes int64 = 256 * 1024 * 1024 // 256 MiB

	// MaxPerRunLogBytes is the cap on per-run standard capture log size (10 MiB).
	MaxPerRunLogBytes int64 = 10 * 1024 * 1024 // 10 MiB
)

// Standard error codes and default retry delays for admission backpressure.
const (
	// AdmissionCodeQueueFull is emitted when the admission queue capacity (512) is saturated.
	AdmissionCodeQueueFull = "ERR_QUEUE_FULL"

	// AdmissionCodeResourceConstrained is emitted when host thread pools, memory, or storage envelopes saturate.
	AdmissionCodeResourceConstrained = "ERR_RESOURCE_CONSTRAINED"

	// DefaultRetryAfterQueueFullMS is the default retry delay when the admission queue is saturated.
	DefaultRetryAfterQueueFullMS int64 = 50

	// DefaultRetryAfterResourceConstrainedMS is the default retry delay when resource envelopes saturate.
	DefaultRetryAfterResourceConstrainedMS int64 = 100
)

// AdmissionError represents a structured, fail-closed backpressure refusal.
// It serializes cleanly to JSON for wire protocols and HTTP status propagation.
type AdmissionError struct {
	Code         string `json:"code"`
	Message      string `json:"message"`
	RetryAfterMS int64  `json:"retry_after_ms"`
}

// Error implements the standard error interface.
func (e *AdmissionError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.RetryAfterMS > 0 {
		return fmt.Sprintf("%s: %s (retry after %dms)", e.Code, e.Message, e.RetryAfterMS)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Is supports errors.Is checks matching target errors by Code.
func (e *AdmissionError) Is(target error) bool {
	if target == nil {
		return e == nil
	}
	t, ok := target.(*AdmissionError)
	if !ok {
		return false
	}
	if e == t {
		return true
	}
	return e.Code == t.Code
}

// Standard sentinel structured admission errors.
var (
	// ErrQueueFull is emitted when the queue reaches MaxQueueCapacity (512).
	ErrQueueFull = &AdmissionError{
		Code:         AdmissionCodeQueueFull,
		Message:      "admission queue capacity exceeded",
		RetryAfterMS: DefaultRetryAfterQueueFullMS,
	}

	// ErrResourceConstrained is emitted when host thread pools or resource envelopes saturate.
	ErrResourceConstrained = &AdmissionError{
		Code:         AdmissionCodeResourceConstrained,
		Message:      "host resource envelope saturated",
		RetryAfterMS: DefaultRetryAfterResourceConstrainedMS,
	}
)

// NewAdmissionError returns a new structured AdmissionError.
func NewAdmissionError(code, message string, retryAfterMS int64) *AdmissionError {
	return &AdmissionError{
		Code:         code,
		Message:      message,
		RetryAfterMS: retryAfterMS,
	}
}

// NewQueueFullError creates a structured ErrQueueFull with a custom retry delay.
func NewQueueFullError(retryAfterMS int64) *AdmissionError {
	if retryAfterMS <= 0 {
		retryAfterMS = DefaultRetryAfterQueueFullMS
	}
	return &AdmissionError{
		Code:         AdmissionCodeQueueFull,
		Message:      "admission queue capacity exceeded",
		RetryAfterMS: retryAfterMS,
	}
}

// NewResourceConstrainedError creates a structured ErrResourceConstrained with custom message and retry delay.
func NewResourceConstrainedError(message string, retryAfterMS int64) *AdmissionError {
	if retryAfterMS <= 0 {
		retryAfterMS = DefaultRetryAfterResourceConstrainedMS
	}
	if message == "" {
		message = "host resource envelope saturated"
	}
	return &AdmissionError{
		Code:         AdmissionCodeResourceConstrained,
		Message:      message,
		RetryAfterMS: retryAfterMS,
	}
}
