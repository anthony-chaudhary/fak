// agent_invariants.go — O(1) agent lifecycle invariants, thread quotas, and admission backpressure.
//
// Frozen ABI contracts: priority tiers, queue/pool ceilings, structured admission
// backpressure, and retention byte ceilings.

package abi

import "fmt"

// ThreadPriority defines the scheduling priority tier for agent execution threads and workers.
type ThreadPriority uint8

const (
	// ThreadPriorityP0System is the highest tier: kernel/system control, health checks, heartbeats, lease renewals, and emergency reclaim.
	ThreadPriorityP0System ThreadPriority = iota

	// ThreadPriorityP1Interactive is for user-facing interactive queries, CLI sessions, and synchronous RPC/tool calls.
	ThreadPriorityP1Interactive

	// ThreadPriorityP2Batch is for autonomous background workers, bulk sweeps, subagent task loops, and scheduled maintenance.
	ThreadPriorityP2Batch

	// ThreadPriorityP3Speculative is for opportunistic speculative work, cache warming, and idle-time indexing; strictly yieldable and preemptible.
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

// Admission queue depth bound, default per-tier worker-pool ceilings (in K), and the global worker-total ceiling.
const (
	// MaxQueueCapacity is the hard bound on pending admission queue depth; enqueues beyond it fail closed with ErrQueueFull.
	MaxQueueCapacity = 512

	// Default worker pool ceilings (K) per priority tier.
	DefaultWorkerPoolP0System      = 4
	DefaultWorkerPoolP1Interactive = 16
	DefaultWorkerPoolP2Batch       = 32
	DefaultWorkerPoolP3Speculative = 8

	// DefaultMaxTotalWorkers is the global default worker ceiling across all tiers.
	DefaultMaxTotalWorkers = 64
)

const (
	// MaxJournalSizeBytes is the hard ceiling for a single rolling journal (64 MiB); segment rotation or compaction triggers when exceeded.
	MaxJournalSizeBytes int64 = 64 * 1024 * 1024

	// MaxScratchStorageBytes is the total workspace scratch storage ceiling (512 MiB).
	MaxScratchStorageBytes int64 = 512 * 1024 * 1024

	// MaxPerRunScratchBytes is the scratch storage ceiling for an individual agent run (50 MiB).
	MaxPerRunScratchBytes int64 = 50 * 1024 * 1024

	// MaxInMemoryCacheBytes is the ceiling for volatile in-memory cache structures (256 MiB).
	MaxInMemoryCacheBytes int64 = 256 * 1024 * 1024

	// MaxPerRunLogBytes is the cap on per-run standard capture log size (10 MiB).
	MaxPerRunLogBytes int64 = 10 * 1024 * 1024
)

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

// AdmissionError is a structured backpressure refusal with a machine-readable code and retry hint.
type AdmissionError struct {
	Code         string `json:"code"`
	Message      string `json:"message"`
	RetryAfterMS int64  `json:"retry_after_ms"`
}

// Error renders the machine-readable code, message, and optional retry hint as the error string.
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

// NewAdmissionError builds a structured backpressure refusal with an explicit code, message, and retry hint.
func NewAdmissionError(code, message string, retryAfterMS int64) *AdmissionError {
	return &AdmissionError{
		Code:         code,
		Message:      message,
		RetryAfterMS: retryAfterMS,
	}
}

// NewQueueFullError builds the queue-saturated refusal, defaulting the retry hint to DefaultRetryAfterQueueFullMS.
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

// NewResourceConstrainedError builds the resource-envelope refusal, defaulting empty messages and non-positive retry hints.
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
