package agentsched

import (
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Invariant: agent scheduler enforces priority queue ordering and shedder admission fail-closed.
// Invariant: priority queue strictly isolates P0 system, P1 interactive, P2 batch, and P3 speculative tiers.
// Invariant: queue capacity is strictly bounded with non-blocking fail-closed rejection on overflow.

// Task represents a unit of work submitted to the agent thread scheduler.
type Task struct {
	ID           string             `json:"id"`
	Priority     abi.ThreadPriority `json:"priority"`
	Lane         string             `json:"lane"`
	Tree         []string           `json:"tree"`
	AccountID    string             `json:"account_id"`
	TokensNeeded int64              `json:"tokens_needed"`
	SessionID    string             `json:"session_id"`
	EnqueuedAt   time.Time          `json:"enqueued_at"`
	Metadata     map[string]string  `json:"metadata,omitempty"`
}

// PriorityQueue implements a multi-level priority queue with 4 distinct priority tiers (P0..P3)
// and a strict fixed-capacity ceiling (abi.MaxQueueCapacity = 512).
type PriorityQueue struct {
	mu       sync.Mutex
	capacity int
	total    int
	buckets  [4][]*Task
}

// NewPriorityQueue creates a PriorityQueue with capacity cap (defaults to abi.MaxQueueCapacity if <= 0).
func NewPriorityQueue(capacity int) *PriorityQueue {
	if capacity <= 0 {
		capacity = abi.MaxQueueCapacity
	}
	return &PriorityQueue{
		capacity: capacity,
	}
}

// Capacity returns the maximum configured task capacity.
func (pq *PriorityQueue) Capacity() int {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	return pq.capacity
}

// Len returns the total number of enqueued tasks across all tiers.
func (pq *PriorityQueue) Len() int {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	return pq.total
}

// LenPriority returns the number of enqueued tasks for a specific priority tier.
func (pq *PriorityQueue) LenPriority(p abi.ThreadPriority) int {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	if !p.IsValid() {
		return 0
	}
	return len(pq.buckets[p])
}

// Enqueue inserts a task into its corresponding priority tier.
// If the total queue count has reached capacity, it returns abi.ErrQueueFull immediately (fail-closed, non-blocking).
func (pq *PriorityQueue) Enqueue(t *Task) error {
	if t == nil {
		return nil
	}
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if pq.total >= pq.capacity {
		return abi.NewQueueFullError(abi.DefaultRetryAfterQueueFullMS)
	}

	p := t.Priority
	if !p.IsValid() {
		p = abi.ThreadPriorityP2Batch
		t.Priority = p
	}

	if t.EnqueuedAt.IsZero() {
		t.EnqueuedAt = time.Now()
	}

	pq.buckets[p] = append(pq.buckets[p], t)
	pq.total++
	return nil
}

// Peek returns the highest priority task waiting at the head of the queue without removing it.
// If allowP3 is false, speculative (P3) tasks are ignored.
func (pq *PriorityQueue) Peek(allowP3 bool) (*Task, bool) {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	maxTier := abi.ThreadPriorityP3Speculative
	if !allowP3 {
		maxTier = abi.ThreadPriorityP2Batch
	}

	for p := abi.ThreadPriorityP0System; p <= maxTier; p++ {
		if len(pq.buckets[p]) > 0 {
			return pq.buckets[p][0], true
		}
	}
	return nil, false
}

// Dequeue removes and returns the highest-priority task waiting in the queue.
// If allowP3 is false, speculative (P3) tasks are skipped.
func (pq *PriorityQueue) Dequeue(allowP3 bool) (*Task, bool) {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	maxTier := abi.ThreadPriorityP3Speculative
	if !allowP3 {
		maxTier = abi.ThreadPriorityP2Batch
	}

	for p := abi.ThreadPriorityP0System; p <= maxTier; p++ {
		if len(pq.buckets[p]) > 0 {
			task := pq.buckets[p][0]
			pq.buckets[p] = pq.buckets[p][1:]
			pq.total--
			return task, true
		}
	}
	return nil, false
}

// Remove deletes a task with taskID from the queue if present.
func (pq *PriorityQueue) Remove(taskID string) bool {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	for p := 0; p < 4; p++ {
		for i, t := range pq.buckets[p] {
			if t.ID == taskID {
				pq.buckets[p] = append(pq.buckets[p][:i], pq.buckets[p][i+1:]...)
				pq.total--
				return true
			}
		}
	}
	return false
}

// DropP3 sheds all speculative P3 tasks currently enqueued and returns the count dropped.
func (pq *PriorityQueue) DropP3() int {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	dropped := len(pq.buckets[abi.ThreadPriorityP3Speculative])
	pq.buckets[abi.ThreadPriorityP3Speculative] = nil
	pq.total -= dropped
	return dropped
}
