package l3kv

import (
	"context"
	"sync/atomic"
)

// AsyncJob is a tagged background KV store or transfer task.
type AsyncJob struct {
	ID      uint64
	Key     string
	Payload []byte
}

// WatermarkStats holds snapshot telemetry for the watermark manager.
type WatermarkStats struct {
	CurrentSeq     uint64 `json:"current_seq"`
	StaleWatermark uint64 `json:"stale_watermark"`
	CompletedCount uint64 `json:"completed_count"`
	DroppedCount   uint64 `json:"dropped_count"`
}

// AsyncWatermarkManager coordinates asynchronous KV store and transfer tasks, providing
// O(1) task invalidation on cache or session reset (#10729, borrowed from vLLM kv_offload).
// Monotonically advancing an integer watermark on Reset() allows in-flight background worker
// jobs to self-identify as obsolete upon completion and discard themselves silently without
// locks, complex cancellation protocols, or thread-stop races.
type AsyncWatermarkManager struct {
	jobSeq         uint64
	staleWatermark uint64
	completedCount uint64
	droppedCount   uint64
}

// NewAsyncWatermarkManager builds a fresh watermark manager.
func NewAsyncWatermarkManager() *AsyncWatermarkManager {
	return &AsyncWatermarkManager{}
}

// Dispatch allocates a monotonic job sequence ID for an outgoing background storage or transfer task.
func (m *AsyncWatermarkManager) Dispatch(key string, payload []byte) AsyncJob {
	if m == nil {
		return AsyncJob{Key: key, Payload: payload}
	}
	id := atomic.AddUint64(&m.jobSeq, 1)
	return AsyncJob{
		ID:      id,
		Key:     key,
		Payload: payload,
	}
}

// Reset advances the stale watermark threshold to the current job sequence counter.
// Any job dispatched prior to or during this reset will have ID <= staleWatermark
// and will be dropped upon completion.
func (m *AsyncWatermarkManager) Reset() uint64 {
	if m == nil {
		return 0
	}
	current := atomic.LoadUint64(&m.jobSeq)
	for {
		old := atomic.LoadUint64(&m.staleWatermark)
		if current <= old {
			return old
		}
		if atomic.CompareAndSwapUint64(&m.staleWatermark, old, current) {
			return current
		}
	}
}

// IsStale checks whether a job ID is obsolete according to the active watermark threshold.
func (m *AsyncWatermarkManager) IsStale(jobID uint64) bool {
	if m == nil {
		return false
	}
	threshold := atomic.LoadUint64(&m.staleWatermark)
	return jobID <= threshold
}

// Complete evaluates a finished job: if the job is stale (dispatched prior to the latest
// Reset), its result is discarded and Complete returns false. If valid, onCommit is executed
// and Complete returns true.
func (m *AsyncWatermarkManager) Complete(job AsyncJob, onCommit func(job AsyncJob)) bool {
	if m == nil {
		if onCommit != nil {
			onCommit(job)
		}
		return true
	}
	if m.IsStale(job.ID) {
		atomic.AddUint64(&m.droppedCount, 1)
		return false
	}
	if onCommit != nil {
		onCommit(job)
	}
	atomic.AddUint64(&m.completedCount, 1)
	return true
}

// Stats returns current sequence and dropped/completed counts.
func (m *AsyncWatermarkManager) Stats() WatermarkStats {
	if m == nil {
		return WatermarkStats{}
	}
	return WatermarkStats{
		CurrentSeq:     atomic.LoadUint64(&m.jobSeq),
		StaleWatermark: atomic.LoadUint64(&m.staleWatermark),
		CompletedCount: atomic.LoadUint64(&m.completedCount),
		DroppedCount:   atomic.LoadUint64(&m.droppedCount),
	}
}

// AsyncStore wraps a synchronous Store with asynchronous job dispatching and stale watermark reset.
type AsyncStore struct {
	store Store
	wm    *AsyncWatermarkManager
}

// NewAsyncStore wraps store with watermark tracking.
func NewAsyncStore(store Store) *AsyncStore {
	return &AsyncStore{
		store: store,
		wm:    NewAsyncWatermarkManager(),
	}
}

// WatermarkManager exposes the underlying watermark coordinator.
func (s *AsyncStore) WatermarkManager() *AsyncWatermarkManager {
	if s == nil {
		return nil
	}
	return s.wm
}

// Reset advances the watermark so in-flight async writes are discarded on arrival.
func (s *AsyncStore) Reset() uint64 {
	if s == nil || s.wm == nil {
		return 0
	}
	return s.wm.Reset()
}

// PutAsync dispatches a background Put operation to the underlying Store.
// The onComplete callback is executed only if the job has not been rendered stale by Reset().
func (s *AsyncStore) PutAsync(ctx context.Context, key string, payload []byte, onComplete func(err error)) AsyncJob {
	job := s.wm.Dispatch(key, payload)
	go func() {
		var err error
		if s.store != nil {
			err = s.store.Put(ctx, key, payload)
		}
		s.wm.Complete(job, func(j AsyncJob) {
			if onComplete != nil {
				onComplete(err)
			}
		})
	}()
	return job
}
