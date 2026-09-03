package metalgemm

// Prior-art: carloslfu/slotstream / MLX (https://github.com/carloslfu/slotstream)

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"
)

// Constants defining canonical Apple Silicon MoE expert streaming defaults.
//
// In Qwen 3.8 MoE on Apple Silicon (105 GB 4-bit weights on 16-36GB Macs), eager
// mmap causes catastrophic macOS swap thrashing (48 GB swap) and host kernel freezes.
// Slotstream and FAK solve this by keeping only the 3.8 GB dense trunk resident in RAM,
// streaming active routed 4-bit experts on-demand from NVMe SSD through a preallocated
// unified memory slot pool via a queue-depth 32 (QD32) asynchronous worker pool.
const (
	// DefaultQueueDepth is the canonical queue depth of 32 worker lanes (QD32)
	// engineered to saturate Apple Silicon NVMe SSD pread bandwidth (10-17+ GB/s).
	DefaultQueueDepth = 32

	// DefaultSlotCount is the default capacity of preallocated unified memory slots.
	// In Qwen 3.8 MoE, typically 8-10 routed experts are active per token, so
	// a pool of 32-64 slots provides generous headroom and temporal reuse.
	DefaultSlotCount = 32

	// DefaultSlotBytes is the default buffer size for each expert slot.
	// In Qwen 3.8 MoE 4-bit, an expert record (gate/up/down tensors) is ~2.76 MB.
	// 3 MiB provides full alignment and headroom for 4-bit expert chunks.
	DefaultSlotBytes = 3 * 1024 * 1024
)

// Common error definitions.
var (
	ErrQueueClosed              = errors.New("expert stream queue is closed")
	ErrBatchExceedsPoolCapacity = errors.New("batch size exceeds total slot pool capacity")
	ErrExpertNotFound           = errors.New("expert location not found")
	ErrSizeExceedsSlot          = errors.New("expert size exceeds preallocated slot capacity")
	ErrInvalidSize              = errors.New("expert size must be greater than zero")
)

// ExpertLocation specifies the file offset and size of an expert's weights.
type ExpertLocation struct {
	Offset int64
	Size   int64
}

// ExpertRequest represents a request to stream an expert into a slot.
// If Size is <= 0, the queue looks up the registered ExpertLocation for ExpertID.
type ExpertRequest struct {
	ExpertID int
	Offset   int64
	Size     int64
}

// StreamConfig configures the ExpertStreamQueue.
type StreamConfig struct {
	// QueueDepth is the number of concurrent worker lanes (defaults to DefaultQueueDepth = 32).
	QueueDepth int

	// SlotCount is the number of preallocated unified memory slots (defaults to DefaultSlotCount = 32).
	SlotCount int

	// SlotBytes is the byte capacity of each preallocated slot buffer (defaults to DefaultSlotBytes = 3 MiB).
	SlotBytes int

	// Reader is the underlying source of weights supporting concurrent positional reads (pread).
	Reader io.ReaderAt

	// ExpertLocations optionally maps expertID -> file location (offset and size).
	ExpertLocations map[int]ExpertLocation
}

// StreamMetrics captures an atomic snapshot of queue and slot pool telemetry.
type StreamMetrics struct {
	BytesTransferred uint64 `json:"bytes_transferred"`
	ActiveQueueDepth int32  `json:"active_queue_depth"`
	PeakQueueDepth   int32  `json:"peak_queue_depth"`
	SlotHits         uint64 `json:"slot_hits"`
	SlotMisses       uint64 `json:"slot_misses"`
	SlotEvictions    uint64 `json:"slot_evictions"`
	TotalRequests    uint64 `json:"total_requests"`
	TotalReads       uint64 `json:"total_reads"`
}

type queueMetrics struct {
	bytesTransferred atomic.Uint64
	activeQueueDepth atomic.Int32
	peakQueueDepth   atomic.Int32
	slotHits         atomic.Uint64
	slotMisses       atomic.Uint64
	slotEvictions    atomic.Uint64
	totalRequests    atomic.Uint64
	totalReads       atomic.Uint64
}

type slotState int

const (
	slotFree slotState = iota
	slotLoading
	slotReady
)

// slot represents one preallocated unified memory buffer.
type slot struct {
	id         int
	buf        []byte
	expertID   int
	validBytes int
	state      slotState
	refCount   int32 // active leases
	lastUsed   int64 // monotonic sequence for LRU eviction
}

type loadingState struct {
	slot *slot
	done chan struct{}
	err  error
}

// slotPool manages preallocated reusable buffers with LRU eviction and lease lifecycles.
type slotPool struct {
	mu        sync.Mutex
	slots     []*slot
	slotCount int
	slotBytes int
	resident  map[int]*slot         // expertID -> ready slot
	loading   map[int]*loadingState // expertID -> in-flight loading state
	seq       int64
	waiters   []chan struct{}
	closed    bool
	metrics   *queueMetrics
}

func newSlotPool(count, bytesPerSlot int, metrics *queueMetrics) *slotPool {
	slots := make([]*slot, count)
	for i := 0; i < count; i++ {
		slots[i] = &slot{
			id:       i,
			buf:      make([]byte, bytesPerSlot),
			expertID: -1,
			state:    slotFree,
		}
	}
	return &slotPool{
		slots:     slots,
		slotCount: count,
		slotBytes: bytesPerSlot,
		resident:  make(map[int]*slot, count),
		loading:   make(map[int]*loadingState),
		metrics:   metrics,
	}
}

func (p *slotPool) nextSeq() int64 {
	p.seq++
	return p.seq
}

func (p *slotPool) findAvailableSlotLocked() *slot {
	// 1. Prefer free slot (never loaded or reset).
	for _, s := range p.slots {
		if s.state == slotFree {
			return s
		}
	}

	// 2. Find unleased slot (refCount == 0, state == slotReady) with minimum lastUsed (LRU).
	var oldest *slot
	var oldestSeq int64 = math.MaxInt64
	for _, s := range p.slots {
		if s.state == slotReady && s.refCount == 0 {
			if s.lastUsed < oldestSeq {
				oldestSeq = s.lastUsed
				oldest = s
			}
		}
	}
	return oldest
}

func (p *slotPool) notifyWaitersLocked() {
	if len(p.waiters) > 0 {
		w := p.waiters[0]
		p.waiters = p.waiters[1:]
		select {
		case w <- struct{}{}:
		default:
		}
	}
}

func (p *slotPool) releaseSlot(s *slot) {
	p.mu.Lock()
	defer p.mu.Unlock()

	s.refCount--
	if s.refCount <= 0 {
		s.refCount = 0
		p.notifyWaitersLocked()
	}
}

func (p *slotPool) close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	for _, w := range p.waiters {
		select {
		case w <- struct{}{}:
		default:
		}
	}
	p.waiters = nil
}

// SlotLease provides leased access to an expert resident in unified memory.
// While held, the underlying slot cannot be evicted or overwritten.
type SlotLease struct {
	slot       *slot
	expertID   int
	validBytes int
	pool       *slotPool
	released   atomic.Bool
}

// ExpertID returns the expert ID associated with this lease.
func (l *SlotLease) ExpertID() int {
	return l.expertID
}

// SlotID returns the physical slot index in the pool.
func (l *SlotLease) SlotID() int {
	return l.slot.id
}

// Bytes returns a slice over the valid preloaded weight bytes.
// If the lease has been released, it returns nil.
func (l *SlotLease) Bytes() []byte {
	if l.released.Load() {
		return nil
	}
	return l.slot.buf[:l.validBytes]
}

// IsReleased reports whether the lease has already been released.
func (l *SlotLease) IsReleased() bool {
	return l.released.Load()
}

// Release releases the lease on the slot, decrementing its reference count.
// Once all leases on a slot are released, it becomes eligible for LRU recycling.
// Safe to call multiple times (idempotent).
func (l *SlotLease) Release() {
	if !l.released.CompareAndSwap(false, true) {
		return
	}
	l.pool.releaseSlot(l.slot)
}

// ReleaseLeases releases a slice of slot leases, ignoring nil entries.
func ReleaseLeases(leases []*SlotLease) {
	for _, l := range leases {
		if l != nil {
			l.Release()
		}
	}
}

func newLease(s *slot, pool *slotPool) *SlotLease {
	return &SlotLease{
		slot:       s,
		expertID:   s.expertID,
		validBytes: s.validBytes,
		pool:       pool,
	}
}

type workerJob struct {
	ctx    context.Context
	reader io.ReaderAt
	offset int64
	size   int64
	dst    []byte
	done   chan workerResult
}

type workerResult struct {
	n   int
	err error
}

// ExpertStreamQueue manages a QD32 worker pool and unified memory slot pool for
// asynchronous pread-based expert streaming.
type ExpertStreamQueue struct {
	cfg        StreamConfig
	queueDepth int
	reader     io.ReaderAt
	readerMu   sync.RWMutex

	// Worker pool
	jobChan   chan workerJob
	closeChan chan struct{}
	workerWg  sync.WaitGroup
	closed    atomic.Bool

	// Slot pool
	slotPool *slotPool

	// Expert location catalog
	locations map[int]ExpertLocation
	locMu     sync.RWMutex

	// Metrics
	metrics queueMetrics
}

// NewExpertStreamQueue initializes an asynchronous expert streaming queue.
func NewExpertStreamQueue(cfg StreamConfig) (*ExpertStreamQueue, error) {
	qd := cfg.QueueDepth
	if qd <= 0 {
		qd = DefaultQueueDepth
	}

	slotCount := cfg.SlotCount
	if slotCount <= 0 {
		slotCount = DefaultSlotCount
	}

	slotBytes := cfg.SlotBytes
	if slotBytes <= 0 {
		slotBytes = DefaultSlotBytes
	}

	q := &ExpertStreamQueue{
		cfg:        cfg,
		queueDepth: qd,
		reader:     cfg.Reader,
		jobChan:    make(chan workerJob, qd),
		closeChan:  make(chan struct{}),
		locations:  make(map[int]ExpertLocation),
	}
	q.slotPool = newSlotPool(slotCount, slotBytes, &q.metrics)

	if cfg.ExpertLocations != nil {
		for k, v := range cfg.ExpertLocations {
			q.locations[k] = v
		}
	}

	for i := 0; i < qd; i++ {
		q.workerWg.Add(1)
		go q.workerLoop(i)
	}

	return q, nil
}

func (q *ExpertStreamQueue) workerLoop(laneID int) {
	defer q.workerWg.Done()
	for {
		select {
		case <-q.closeChan:
			return
		case job, ok := <-q.jobChan:
			if !ok {
				return
			}
			q.executeJob(job)
		}
	}
}

func (q *ExpertStreamQueue) executeJob(job workerJob) {
	if job.ctx.Err() != nil {
		job.done <- workerResult{n: 0, err: job.ctx.Err()}
		return
	}

	q.metrics.activeQueueDepth.Add(1)
	for {
		cur := q.metrics.activeQueueDepth.Load()
		peak := q.metrics.peakQueueDepth.Load()
		if cur <= peak || q.metrics.peakQueueDepth.CompareAndSwap(peak, cur) {
			break
		}
	}

	n, err := job.reader.ReadAt(job.dst[:job.size], job.offset)
	q.metrics.activeQueueDepth.Add(-1)

	if n == int(job.size) && errors.Is(err, io.EOF) {
		err = nil
	} else if n < int(job.size) && errors.Is(err, io.EOF) {
		err = io.ErrUnexpectedEOF
	}

	if n > 0 {
		q.metrics.bytesTransferred.Add(uint64(n))
	}
	q.metrics.totalReads.Add(1)

	job.done <- workerResult{n: n, err: err}
}

func (q *ExpertStreamQueue) dispatchReadAsync(ctx context.Context, slot *slot, offset, size int64) (<-chan workerResult, error) {
	q.readerMu.RLock()
	r := q.reader
	q.readerMu.RUnlock()
	if r == nil {
		return nil, errors.New("no reader configured for expert streaming")
	}

	done := make(chan workerResult, 1)
	job := workerJob{
		ctx:    ctx,
		reader: r,
		offset: offset,
		size:   size,
		dst:    slot.buf,
		done:   done,
	}

	select {
	case <-q.closeChan:
		return nil, ErrQueueClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	case q.jobChan <- job:
		return done, nil
	}
}

// SetReader configures the underlying io.ReaderAt source.
func (q *ExpertStreamQueue) SetReader(r io.ReaderAt) {
	q.readerMu.Lock()
	defer q.readerMu.Unlock()
	q.reader = r
}

// Reader returns the current io.ReaderAt source.
func (q *ExpertStreamQueue) Reader() io.ReaderAt {
	q.readerMu.RLock()
	defer q.readerMu.RUnlock()
	return q.reader
}

// RegisterLocation registers the offset and size for a given expert ID.
func (q *ExpertStreamQueue) RegisterLocation(expertID int, offset, size int64) {
	q.locMu.Lock()
	defer q.locMu.Unlock()
	q.locations[expertID] = ExpertLocation{Offset: offset, Size: size}
}

// Location retrieves the registered location for an expert ID.
func (q *ExpertStreamQueue) Location(expertID int) (ExpertLocation, bool) {
	q.locMu.RLock()
	defer q.locMu.RUnlock()
	loc, ok := q.locations[expertID]
	return loc, ok
}

// StreamExpert streams a single requested expert ID into a slot and returns its lease.
func (q *ExpertStreamQueue) StreamExpert(ctx context.Context, req ExpertRequest) (*SlotLease, error) {
	if q.closed.Load() {
		return nil, ErrQueueClosed
	}
	q.metrics.totalRequests.Add(1)

	offset := req.Offset
	size := req.Size
	if size <= 0 {
		loc, ok := q.Location(req.ExpertID)
		if !ok {
			return nil, fmt.Errorf("%w: expert %d has no registered location", ErrExpertNotFound, req.ExpertID)
		}
		offset = loc.Offset
		size = loc.Size
	}

	if size <= 0 {
		return nil, ErrInvalidSize
	}
	if offset < 0 {
		return nil, errors.New("expert offset cannot be negative")
	}
	if size > int64(q.slotPool.slotBytes) {
		return nil, fmt.Errorf("%w: requested %d bytes, slot capacity is %d bytes",
			ErrSizeExceedsSlot, size, q.slotPool.slotBytes)
	}

	return q.acquireExpert(ctx, req.ExpertID, offset, size)
}

// StreamExperts streams a slice of expert IDs using their registered locations.
func (q *ExpertStreamQueue) StreamExperts(ctx context.Context, expertIDs []int) ([]*SlotLease, error) {
	reqs := make([]ExpertRequest, len(expertIDs))
	for i, id := range expertIDs {
		reqs[i] = ExpertRequest{ExpertID: id}
	}
	return q.StreamBatch(ctx, reqs)
}

// StreamBatch streams requested expert IDs concurrently across the worker lanes into slots.
// Leases are returned in the exact order of reqs. If any error occurs, any acquired leases
// in this batch are safely released.
func (q *ExpertStreamQueue) StreamBatch(ctx context.Context, reqs []ExpertRequest) ([]*SlotLease, error) {
	if q.closed.Load() {
		return nil, ErrQueueClosed
	}
	if len(reqs) == 0 {
		return nil, nil
	}

	// Validate distinct count against pool capacity to detect deadlocks early.
	distinct := make(map[int]struct{}, len(reqs))
	for _, r := range reqs {
		distinct[r.ExpertID] = struct{}{}
	}
	if len(distinct) > q.slotPool.slotCount {
		return nil, fmt.Errorf("%w: batch requires %d distinct slots, but pool capacity is %d",
			ErrBatchExceedsPoolCapacity, len(distinct), q.slotPool.slotCount)
	}

	leases := make([]*SlotLease, len(reqs))
	var (
		wg       sync.WaitGroup
		firstErr error
		errMu    sync.Mutex
	)

	for i := range reqs {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			lease, err := q.StreamExpert(ctx, reqs[idx])
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				errMu.Unlock()
				return
			}
			leases[idx] = lease
		}(i)
	}
	wg.Wait()

	if firstErr != nil {
		ReleaseLeases(leases)
		return nil, firstErr
	}
	return leases, nil
}

func (q *ExpertStreamQueue) acquireExpert(ctx context.Context, expertID int, offset, size int64) (*SlotLease, error) {
	p := q.slotPool

	for {
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			return nil, ErrQueueClosed
		}

		// 1. Check resident cache hit.
		if s, ok := p.resident[expertID]; ok && s.state == slotReady {
			s.refCount++
			s.lastUsed = p.nextSeq()
			p.metrics.slotHits.Add(1)
			p.mu.Unlock()
			return newLease(s, p), nil
		}

		// 2. Check in-flight deduplication.
		if load, ok := p.loading[expertID]; ok {
			done := load.done
			s := load.slot
			p.mu.Unlock()

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-q.closeChan:
				return nil, ErrQueueClosed
			case <-done:
			}

			p.mu.Lock()
			if load.err != nil {
				err := load.err
				p.mu.Unlock()
				return nil, err
			}
			if s.state == slotReady && s.expertID == expertID {
				s.refCount++
				s.lastUsed = p.nextSeq()
				p.metrics.slotHits.Add(1)
				p.mu.Unlock()
				return newLease(s, p), nil
			}
			// If state changed unexpectedly, retry loop.
			p.mu.Unlock()
			continue
		}

		// 3. Cache miss: find an available slot (free or unleased LRU).
		slot := p.findAvailableSlotLocked()
		if slot != nil {
			p.metrics.slotMisses.Add(1)
			if slot.expertID >= 0 && slot.state == slotReady {
				delete(p.resident, slot.expertID)
				p.metrics.slotEvictions.Add(1)
			}

			slot.expertID = expertID
			slot.refCount = 1 // reserved for this caller
			slot.lastUsed = p.nextSeq()
			slot.state = slotLoading
			slot.validBytes = 0

			load := &loadingState{
				slot: slot,
				done: make(chan struct{}),
			}
			p.loading[expertID] = load
			p.mu.Unlock()

			// Dispatch read to QD32 worker lanes.
			doneCh, err := q.dispatchReadAsync(ctx, slot, offset, size)
			if err != nil {
				p.mu.Lock()
				slot.state = slotFree
				slot.expertID = -1
				slot.refCount = 0
				delete(p.loading, expertID)
				close(load.done)
				p.notifyWaitersLocked()
				p.mu.Unlock()
				return nil, err
			}

			select {
			case res := <-doneCh:
				p.mu.Lock()
				load.err = res.err
				if res.err == nil {
					slot.state = slotReady
					slot.validBytes = res.n
					p.resident[expertID] = slot
				} else {
					slot.state = slotFree
					slot.expertID = -1
					slot.validBytes = 0
					slot.refCount = 0
					p.notifyWaitersLocked()
				}
				delete(p.loading, expertID)
				close(load.done)
				p.mu.Unlock()

				if res.err != nil {
					return nil, res.err
				}
				return newLease(slot, p), nil

			case <-ctx.Done():
				// Caller cancelled while read is in-flight. Detach cleanup goroutine
				// to safely retain or free the slot once the worker finishes.
				go func() {
					res := <-doneCh
					p.mu.Lock()
					defer p.mu.Unlock()
					load.err = res.err
					if res.err == nil {
						slot.state = slotReady
						slot.validBytes = res.n
						p.resident[expertID] = slot
					} else {
						slot.state = slotFree
						slot.expertID = -1
						slot.validBytes = 0
					}
					slot.refCount--
					if slot.refCount <= 0 {
						slot.refCount = 0
						p.notifyWaitersLocked()
					}
					delete(p.loading, expertID)
					close(load.done)
				}()
				return nil, ctx.Err()

			case <-q.closeChan:
				return nil, ErrQueueClosed
			}
		}

		// 4. All slots currently leased or loading. Register waiter and block.
		waiter := make(chan struct{}, 1)
		p.waiters = append(p.waiters, waiter)
		p.mu.Unlock()

		select {
		case <-ctx.Done():
			p.mu.Lock()
			select {
			case <-waiter:
				// If a notification was consumed, pass it along.
				p.notifyWaitersLocked()
			default:
				for i, w := range p.waiters {
					if w == waiter {
						p.waiters = append(p.waiters[:i], p.waiters[i+1:]...)
						break
					}
				}
			}
			p.mu.Unlock()
			return nil, ctx.Err()

		case <-q.closeChan:
			return nil, ErrQueueClosed

		case <-waiter:
			// Woken up by a release; retry slot acquisition.
		}
	}
}

// Metrics returns an atomic snapshot of current queue and pool metrics.
func (q *ExpertStreamQueue) Metrics() StreamMetrics {
	return StreamMetrics{
		BytesTransferred: q.metrics.bytesTransferred.Load(),
		ActiveQueueDepth: q.metrics.activeQueueDepth.Load(),
		PeakQueueDepth:   q.metrics.peakQueueDepth.Load(),
		SlotHits:         q.metrics.slotHits.Load(),
		SlotMisses:       q.metrics.slotMisses.Load(),
		SlotEvictions:    q.metrics.slotEvictions.Load(),
		TotalRequests:    q.metrics.totalRequests.Load(),
		TotalReads:       q.metrics.totalReads.Load(),
	}
}

// BytesTransferred returns the total bytes read from disk/reader.
func (q *ExpertStreamQueue) BytesTransferred() uint64 {
	return q.metrics.bytesTransferred.Load()
}

// ActiveQueueDepth returns the current number of in-flight pread reads across worker lanes.
func (q *ExpertStreamQueue) ActiveQueueDepth() int32 {
	return q.metrics.activeQueueDepth.Load()
}

// PeakQueueDepth returns the peak active queue depth observed.
func (q *ExpertStreamQueue) PeakQueueDepth() int32 {
	return q.metrics.peakQueueDepth.Load()
}

// SlotHits returns the count of expert requests satisfied without disk I/O.
func (q *ExpertStreamQueue) SlotHits() uint64 {
	return q.metrics.slotHits.Load()
}

// SlotMisses returns the count of expert requests requiring disk I/O.
func (q *ExpertStreamQueue) SlotMisses() uint64 {
	return q.metrics.slotMisses.Load()
}

// SlotEvictions returns the count of slots recycled via LRU.
func (q *ExpertStreamQueue) SlotEvictions() uint64 {
	return q.metrics.slotEvictions.Load()
}

// TotalRequests returns the total count of requested expert streams.
func (q *ExpertStreamQueue) TotalRequests() uint64 {
	return q.metrics.totalRequests.Load()
}

// TotalReads returns the total count of completed physical reads.
func (q *ExpertStreamQueue) TotalReads() uint64 {
	return q.metrics.totalReads.Load()
}

// ResetMetrics resets all telemetry counters.
func (q *ExpertStreamQueue) ResetMetrics() {
	q.metrics.bytesTransferred.Store(0)
	q.metrics.peakQueueDepth.Store(q.metrics.activeQueueDepth.Load())
	q.metrics.slotHits.Store(0)
	q.metrics.slotMisses.Store(0)
	q.metrics.slotEvictions.Store(0)
	q.metrics.totalRequests.Store(0)
	q.metrics.totalReads.Store(0)
}

// SlotCount returns the total number of preallocated unified memory slots.
func (q *ExpertStreamQueue) SlotCount() int {
	return q.slotPool.slotCount
}

// SlotBytes returns the byte capacity of each slot buffer.
func (q *ExpertStreamQueue) SlotBytes() int {
	return q.slotPool.slotBytes
}

// ActiveLeases returns the count of currently active slot leases.
func (q *ExpertStreamQueue) ActiveLeases() int {
	q.slotPool.mu.Lock()
	defer q.slotPool.mu.Unlock()
	count := 0
	for _, s := range q.slotPool.slots {
		count += int(s.refCount)
	}
	return count
}

// ResidentCount returns the count of experts currently cached in slots.
func (q *ExpertStreamQueue) ResidentCount() int {
	q.slotPool.mu.Lock()
	defer q.slotPool.mu.Unlock()
	return len(q.slotPool.resident)
}

// Close gracefully closes the queue, waking up waiters and shutting down worker lanes.
func (q *ExpertStreamQueue) Close() error {
	if !q.closed.CompareAndSwap(false, true) {
		return nil
	}
	q.slotPool.close()
	close(q.closeChan)
	q.workerWg.Wait()
	return nil
}
