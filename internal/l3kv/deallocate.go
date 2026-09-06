package l3kv

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
)

var (
	// ErrDeallocatorClosed indicates the deallocator is shutting down or closed.
	ErrDeallocatorClosed = errors.New("l3kv: deallocator is closed")

	// ErrQueueFull indicates the bounded async deallocation queue is saturated.
	// Returning this error ensures eviction is never blocked by slow disk operations.
	ErrQueueFull = errors.New("l3kv: deallocator queue is full")
)

// DefaultDeallocatorQueueDepth is the default queue depth for asynchronous deallocations.
const DefaultDeallocatorQueueDepth = 1024

// DeallocateRequest encapsulates an asynchronous block deallocation / TRIM request.
type DeallocateRequest struct {
	File       *os.File
	Offset     int64
	Length     int64
	OnComplete func(err error)
}

// DeallocatorStats provides telemetry on deallocation operations.
type DeallocatorStats struct {
	SubmittedCount uint64 `json:"submitted_count"`
	ProcessedCount uint64 `json:"processed_count"`
	DroppedCount   uint64 `json:"dropped_count"`
	ErrorCount     uint64 `json:"error_count"`
	QueueDepth     int    `json:"queue_depth"`
	QueueLen       int    `json:"queue_len"`
}

// AsyncDeallocator coordinates asynchronous block deallocation and NVMe TRIM commands
// on cache span eviction without stalling hot eviction pathways.
type AsyncDeallocator struct {
	mu             sync.RWMutex
	queue          chan DeallocateRequest
	queueDepth     int
	wg             sync.WaitGroup
	closed         bool
	submittedCount uint64
	processedCount uint64
	droppedCount   uint64
	errorCount     uint64
}

// NewAsyncDeallocator creates and starts a bounded background worker queue for deallocations.
// queueDepth configures the request buffer size (defaults to DefaultDeallocatorQueueDepth if <= 0).
// An optional workers argument specifies worker goroutine concurrency (defaults to 1).
func NewAsyncDeallocator(queueDepth int, workers ...int) *AsyncDeallocator {
	if queueDepth <= 0 {
		queueDepth = DefaultDeallocatorQueueDepth
	}
	numWorkers := 1
	if len(workers) > 0 && workers[0] > 0 {
		numWorkers = workers[0]
	}

	d := &AsyncDeallocator{
		queue:      make(chan DeallocateRequest, queueDepth),
		queueDepth: queueDepth,
	}

	for i := 0; i < numWorkers; i++ {
		d.wg.Add(1)
		go d.workerLoop()
	}

	return d
}

func (d *AsyncDeallocator) workerLoop() {
	defer d.wg.Done()
	for req := range d.queue {
		err := DeallocateFileRange(req.File, req.Offset, req.Length)
		if err != nil {
			atomic.AddUint64(&d.errorCount, 1)
		} else {
			atomic.AddUint64(&d.processedCount, 1)
		}
		if req.OnComplete != nil {
			req.OnComplete(err)
		}
	}
}

// Submit enqueues a deallocation request without blocking. If the queue is saturated,
// it immediately returns ErrQueueFull, preserving eviction throughput and telemetry.
func (d *AsyncDeallocator) Submit(file *os.File, offset, length int64) error {
	return d.SubmitWithCallback(file, offset, length, nil)
}

// SubmitWithCallback enqueues a deallocation request with an optional completion callback.
func (d *AsyncDeallocator) SubmitWithCallback(file *os.File, offset, length int64, onComplete func(err error)) error {
	if d == nil {
		return nil
	}
	d.mu.RLock()
	defer d.mu.RUnlock()

	if d.closed {
		return ErrDeallocatorClosed
	}

	req := DeallocateRequest{
		File:       file,
		Offset:     offset,
		Length:     length,
		OnComplete: onComplete,
	}

	select {
	case d.queue <- req:
		atomic.AddUint64(&d.submittedCount, 1)
		return nil
	default:
		atomic.AddUint64(&d.droppedCount, 1)
		if onComplete != nil {
			onComplete(ErrQueueFull)
		}
		return ErrQueueFull
	}
}

// DeallocateSpan issues an asynchronous deallocate / TRIM command for a span file range.
// It returns immediately without blocking eviction.
func (d *AsyncDeallocator) DeallocateSpan(file *os.File, offset, length int64) error {
	return d.Submit(file, offset, length)
}

// Close gracefully stops all worker goroutines and drains queued deallocations.
func (d *AsyncDeallocator) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return nil
	}
	d.closed = true
	close(d.queue)
	d.mu.Unlock()

	d.wg.Wait()
	return nil
}

// Stats returns a point-in-time telemetry snapshot of the async deallocator.
func (d *AsyncDeallocator) Stats() DeallocatorStats {
	if d == nil {
		return DeallocatorStats{}
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return DeallocatorStats{
		SubmittedCount: atomic.LoadUint64(&d.submittedCount),
		ProcessedCount: atomic.LoadUint64(&d.processedCount),
		DroppedCount:   atomic.LoadUint64(&d.droppedCount),
		ErrorCount:     atomic.LoadUint64(&d.errorCount),
		QueueDepth:     d.queueDepth,
		QueueLen:       len(d.queue),
	}
}

// DeallocateFileRange issues an explicit OS-level deallocate/TRIM command for the byte range
// [offset, offset+length) of the given file. On Linux it issues punch-hole fallocate; on Windows
// it issues DeviceIoControl zero-data / sparse range deallocation; with graceful fallback on
// unsupported environments.
func DeallocateFileRange(file *os.File, offset, length int64) error {
	if file == nil {
		return errors.New("l3kv: file is nil")
	}
	if offset < 0 || length < 0 {
		return fmt.Errorf("l3kv: invalid range: offset=%d length=%d", offset, length)
	}
	if length == 0 {
		return nil
	}
	return deallocateFileRangeOS(file, offset, length)
}

// fallbackDeallocate provides a portable fallback when native filesystem hole punching
// is unavailable or unsupported (e.g. non-sparse NTFS files, tmpfs, older kernels, or exFAT).
func fallbackDeallocate(file *os.File, offset, length int64) error {
	fi, err := file.Stat()
	if err != nil {
		return err
	}
	size := fi.Size()
	// If the range begins at or beyond EOF, there is nothing to deallocate.
	if offset >= size {
		return nil
	}
	// If the range reaches or extends beyond the end of the file, truncate.
	if offset+length >= size {
		return file.Truncate(offset)
	}

	// For ranges in the middle of the file, zero-fill the range.
	const chunk = 64 * 1024
	zeroes := make([]byte, chunk)
	remaining := length
	cur := offset
	for remaining > 0 {
		toWrite := int64(len(zeroes))
		if remaining < toWrite {
			toWrite = remaining
		}
		n, err := file.WriteAt(zeroes[:toWrite], cur)
		if err != nil {
			return err
		}
		cur += int64(n)
		remaining -= int64(n)
	}
	return nil
}
