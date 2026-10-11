package metalgemm

import (
	"bytes"
	"context"
	"errors"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const lifetimeSlotBytes = 32

type lifetimeReadResult struct {
	n   int
	err error
}

type lifetimeRead struct {
	dst      []byte
	offset   int64
	finish   chan lifetimeReadResult
	returned chan struct{}
}

type lifetimeReader struct {
	calls  chan *lifetimeRead
	abort  chan struct{}
	once   sync.Once
	active atomic.Int32
	peak   atomic.Int32
	reads  atomic.Uint64
	bytes  atomic.Uint64
}

func (r *lifetimeReader) ReadAt(dst []byte, offset int64) (int, error) {
	active := r.active.Add(1)
	for peak := r.peak.Load(); active > peak; peak = r.peak.Load() {
		if r.peak.CompareAndSwap(peak, active) {
			break
		}
	}
	call := &lifetimeRead{dst: dst, offset: offset, finish: make(chan lifetimeReadResult, 1), returned: make(chan struct{})}
	r.calls <- call
	var result lifetimeReadResult
	select {
	case result = <-call.finish:
	case <-r.abort:
		result.err = context.Canceled
	}
	for i := 0; i < result.n; i++ {
		dst[i] = byte(offset + int64(i) + 1)
	}
	r.bytes.Add(uint64(result.n))
	r.reads.Add(1)
	r.active.Add(-1)
	close(call.returned)
	return result.n, result.err
}

func newLifetimeQueue(t *testing.T, depth, slots int) (*ExpertStreamQueue, *lifetimeReader) {
	t.Helper()
	r := &lifetimeReader{calls: make(chan *lifetimeRead, 8), abort: make(chan struct{})}
	q, err := NewExpertStreamQueue(StreamConfig{QueueDepth: depth, SlotCount: slots, SlotBytes: lifetimeSlotBytes, Reader: r})
	if err != nil {
		t.Fatal(err)
	}
	for id := 0; id < 4; id++ {
		q.RegisterLocation(id, int64(id*lifetimeSlotBytes), lifetimeSlotBytes)
	}
	t.Cleanup(func() {
		r.once.Do(func() { close(r.abort) })
		done := make(chan error, 1)
		go func() { done <- q.Close() }()
		if err := awaitLifetime(t, done, "queue cleanup"); err != nil {
			t.Error(err)
		}
	})
	return q, r
}

func awaitLifetime[T any](t *testing.T, done <-chan T, phase string) T {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-time.After(5 * time.Second):
		t.Fatalf("%s did not complete", phase)
		var zero T
		return zero
	}
}

func awaitLifetimeState(t *testing.T, ready func() bool, phase string) {
	t.Helper()
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	for !ready() {
		select {
		case <-timeout.C:
			t.Fatalf("%s was not reached", phase)
		default:
			runtime.Gosched()
		}
	}
}

type lifetimeObservedContext struct {
	context.Context
	observed chan struct{}
	once     sync.Once
}

func (c *lifetimeObservedContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.observed) })
	return c.Context.Done()
}

func observeLifetimeContext(ctx context.Context) *lifetimeObservedContext {
	return &lifetimeObservedContext{Context: ctx, observed: make(chan struct{})}
}

type lifetimeLeaseResult struct {
	lease *SlotLease
	err   error
}

func requestLifetime(q *ExpertStreamQueue, ctx context.Context, id int) <-chan lifetimeLeaseResult {
	done := make(chan lifetimeLeaseResult, 1)
	go func() {
		lease, err := q.StreamExpert(ctx, ExpertRequest{ExpertID: id})
		done <- lifetimeLeaseResult{lease: lease, err: err}
	}()
	return done
}

func assertLifetimeMetrics(t *testing.T, q *ExpertStreamQueue, r *lifetimeReader, reads, transferred uint64, peak int32) {
	t.Helper()
	metrics := q.Metrics()
	t.Logf("reads=%d bytes=%d peak=%d active=%d leases=%d", metrics.TotalReads, metrics.BytesTransferred, metrics.PeakQueueDepth, metrics.ActiveQueueDepth, q.ActiveLeases())
	if metrics.TotalReads != reads || r.reads.Load() != reads {
		t.Errorf("completed reads: queue=%d reader=%d want=%d", metrics.TotalReads, r.reads.Load(), reads)
	}
	if metrics.BytesTransferred != transferred || r.bytes.Load() != transferred {
		t.Errorf("transferred bytes: queue=%d reader=%d want=%d", metrics.BytesTransferred, r.bytes.Load(), transferred)
	}
	if metrics.PeakQueueDepth != peak || r.peak.Load() != peak {
		t.Errorf("peak reads: queue=%d reader=%d want=%d", metrics.PeakQueueDepth, r.peak.Load(), peak)
	}
	if metrics.ActiveQueueDepth != 0 || r.active.Load() != 0 || q.ActiveLeases() != 0 {
		t.Errorf("unfinished work: queue=%d reader=%d leases=%d", metrics.ActiveQueueDepth, r.active.Load(), q.ActiveLeases())
	}
}

// fak-test:justify why=regression when=changed:internal/metalgemm/**
// fak-test:runtime fast est=10ms
func TestExpertStreamLifetime_CloseReleasesActiveAndQueuedLoads(t *testing.T) {
	q, reader := newLifetimeQueue(t, 1, 2)
	active := requestLifetime(q, context.Background(), 0)
	call := awaitLifetime(t, reader.calls, "active ReadAt")
	duplicateContext := observeLifetimeContext(context.Background())
	duplicate := requestLifetime(q, duplicateContext, 0)
	awaitLifetime(t, duplicateContext.observed, "duplicate wait")
	queuedContext, cancelQueued := context.WithCancel(context.Background())
	defer cancelQueued()
	queued := requestLifetime(q, queuedContext, 1)
	awaitLifetimeState(t, func() bool { return len(q.jobChan) == 1 }, "queued read")
	q.slotPool.mu.Lock()
	activeLoadingDone := q.slotPool.loading[0].done
	queuedLoadingDone := q.slotPool.loading[1].done
	q.slotPool.mu.Unlock()
	cancelQueued()
	if result := awaitLifetime(t, queued, "queued cancellation"); !errors.Is(result.err, context.Canceled) || result.lease != nil {
		ReleaseLeases([]*SlotLease{result.lease})
		t.Fatalf("queued cancellation: lease=%v error=%v", result.lease != nil, result.err)
	}
	closed := make(chan error, 1)
	go func() { closed <- q.Close() }()
	for _, result := range []lifetimeLeaseResult{awaitLifetime(t, active, "active shutdown waiter"), awaitLifetime(t, duplicate, "duplicate shutdown waiter")} {
		if !errors.Is(result.err, ErrQueueClosed) || result.lease != nil {
			ReleaseLeases([]*SlotLease{result.lease})
			t.Errorf("shutdown waiter: lease=%v error=%v", result.lease != nil, result.err)
		}
	}
	select {
	case err := <-closed:
		t.Fatalf("Close returned while ReaderAt still owned the buffer: %v", err)
	default:
	}
	call.finish <- lifetimeReadResult{n: lifetimeSlotBytes}
	awaitLifetime(t, call.returned, "active ReadAt completion")
	if err := awaitLifetime(t, closed, "Close completion"); err != nil {
		t.Fatal(err)
	}
	assertLifetimeMetrics(t, q, reader, 1, lifetimeSlotBytes, 1)
	for name, done := range map[string]<-chan struct{}{"active": activeLoadingDone, "queued": queuedLoadingDone} {
		select {
		case <-done:
		default:
			t.Errorf("Close abandoned %s loading completion", name)
		}
	}
}

// fak-test:justify why=invariant when=changed:internal/metalgemm/**
// fak-test:runtime fast est=10ms
func TestExpertStreamLifetime_CancelRetainsBuffer(t *testing.T) {
	q, reader := newLifetimeQueue(t, 2, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first := requestLifetime(q, ctx, 0)
	firstRead := awaitLifetime(t, reader.calls, "first ReadAt")
	cancel()
	if result := awaitLifetime(t, first, "active cancellation"); !errors.Is(result.err, context.Canceled) || result.lease != nil {
		ReleaseLeases([]*SlotLease{result.lease})
		t.Fatalf("active cancellation: lease=%v error=%v", result.lease != nil, result.err)
	}
	if q.ActiveLeases() != 1 || q.ActiveQueueDepth() != 1 {
		t.Fatalf("cancelled active read lost its reservation: leases=%d reads=%d", q.ActiveLeases(), q.ActiveQueueDepth())
	}
	nextContext := observeLifetimeContext(context.Background())
	next := requestLifetime(q, nextContext, 1)
	awaitLifetime(t, nextContext.observed, "capacity wait")
	select {
	case <-reader.calls:
		t.Fatal("slot reused before its active ReaderAt completed")
	default:
	}
	firstRead.finish <- lifetimeReadResult{n: lifetimeSlotBytes}
	awaitLifetime(t, firstRead.returned, "cancelled ReadAt completion")
	secondRead := awaitLifetime(t, reader.calls, "next ReadAt")
	if secondRead.offset != lifetimeSlotBytes {
		t.Fatalf("next read offset=%d", secondRead.offset)
	}
	secondRead.finish <- lifetimeReadResult{n: lifetimeSlotBytes}
	result := awaitLifetime(t, next, "next expert lease")
	if result.err != nil || result.lease == nil {
		t.Fatalf("next expert: %v", result.err)
	}
	defer result.lease.Release()
	want := make([]byte, lifetimeSlotBytes)
	for i := range want {
		want[i] = byte(lifetimeSlotBytes + i + 1)
	}
	if !bytes.Equal(result.lease.Bytes(), want) {
		t.Error("recycled slot returned corrupt expert bytes")
	}
	result.lease.Release()
	result.lease.Release()
	assertLifetimeMetrics(t, q, reader, 2, 2*lifetimeSlotBytes, 1)
}

// fak-test:justify why=contract when=changed:internal/metalgemm/**
// fak-test:runtime fast est=10ms
func TestExpertStreamLifetime_DuplicateWaiters(t *testing.T) {
	q, reader := newLifetimeQueue(t, 2, 1)
	owner := requestLifetime(q, context.Background(), 0)
	call := awaitLifetime(t, reader.calls, "owner ReadAt")
	cancelContext, cancel := context.WithCancel(context.Background())
	defer cancel()
	observedCancel := observeLifetimeContext(cancelContext)
	cancelled := requestLifetime(q, observedCancel, 0)
	awaitLifetime(t, observedCancel.observed, "cancellable duplicate wait")
	survivorContext := observeLifetimeContext(context.Background())
	survivor := requestLifetime(q, survivorContext, 0)
	awaitLifetime(t, survivorContext.observed, "surviving duplicate wait")
	cancel()
	if result := awaitLifetime(t, cancelled, "duplicate cancellation"); !errors.Is(result.err, context.Canceled) || result.lease != nil {
		ReleaseLeases([]*SlotLease{result.lease})
		t.Fatalf("duplicate cancellation: lease=%v error=%v", result.lease != nil, result.err)
	}
	if q.ActiveLeases() != 1 || q.ActiveQueueDepth() != 1 {
		t.Fatalf("duplicate cancellation disturbed owner: leases=%d reads=%d", q.ActiveLeases(), q.ActiveQueueDepth())
	}
	call.finish <- lifetimeReadResult{n: lifetimeSlotBytes}
	first := awaitLifetime(t, owner, "owner lease")
	second := awaitLifetime(t, survivor, "duplicate lease")
	defer ReleaseLeases([]*SlotLease{first.lease, second.lease})
	if first.err != nil || second.err != nil || first.lease == nil || second.lease == nil {
		t.Fatalf("shared read results: owner=%v duplicate=%v", first.err, second.err)
	}
	if first.lease.SlotID() != second.lease.SlotID() || !bytes.Equal(first.lease.Bytes(), second.lease.Bytes()) {
		t.Error("duplicate leases did not share the loaded expert")
	}
	if q.ActiveLeases() != 2 {
		t.Errorf("live duplicate leases=%d want=2", q.ActiveLeases())
	}
	ReleaseLeases([]*SlotLease{first.lease, second.lease})
	assertLifetimeMetrics(t, q, reader, 1, lifetimeSlotBytes, 1)
}

// fak-test:justify why=contract when=changed:internal/metalgemm/**
// fak-test:runtime fast est=20ms
func TestExpertStreamLifetime_BatchFailureReleasesLeases(t *testing.T) {
	readFailure := errors.New("injected expert read failure")
	for _, test := range []struct {
		name string
		read lifetimeReadResult
		want error
	}{
		{name: "read_failed", read: lifetimeReadResult{err: readFailure}, want: readFailure},
		{name: "partial_read", read: lifetimeReadResult{n: 7, err: io.EOF}, want: io.ErrUnexpectedEOF},
	} {
		t.Run(test.name, func(t *testing.T) {
			q, reader := newLifetimeQueue(t, 2, 2)
			warm := requestLifetime(q, context.Background(), 0)
			call := awaitLifetime(t, reader.calls, "cache warm ReadAt")
			call.finish <- lifetimeReadResult{n: lifetimeSlotBytes}
			cached := awaitLifetime(t, warm, "cache warm lease")
			if cached.err != nil || cached.lease == nil {
				t.Fatalf("warm cache: %v", cached.err)
			}
			cached.lease.Release()
			type batchResult struct {
				leases []*SlotLease
				err    error
			}
			done := make(chan batchResult, 1)
			go func() {
				leases, err := q.StreamBatch(context.Background(), []ExpertRequest{{ExpertID: 0}, {ExpertID: 1}})
				done <- batchResult{leases: leases, err: err}
			}()
			failedRead := awaitLifetime(t, reader.calls, "failing ReadAt")
			failedRead.finish <- test.read
			result := awaitLifetime(t, done, "failed batch cleanup")
			defer ReleaseLeases(result.leases)
			if !errors.Is(result.err, test.want) || len(result.leases) != 0 {
				t.Errorf("failed batch: leases=%d error=%v", len(result.leases), result.err)
			}
			if q.SlotHits() != 1 {
				t.Error("failed batch did not exercise an acquired cached lease")
			}
			assertLifetimeMetrics(t, q, reader, 2, lifetimeSlotBytes+uint64(test.read.n), 1)
		})
	}
	t.Run("oversized_active_set", func(t *testing.T) {
		q, reader := newLifetimeQueue(t, 1, 2)
		leases, err := q.StreamBatch(context.Background(), []ExpertRequest{{ExpertID: 0}, {ExpertID: 1}, {ExpertID: 2}})
		defer ReleaseLeases(leases)
		if !errors.Is(err, ErrBatchExceedsPoolCapacity) || len(leases) != 0 {
			t.Errorf("oversized active set: leases=%d error=%v", len(leases), err)
		}
		assertLifetimeMetrics(t, q, reader, 0, 0, 0)
	})
}

// fak-test:justify why=contract when=changed:internal/metalgemm/**
// fak-test:runtime fast est=10ms
func TestExpertStreamLifetime_ClosePreservesReturnedLease(t *testing.T) {
	q, reader := newLifetimeQueue(t, 2, 1)
	request := requestLifetime(q, context.Background(), 0)
	call := awaitLifetime(t, reader.calls, "held lease ReadAt")
	call.finish <- lifetimeReadResult{n: lifetimeSlotBytes}
	result := awaitLifetime(t, request, "held lease acquisition")
	if result.err != nil || result.lease == nil {
		t.Fatalf("held lease acquisition: %v", result.err)
	}
	defer result.lease.Release()
	want := append([]byte(nil), result.lease.Bytes()...)
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	if result.lease.IsReleased() || !bytes.Equal(result.lease.Bytes(), want) || q.ActiveLeases() != 1 {
		t.Errorf("Close invalidated returned lease: released=%v leases=%d", result.lease.IsReleased(), q.ActiveLeases())
	}
	rejected, err := q.StreamExpert(context.Background(), ExpertRequest{ExpertID: 1})
	defer ReleaseLeases([]*SlotLease{rejected})
	if !errors.Is(err, ErrQueueClosed) || rejected != nil {
		t.Errorf("closed request: lease=%v error=%v", rejected != nil, err)
	}
	if !bytes.Equal(result.lease.Bytes(), want) {
		t.Error("closed request overwrote returned lease bytes")
	}
	result.lease.Release()
	result.lease.Release()
	if !result.lease.IsReleased() || result.lease.Bytes() != nil {
		t.Error("returned lease was not releaseable after Close")
	}
	assertLifetimeMetrics(t, q, reader, 1, lifetimeSlotBytes, 1)
}
