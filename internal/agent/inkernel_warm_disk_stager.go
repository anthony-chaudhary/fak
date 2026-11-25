package agent

import (
	"context"
	"sync"
	"time"
)

// warmDiskRecordWriter is the durability seam the conversation-state stager
// publishes through. *l3kv.DiskStore satisfies it: Put commits via a temp file
// that is fsync'd and atomically renamed, so a record is reusable only after
// all of its files are written and synced.
type warmDiskRecordWriter interface {
	Put(ctx context.Context, key string, payload []byte) error
}

// learnedGuardedWriter is the durability seam for record writers whose records
// must also land in the learned state index: publishes go through the
// pending-write guard so a crash leaves a reconcilable marker instead of an
// unindexed record that the next open would reap.
type learnedGuardedWriter interface {
	warmDiskRecordWriter
	beginWrite(key string) error
	commitWrite(ctx context.Context, key string, payloadBytes int64) error
	abortWrite(key string)
}

// warmDiskStagedSave is one encoded conversation-state record awaiting durable
// write. The payload is an owned copy: the RAM state may leave while it waits,
// and a cancelled save simply loses its copy.
type warmDiskStagedSave struct {
	key     string
	payload []byte
	done    chan error // one slot: nil published, sentinel when cancelled
}

// warmDiskStager is the bounded I/O queue behind opt-in conversation-state
// persistence. One writer goroutine (the I/O thread) takes saves FIFO and hands
// each to the record writer; a save becomes visible to readers only after that
// writer has committed it. Staging charges the byte permit for queued plus
// in-flight payloads; a save that finds the capacity full but would fit alone
// waits for the next write to finish, and is skipped if it still does not fit.
// A killed process loses unfinished saves (best effort): nothing half-written
// is ever published, and the private temp files are reaped by PruneContext.
type warmDiskStager struct {
	mu       sync.Mutex
	cond     *sync.Cond
	store    warmDiskRecordWriter
	queue    []warmDiskStagedSave
	waiters  []chan struct{}
	charged  int64 // queued + in-flight payload bytes
	inflight int
	permit   int64
	maxJobs  int
	closed   bool
	written  int
	failed   int
	stopped  chan struct{}
}

func newWarmDiskStager(store warmDiskRecordWriter, permit int64, maxJobs int) *warmDiskStager {
	if permit <= 0 {
		permit = ConversationStatePermitDefault
	}
	if maxJobs <= 0 {
		maxJobs = ConversationStateJobsDefault
	}
	s := &warmDiskStager{store: store, permit: permit, maxJobs: maxJobs, stopped: make(chan struct{})}
	s.cond = sync.NewCond(&s.mu)
	go s.run()
	return s
}

// stage enqueues payload under key, charging the permit. It reports
// errConversationStateCapacityFull for a save that cannot be taken (skipped),
// errConversationStateClosed after the drain, or ctx.Err() when the caller's
// context ends while waiting for room. The returned save's done channel
// delivers nil once the record is published or the cancellation sentinel.
func (s *warmDiskStager) stage(ctx context.Context, key string, payload []byte) (*warmDiskStagedSave, error) {
	size := int64(len(payload))
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, errConversationStateClosed
	}
	if s.fitsLocked(size) {
		save := s.enqueueLocked(key, payload, size)
		s.mu.Unlock()
		return save, nil
	}
	if size > s.permit {
		// Cannot fit even alone: skipped, never written.
		s.mu.Unlock()
		return nil, errConversationStateCapacityFull
	}
	waiter := make(chan struct{})
	s.waiters = append(s.waiters, waiter)
	s.mu.Unlock()
	select {
	case <-waiter:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, errConversationStateClosed
	}
	if !s.fitsLocked(size) {
		return nil, errConversationStateCapacityFull
	}
	return s.enqueueLocked(key, payload, size), nil
}

// fitsLocked reports whether one more save of size bytes fits under the job cap
// and the byte permit.
func (s *warmDiskStager) fitsLocked(size int64) bool {
	return len(s.queue)+s.inflight < s.maxJobs && s.charged+size <= s.permit
}

func (s *warmDiskStager) enqueueLocked(key string, payload []byte, size int64) *warmDiskStagedSave {
	save := &warmDiskStagedSave{key: key, payload: payload, done: make(chan error, 1)}
	s.queue = append(s.queue, *save)
	s.charged += size
	s.cond.Signal()
	return save
}

// run is the I/O thread: it takes queued saves FIFO and publishes each through
// the record writer. It exits only after the drain closes the queue; a write
// already in flight may finish after the deadline using only its private temp
// file and staged bytes.
func (s *warmDiskStager) run() {
	defer close(s.stopped)
	for {
		s.mu.Lock()
		for len(s.queue) == 0 && !s.closed {
			s.cond.Wait()
		}
		if len(s.queue) == 0 {
			s.mu.Unlock()
			return
		}
		save := s.queue[0]
		s.queue = s.queue[1:]
		s.inflight++
		s.mu.Unlock()
		err := s.publishSave(&save)
		s.mu.Lock()
		s.inflight--
		s.charged -= int64(len(save.payload))
		if err != nil {
			s.failed++
		} else {
			s.written++
		}
		s.broadcastLocked()
		s.mu.Unlock()
		save.done <- err // nil when published
	}
}

// publishSave lands one staged save. Writers that also maintain a learned
// state index (the warm-disk store) go through their pending-write guard so
// the record is indexed before it becomes visible and a crash leaves a
// reconcilable marker; plain writers fall back to a direct Put. A deferral by
// an in-flight write on the same identity counts as a failed save: the stager
// is bounded and the caller sees the error, it never blocks the I/O thread.
func (s *warmDiskStager) publishSave(save *warmDiskStagedSave) error {
	g, ok := s.store.(learnedGuardedWriter)
	if !ok {
		return s.store.Put(context.Background(), save.key, save.payload)
	}
	ctx := context.Background()
	if err := g.beginWrite(save.key); err != nil {
		return err
	}
	if err := g.Put(ctx, save.key, save.payload); err != nil {
		g.abortWrite(save.key)
		return err
	}
	return g.commitWrite(ctx, save.key, int64(len(save.payload)))
}

// broadcastLocked wakes every stage caller waiting for room.
func (s *warmDiskStager) broadcastLocked() {
	for _, w := range s.waiters {
		close(w)
	}
	s.waiters = nil
}

// drain runs the planned-stop protocol: keep publishing until the queue is
// empty or the deadline passes, then close the queue and cancel whatever is
// still queued without writing it. It reports saves published during the drain
// and saves cancelled.
func (s *warmDiskStager) drain(deadline time.Duration) (published, cancelled int) {
	s.mu.Lock()
	start := s.written
	alreadyClosed := s.closed
	s.mu.Unlock()
	if !alreadyClosed {
		deadlineAt := time.Now().Add(deadline)
		for {
			s.mu.Lock()
			empty := len(s.queue) == 0 && s.inflight == 0
			s.mu.Unlock()
			if empty || time.Now().After(deadlineAt) {
				break
			}
			time.Sleep(2 * time.Millisecond)
		}
	}
	s.mu.Lock()
	for i := range s.queue {
		s.queue[i].done <- errConversationStateCancelled
	}
	cancelled = len(s.queue)
	s.queue = nil
	s.charged = 0
	s.closed = true
	s.broadcastLocked()
	s.cond.Signal()
	waitWriter := s.inflight == 0
	s.mu.Unlock()
	if waitWriter {
		<-s.stopped // the writer exits promptly once the queue is empty and closed
	}
	// A write still in flight past the deadline may finish later using only its
	// private temp file and staged bytes; it never blocks the stop.
	s.mu.Lock()
	published = s.written - start
	s.mu.Unlock()
	return published, cancelled
}

// warmDiskStagerByDir binds each conversation-state store directory to its
// single I/O queue for the process lifetime. The queue deliberately outlives
// any one planner: a restarted server re-opens the same directory and must
// keep publishing behind the same byte permit, and the planned-stop drain
// closes it explicitly. Bounded by the number of distinct store directories a
// process ever persists into.
var (
	warmDiskStagerMu    sync.Mutex
	warmDiskStagerByDir = map[string]*warmDiskStager{}
)

// conversationStager returns the I/O queue bound to the resolved store
// directory, creating it on first use with the store first opened for
// persistence.
func conversationStager(dir string, store warmDiskRecordWriter, permit int64, maxJobs int) *warmDiskStager {
	warmDiskStagerMu.Lock()
	defer warmDiskStagerMu.Unlock()
	if s, ok := warmDiskStagerByDir[dir]; ok {
		return s
	}
	s := newWarmDiskStager(store, permit, maxJobs)
	warmDiskStagerByDir[dir] = s
	return s
}

// DrainConversationState runs the planned-stop protocol for opt-in
// conversation-state staging: wait up to deadline (default
// ConversationStateStopDrain) for queued saves to publish, then cancel the
// remainder. It reports saves published during the drain and saves cancelled.
// A killed process instead loses unfinished saves outright; the stale temp
// files stay private and are reaped by PruneContext.
func (p *InKernelPlanner) DrainConversationState(deadline time.Duration) (published, cancelled int) {
	if p == nil {
		return 0, 0
	}
	p.mu.Lock()
	cfg := p.warmDiskConfig
	p.mu.Unlock()
	if !cfg.ConversationState || cfg.Disabled {
		return 0, 0
	}
	resolved, err := resolveWarmDiskConfig(cfg)
	if err != nil {
		return 0, 0
	}
	warmDiskStagerMu.Lock()
	stager := warmDiskStagerByDir[resolved.Dir]
	warmDiskStagerMu.Unlock()
	if stager == nil {
		return 0, 0
	}
	if deadline <= 0 {
		deadline = ConversationStateStopDrain
	}
	return stager.drain(deadline)
}
