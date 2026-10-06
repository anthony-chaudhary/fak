package perfledger

import (
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	writerQueue      = 1024
	writerBatch      = 256
	writerCloseGrace = 2 * time.Second
)

// ErrCloseTimeout reports that Close gave up waiting for the background drain;
// rows still queued are lost rather than holding shutdown on a slow disk.
var ErrCloseTimeout = errors.New("perfledger: close timed out draining queued rows")

// Writer appends Records to a size-capped JSONL file from one background
// goroutine. Offer never blocks: a full queue drops the row and counts it.
type Writer struct {
	path     string
	maxBytes int64
	ch       chan Record
	stop     chan struct{}
	done     chan struct{}
	closed   atomic.Bool
	once     sync.Once

	dropped atomic.Uint64
	written atomic.Uint64
	errMu   sync.Mutex
	lastErr error
}

// OpenWriter starts the background appender. maxBytes<=0 takes DefaultMaxBytes;
// past it the active file rotates to path+".1" (one sealed generation).
func OpenWriter(path string, maxBytes int64) *Writer {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxBytes
	}
	w := &Writer{
		path:     path,
		maxBytes: maxBytes,
		ch:       make(chan Record, writerQueue),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	go w.run()
	return w
}

func (w *Writer) Path() string {
	if w == nil {
		return ""
	}
	return w.path
}

// Offer enqueues r without blocking; it reports false when the row was dropped.
func (w *Writer) Offer(r Record) bool {
	if w == nil {
		return false
	}
	if w.closed.Load() {
		w.dropped.Add(1)
		return false
	}
	select {
	case w.ch <- r:
		return true
	default:
		w.dropped.Add(1)
		return false
	}
}

// Dropped counts rows never written: queue-full drops, post-Close offers, and
// rows in a batch whose append failed.
func (w *Writer) Dropped() uint64 {
	if w == nil {
		return 0
	}
	return w.dropped.Load()
}

func (w *Writer) Written() uint64 {
	if w == nil {
		return 0
	}
	return w.written.Load()
}

func (w *Writer) LastError() error {
	if w == nil {
		return nil
	}
	w.errMu.Lock()
	defer w.errMu.Unlock()
	return w.lastErr
}

// Close stops accepting rows, flushes what is queued, and waits at most
// writerCloseGrace. Safe to call more than once.
func (w *Writer) Close() error {
	if w == nil {
		return nil
	}
	w.once.Do(func() {
		w.closed.Store(true)
		close(w.stop)
	})
	select {
	case <-w.done:
		return w.LastError()
	case <-time.After(writerCloseGrace):
		return ErrCloseTimeout
	}
}

func (w *Writer) run() {
	defer close(w.done)
	batch := make([]Record, 0, writerBatch)
	for {
		select {
		case r := <-w.ch:
			batch = append(batch[:0], r)
			batch = w.drainInto(batch, writerBatch)
			w.flush(batch)
		case <-w.stop:
			for {
				batch = w.drainInto(batch[:0], writerBatch)
				if len(batch) == 0 {
					return
				}
				w.flush(batch)
			}
		}
	}
}

func (w *Writer) drainInto(batch []Record, max int) []Record {
	for len(batch) < max {
		select {
		case r := <-w.ch:
			batch = append(batch, r)
		default:
			return batch
		}
	}
	return batch
}

func (w *Writer) flush(batch []Record) {
	if len(batch) == 0 {
		return
	}
	var buf []byte
	rows := 0
	for _, r := range batch {
		line, err := json.Marshal(r)
		if err != nil {
			w.dropped.Add(1)
			continue
		}
		buf = append(append(buf, line...), '\n')
		rows++
	}
	if rows == 0 {
		return
	}
	if err := jsonlledger.AppendBounded(w.path, buf, w.maxBytes); err != nil {
		w.dropped.Add(uint64(rows))
		w.errMu.Lock()
		w.lastErr = err
		w.errMu.Unlock()
		return
	}
	w.written.Add(uint64(rows))
}
