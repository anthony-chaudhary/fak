package gateway

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// phase_ledger_writer.go — the durable sibling of the default perf ledger for
// per-request native phase receipts. The ring at /v1/fak/perf/phases dies with
// the process; this writer persists each committed fak-request-phases/1
// envelope beside gateway-perf.jsonl so the six-bucket same-run breakdown
// (CPU, copies, weight loading, kernels, sync, unexplained) is computable from
// default artifacts alone, server down or after a restart. Offer never blocks
// the served turn: a full queue drops the receipt and counts it, the same
// non-blocking discipline as the perf sink it shadows.

const (
	phaseWriterQueue      = 1024
	phaseWriterBatch      = 256
	phaseWriterCloseGrace = 2 * time.Second
)

// ErrPhaseCloseTimeout reports that Close gave up waiting for the background
// drain; receipts still queued are lost rather than holding shutdown on a slow
// disk.
var ErrPhaseCloseTimeout = errors.New("gateway: phase ledger close timed out draining queued receipts")

// phaseLedgerWriter appends RequestPhaseReceipts to a size-capped JSONL file
// from one background goroutine.
type phaseLedgerWriter struct {
	path     string
	maxBytes int64
	ch       chan modelperfobs.RequestPhaseReceipt
	stop     chan struct{}
	done     chan struct{}
	closed   atomic.Bool
	once     sync.Once

	dropped atomic.Uint64
	written atomic.Uint64
	errMu   sync.Mutex
	lastErr error
}

// OpenPhaseLedgerWriter starts the background appender. maxBytes<=0 takes
// perfledger.DefaultMaxBytes; past it the active file rotates to path+".1"
// (one sealed generation), matching the perf sink it shadows.
func OpenPhaseLedgerWriter(path string, maxBytes int64) *phaseLedgerWriter {
	if maxBytes <= 0 {
		maxBytes = perfledger.DefaultMaxBytes
	}
	w := &phaseLedgerWriter{
		path:     path,
		maxBytes: maxBytes,
		ch:       make(chan modelperfobs.RequestPhaseReceipt, phaseWriterQueue),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	go w.run()
	return w
}

func (w *phaseLedgerWriter) Path() string {
	if w == nil {
		return ""
	}
	return w.path
}

// Offer enqueues r without blocking; it reports false when the receipt was
// dropped.
func (w *phaseLedgerWriter) Offer(r modelperfobs.RequestPhaseReceipt) bool {
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

// Dropped counts receipts never written: queue-full drops, post-Close offers,
// and rows in a batch whose append failed.
func (w *phaseLedgerWriter) Dropped() uint64 {
	if w == nil {
		return 0
	}
	return w.dropped.Load()
}

func (w *phaseLedgerWriter) Written() uint64 {
	if w == nil {
		return 0
	}
	return w.written.Load()
}

func (w *phaseLedgerWriter) LastError() error {
	if w == nil {
		return nil
	}
	w.errMu.Lock()
	defer w.errMu.Unlock()
	return w.lastErr
}

// Close stops accepting receipts, flushes what is queued, and waits at most
// phaseWriterCloseGrace. Safe to call more than once.
func (w *phaseLedgerWriter) Close() error {
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
	case <-time.After(phaseWriterCloseGrace):
		return ErrPhaseCloseTimeout
	}
}

func (w *phaseLedgerWriter) run() {
	defer close(w.done)
	batch := make([]modelperfobs.RequestPhaseReceipt, 0, phaseWriterBatch)
	for {
		select {
		case r := <-w.ch:
			batch = append(batch[:0], r)
			batch = w.drainInto(batch, phaseWriterBatch)
			w.flush(batch)
		case <-w.stop:
			for {
				batch = w.drainInto(batch[:0], phaseWriterBatch)
				if len(batch) == 0 {
					return
				}
				w.flush(batch)
			}
		}
	}
}

func (w *phaseLedgerWriter) drainInto(batch []modelperfobs.RequestPhaseReceipt, max int) []modelperfobs.RequestPhaseReceipt {
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

func (w *phaseLedgerWriter) flush(batch []modelperfobs.RequestPhaseReceipt) {
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

// PhaseLedgerSiblingPath is the durable phase-receipt file next to the perf
// ledger: gateway-perf.jsonl -> gateway-perf.phases.jsonl. A path without the
// .jsonl suffix gets .phases.jsonl appended instead.
func PhaseLedgerSiblingPath(perfPath string) string {
	if p := strings.TrimSuffix(perfPath, ".jsonl"); p != perfPath {
		return p + ".phases.jsonl"
	}
	return perfPath + ".phases.jsonl"
}
