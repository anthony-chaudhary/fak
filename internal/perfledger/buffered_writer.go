// Adapted from ai-dynamo/aiperf
// src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:21@ab610add4e8876fca223197596c728e17365f381
// (Apache-2.0, Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES;
// modified from the original): buffered JSONL writer with batch-size and
// periodic freshness flushes, bounded flush-task concurrency, and wire-only-field exclusion.
package perfledger

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// BufferedWriterConfig configures a BufferedJSONLWriter.
type BufferedWriterConfig struct {
	BatchSize          int
	FlushInterval      time.Duration
	MaxInFlightFlushes int
	WireOnlyFields     []string
}

// BufferedJSONLWriter batches JSONL records, flushes on batch threshold or periodic
// ticker, bounds concurrent flush tasks, and strips wire-only fields.
type BufferedJSONLWriter struct {
	path          string
	batchSize     int
	flushInterval time.Duration
	sem           chan struct{}
	wireOnly      map[string]struct{}

	mu     sync.Mutex
	buf    []map[string]any
	closed bool

	stop    chan struct{}
	done    chan struct{}
	wg      sync.WaitGroup
	flushed atomic.Uint64
	errMu   sync.Mutex
	lastErr error
}

// NewBufferedJSONLWriter opens a self-managed buffered JSONL writer at path.
func NewBufferedJSONLWriter(path string, cfg BufferedWriterConfig) (*BufferedJSONLWriter, error) {
	if path == "" {
		return nil, errors.New("perfledger: empty buffered writer path")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = 50 * time.Millisecond
	}
	if cfg.MaxInFlightFlushes <= 0 {
		cfg.MaxInFlightFlushes = 2
	}
	wire := make(map[string]struct{}, len(cfg.WireOnlyFields))
	for _, f := range cfg.WireOnlyFields {
		wire[f] = struct{}{}
	}
	w := &BufferedJSONLWriter{
		path:          path,
		batchSize:     cfg.BatchSize,
		flushInterval: cfg.FlushInterval,
		sem:           make(chan struct{}, cfg.MaxInFlightFlushes),
		wireOnly:      wire,
		stop:          make(chan struct{}),
		done:          make(chan struct{}),
	}
	go w.periodicLoop()
	return w, nil
}

// Write enqueues a record map, excluding any configured wire-only fields, and
// triggers a bounded async flush when the batch threshold is reached.
func (w *BufferedJSONLWriter) Write(rec map[string]any) error {
	clean := make(map[string]any, len(rec))
	for k, v := range rec {
		if _, skip := w.wireOnly[k]; skip {
			continue
		}
		clean[k] = v
	}

	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return errors.New("perfledger: buffered writer closed")
	}
	w.buf = append(w.buf, clean)
	var batch []map[string]any
	if len(w.buf) >= w.batchSize {
		batch = w.buf
		w.buf = nil
	}
	w.mu.Unlock()

	if len(batch) > 0 {
		w.dispatchFlush(batch)
	}
	return nil
}

// FlushedCount returns the total number of records persisted by completed flushes.
func (w *BufferedJSONLWriter) FlushedCount() uint64 {
	return w.flushed.Load()
}

// LastError returns the most recent flush error, if any.
func (w *BufferedJSONLWriter) LastError() error {
	w.errMu.Lock()
	defer w.errMu.Unlock()
	return w.lastErr
}

// Close stops the periodic ticker, flushes any remaining buffered records, and
// waits for all in-flight flush tasks to finish.
func (w *BufferedJSONLWriter) Close() error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		<-w.done
		return w.LastError()
	}
	w.closed = true
	rem := w.buf
	w.buf = nil
	w.mu.Unlock()

	close(w.stop)
	<-w.done
	if len(rem) > 0 {
		w.dispatchFlush(rem)
	}
	w.wg.Wait()
	return w.LastError()
}

func (w *BufferedJSONLWriter) periodicLoop() {
	defer close(w.done)
	ticker := time.NewTicker(w.flushInterval)
	defer ticker.Stop()
	for {
		select {
		case <-w.stop:
			return
		case <-ticker.C:
			w.mu.Lock()
			if len(w.buf) == 0 {
				w.mu.Unlock()
				continue
			}
			batch := w.buf
			w.buf = nil
			w.mu.Unlock()
			w.dispatchFlush(batch)
		}
	}
}

func (w *BufferedJSONLWriter) dispatchFlush(batch []map[string]any) {
	w.wg.Add(1)
	w.sem <- struct{}{}
	go func() {
		defer w.wg.Done()
		defer func() { <-w.sem }()
		if err := w.writeBatch(batch); err != nil {
			w.errMu.Lock()
			w.lastErr = err
			w.errMu.Unlock()
			return
		}
		w.flushed.Add(uint64(len(batch)))
	}()
}

func (w *BufferedJSONLWriter) writeBatch(batch []map[string]any) error {
	var payload []byte
	for _, rec := range batch {
		line, err := json.Marshal(rec)
		if err != nil {
			return err
		}
		payload = append(payload, line...)
		payload = append(payload, '\n')
	}
	f, err := os.OpenFile(w.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(payload)
	return err
}
