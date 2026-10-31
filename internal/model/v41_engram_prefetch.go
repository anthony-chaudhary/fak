package model

// Known-address batch preparation is adapted from JigSawPT/llama.cpp
// src/llama-graph.cpp:187-224@8ce0968539b1a6795f3413c2783dfa5646b4688a.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"sync/atomic"
)

const v41EngramBatchMaxBytes int64 = 64 << 10

func (c *V41EngramRowCache) validateGatherGeometry() error {
	if c == nil || c.src == nil || c.inner == nil || c.opts.TableRows <= 0 || c.opts.RowBytes <= 0 ||
		int64(c.opts.RowBytes) > c.opts.BudgetBytes ||
		c.src.RowBytes() != c.opts.RowBytes {
		return &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1,
			Err: fmt.Errorf("invalid bounded Engram gather cache geometry")}
	}
	return nil
}

func (c *V41EngramRowCache) gatherKnownRows(rows []uint32) ([][]byte, error) {
	out, _, err := c.gatherKnownRowsOwned(rows, false)
	return out, err
}

// gatherKnownRowsOwned optionally owns one encoded panel for the resident
// decoder. The public gather keeps independent row allocations: retaining one
// public row must not retain an otherwise dead batch. Neither form borrows
// cache payloads or source read buffers.
func (c *V41EngramRowCache) gatherKnownRowsOwned(rows []uint32, contiguous bool) ([][]byte, []byte, error) {
	panelBytes := 0
	if contiguous {
		if err := c.validateGatherGeometry(); err != nil {
			return nil, nil, err
		}
		var ok bool
		panelBytes, ok = checkedMulInt(len(rows), c.opts.RowBytes)
		if !ok {
			return nil, nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1,
				Err: fmt.Errorf("encoded Engram panel byte size overflows")}
		}
		for _, row := range rows {
			if uint64(row) >= uint64(c.opts.TableRows) {
				return nil, nil, &V41EngramStreamError{Kind: V41EngramStreamOutOfBounds, Row: int(row),
					Err: fmt.Errorf("gather layer 0 row outside [0,%d)", c.opts.TableRows)}
			}
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([][]byte, len(rows))
	var panel []byte
	if contiguous {
		panel = make([]byte, panelBytes)
	}
	positions := make(map[uint32][]int, len(rows))
	unique := make([]uint32, 0, len(rows))
	for position, row := range rows {
		if _, ok := positions[row]; !ok {
			unique = append(unique, row)
		}
		positions[row] = append(positions[row], position)
	}
	serve := func(row uint32, payload []byte, hits int64) {
		for _, position := range positions[row] {
			if contiguous {
				lo := position * c.opts.RowBytes
				hi := lo + c.opts.RowBytes
				out[position] = panel[lo:hi:hi]
				copy(out[position], payload)
			} else {
				out[position] = append([]byte(nil), payload...)
			}
		}
		served := int64(len(positions[row]))
		c.stats.Hits += hits
		c.stats.RowsServed += served
		c.stats.BytesServed += served * int64(c.opts.RowBytes)
		c.stats.UsefulBytes += served * int64(c.opts.RowBytes)
	}
	serveResident := func(row uint32) (bool, error) {
		payload, ok := c.inner.Get(0, int(row))
		if !ok {
			return false, nil
		}
		if len(payload) != c.opts.RowBytes {
			return false, &V41EngramStreamError{Kind: V41EngramStreamShortRead, Row: int(row),
				Err: io.ErrUnexpectedEOF}
		}
		serve(row, payload, int64(len(positions[row])))
		return true, nil
	}
	missing := make([]uint32, 0, len(unique))
	for _, row := range unique {
		resident, err := serveResident(row)
		if err != nil {
			return nil, nil, err
		}
		if !resident {
			missing = append(missing, row)
		}
	}
	sort.Slice(missing, func(i, j int) bool { return missing[i] < missing[j] })
	maxBytes := c.opts.BudgetBytes
	if maxBytes > v41EngramBatchMaxBytes {
		maxBytes = v41EngramBatchMaxBytes
	}
	maxRows := int(maxBytes / int64(c.opts.RowBytes))
	// A legal row wider than the coalescing ceiling remains a budget-bounded singleton.
	if maxRows < 1 {
		maxRows = 1
	}
	parallel, err := c.gatherParallelMissRows(missing, maxRows, func(row uint32, payload []byte) {
		serve(row, payload, int64(len(positions[row])-1))
	})
	if err != nil {
		return nil, nil, err
	}
	if parallel {
		// Joined, owned outputs are ready. Both paths share the demand-order
		// residency publication below; a failed parallel wave never reaches it.
		missing = nil
	}
	// One bounded source-read slab covers every coalesced miss span in this
	// gather: size it to the largest planned span - the longest run of
	// consecutive missing rows, capped at the coalescer ceiling. Residency can
	// only shorten a span mid-gather, never extend one, so no read needs more
	// than slabRows. Reusing the slab across the sequential reads drops one
	// transient allocation per coalesced miss; served payloads copy out
	// immediately, so nothing aliases the slab past its span.
	slabRows := 0
	for i, run := 0, 0; i < len(missing); i++ {
		if i > 0 && missing[i] == missing[i-1]+1 {
			run++
		} else {
			run = 1
		}
		if run > slabRows {
			slabRows = run
		}
	}
	if slabRows > maxRows {
		slabRows = maxRows
	}
	var slab []byte
	if slabRows > 0 {
		slab = make([]byte, slabRows*c.opts.RowBytes)
	}
	for lo := 0; lo < len(missing); {
		// Readahead from an earlier range may already hold this row.
		resident, err := serveResident(missing[lo])
		if err != nil {
			return nil, nil, err
		}
		if resident {
			lo++
			continue
		}
		hi := lo + 1
		for hi < len(missing) && hi-lo < maxRows && uint64(missing[hi]) == uint64(missing[hi-1])+1 {
			if _, ok := c.inner.Get(0, int(missing[hi])); ok {
				break
			}
			hi++
		}
		count := hi - lo
		want := count * c.opts.RowBytes
		buf := slab[:want]
		c.stats.Misses += int64(count)
		n, err := c.src.ReadRows(int(missing[lo]), count, buf)
		if n < 0 || n > want {
			return nil, nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: int(missing[lo]),
				Err: fmt.Errorf("source returned invalid byte count %d for %d-byte range", n, want)}
		}
		c.stats.BytesRead += int64(n)
		if n != want {
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return nil, nil, &V41EngramStreamError{Kind: V41EngramStreamShortRead, Row: int(missing[lo]),
				Err: fmt.Errorf("source returned bytes=%d want=%d: %w", n, want, err)}
		}
		if err != nil {
			return nil, nil, fmt.Errorf("model: V41 Engram row read: %w", err)
		}
		for i, row := range missing[lo:hi] {
			payload := buf[i*c.opts.RowBytes : (i+1)*c.opts.RowBytes]
			serve(row, payload, int64(len(positions[row])-1))
		}
		if c.opts.PrefetchRows > 0 {
			c.prefetch(int(missing[hi-1]))
		}
		lo = hi
	}
	// Demand order owns final residency, including resident rows skipped by sorted reads.
	for position, row := range rows {
		if _, resident := c.inner.Get(0, int(row)); !resident {
			_ = c.inner.Put(0, int(row), out[position])
		}
	}
	return out, panel, nil
}

// The bounded worker scheduling is adapted from Strata/Project Maya EgReader:
// Niko1221/Strata src/dsv41/ds4_engram_io.hpp:129-203 at
// c62d5e3a15b40ea51c96aae9e8d36928965f53ab (MIT; complete license and Maya
// attribution in THIRD_PARTY_NOTICES.md). Unlike the donor's file-handle pool,
// FAK reuses its explicit producer policy and source hold, exact coalesced spans,
// immutable copied output and existing transaction/cache publication boundary.
// One process-wide additional read group bounds total new workers to four.
var v41EngramParallelReadBusy atomic.Bool

const v41EngramParallelReadBudget int64 = 1 << 20
const v41EngramParallelReadMetadata int64 = 64 << 10

type v41EngramParallelSpan struct {
	lo, hi     int
	buf        []byte
	n          int
	err        error
	panicValue any
	returned   bool
	done       *sync.WaitGroup
}

func v41EngramMissSpanEnd(missing []uint32, lo, maxRows int) int {
	hi := lo + 1
	for hi < len(missing) && hi-lo < maxRows && uint64(missing[hi]) == uint64(missing[hi-1])+1 {
		hi++
	}
	return hi
}

// Called with c.mu held. Workers only fill disjoint read buffers/result slots;
// this owner alone updates statistics and copies outputs. All workers are joined
// before releasing the exact source hold, even on panic or a source's Goexit.
// No ReaderAt interruption is promised: closing may wait for a read to return.
// Failed waves publish no rows; earlier successful waves only filled the private
// gather output. Cache residency is published by the caller only after success.
func (c *V41EngramRowCache) gatherParallelMissRows(missing []uint32, maxRows int, serve func(uint32, []byte)) (bool, error) {
	workers := c.opts.ReadWorkers
	if workers < 2 || len(missing) < 2 || maxRows < 2 || c.opts.PrefetchRows > 0 ||
		v41EngramMissSpanEnd(missing, 0, maxRows) == len(missing) {
		return false, nil
	}
	staging, eligible := v41EngramForwardLookAheadPolicy(c.src, c.opts.RowBytes)
	owner, owned := c.src.(v41EngramSourceReadOwner)
	if !eligible || !owned {
		return false, nil
	}
	if staging < 0 || workers > v41EngramMaxReadWorkers {
		return true, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1,
			Err: fmt.Errorf("invalid parallel read producer/worker envelope")}
	}
	// One aggregate wave, not workers copies of the serial read budget. Its
	// source inputs fit min(cache budget,64KiB). Producer staging plus a fixed
	// worker/metadata allowance also fits one process-wide logical MiB; this
	// is not a bound on Go RSS/GC or on the existing caller-owned output panel.
	perRow := int64(c.opts.RowBytes) + int64(staging)
	if perRow <= 0 || int64(maxRows) > (v41EngramParallelReadBudget-v41EngramParallelReadMetadata)/perRow {
		return false, nil
	}
	if !v41EngramParallelReadBusy.CompareAndSwap(false, true) {
		return false, nil
	}
	defer v41EngramParallelReadBusy.Store(false)
	release, closing, held := owner.V41EngramHoldSourceReads()
	if release != nil {
		defer release()
	}
	if !held || release == nil || closing == nil {
		return true, fmt.Errorf("model: V41 Engram parallel read source retention refused")
	}
	stopError := func() error {
		select {
		case <-closing:
			return fmt.Errorf("model: V41 Engram parallel read source is closing")
		default:
			return nil
		}
	}
	if err := stopError(); err != nil {
		return true, err
	}
	wave := make([]byte, maxRows*c.opts.RowBytes)
	jobs := make(chan *v41EngramParallelSpan, workers)
	var pool sync.WaitGroup
	pool.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer pool.Done()
			for job := range jobs {
				func() {
					defer job.done.Done()
					defer func() { job.panicValue = recover() }()
					job.n, job.err = c.src.ReadRows(int(missing[job.lo]), job.hi-job.lo, job.buf)
					job.returned = true
				}()
			}
		}()
	}
	// Registered after the source release: pool drain runs first on every exit.
	defer func() { close(jobs); pool.Wait() }()
	for lo := 0; lo < len(missing); {
		if err := stopError(); err != nil {
			return true, err
		}
		var spans [v41EngramMaxReadWorkers]v41EngramParallelSpan
		var done sync.WaitGroup
		nspans, usedRows := 0, 0
		for lo < len(missing) && nspans < workers {
			hi := v41EngramMissSpanEnd(missing, lo, maxRows)
			rows := hi - lo
			if usedRows+rows > maxRows {
				break // never split or amplify an already-coalesced span
			}
			start, end := usedRows*c.opts.RowBytes, (usedRows+rows)*c.opts.RowBytes
			spans[nspans] = v41EngramParallelSpan{lo: lo, hi: hi, buf: wave[start:end:end], done: &done}
			nspans++
			usedRows += rows
			lo = hi
		}
		done.Add(nspans)
		for i := 0; i < nspans; i++ {
			jobs <- &spans[i]
		}
		done.Wait()
		// Count all actually started reads, including later reads in a failed
		// wave. Unlike serial I/O, an earlier failure may have peers in flight.
		c.stats.Misses += int64(usedRows)
		for i := 0; i < nspans; i++ {
			if n := spans[i].n; n >= 0 && n <= len(spans[i].buf) {
				c.stats.BytesRead += int64(n)
			}
		}
		// First failure is selected by the original sorted span order, not by
		// completion order. No worker owns cache state or caller publication.
		for i := 0; i < nspans; i++ {
			job := &spans[i]
			if job.panicValue != nil {
				panic(job.panicValue)
			}
			if !job.returned {
				return true, fmt.Errorf("model: V41 Engram parallel read exited without completion")
			}
			want := len(job.buf)
			if job.n < 0 || job.n > want {
				return true, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: int(missing[job.lo]),
					Err: fmt.Errorf("source returned invalid byte count %d for %d-byte range", job.n, want)}
			}
			if job.n != want {
				err := job.err
				if err == nil {
					err = io.ErrUnexpectedEOF
				}
				return true, &V41EngramStreamError{Kind: V41EngramStreamShortRead, Row: int(missing[job.lo]),
					Err: fmt.Errorf("source returned bytes=%d want=%d: %w", job.n, want, err)}
			}
			if job.err != nil {
				return true, fmt.Errorf("model: V41 Engram row read: %w", job.err)
			}
		}
		if err := stopError(); err != nil {
			return true, err
		}
		for i := 0; i < nspans; i++ {
			job := &spans[i]
			for j, row := range missing[job.lo:job.hi] {
				serve(row, job.buf[j*c.opts.RowBytes:(j+1)*c.opts.RowBytes])
			}
		}
	}
	return true, stopError()
}
