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
