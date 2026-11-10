// MappedRegion gives loadgen workers a read-only, concurrently sliceable
// window over a dataset file, with optional prefaulting at open.
//
// Direct port from ai-dynamo/aiperf (Apache-2.0):
//
//	Source file:  src/aiperf/dataset/memory_map_utils.py
//	URL (pinned): https://github.com/ai-dynamo/aiperf/blob/ab610add4e8876fca223197596c728e17365f381/src/aiperf/dataset/memory_map_utils.py
//	Revision:     ab610add4e8876fca223197596c728e17365f381
//	License:      Apache-2.0
//	Upstream copyright (retained verbatim):
//	SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//	SPDX-License-Identifier: Apache-2.0
//
// Ported mechanisms:
//   - MemoryMapDatasetClient._prefault_data_mmap: at open (when prefault is
//     requested), advise MADV_WILLNEED best-effort (upstream suppresses the
//     failure) and touch one byte per page. Upstream rationale, kept:
//     "Without this, the first read of each page takes a major fault whose
//     latency lands inside a measured request. Workers share the kernel page
//     cache, so N workers cost one disk read rather than N serialized ones."
//     The Go walk folds each touched byte into a checksum so the
//     one-byte-per-page loads are observable work the compiler cannot elide.
//   - Slice-not-seek reads: upstream slices because "the mmap file position is
//     shared state, so two concurrent readers can interleave a seek between
//     another's seek and read and serve foreign bytes." MappedRegion.Slice
//     copies straight out of the mapping; no position state exists at all.
package main

import (
	"errors"
	"os"
	"sync"
)

// Sentinel errors. Callers match with errors.Is, never against prose.
var (
	errEmptyRegion      = errors.New("loadgen: mapped region is empty")
	errClosedRegion     = errors.New("loadgen: mapped region is closed")
	errOffsetOutOfRange = errors.New("loadgen: slice offset or size out of range")
)

// PrefaultReport records what OpenMappedRegion did before the first measured
// request could run.
type PrefaultReport struct {
	Bytes    int64 // total bytes walked
	PageSize int64 // page size the walk stepped by
	Pages    int64 // pages touched; ceil(Bytes/PageSize)
	Advised  bool  // madvise(MADV_WILLNEED) issued and accepted (unix only)
}

// MappedRegion holds a whole dataset file readable by any number of
// concurrent workers. Reads are plain byte copies out of the mapping; no
// shared cursor exists, so concurrent Slices cannot serve foreign bytes.
type MappedRegion struct {
	path       string
	data       []byte // mapped view (unix) or materialized copy (!unix)
	mu         sync.Mutex
	closed     bool
	report     PrefaultReport
	touchedSum uint64 // checksum of the prefault walk; stored so the compiler cannot elide the loads
}

// OpenMappedRegion opens path read-only and maps the whole file. With
// prefault true it advises WILLNEED and touches one byte per page to reduce
// first-read page faults on a best-effort basis (upstream MMAP_PREFAULT
// behavior). Empty files are rejected: there is nothing to map.
func OpenMappedRegion(path string, prefault bool) (*MappedRegion, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.Size() == 0 {
		return nil, errEmptyRegion
	}
	r := &MappedRegion{path: path}
	if r.data, err = r.mapView(f); err != nil {
		return nil, err
	}
	if prefault {
		r.report = r.prefault()
	}
	return r, nil
}

// prefault populates the page cache for the whole region, in upstream order:
// best-effort madvise(MADV_WILLNEED) first (suppressed when unavailable),
// then a one-byte-per-page walk. The walk folds each touched byte into
// touchedSum so the loads are real work the compiler cannot drop.
func (r *MappedRegion) prefault() PrefaultReport {
	ps := r.pageUnit()
	advised := r.advise()
	var sum uint64
	for off := 0; off < len(r.data); off += int(ps) {
		sum ^= uint64(r.data[off])
	}
	r.touchedSum = sum
	n := int64(len(r.data))
	return PrefaultReport{
		Bytes:    n,
		PageSize: ps,
		Pages:    (n + ps - 1) / ps,
		Advised:  advised,
	}
}

// Slice copies [offset, offset+size) out of the region into a fresh buffer.
// It never mutates any shared position state (the upstream slice-not-seek
// contract), so concurrent readers each get exactly their own bytes. The
// copy is safe to use after Close.
func (r *MappedRegion) Slice(offset, size int64) ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errClosedRegion
	}
	n := int64(len(r.data))
	if offset < 0 || size < 0 || offset > n || size > n-offset {
		return nil, errOffsetOutOfRange
	}
	out := make([]byte, size)
	copy(out, r.data[offset:offset+size])
	return out, nil
}

// Len reports the region size in bytes, or 0 once the region is closed
// (empty regions cannot exist: OpenMappedRegion rejects them).
func (r *MappedRegion) Len() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return 0
	}
	return int64(len(r.data))
}

// Report returns the prefault report; zero-valued when opened with
// prefault=false.
func (r *MappedRegion) Report() PrefaultReport {
	return r.report
}

// Close releases the mapping. It is idempotent; Slices after Close fail
// with errClosedRegion.
func (r *MappedRegion) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	return r.release()
}
