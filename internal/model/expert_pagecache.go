package model

import (
	"fmt"
	"os"
	"sync"
)

// expert_pagecache.go - a page-cache/mmap-aware expert source for the fused GGUF routed-expert
// slabs ggufExpertSource indexes (#1302; the llama.cpp 3-D-repack-vs-mmap tension tracked as
// llama.cpp #6387 / #6798). It is the read path that lets a kernel-resident page cache absorb a
// warm expert fault, so the second read of an expert already in cache moves ZERO bytes from the
// device, and reports that fact in a ledger instead of asserting it.
//
// THE TENSION. llama.cpp repacks a fused MoE checkpoint's per-expert 3-D tensor into per-expert
// 2-D tensors (#6387) so the runtime can mmap one expert at a time without touching the whole
// slab. The repack fixes read AMPLIFICATION but forfeits the single-file mmap and its page-cache
// residency: the repacked tensors are a different on-disk layout, so a route now faults distinct
// file offsets per expert and re-pays device IO for an expert the kernel already held under the
// original layout. #6798 is the fallout - a repacked file that no longer doubles as the mapped
// training artifact. The choice looked binary because the two layouts were seen as exclusive:
// either repack for streaming or mmap the original for cache reuse.
//
// THE THROUGH-LINE. ggufExpertSource already reads exactly one expert's stride out of the
// ORIGINAL fused tensor through a caller-supplied io.ReaderAt - no repack, no format change. So
// the fused layout can stay the on-disk artifact (mmap-friendly, cache-shared) AND still stream
// one expert at a time, IF the reader can (a) tell the kernel to fault the expert's pages in and
// (b) observe that a repeat read was served from the page cache rather than the device. This file
// adds exactly those two: an aligned read-through into a caller-supplied mapped region
// (enablePageCache), and an advice shim (adviseRange) that mirrors ggufload's MmapAdvice
// vocabulary WITHOUT importing it (model must not depend on ggufload in non-test code).
//
// ALIGNMENT IS THE OVER-READ, AND IT IS THE POINT. A page-cache-backed read costs whole pages
// regardless of the expert stride: the kernel pages in [pageAlignDown(start), pageAlignUp(end))
// and discards the tail. readExpertPageCache therefore issues an ALIGNED ReadAt and slices back
// to the exact stride - byte-identical to readExpert - and books the aligned-minus-stride bytes
// as AdvantageBytes. That number is the honest, measurable difference between "double-buffered"
// (read stride bytes, then a second copy serviced the same pages) and "not double-buffered" (one
// aligned fault, stride handed out zero-copy from the mapped region on a warm hit).
//
// PAGE-GRANULAR RESIDENCY. The page cache holds PAGES, not experts, so residency is tracked per
// aligned page index rather than per expert: a cold read marks every page its aligned range
// covers, and a later read is warm iff EVERY page its own aligned range covers is already
// resident. That means the witness is real page-cache behaviour - a second read of the same
// expert is warm because the first faulted its pages, and two experts sharing a page warm each
// other - rather than a memo of the exact (tensor, expert) pair.
//
// DEFAULT OFF, NEVER EXPOSED. enablePageCache is opt-in and refuses unless the mapped data is
// exactly the declared source extent - a partial map cannot answer a read without falling back to
// the device, so a mismatch is reported false rather than silently mixed. With no enable call,
// readExpertPageCache falls through to readExpert and reports wasWarmHit=false, so no default
// load/decode path changes. PROMOTION evidence: expert_pagecache_test.go proves the FIRST read of
// an expert issues exactly one ALIGNED device ReadAt, the SECOND issues none and is a warm hit,
// and both are bit-identical to readExpert; it runs on Windows because the advice shim's no-op
// arm must keep the whole path correct where madvise does not exist. DEMOTION/retirement: retire
// if the loader grows a resident path that never faults from disk (the page cache then has nothing
// to absorb) or if llama.cpp lands a repack that also yields per-expert cache reuse. INVALIDATING
// assumption: an expert's stride is far smaller than a page-cache residency budget and a warm read
// is genuinely served from cache; on a workload that evicts before the repeat, "warm" collapses to
// a cold aligned read and AdvantageBytes is retained IO rather than a win - so any promotion must
// carry the measured bytes/token witness, never assume the reuse. This leaf models the HOST page
// cache it drives, not the kernel's private eviction; a caller that wants the kernel's truth must
// carry the measured witness, and the modeled set is the deliberately conservative one.

// PageCacheMode selects the expert read path. The zero value PageCacheOff is the default: the
// source behaves exactly as before and readExpertPageCache falls through to readExpert.
type PageCacheMode uint8

const (
	// PageCacheOff is the zero value - the default. No mapped region is retained and every read
	// goes to the device path, so nothing about the shipped behaviour changes.
	PageCacheOff PageCacheMode = iota
	// PageCacheMapped enables the aligned read-through path over a caller-supplied mapped region:
	// a cold read faults whole pages and is sliced back to the expert stride, and a repeat read is
	// served zero-copy from the mapping as a warm page-cache hit.
	PageCacheMapped
)

// String names the mode for a report or a diagnostic.
func (m PageCacheMode) String() string {
	switch m {
	case PageCacheMapped:
		return "mapped"
	default:
		return "off"
	}
}

// PageAdvice is the model-local mirror of ggufload.MmapAdvice (#5284), kept here because a
// non-test package in model must not import internal/ggufload. It selects the access hint
// adviseRange issues over a mapped expert range. The zero value PageAdviceNormal is the neutral,
// fail-closed default.
type PageAdvice uint8

const (
	// PageAdviceNormal is the neutral, no-preference hint (maps to MADV_NORMAL).
	PageAdviceNormal PageAdvice = iota
	// PageAdviceRandom hints random access (maps to MADV_RANDOM): no eager read-ahead, which is
	// the honest hint for scattered routed-expert faults with no sequential locality.
	PageAdviceRandom
	// PageAdviceSequential hints a forward streaming scan (maps to MADV_SEQUENTIAL): aggressive
	// read-ahead, the honest hint when the activated experts are read in slab order.
	PageAdviceSequential
)

// String formats the hint as a stable lowercase token mirroring ggufload.MmapAdvice.String.
func (a PageAdvice) String() string {
	switch a {
	case PageAdviceRandom:
		return "random"
	case PageAdviceSequential:
		return "sequential"
	default:
		return "normal"
	}
}

// pageAlignDown rounds off DOWN to the next page boundary - the madvise/read requirement that a
// backed range start on a page. A non-positive page leaves off unchanged (nothing to align to).
func pageAlignDown(off int64, page int) int64 {
	if page <= 0 {
		return off
	}
	rem := off % int64(page)
	if rem < 0 {
		rem += int64(page)
	}
	return off - rem
}

// pageAlignUp rounds off UP to the next page boundary - the exclusive end of the pages that back
// [off, off+length). A non-positive page leaves off unchanged.
func pageAlignUp(off int64, page int) int64 {
	if page <= 0 {
		return off
	}
	down := pageAlignDown(off, page)
	if down == off {
		return off
	}
	return down + int64(page)
}

// cachedPageSize returns the OS page size with a sane fallback: os.Getpagesize is positive on
// every supported host, but a 0/negative reading (a stubbed or exotic runtime) falls back to 4096
// so alignment is never a division by zero, only a coarser over-read.
func cachedPageSize() int {
	if ps := os.Getpagesize(); ps > 0 {
		return ps
	}
	return 4096
}

// expertPageCacheStats is the page-cache ledger - the evidence for the "not double-buffered" claim
// (#1302). BytesFromDevice is what the aligned cold reads requested and the kernel paid to serve;
// BytesFromMapping is what warm hits handed out. AdvantageBytes is the alignment over-read
// (aligned length minus stride) accumulated on cold reads, i.e. the bytes a page-cache-aware read
// moves but never exposes as expert data - the measurable difference between an aligned fault and
// a double-buffered copy. The whole struct is a ledger; nothing here drives control flow.
type expertPageCacheStats struct {
	Enabled          bool  `json:"enabled"`
	ExpertsRead      int   `json:"experts_read"`
	BytesFromDevice  int64 `json:"bytes_from_device"`
	BytesFromMapping int64 `json:"bytes_from_mapping"`
	WarmHits         int   `json:"warm_hits"`
	FillPins         int   `json:"fill_pins"`
	AdvantageBytes   int64 `json:"advantage_bytes"`
}

// ExpertPageCacheStats is the exported accessor form of expertPageCacheStats, so a caller outside
// the package can read the ledger without the unexported fields leaking into its API.
type ExpertPageCacheStats = expertPageCacheStats

// expertPageCacheState is the per-source page-cache state, held on ggufExpertSource.page and nil
// until enablePageCache is called, so the default path allocates nothing. `resident` is the
// modeled host page cache: the set of aligned page indices this source has faulted through.
type expertPageCacheState struct {
	mu       sync.Mutex
	data     []byte
	mode     PageCacheMode
	resident map[int64]struct{}
	stats    expertPageCacheStats
}

// state returns this source's page-cache state, or nil when enablePageCache was never called.
// It NEVER allocates: the field is written once by enablePageCache, before the source is
// published to any reader, so a concurrent read of s.page here is race-free. A nil return is the
// default-off signal every reader treats as "use the historical readExpert path".
func (s *ggufExpertSource) state() *expertPageCacheState { return s.page }

// enablePageCache opts this source into the aligned read-through path over data, a mapped view of
// the source's bytes. It enables ONLY when data is non-empty AND its length equals the source's
// declared extent: a partial or over-long mapping cannot answer an expert read without falling
// back to the device, and mixing the two would make the ledger lie about where bytes came from.
// It is idempotent - re-enabling with the same mapping is a no-op - and returns whether the page
// cache is enabled (true for a valid mapping, including a repeat call; false for a refused one).
func (s *ggufExpertSource) enablePageCache(data []byte) bool {
	if s == nil || len(data) == 0 || s.size != int64(len(data)) {
		return false
	}
	if s.page == nil {
		s.page = &expertPageCacheState{resident: map[int64]struct{}{}}
	}
	st := s.page
	st.mu.Lock()
	defer st.mu.Unlock()
	st.data = data
	st.mode = PageCacheMapped
	st.stats.Enabled = true
	if st.resident == nil {
		st.resident = map[int64]struct{}{}
	}
	return true
}

// readExpertPageCache reads expert e of the named fused tensor, preferring the mapped page-cache
// path when it is enabled. It returns (bytes, wasWarmHit, err). A warm hit - every aligned page
// its range covers is already in the modeled page cache - is served ZERO-COPY from the mapping (a
// three-index-capped slice of the mapped region, no device ReadAt). A cold read issues ONE
// aligned device ReadAt: offset rounded DOWN and length rounded UP to a page boundary, clamped to
// the source's bounds, into a freshly-owned buffer, then sliced back to the exact expert stride,
// and marks its pages resident. The returned bytes are always byte-identical to readExpert. When
// the page cache is not enabled this falls through to readExpert and reports wasWarmHit=false, so
// an unmapped source never changes behaviour. The ledger is updated under the state lock: a cold
// read books BytesFromDevice += aligned length, ExpertsRead++, and AdvantageBytes += alignedLen -
// stride (the alignment over-read); a warm read books WarmHits++ and BytesFromMapping += stride.
func (s *ggufExpertSource) readExpertPageCache(name string, e int) ([]byte, bool, error) {
	entry, ok := s.tensors[name]
	if !ok {
		return nil, false, fmt.Errorf("%w: %s", ErrGGUFExpertNotFound, name)
	}
	if e < 0 || e >= entry.desc.Experts {
		return nil, false, fmt.Errorf("%w: expert %d of %s outside [0,%d)", ErrGGUFExpertOutOfRange, e, name, entry.desc.Experts)
	}
	maxInt := int64(^uint(0) >> 1)
	if entry.stride > maxInt {
		return nil, false, fmt.Errorf("%w: %s expert stride %d exceeds addressable buffer size", ErrGGUFExpertMetadata, name, entry.stride)
	}
	st := s.state()
	if st == nil {
		buf, err := s.readExpert(name, e)
		return buf, false, err
	}
	st.mu.Lock()
	mode, data := st.mode, st.data
	st.mu.Unlock()
	if mode != PageCacheMapped || data == nil {
		buf, err := s.readExpert(name, e)
		return buf, false, err
	}
	stride := entry.stride
	start := entry.desc.Offset + int64(e)*stride
	if start < 0 || stride > int64(len(data)) || start > int64(len(data))-stride {
		// The mapping does not cover this expert (should be impossible after enablePageCache's
		// length gate, but a concurrent truncation must fail loudly, never slice out of range).
		return nil, false, fmt.Errorf("%w: %s expert %d range [%d,%d) outside mapping [0,%d)",
			ErrGGUFExpertMetadata, name, e, start, start+stride, len(data))
	}

	page := cachedPageSize()
	alignedStart := pageAlignDown(start, page)
	alignedEnd := pageAlignUp(start+stride, page)
	if alignedStart < 0 {
		alignedStart = 0
	}
	if alignedEnd > int64(len(data)) {
		alignedEnd = int64(len(data))
	}
	firstPage := alignedStart / int64(page)
	lastPage := (alignedEnd - 1) / int64(page)

	st.mu.Lock()
	warm := true
	for p := firstPage; p <= lastPage; p++ {
		if _, ok := st.resident[p]; !ok {
			warm = false
			break
		}
	}
	if warm {
		st.stats.WarmHits++
		st.stats.BytesFromMapping += stride
		st.mu.Unlock()
		rel := start - alignedStart
		return data[alignedStart+rel : alignedStart+rel+stride : alignedStart+rel+stride], true, nil
	}
	st.mu.Unlock()

	// Cold: one aligned device read, sliced back to the exact stride, then mark its pages resident
	// so a repeat read of this expert - or an expert sharing one of these pages - is a warm hit.
	alignedLen := alignedEnd - alignedStart
	if alignedLen > maxInt {
		return nil, false, fmt.Errorf("%w: %s aligned length %d exceeds addressable buffer size", ErrGGUFExpertMetadata, name, alignedLen)
	}
	buf := make([]byte, alignedLen)
	if _, err := s.r.ReadAt(buf, alignedStart); err != nil {
		return nil, false, fmt.Errorf("model: gguf expert page-cache read %s[%d]: %w", name, e, err)
	}
	rel := start - alignedStart
	out := buf[rel : rel+stride : rel+stride]
	st.mu.Lock()
	for p := firstPage; p <= lastPage; p++ {
		st.resident[p] = struct{}{}
	}
	st.stats.ExpertsRead++
	st.stats.BytesFromDevice += alignedLen
	if alignedLen > stride {
		st.stats.AdvantageBytes += alignedLen - stride
	}
	// A cold read that faults an expert's pages IS a fill: record it as eligible to be pinned in L2
	// by a residency tier above this source (#1302 acceptance criterion 3). It is a ledger event,
	// not a kernel pin - see markFillPin.
	st.stats.FillPins++
	st.mu.Unlock()
	return out, false, nil
}

// pageCacheSession returns a snapshot of the page-cache ledger plus whether the page cache is
// enabled. The snapshot is copied under the state lock so a caller can read the counts without
// racing a concurrent read.
func (s *ggufExpertSource) pageCacheSession() (ExpertPageCacheStats, bool) {
	if s == nil || s.page == nil {
		return ExpertPageCacheStats{}, false
	}
	st := s.page
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.stats, st.mode == PageCacheMapped && st.data != nil
}

// PageCacheEnabled reports whether this source's page-cache read path is enabled.
func (s *ggufExpertSource) PageCacheEnabled() bool {
	if s == nil {
		return false
	}
	_, on := s.pageCacheSession()
	return on
}
