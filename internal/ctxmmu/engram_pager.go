package ctxmmu

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
)

// Default configuration constants for 51B Engram / PLE tables.
const (
	// DefaultRowSizeBytes defines the per-layer embedding row width (130 bytes for 4-bit quant PLE).
	DefaultRowSizeBytes uint64 = 130

	// DefaultMaxResidentBytes defines the resident RAM budget ceiling (2 GiB RSS ceiling).
	DefaultMaxResidentBytes uint64 = 2 * 1024 * 1024 * 1024

	// DefaultPageSizeBytes defines standard OS virtual memory page granularity (4 KiB).
	DefaultPageSizeBytes uint64 = 4096

	// StandardReadaheadBytes defines standard sequential OS readahead window (128 KiB).
	StandardReadaheadBytes uint64 = 128 * 1024
)

// POSIX madvise advisory flags.
const (
	// MadvNormal signals default kernel clustering/readahead heuristics.
	MadvNormal = 0

	// MadvRandom advises kernel that access will be random; suppresses sequential readahead.
	MadvRandom = 1

	// MadvSequential advises kernel that access will be sequential; enables aggressive readahead.
	MadvSequential = 2

	// MadvWillNeed advises kernel to prefetch the specified page range into RAM.
	MadvWillNeed = 3

	// MadvDontNeed advises kernel that the specified page range can be dropped from RSS.
	MadvDontNeed = 4
)

var (
	// ErrTotalRowsZero indicates total rows configuration cannot be zero.
	ErrTotalRowsZero = errors.New("engram_pager: total rows must be greater than zero")

	// ErrRowOutOfBounds indicates requested row index exceeds total table rows.
	ErrRowOutOfBounds = errors.New("engram_pager: row index out of bounds")

	// ErrOutputBufferSmall indicates output gather slice is smaller than required row bytes.
	ErrOutputBufferSmall = errors.New("engram_pager: output buffer too small")

	// ErrPagerClosed indicates operations were attempted on a closed pager instance.
	ErrPagerClosed = errors.New("engram_pager: pager is closed")
)

// EngramPagerConfig configures SSD-backed batched row readahead for 51B PLE tables.
type EngramPagerConfig struct {
	// TablePath is the filesystem path to the raw tensor table on SSD/NVMe (empty for anonymous).
	TablePath string `json:"table_path"`

	// RowSizeBytes is the byte width of an individual embedding row (e.g. 130 bytes).
	RowSizeBytes uint64 `json:"row_size_bytes"`

	// TotalRows is the total count of embedding rows in the table.
	TotalRows uint64 `json:"total_rows"`

	// MaxResidentBytes sets the maximum resident RAM budget (e.g. 2 GiB RSS limit).
	MaxResidentBytes uint64 `json:"max_resident_bytes"`

	// PinFullRAM when true disables lazy page eviction, keeping the full working set in DRAM.
	PinFullRAM bool `json:"pin_full_ram"`

	// PageSizeBytes defines memory page alignment granularity (defaults to 4096).
	PageSizeBytes uint64 `json:"page_size_bytes"`

	// AnonymousData supplies pre-allocated in-memory backing data for synthetic tests.
	AnonymousData []byte `json:"-"`

	// MadviseSyscall is an optional syscall or simulation hook for OS madvise advisories.
	MadviseSyscall func(addr uintptr, length uintptr, advice int) error `json:"-"`
}

// AdvisoryCall records an issued madvise OS hint for auditing and verification.
type AdvisoryCall struct {
	Offset uintptr `json:"offset"`
	Length uintptr `json:"length"`
	Advice int     `json:"advice"`
}

// EngramTelemetry provides real-time telemetry on paging efficiency, RSS footprint,
// and NVMe read amplification reduction.
type EngramTelemetry struct {
	ResidentBytes           uint64  `json:"resident_bytes"`
	ResidentPages           uint64  `json:"resident_pages"`
	TotalGatheredBytes      uint64  `json:"total_gathered_bytes"`
	NVMeReadBytes           uint64  `json:"nvme_read_bytes"`
	SequentialBaselineBytes uint64  `json:"sequential_baseline_bytes"`
	CacheHits               uint64  `json:"cache_hits"`
	CacheMisses             uint64  `json:"cache_misses"`
	PrefetchBatches         uint64  `json:"prefetch_batches"`
	PrefetchPages           uint64  `json:"prefetch_pages"`
	EvictedPages            uint64  `json:"evicted_pages"`
	ReadAmplification       float64 `json:"read_amplification"`
	StandardAmplification   float64 `json:"standard_amplification"`
}

type pageEntry struct {
	mu     sync.Mutex
	pins   int
	loaded bool
	pageID uint64
	data   []byte
	prev   *pageEntry
	next   *pageEntry
}

// EngramPager manages lazy batched row readahead for SSD-backed embedding tables.
// It applies MADV_RANDOM to eliminate wasteful 128 KiB OS readahead, pre-fetches
// exact micro-batch row footprints via batched MADV_WILLNEED, and enforces strict
// resident RAM limits (<= 2.0 GiB RSS) via LRU-clock MADV_DONTNEED eviction.
type EngramPager struct {
	mu         sync.Mutex
	life       sync.RWMutex
	fileReader func([]byte, int64) (int, error)
	fileWriter func([]byte, int64) (int, error)

	cfg        EngramPagerConfig
	file       *os.File
	anonData   []byte
	totalBytes uint64

	residentPages        map[uint64]*pageEntry
	lruHead              *pageEntry
	lruTail              *pageEntry
	currentResidentBytes uint64

	telemetry   EngramTelemetry
	madviseHook func(addr uintptr, length uintptr, advice int) error
	advisories  []AdvisoryCall
	closed      bool
}

// NewEngramPager initializes an engram table pager with MADV_RANDOM advisory.
func NewEngramPager(cfg EngramPagerConfig) (*EngramPager, error) {
	if cfg.TotalRows == 0 {
		return nil, ErrTotalRowsZero
	}
	if cfg.RowSizeBytes == 0 {
		cfg.RowSizeBytes = DefaultRowSizeBytes
	}
	if cfg.MaxResidentBytes == 0 {
		cfg.MaxResidentBytes = DefaultMaxResidentBytes
	}
	if cfg.PageSizeBytes == 0 {
		cfg.PageSizeBytes = DefaultPageSizeBytes
	}

	totalBytes := cfg.TotalRows * cfg.RowSizeBytes

	var file *os.File
	var anonData []byte

	if cfg.TablePath != "" {
		f, err := os.OpenFile(cfg.TablePath, os.O_RDWR, 0644)
		if err != nil {
			// Fall back to read-only if write not permitted
			f, err = os.Open(cfg.TablePath)
			if err != nil {
				return nil, fmt.Errorf("engram_pager: open table file: %w", err)
			}
		}
		file = f
	} else if len(cfg.AnonymousData) > 0 {
		anonData = cfg.AnonymousData
	} else if totalBytes <= 64*1024*1024 {
		// Small to medium table in anonymous mode: allocate direct memory buffer
		anonData = make([]byte, totalBytes)
		for i := range anonData {
			anonData[i] = byte(i % 251)
		}
	}

	p := &EngramPager{
		cfg:           cfg,
		file:          file,
		anonData:      anonData,
		totalBytes:    totalBytes,
		residentPages: make(map[uint64]*pageEntry),
		madviseHook:   cfg.MadviseSyscall,
	}

	if file != nil {
		p.fileReader, p.fileWriter = file.ReadAt, file.WriteAt
	}
	// Issue MADV_RANDOM advisory across the entire virtual table space
	p.issueMadvise(0, uintptr(totalBytes), MadvRandom)

	return p, nil
}

// PrefetchUbatchRows deduplicates and sorts row offsets for the upcoming micro-batch,
// coalesces contiguous page intervals, and issues batched madvise(MADV_WILLNEED) hints
// over aligned page boundaries.
func (p *EngramPager) PrefetchUbatchRows(ctx context.Context, rowIndices []uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	p.life.RLock()
	defer p.life.RUnlock()
	if err := p.validateRows(rowIndices); err != nil {
		return err
	}
	if len(rowIndices) == 0 {
		return nil
	}
	pages := p.rowPages(rowIndices)
	// Keep the original coalesced page-span advisory shape.
	for i := 0; i < len(pages); {
		j := i + 1
		for j < len(pages) && pages[j] == pages[j-1]+1 {
			j++
		}
		p.issueMadvise(uintptr(pages[i]*p.cfg.PageSizeBytes), uintptr(uint64(j-i)*p.cfg.PageSizeBytes), MadvWillNeed)
		i = j
	}
	for _, pg := range pages {
		if err := ctx.Err(); err != nil {
			return err
		}
		entry, err := p.acquirePage(pg, true)
		if err != nil {
			return err
		}
		entry.mu.Unlock()
		p.releasePage(entry)
	}
	p.mu.Lock()
	p.telemetry.PrefetchBatches++
	p.updateTelemetryLocked()
	p.mu.Unlock()
	return nil
}

func (p *EngramPager) validateRows(rows []uint64) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return ErrPagerClosed
	}
	for _, row := range rows {
		if row >= p.cfg.TotalRows {
			return fmt.Errorf("%w: row %d >= %d", ErrRowOutOfBounds, row, p.cfg.TotalRows)
		}
	}
	return nil
}

func (p *EngramPager) rowPages(rows []uint64) []uint64 {
	set := make(map[uint64]struct{})
	for _, row := range rows {
		begin := row * p.cfg.RowSizeBytes
		for pg, last := begin/p.cfg.PageSizeBytes, (begin+p.cfg.RowSizeBytes-1)/p.cfg.PageSizeBytes; pg <= last; pg++ {
			set[pg] = struct{}{}
		}
	}
	pages := make([]uint64, 0, len(set))
	for pg := range set {
		pages = append(pages, pg)
	}
	sort.Slice(pages, func(i, j int) bool { return pages[i] < pages[j] })
	return pages
}

// acquirePage reserves ownership under the global mutex, then faults and
// publishes bytes under the page mutex. Unrelated pages and telemetry proceed
// while a disk read is blocked. Returned entries hold their byte mutex.
func (p *EngramPager) acquirePage(pg uint64, prefetch bool) (*pageEntry, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, ErrPagerClosed
	}
	entry := p.residentPages[pg]
	if entry == nil {
		entry = &pageEntry{pageID: pg}
		p.residentPages[pg] = entry
		p.lruPushFront(entry)
		p.currentResidentBytes += p.cfg.PageSizeBytes
	} else {
		p.lruMoveToFront(entry)
	}
	entry.pins++
	p.mu.Unlock()
	entry.mu.Lock()
	if !entry.loaded {
		entry.data = p.loadPageData(pg)
		entry.loaded = true
		p.mu.Lock()
		p.telemetry.NVMeReadBytes += p.cfg.PageSizeBytes
		if prefetch {
			p.telemetry.PrefetchPages++
		}
		p.updateTelemetryLocked()
		p.mu.Unlock()
	}
	return entry, nil
}

func (p *EngramPager) releasePage(entry *pageEntry) {
	p.mu.Lock()
	entry.pins--
	evicted := p.enforceResidentLimit()
	p.updateTelemetryLocked()
	p.mu.Unlock()
	for _, pg := range evicted {
		p.issueMadvise(uintptr(pg*p.cfg.PageSizeBytes), uintptr(p.cfg.PageSizeBytes), MadvDontNeed)
	}
}

// GatherRows preserves flat row order, including rows crossing page boundaries.
func (p *EngramPager) GatherRows(rowIndices []uint64, out []byte) error {
	p.life.RLock()
	defer p.life.RUnlock()
	if err := p.validateRows(rowIndices); err != nil {
		return err
	}
	needed := uint64(len(rowIndices)) * p.cfg.RowSizeBytes
	if uint64(len(out)) < needed {
		return fmt.Errorf("%w: have %d bytes, need %d", ErrOutputBufferSmall, len(out), needed)
	}
	for i, row := range rowIndices {
		start := row * p.cfg.RowSizeBytes
		pages := p.rowPages([]uint64{row})
		p.mu.Lock()
		hit := true
		for _, pg := range pages {
			if p.residentPages[pg] == nil {
				hit = false
				break
			}
		}
		if hit {
			p.telemetry.CacheHits++
		} else {
			p.telemetry.CacheMisses++
		}
		p.mu.Unlock()
		copied := uint64(0)
		for _, pg := range pages {
			entry, err := p.acquirePage(pg, false)
			if err != nil {
				return err
			}
			offset := (start + copied) % p.cfg.PageSizeBytes
			n := min(p.cfg.PageSizeBytes-offset, p.cfg.RowSizeBytes-copied)
			dst := uint64(i)*p.cfg.RowSizeBytes + copied
			copy(out[dst:dst+n], entry.data[offset:offset+n])
			entry.mu.Unlock()
			p.releasePage(entry)
			copied += n
		}
	}
	p.mu.Lock()
	p.telemetry.TotalGatheredBytes += needed
	p.telemetry.SequentialBaselineBytes += uint64(len(rowIndices)) * StandardReadaheadBytes
	p.updateTelemetryLocked()
	p.mu.Unlock()
	return nil
}

// WriteRow locks affected page ownership in ascending order. File writes and
// byte mutation never hold the global LRU/telemetry mutex.
func (p *EngramPager) WriteRow(rowIdx uint64, data []byte) error {
	p.life.RLock()
	defer p.life.RUnlock()
	if err := p.validateRows([]uint64{rowIdx}); err != nil {
		return err
	}
	size := p.cfg.RowSizeBytes
	if uint64(len(data)) < size {
		return fmt.Errorf("engram_pager: write data slice too short (%d < %d)", len(data), size)
	}
	pages := p.rowPages([]uint64{rowIdx})
	entries := make([]*pageEntry, 0, len(pages))
	defer func() {
		for _, entry := range entries {
			entry.mu.Unlock()
			p.releasePage(entry)
		}
	}()
	for _, pg := range pages {
		entry, err := p.acquirePage(pg, false)
		if err != nil {
			return err
		}
		entries = append(entries, entry)
	}
	start := rowIdx * size
	if p.fileWriter != nil {
		if _, err := p.fileWriter(data[:size], int64(start)); err != nil {
			return fmt.Errorf("engram_pager: write to file: %w", err)
		}
	}
	if uint64(len(p.anonData)) >= start+size {
		copy(p.anonData[start:start+size], data[:size])
	}
	for _, entry := range entries {
		begin := max(start, entry.pageID*p.cfg.PageSizeBytes)
		end := min(start+size, (entry.pageID+1)*p.cfg.PageSizeBytes)
		copy(entry.data[begin-entry.pageID*p.cfg.PageSizeBytes:end-entry.pageID*p.cfg.PageSizeBytes], data[begin-start:end-start])
	}
	return nil
}

func (p *EngramPager) loadPageData(pg uint64) []byte {
	size := p.cfg.PageSizeBytes
	data := make([]byte, size)
	offset := pg * size
	if p.fileReader != nil {
		_, _ = p.fileReader(data, int64(offset))
		return data
	}
	if len(p.anonData) > 0 {
		if offset < uint64(len(p.anonData)) {
			copy(data, p.anonData[offset:min(offset+size, uint64(len(p.anonData)))])
		}
		return data
	}
	for i := range data {
		data[i] = byte((offset + uint64(i)) % 251)
	}
	return data
}

// enforceResidentLimit mutates only bookkeeping; callers run returned advice
// outside p.mu. Pinned in-flight pages cannot be evicted and reloaded under a
// different byte mutex. The budget is restored when their last user releases.
func (p *EngramPager) enforceResidentLimit() []uint64 {
	var evicted []uint64
	if p.cfg.PinFullRAM {
		return nil
	}
	for p.currentResidentBytes > p.cfg.MaxResidentBytes {
		victim := p.lruTail
		for victim != nil && victim.pins > 0 {
			victim = victim.prev
		}
		if victim == nil {
			break
		}
		p.lruRemove(victim)
		delete(p.residentPages, victim.pageID)
		p.currentResidentBytes -= p.cfg.PageSizeBytes
		p.telemetry.EvictedPages++
		evicted = append(evicted, victim.pageID)
	}
	return evicted
}

func (p *EngramPager) issueMadvise(offset uintptr, length uintptr, advice int) {
	p.mu.Lock()
	p.advisories = append(p.advisories, AdvisoryCall{Offset: offset, Length: length, Advice: advice})
	hook := p.madviseHook
	p.mu.Unlock()
	if hook != nil {
		_ = hook(offset, length, advice)
	}
}

func (p *EngramPager) updateTelemetryLocked() {
	p.telemetry.ResidentBytes = p.currentResidentBytes
	p.telemetry.ResidentPages = uint64(len(p.residentPages))
	if p.telemetry.TotalGatheredBytes > 0 {
		p.telemetry.ReadAmplification = float64(p.telemetry.NVMeReadBytes) / float64(p.telemetry.TotalGatheredBytes)
		p.telemetry.StandardAmplification = float64(p.telemetry.SequentialBaselineBytes) / float64(p.telemetry.TotalGatheredBytes)
	}
}

// Telemetry returns an atomic copy of active paging and amplification metrics.
func (p *EngramPager) Telemetry() EngramTelemetry {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.telemetry
}

// ResidentBytes returns the current estimated resident RSS footprint in bytes.
func (p *EngramPager) ResidentBytes() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.currentResidentBytes
}

// ReadAmplificationRatio returns the actual NVMe read amplification ratio
// (NVMeReadBytes / TotalGatheredBytes).
func (p *EngramPager) ReadAmplificationRatio() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.telemetry.ReadAmplification
}

// CacheHitRatio returns the ratio of cache hits to total row gathers.
func (p *EngramPager) CacheHitRatio() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	total := p.telemetry.CacheHits + p.telemetry.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(p.telemetry.CacheHits) / float64(total)
}

// TotalGatheredBytes returns the cumulative payload bytes gathered.
func (p *EngramPager) TotalGatheredBytes() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.telemetry.TotalGatheredBytes
}

// CacheHits returns count of prefetched hits during gather.
func (p *EngramPager) CacheHits() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.telemetry.CacheHits
}

// CacheMisses returns count of unprefetched misses during gather.
func (p *EngramPager) CacheMisses() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.telemetry.CacheMisses
}

// NVMeReadBytes returns the total bytes read from disk/backing store.
func (p *EngramPager) NVMeReadBytes() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.telemetry.NVMeReadBytes
}

// Advisories returns a copy of all recorded OS madvise calls.
func (p *EngramPager) Advisories() []AdvisoryCall {
	p.mu.Lock()
	defer p.mu.Unlock()
	cp := make([]AdvisoryCall, len(p.advisories))
	copy(cp, p.advisories)
	return cp
}

// Close releases open file handles and flushes memory structures.
func (p *EngramPager) Close() error {
	p.life.Lock()
	defer p.life.Unlock()
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil
	}
	p.closed = true

	file := p.file
	p.file = nil
	p.mu.Unlock()
	var err error
	if file != nil {
		err = file.Close()
	}
	p.mu.Lock()
	p.residentPages = nil
	p.lruHead = nil
	p.lruTail = nil
	return err
}

// LRU list maintenance helpers (must be called with p.mu held)

func (p *EngramPager) lruPushFront(entry *pageEntry) {
	entry.prev = nil
	entry.next = p.lruHead
	if p.lruHead != nil {
		p.lruHead.prev = entry
	}
	p.lruHead = entry
	if p.lruTail == nil {
		p.lruTail = entry
	}
}

func (p *EngramPager) lruMoveToFront(entry *pageEntry) {
	if p.lruHead == entry {
		return
	}
	p.lruRemove(entry)
	p.lruPushFront(entry)
}

func (p *EngramPager) lruRemove(entry *pageEntry) {
	if entry.prev != nil {
		entry.prev.next = entry.next
	} else {
		p.lruHead = entry.next
	}
	if entry.next != nil {
		entry.next.prev = entry.prev
	} else {
		p.lruTail = entry.prev
	}
	entry.prev = nil
	entry.next = nil
}
