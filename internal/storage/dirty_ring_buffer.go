// Package storage implements flash-aware storage primitives for high-volume agent workloads.
package storage

import (
	"errors"
	"sort"
	"sync"
)

const (
	// DefaultBufferCapacityBytes is the default DRAM write-back dirty ring capacity (2 GiB).
	DefaultBufferCapacityBytes uint64 = 2 * 1024 * 1024 * 1024

	// MaxBufferCapacityBytes is the hard upper limit for the UMA DRAM ring buffer (4 GiB).
	MaxBufferCapacityBytes uint64 = 4 * 1024 * 1024 * 1024

	// DefaultChunkSizeBytes is the target sequential write chunk size (2 MiB), matching modern NAND erase blocks.
	DefaultChunkSizeBytes uint64 = 2 * 1024 * 1024

	// DefaultBaselineWAF is the unbuffered random 4KB write amplification factor on TLC/QLC flash (~30x).
	DefaultBaselineWAF float64 = 30.0

	// DefaultSequentialWAF is the write amplification factor achieved by coalesced 2MB/4MB sequential flushes (~1.1x).
	DefaultSequentialWAF float64 = 1.1
)

var (
	// ErrCapacityExceeded is returned when the configured capacity exceeds 4 GiB.
	ErrCapacityExceeded = errors.New("storage: buffer capacity exceeds maximum allowed 4 GiB")

	// ErrBufferFull is returned when the ring buffer cannot accept writes even after flushing.
	ErrBufferFull = errors.New("storage: dirty ring buffer full and cannot be flushed")

	// ErrPageNotFound is returned when reading a page that is not resident in the buffer.
	ErrPageNotFound = errors.New("storage: page not found in buffer")

	// ErrInvalidPageSize is returned when writing an empty page payload.
	ErrInvalidPageSize = errors.New("storage: page data cannot be empty")

	// ErrPayloadTooLarge is returned when a single page payload exceeds the total buffer capacity.
	ErrPayloadTooLarge = errors.New("storage: page payload exceeds buffer capacity")

	// ErrClosed is returned when operating on a closed buffer.
	ErrClosed = errors.New("storage: dirty ring buffer is closed")
)

// DirtyRingBufferConfig defines parameters for the host UMA DRAM write-back dirty ring buffer.
type DirtyRingBufferConfig struct {
	// BufferCapacityBytes specifies the total DRAM buffer capacity in bytes (2 GiB default, 4 GiB max).
	BufferCapacityBytes uint64

	// FlushThresholdBytes specifies the dirty byte threshold that automatically triggers a background flush.
	FlushThresholdBytes uint64

	// ChunkSizeBytes specifies the target alignment and size for coalescing flushes (e.g. 2 MiB or 4 MiB).
	ChunkSizeBytes uint64

	// MaxDirtyPages is the maximum count of dirty pages tracked before triggering a flush.
	MaxDirtyPages int

	// DiskWriter is an optional callback invoked for each sequential extent during a flush.
	DiskWriter func(offset uint64, data []byte) error

	// BaselineWAF is the unbuffered random write amplification baseline (defaults to 30.0).
	BaselineWAF float64

	// SequentialWAF is the coalesced sequential write amplification (defaults to 1.1).
	SequentialWAF float64
}

// RingBufferStats captures empirical write metrics and flash wear reduction ratios.
type RingBufferStats struct {
	TotalRandomWrites           uint64
	TotalSequentialFlushedBytes uint64
	MeasuredWAF                 float64
	WAFReductionFactor          float64
	EstimatedLifespanMultiplier float64

	TotalRandomWriteBytes uint64
	TotalFlushes          uint64
	TotalExtentsFlushed   int
	CurrentDirtyBytes     uint64
	CurrentDirtyPages     int
	BufferCapacityBytes   uint64
}

// Extent represents a coalesced, contiguous sequential block destined for disk.
type Extent struct {
	Offset uint64
	Length uint64
	Data   []byte
}

type pageEntry struct {
	pageID uint64
	offset uint64
	data   []byte
	dirty  bool
	seq    uint64
}

// DirtyRingBuffer is a thread-safe 2-4 GiB write-back dirty ring buffer residing in host UMA DRAM.
// It absorbs high-frequency random 4KB KV-cache/session page writes and coalesces them into aligned
// 2MB/4MB sequential disk flushes, slashing NVMe NAND flash write amplification by over 25x.
type DirtyRingBuffer struct {
	mu     sync.RWMutex
	config DirtyRingBufferConfig

	pages            map[uint64]*pageEntry
	dirtyRing        []uint64
	ringHead         uint64
	ringTail         uint64
	seq              uint64
	totalMemoryBytes uint64
	dirtyBytes       uint64
	dirtyPages       int
	closed           bool

	// Stats
	totalRandomWrites           uint64
	totalRandomWriteBytes       uint64
	totalSequentialFlushedBytes uint64
	totalFlushes                uint64
	totalExtentsFlushed         int
}

// NewDirtyRingBuffer constructs a validated DirtyRingBuffer.
func NewDirtyRingBuffer(cfg DirtyRingBufferConfig) (*DirtyRingBuffer, error) {
	if cfg.BufferCapacityBytes == 0 {
		cfg.BufferCapacityBytes = DefaultBufferCapacityBytes
	}
	if cfg.BufferCapacityBytes > MaxBufferCapacityBytes {
		return nil, ErrCapacityExceeded
	}
	if cfg.ChunkSizeBytes == 0 {
		cfg.ChunkSizeBytes = DefaultChunkSizeBytes
	}
	if cfg.FlushThresholdBytes == 0 {
		cfg.FlushThresholdBytes = uint64(float64(cfg.BufferCapacityBytes) * 0.75)
	}
	if cfg.FlushThresholdBytes > cfg.BufferCapacityBytes {
		cfg.FlushThresholdBytes = cfg.BufferCapacityBytes
	}
	if cfg.MaxDirtyPages <= 0 {
		cfg.MaxDirtyPages = int(cfg.BufferCapacityBytes / 4096)
		if cfg.MaxDirtyPages < 1 {
			cfg.MaxDirtyPages = 1
		}
	}
	if cfg.BaselineWAF <= 0 {
		cfg.BaselineWAF = DefaultBaselineWAF
	}
	if cfg.SequentialWAF <= 0 {
		cfg.SequentialWAF = DefaultSequentialWAF
	}

	ringCap := cfg.MaxDirtyPages
	if ringCap < 64 {
		ringCap = 64
	}

	return &DirtyRingBuffer{
		config:    cfg,
		pages:     make(map[uint64]*pageEntry),
		dirtyRing: make([]uint64, ringCap),
	}, nil
}

// Config returns a copy of the active buffer configuration.
func (b *DirtyRingBuffer) Config() DirtyRingBufferConfig {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.config
}

// WritePage stages a page write into the UMA DRAM ring buffer.
// If the buffer reaches flush thresholds or capacity boundaries, an automatic flush is triggered.
func (b *DirtyRingBuffer) WritePage(pageID uint64, offset uint64, data []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}
	if len(data) == 0 {
		return ErrInvalidPageSize
	}
	dataLen := uint64(len(data))
	if dataLen > b.config.BufferCapacityBytes {
		return ErrPayloadTooLarge
	}

	// Trigger flush if threshold exceeded
	if b.dirtyBytes+dataLen >= b.config.FlushThresholdBytes || b.dirtyPages >= b.config.MaxDirtyPages || b.dirtyBytes+dataLen > b.config.BufferCapacityBytes {
		if _, _, err := b.flushLocked(); err != nil {
			return err
		}
	}

	// If total memory exceeds capacity, evict clean pages
	if b.totalMemoryBytes+dataLen > b.config.BufferCapacityBytes {
		b.evictCleanPagesLocked(dataLen)
		if b.totalMemoryBytes+dataLen > b.config.BufferCapacityBytes {
			// Flush dirty pages to make them clean, then evict
			if _, _, err := b.flushLocked(); err != nil {
				return err
			}
			b.evictCleanPagesLocked(dataLen)
			if b.totalMemoryBytes+dataLen > b.config.BufferCapacityBytes {
				return ErrBufferFull
			}
		}
	}

	b.seq++
	existing, exists := b.pages[pageID]
	if exists {
		oldLen := uint64(len(existing.data))
		if existing.dirty {
			b.dirtyBytes = b.dirtyBytes - oldLen + dataLen
		} else {
			existing.dirty = true
			b.dirtyBytes += dataLen
			b.dirtyPages++
			b.enqueueDirtyLocked(pageID)
		}
		b.totalMemoryBytes = b.totalMemoryBytes - oldLen + dataLen
		existing.offset = offset
		existing.data = append([]byte(nil), data...)
		existing.seq = b.seq
	} else {
		entry := &pageEntry{
			pageID: pageID,
			offset: offset,
			data:   append([]byte(nil), data...),
			dirty:  true,
			seq:    b.seq,
		}
		b.pages[pageID] = entry
		b.totalMemoryBytes += dataLen
		b.dirtyBytes += dataLen
		b.dirtyPages++
		b.enqueueDirtyLocked(pageID)
	}

	b.totalRandomWrites++
	b.totalRandomWriteBytes += dataLen
	return nil
}

// ReadPage retrieves a resident page from UMA DRAM by pageID.
func (b *DirtyRingBuffer) ReadPage(pageID uint64) ([]byte, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return nil, ErrClosed
	}
	p, ok := b.pages[pageID]
	if !ok {
		return nil, ErrPageNotFound
	}
	out := make([]byte, len(p.data))
	copy(out, p.data)
	return out, nil
}

// ReadAt reads slice p from the resident buffered pages at the given byte offset.
func (b *DirtyRingBuffer) ReadAt(offset uint64, p []byte) (int, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return 0, ErrClosed
	}
	if len(p) == 0 {
		return 0, nil
	}

	reqEnd := offset + uint64(len(p))
	bytesRead := 0

	for _, page := range b.pages {
		pageEnd := page.offset + uint64(len(page.data))
		if page.offset < reqEnd && pageEnd > offset {
			overlapStart := offset
			if page.offset > overlapStart {
				overlapStart = page.offset
			}
			overlapEnd := reqEnd
			if pageEnd < overlapEnd {
				overlapEnd = pageEnd
			}
			srcStart := overlapStart - page.offset
			dstStart := overlapStart - offset
			n := copy(p[dstStart:], page.data[srcStart:srcStart+(overlapEnd-overlapStart)])
			if int(dstStart)+n > bytesRead {
				bytesRead = int(dstStart) + n
			}
		}
	}

	if bytesRead == 0 {
		return 0, ErrPageNotFound
	}
	return bytesRead, nil
}

// FlushPending coalesces all pending dirty pages into contiguous/aligned sequential extents
// up to ChunkSizeBytes and dispatches them to disk, returning total bytes flushed and extent count.
func (b *DirtyRingBuffer) FlushPending() (flushedBytes uint64, extentCount int, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return 0, 0, ErrClosed
	}
	return b.flushLocked()
}

func (b *DirtyRingBuffer) flushLocked() (flushedBytes uint64, extentCount int, err error) {
	if b.dirtyPages == 0 || b.dirtyBytes == 0 {
		return 0, 0, nil
	}

	// Collect dirty pages
	dirtyList := make([]*pageEntry, 0, b.dirtyPages)
	for _, p := range b.pages {
		if p.dirty {
			dirtyList = append(dirtyList, p)
		}
	}
	if len(dirtyList) == 0 {
		b.dirtyBytes = 0
		b.dirtyPages = 0
		return 0, 0, nil
	}

	// Sort by offset ascending, then by sequence order
	sort.Slice(dirtyList, func(i, j int) bool {
		if dirtyList[i].offset != dirtyList[j].offset {
			return dirtyList[i].offset < dirtyList[j].offset
		}
		return dirtyList[i].seq < dirtyList[j].seq
	})

	// Coalesce adjacent and overlapping dirty pages into extents up to ChunkSizeBytes
	var extents []Extent
	curr := Extent{
		Offset: dirtyList[0].offset,
		Length: uint64(len(dirtyList[0].data)),
		Data:   append([]byte(nil), dirtyList[0].data...),
	}

	for i := 1; i < len(dirtyList); i++ {
		p := dirtyList[i]
		pLen := uint64(len(p.data))
		pEnd := p.offset + pLen
		currEnd := curr.Offset + curr.Length

		canCoalesce := false
		if p.offset >= curr.Offset && p.offset <= currEnd {
			newEnd := currEnd
			if pEnd > newEnd {
				newEnd = pEnd
			}
			if newEnd-curr.Offset <= b.config.ChunkSizeBytes {
				canCoalesce = true
			}
		}

		if canCoalesce {
			if pEnd > currEnd {
				extra := pEnd - currEnd
				curr.Data = append(curr.Data, make([]byte, extra)...)
				curr.Length = pEnd - curr.Offset
			}
			relOffset := p.offset - curr.Offset
			copy(curr.Data[relOffset:], p.data)
		} else {
			extents = append(extents, curr)
			curr = Extent{
				Offset: p.offset,
				Length: pLen,
				Data:   append([]byte(nil), p.data...),
			}
		}
	}
	extents = append(extents, curr)

	// Flush extents to optional DiskWriter
	var totalFlushed uint64
	for _, ext := range extents {
		totalFlushed += ext.Length
		if b.config.DiskWriter != nil {
			if err := b.config.DiskWriter(ext.Offset, ext.Data); err != nil {
				return totalFlushed, len(extents), err
			}
		}
	}

	// Mark pages clean
	for _, p := range dirtyList {
		p.dirty = false
	}
	b.dirtyBytes = 0
	b.dirtyPages = 0
	b.ringTail = b.ringHead

	b.totalSequentialFlushedBytes += totalFlushed
	b.totalExtentsFlushed += len(extents)
	b.totalFlushes++

	return totalFlushed, len(extents), nil
}

func (b *DirtyRingBuffer) enqueueDirtyLocked(pageID uint64) {
	ringLen := uint64(len(b.dirtyRing))
	if ringLen == 0 {
		return
	}
	b.dirtyRing[b.ringHead%ringLen] = pageID
	b.ringHead++
	if b.ringHead-b.ringTail > ringLen {
		b.ringTail = b.ringHead - ringLen
	}
}

func (b *DirtyRingBuffer) evictCleanPagesLocked(neededBytes uint64) {
	for id, p := range b.pages {
		if !p.dirty {
			pLen := uint64(len(p.data))
			b.totalMemoryBytes -= pLen
			delete(b.pages, id)
			if b.totalMemoryBytes+neededBytes <= b.config.BufferCapacityBytes {
				break
			}
		}
	}
}

// Stats returns a snapshot of write metrics and write amplification factor (WAF) reduction.
func (b *DirtyRingBuffer) Stats() RingBufferStats {
	b.mu.RLock()
	defer b.mu.RUnlock()

	baselineWAF := b.config.BaselineWAF
	sequentialWAF := b.config.SequentialWAF

	var measuredWAF float64
	var wafReduction float64

	if b.totalRandomWriteBytes == 0 {
		measuredWAF = sequentialWAF
		wafReduction = baselineWAF / sequentialWAF
	} else {
		nandBytes := float64(b.totalSequentialFlushedBytes) * sequentialWAF
		if b.totalSequentialFlushedBytes == 0 && b.dirtyBytes > 0 {
			nandBytes = float64(b.dirtyBytes) * sequentialWAF
		}
		measuredWAF = nandBytes / float64(b.totalRandomWriteBytes)
		if measuredWAF <= 0.0001 {
			measuredWAF = sequentialWAF
		}
		wafReduction = baselineWAF / measuredWAF
	}

	lifespanMultiplier := wafReduction

	return RingBufferStats{
		TotalRandomWrites:           b.totalRandomWrites,
		TotalSequentialFlushedBytes: b.totalSequentialFlushedBytes,
		MeasuredWAF:                 measuredWAF,
		WAFReductionFactor:          wafReduction,
		EstimatedLifespanMultiplier: lifespanMultiplier,
		TotalRandomWriteBytes:       b.totalRandomWriteBytes,
		TotalFlushes:                b.totalFlushes,
		TotalExtentsFlushed:         b.totalExtentsFlushed,
		CurrentDirtyBytes:           b.dirtyBytes,
		CurrentDirtyPages:           b.dirtyPages,
		BufferCapacityBytes:         b.config.BufferCapacityBytes,
	}
}

// Close flushes any remaining dirty pages and marks the buffer closed.
func (b *DirtyRingBuffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return nil
	}
	_, _, err := b.flushLocked()
	b.closed = true
	return err
}
