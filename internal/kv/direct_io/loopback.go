package directio

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
	"unsafe"
)

// BlockSize is the canonical 4096-byte (4KB) page size for direct-I/O block storage.
const BlockSize = 4096

// Errors returned by LoopbackKVStore.
var (
	ErrStoreClosed      = errors.New("kv direct_io: loopback store is closed")
	ErrStoreFull        = errors.New("kv direct_io: loopback store capacity reached")
	ErrBlockNotFound    = errors.New("kv direct_io: block not found or unallocated")
	ErrInvalidBlockID   = errors.New("kv direct_io: invalid block id")
	ErrDataTooLarge     = errors.New("kv direct_io: data exceeds block size")
	ErrChecksumMismatch = errors.New("kv direct_io: block checksum mismatch")
)

// BlockMetadata records placement, ownership, token range, and access metadata for a KV block.
type BlockMetadata struct {
	BlockID      int       `json:"block_id"`
	SessionID    string    `json:"session_id"`
	Turn         int       `json:"turn"`
	TokenOffset  int       `json:"token_offset"`
	NumTokens    int       `json:"num_tokens"`
	Layer        int       `json:"layer"`
	BytesUsed    int       `json:"bytes_used"`
	AllocatedAt  time.Time `json:"allocated_at"`
	LastAccessed time.Time `json:"last_accessed"`
	Checksum     uint32    `json:"checksum"`
	DirectIO     bool      `json:"direct_io"`
}

// ResumeManifest stores multi-turn session KV checkpoint state for sub-second resume.
type ResumeManifest struct {
	Version     int             `json:"version"`
	SessionID   string          `json:"session_id"`
	Turn        int             `json:"turn"`
	TotalTokens int             `json:"total_tokens"`
	Blocks      []BlockMetadata `json:"blocks"`
	CreatedAt   time.Time       `json:"created_at"`
}

// LoopbackKVStore is a page-aligned (4096 bytes) Direct-I/O file block store for KV cache slabs.
type LoopbackKVStore struct {
	mu          sync.RWMutex
	file        *os.File
	filePath    string
	maxBlocks   int
	nextBlockID int
	freeBlocks  []int
	allocated   map[int]*BlockMetadata
	closed      bool
}

// AlignedBuffer returns a 4096-byte aligned memory buffer.
// If size is not a multiple of BlockSize, it is rounded up to the nearest multiple.
func AlignedBuffer(size int) []byte {
	if size <= 0 {
		size = BlockSize
	}
	if size%BlockSize != 0 {
		size = ((size + BlockSize - 1) / BlockSize) * BlockSize
	}
	raw := make([]byte, size+BlockSize)
	addr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((uintptr(BlockSize) - (addr % uintptr(BlockSize))) % uintptr(BlockSize))
	return raw[offset : offset+size : offset+size]
}

// OpenLoopbackKVStore opens or creates a Direct-I/O loopback KV store file at the specified path.
func OpenLoopbackKVStore(path string, maxBlocks int) (*LoopbackKVStore, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("kv direct_io: create directory %s: %w", dir, err)
	}

	// Open with O_RDWR | O_CREATE | O_SYNC for direct synchronized I/O
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_SYNC, 0600)
	if err != nil {
		return nil, fmt.Errorf("kv direct_io: open loopback file %s: %w", path, err)
	}

	return &LoopbackKVStore{
		file:        f,
		filePath:    path,
		maxBlocks:   maxBlocks,
		allocated:   make(map[int]*BlockMetadata),
		freeBlocks:  make([]int, 0),
		nextBlockID: 0,
	}, nil
}

// NewLoopbackKVStore is an alias for OpenLoopbackKVStore.
func NewLoopbackKVStore(path string, maxBlocks int) (*LoopbackKVStore, error) {
	return OpenLoopbackKVStore(path, maxBlocks)
}

// AllocateBlock reserves a new 4KB block in the loopback store and records initial metadata.
func (s *LoopbackKVStore) AllocateBlock(meta BlockMetadata) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return -1, ErrStoreClosed
	}

	var blockID int
	if len(s.freeBlocks) > 0 {
		blockID = s.freeBlocks[len(s.freeBlocks)-1]
		s.freeBlocks = s.freeBlocks[:len(s.freeBlocks)-1]
	} else {
		if s.maxBlocks > 0 && s.nextBlockID >= s.maxBlocks {
			return -1, ErrStoreFull
		}
		blockID = s.nextBlockID
		s.nextBlockID++
	}

	meta.BlockID = blockID
	if meta.AllocatedAt.IsZero() {
		meta.AllocatedAt = time.Now()
	}
	meta.LastAccessed = time.Now()
	meta.DirectIO = true

	copied := meta
	s.allocated[blockID] = &copied
	return blockID, nil
}

// AllocateBlocks reserves a sequence of blocks for a session turn.
func (s *LoopbackKVStore) AllocateBlocks(count int, sessionID string, turn int) ([]int, error) {
	if count <= 0 {
		return nil, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	blocks := make([]int, count)
	for i := 0; i < count; i++ {
		var blockID int
		if len(s.freeBlocks) > 0 {
			blockID = s.freeBlocks[len(s.freeBlocks)-1]
			s.freeBlocks = s.freeBlocks[:len(s.freeBlocks)-1]
		} else {
			if s.maxBlocks > 0 && s.nextBlockID >= s.maxBlocks {
				for j := 0; j < i; j++ {
					delete(s.allocated, blocks[j])
					s.freeBlocks = append(s.freeBlocks, blocks[j])
				}
				return nil, ErrStoreFull
			}
			blockID = s.nextBlockID
			s.nextBlockID++
		}

		meta := BlockMetadata{
			BlockID:      blockID,
			SessionID:    sessionID,
			Turn:         turn,
			AllocatedAt:  time.Now(),
			LastAccessed: time.Now(),
			DirectIO:     true,
		}
		s.allocated[blockID] = &meta
		blocks[i] = blockID
	}
	return blocks, nil
}

// FreeBlock releases an allocated block back to the pool and removes its metadata.
func (s *LoopbackKVStore) FreeBlock(blockID int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}
	if _, exists := s.allocated[blockID]; !exists {
		return ErrBlockNotFound
	}
	delete(s.allocated, blockID)
	s.freeBlocks = append(s.freeBlocks, blockID)
	return nil
}

// FreeSessionBlocks releases all blocks associated with the given session ID.
func (s *LoopbackKVStore) FreeSessionBlocks(sessionID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}
	count := 0
	for id, meta := range s.allocated {
		if meta.SessionID == sessionID {
			delete(s.allocated, id)
			s.freeBlocks = append(s.freeBlocks, id)
			count++
		}
	}
	return count, nil
}

// WriteBlock writes up to 4096 bytes of KV slab data into the block aligned to 4KB.
func (s *LoopbackKVStore) WriteBlock(blockID int, data []byte) error {
	return s.WriteBlockWithMeta(blockID, BlockMetadata{}, data)
}

// WriteBlockWithMeta writes aligned KV slab data and attaches/updates block metadata.
func (s *LoopbackKVStore) WriteBlockWithMeta(blockID int, meta BlockMetadata, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}
	if blockID < 0 {
		return ErrInvalidBlockID
	}
	if len(data) > BlockSize {
		return fmt.Errorf("%w: got %d bytes, max %d", ErrDataTooLarge, len(data), BlockSize)
	}

	// Page-aligned direct-I/O buffer (4096 bytes)
	buf := AlignedBuffer(BlockSize)
	copy(buf, data)
	for i := len(data); i < BlockSize; i++ {
		buf[i] = 0
	}

	offset := int64(blockID) * int64(BlockSize)
	n, err := s.file.WriteAt(buf, offset)
	if err != nil {
		return fmt.Errorf("kv direct_io: write at offset %d: %w", offset, err)
	}
	if n != BlockSize {
		return fmt.Errorf("kv direct_io: short write at offset %d: wrote %d of %d bytes", offset, n, BlockSize)
	}

	existing, ok := s.allocated[blockID]
	if !ok {
		meta.BlockID = blockID
		if meta.AllocatedAt.IsZero() {
			meta.AllocatedAt = time.Now()
		}
		existing = &meta
		s.allocated[blockID] = existing
	} else {
		if meta.SessionID != "" {
			existing.SessionID = meta.SessionID
		}
		if meta.Turn != 0 {
			existing.Turn = meta.Turn
		}
		if meta.TokenOffset != 0 {
			existing.TokenOffset = meta.TokenOffset
		}
		if meta.NumTokens != 0 {
			existing.NumTokens = meta.NumTokens
		}
		if meta.Layer != 0 {
			existing.Layer = meta.Layer
		}
	}
	existing.BytesUsed = len(data)
	existing.Checksum = crc32.ChecksumIEEE(data)
	existing.LastAccessed = time.Now()
	existing.DirectIO = true

	return nil
}

// ReadBlock reads KV slab data directly from the block using page-aligned I/O and verifies checksum.
func (s *LoopbackKVStore) ReadBlock(blockID int) ([]byte, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, ErrStoreClosed
	}
	meta, ok := s.allocated[blockID]
	bytesUsed := BlockSize
	var expectedChecksum uint32
	if ok {
		if meta.BytesUsed > 0 && meta.BytesUsed <= BlockSize {
			bytesUsed = meta.BytesUsed
		}
		expectedChecksum = meta.Checksum
	}
	s.mu.RUnlock()

	buf := AlignedBuffer(BlockSize)
	offset := int64(blockID) * int64(BlockSize)
	n, err := s.file.ReadAt(buf, offset)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("kv direct_io: read at offset %d: %w", offset, err)
	}
	if n < bytesUsed {
		return nil, fmt.Errorf("kv direct_io: short read at block %d: got %d, want %d", blockID, n, bytesUsed)
	}

	result := make([]byte, bytesUsed)
	copy(result, buf[:bytesUsed])

	if expectedChecksum != 0 {
		c := crc32.ChecksumIEEE(result)
		if c != expectedChecksum {
			return nil, fmt.Errorf("%w: block %d got %08x, want %08x", ErrChecksumMismatch, blockID, c, expectedChecksum)
		}
	}

	s.mu.Lock()
	if m, exists := s.allocated[blockID]; exists {
		m.LastAccessed = time.Now()
	}
	s.mu.Unlock()

	return result, nil
}

// ReadBlockDirect reads an entire 4096-byte page directly into the provided dst buffer.
func (s *LoopbackKVStore) ReadBlockDirect(blockID int, dst []byte) (int, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return 0, ErrStoreClosed
	}
	s.mu.RUnlock()

	if len(dst) < BlockSize {
		return 0, fmt.Errorf("kv direct_io: destination buffer size %d < %d", len(dst), BlockSize)
	}
	offset := int64(blockID) * int64(BlockSize)
	return s.file.ReadAt(dst[:BlockSize], offset)
}

// GetMetadata returns a copy of the block metadata.
func (s *LoopbackKVStore) GetMetadata(blockID int) (BlockMetadata, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	meta, ok := s.allocated[blockID]
	if !ok {
		return BlockMetadata{}, false
	}
	return *meta, true
}

// UpdateMetadata updates user metadata fields for an allocated block.
func (s *LoopbackKVStore) UpdateMetadata(blockID int, meta BlockMetadata) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}
	existing, ok := s.allocated[blockID]
	if !ok {
		return ErrBlockNotFound
	}
	if meta.SessionID != "" {
		existing.SessionID = meta.SessionID
	}
	if meta.Turn != 0 {
		existing.Turn = meta.Turn
	}
	if meta.TokenOffset != 0 {
		existing.TokenOffset = meta.TokenOffset
	}
	if meta.NumTokens != 0 {
		existing.NumTokens = meta.NumTokens
	}
	if meta.Layer != 0 {
		existing.Layer = meta.Layer
	}
	existing.LastAccessed = time.Now()
	return nil
}

// ListBlocks returns copies of all allocated block metadata records.
func (s *LoopbackKVStore) ListBlocks() []BlockMetadata {
	s.mu.RLock()
	defer s.mu.RUnlock()
	res := make([]BlockMetadata, 0, len(s.allocated))
	for _, meta := range s.allocated {
		res = append(res, *meta)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].BlockID < res[j].BlockID
	})
	return res
}

// ListSessionBlocks returns block metadata for a given session.
func (s *LoopbackKVStore) ListSessionBlocks(sessionID string) []BlockMetadata {
	s.mu.RLock()
	defer s.mu.RUnlock()
	res := make([]BlockMetadata, 0)
	for _, meta := range s.allocated {
		if meta.SessionID == sessionID {
			res = append(res, *meta)
		}
	}
	sort.Slice(res, func(i, j int) bool {
		if res[i].Turn != res[j].Turn {
			return res[i].Turn < res[j].Turn
		}
		return res[i].TokenOffset < res[j].TokenOffset
	})
	return res
}

// EvictOldest frees up to count blocks with the oldest access times.
func (s *LoopbackKVStore) EvictOldest(count int) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}
	if count <= 0 || len(s.allocated) == 0 {
		return 0, nil
	}

	type item struct {
		id int
		t  time.Time
	}
	items := make([]item, 0, len(s.allocated))
	for id, meta := range s.allocated {
		items = append(items, item{id: id, t: meta.LastAccessed})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].t.Before(items[j].t)
	})

	freed := 0
	for i := 0; i < count && i < len(items); i++ {
		id := items[i].id
		delete(s.allocated, id)
		s.freeBlocks = append(s.freeBlocks, id)
		freed++
	}
	return freed, nil
}

// PruneIdle removes blocks that have not been accessed within the idle duration.
func (s *LoopbackKVStore) PruneIdle(olderThan time.Duration) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}
	cutoff := time.Now().Add(-olderThan)
	freed := 0
	for id, meta := range s.allocated {
		if meta.LastAccessed.Before(cutoff) {
			delete(s.allocated, id)
			s.freeBlocks = append(s.freeBlocks, id)
			freed++
		}
	}
	return freed, nil
}

// EvictKV implements the gateway KVEvictor interface, evicting oldest blocks when triggered.
func (s *LoopbackKVStore) EvictKV(ctx context.Context) (int, error) {
	s.mu.RLock()
	n := len(s.allocated)
	s.mu.RUnlock()
	if n == 0 {
		return 0, nil
	}
	batch := 16
	if n < batch {
		batch = n
	}
	return s.EvictOldest(batch)
}

// SerializeResume exports the session's multi-turn KV block metadata into a compact serialized manifest.
// Completes in sub-millisecond to sub-second time.
func (s *LoopbackKVStore) SerializeResume(sessionID string) ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	var blocks []BlockMetadata
	maxTurn := 0
	totalTokens := 0
	for _, meta := range s.allocated {
		if meta.SessionID == sessionID {
			blocks = append(blocks, *meta)
			if meta.Turn > maxTurn {
				maxTurn = meta.Turn
			}
			totalTokens += meta.NumTokens
		}
	}

	sort.Slice(blocks, func(i, j int) bool {
		if blocks[i].Turn != blocks[j].Turn {
			return blocks[i].Turn < blocks[j].Turn
		}
		return blocks[i].TokenOffset < blocks[j].TokenOffset
	})

	manifest := ResumeManifest{
		Version:     1,
		SessionID:   sessionID,
		Turn:        maxTurn,
		TotalTokens: totalTokens,
		Blocks:      blocks,
		CreatedAt:   time.Now(),
	}

	return json.Marshal(manifest)
}

// DeserializeResume restores session multi-turn KV block metadata from a serialized manifest.
// Completes in sub-millisecond to sub-second time.
func (s *LoopbackKVStore) DeserializeResume(data []byte) (*ResumeManifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	var manifest ResumeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("kv direct_io: unmarshal resume manifest: %w", err)
	}

	for _, b := range manifest.Blocks {
		metaCopy := b
		s.allocated[b.BlockID] = &metaCopy
		if b.BlockID >= s.nextBlockID {
			s.nextBlockID = b.BlockID + 1
		}
	}

	return &manifest, nil
}

// SaveResumeToFile writes the session resume manifest directly to a file.
func (s *LoopbackKVStore) SaveResumeToFile(sessionID string, filePath string) error {
	data, err := s.SerializeResume(sessionID)
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0600)
}

// LoadResumeFromFile loads and restores session resume manifest from a file.
func (s *LoopbackKVStore) LoadResumeFromFile(filePath string) (*ResumeManifest, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	return s.DeserializeResume(data)
}

// AllocatedCount returns the number of currently allocated blocks.
func (s *LoopbackKVStore) AllocatedCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.allocated)
}

// BlockCount returns the high-water count of blocks written.
func (s *LoopbackKVStore) BlockCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.nextBlockID
}

// Capacity returns the maximum block capacity (0 = unlimited).
func (s *LoopbackKVStore) Capacity() int {
	return s.maxBlocks
}

// Path returns the backing file path.
func (s *LoopbackKVStore) Path() string {
	return s.filePath
}

// Close syncs and closes the backing store file.
func (s *LoopbackKVStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.file != nil {
		_ = s.file.Sync()
		return s.file.Close()
	}
	return nil
}
