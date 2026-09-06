package l3kv

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

const (
	// DefaultChunkSize is the default chunk size (8 MiB), matching physical NAND flash erase blocks
	// on modern TLC/QLC SSDs to minimize flash write amplification factor (WAF).
	DefaultChunkSize uint64 = 8 * 1024 * 1024

	// DefaultAlignment is the minimum block/sector boundary (4096 bytes) for committed segments.
	DefaultAlignment uint64 = 4096

	// manifestFileName is the persistent index manifest file stored in the chunk store directory.
	manifestFileName = "manifest.json"
)

var (
	// ErrNotFound is returned when a requested span digest is not present in the index manifest.
	ErrNotFound = errors.New("l3kv: span not found")

	// ErrPayloadTooLarge is returned when a single span payload exceeds the configured chunk capacity.
	ErrPayloadTooLarge = errors.New("l3kv: payload exceeds chunk size")

	// ErrClosed is returned when an operation is performed on a closed chunk store.
	ErrClosed = errors.New("l3kv: chunk store is closed")
)

// SpanLocation records the exact location of a stored span payload within a chunk file.
type SpanLocation struct {
	ChunkID uint64 `json:"chunk_id"`
	Offset  uint64 `json:"offset"`
	Length  uint64 `json:"length"`
}

// ChunkStoreConfig defines configuration parameters for ChunkStore.
type ChunkStoreConfig struct {
	// Dir is the root directory where chunk files and the manifest are persisted.
	Dir string

	// ChunkSize specifies the capacity of each chunk in bytes (default: 8 MiB).
	ChunkSize uint64

	// Alignment specifies the byte boundary each committed chunk must align to (default: 4096).
	Alignment uint64

	// PadToChunk determines whether committed chunks are zero-padded up to ChunkSize
	// (creating fixed-size erase-block files) or padded to the nearest Alignment boundary.
	// When true (default), every committed chunk is exactly ChunkSize bytes.
	PadToChunk bool

	// AutoSync flushes the active chunk and syncs the manifest on every Put operation.
	AutoSync bool
}

// chunkManifest is the persistent JSON envelope for the index manifest.
type chunkManifest struct {
	Version     int                     `json:"version"`
	ChunkSize   uint64                  `json:"chunk_size"`
	Alignment   uint64                  `json:"alignment"`
	NextChunkID uint64                  `json:"next_chunk_id"`
	Spans       map[string]SpanLocation `json:"spans"`
}

// ChunkStore is an append-only, log-structured, erase-block-aligned KV span store
// engineered to minimize NAND flash write amplification on local NVMe SSDs.
// Incoming span payloads are sequentially packed into an in-memory active chunk buffer
// (default 8 MiB) and committed to disk as aligned extents when full or synced.
type ChunkStore struct {
	mu sync.RWMutex

	dir        string
	chunkSize  uint64
	alignment  uint64
	padToChunk bool
	autoSync   bool

	// in-memory index manifest recording map[string]SpanLocation
	spans map[string]SpanLocation

	// active chunk state
	activeChunkID uint64
	activeOffset  uint64
	activeBuf     []byte

	closed bool
}

// NewChunkStore creates or opens an append-only chunk store rooted at cfg.Dir.
// If an existing manifest is found, it automatically recovers the index manifest
// and prepares the next sequential chunk ID for writes.
func NewChunkStore(cfg ChunkStoreConfig) (*ChunkStore, error) {
	if cfg.Dir == "" {
		return nil, fmt.Errorf("l3kv: empty chunk store directory")
	}
	if cfg.ChunkSize == 0 {
		cfg.ChunkSize = DefaultChunkSize
	}
	if cfg.Alignment == 0 {
		cfg.Alignment = DefaultAlignment
	}
	if cfg.ChunkSize%cfg.Alignment != 0 {
		return nil, fmt.Errorf("l3kv: chunk size %d must be a multiple of alignment %d", cfg.ChunkSize, cfg.Alignment)
	}
	if err := os.MkdirAll(cfg.Dir, 0o755); err != nil {
		return nil, fmt.Errorf("l3kv: create chunk store dir %s: %w", cfg.Dir, err)
	}

	cs := &ChunkStore{
		dir:        cfg.Dir,
		chunkSize:  cfg.ChunkSize,
		alignment:  cfg.Alignment,
		padToChunk: cfg.PadToChunk,
		autoSync:   cfg.AutoSync,
		spans:      make(map[string]SpanLocation),
		activeBuf:  make([]byte, cfg.ChunkSize),
	}

	// Load existing manifest if present (recovers index from disk)
	if err := cs.loadManifestLocked(); err != nil {
		return nil, err
	}

	return cs, nil
}

// NewDefaultChunkStore creates a ChunkStore with default 8 MiB erase-block size and 4 KiB alignment.
func NewDefaultChunkStore(dir string) (*ChunkStore, error) {
	return NewChunkStore(ChunkStoreConfig{
		Dir:        dir,
		ChunkSize:  DefaultChunkSize,
		Alignment:  DefaultAlignment,
		PadToChunk: true,
	})
}

// ChunkPath returns the filesystem path of the chunk file for the given chunkID.
func (s *ChunkStore) ChunkPath(chunkID uint64) string {
	return filepath.Join(s.dir, fmt.Sprintf("chunk_%06d.chunk", chunkID))
}

// Put packs the incoming span payload sequentially into the active aligned chunk buffer.
// If the payload does not fit in the remaining space of the active chunk, the active chunk
// is sealed, aligned, and flushed to disk, and the payload is placed into the new active chunk.
func (s *ChunkStore) Put(ctx context.Context, digest string, payload []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if digest == "" {
		return fmt.Errorf("l3kv: empty span digest")
	}

	payloadLen := uint64(len(payload))

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrClosed
	}

	if payloadLen > s.chunkSize {
		return fmt.Errorf("l3kv: payload size %d exceeds chunk size %d: %w", payloadLen, s.chunkSize, ErrPayloadTooLarge)
	}

	// Boundary wrapping: if payload cannot fit in the remaining active chunk space, flush current chunk.
	if s.activeOffset+payloadLen > s.chunkSize {
		if err := s.flushActiveChunkLocked(); err != nil {
			return fmt.Errorf("l3kv: boundary wrap flush chunk %d: %w", s.activeChunkID, err)
		}
	}

	// Record location in index manifest
	loc := SpanLocation{
		ChunkID: s.activeChunkID,
		Offset:  s.activeOffset,
		Length:  payloadLen,
	}

	// Pack payload into active chunk buffer
	if payloadLen > 0 {
		copy(s.activeBuf[s.activeOffset:], payload)
		s.activeOffset += payloadLen
	}

	s.spans[digest] = loc

	// If chunk is now completely full, flush immediately
	if s.activeOffset >= s.chunkSize {
		if err := s.flushActiveChunkLocked(); err != nil {
			return fmt.Errorf("l3kv: full chunk flush %d: %w", s.activeChunkID-1, err)
		}
	} else if s.autoSync {
		if err := s.flushActiveChunkLocked(); err != nil {
			return fmt.Errorf("l3kv: autosync flush: %w", err)
		}
	}

	return nil
}

// Get retrieves the span payload for digest. If the span resides in the active in-memory
// buffer, it is returned without disk I/O; otherwise it is read from the committed chunk file.
func (s *ChunkStore) Get(ctx context.Context, digest string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if digest == "" {
		return nil, fmt.Errorf("l3kv: empty span digest")
	}

	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, ErrClosed
	}

	loc, ok := s.spans[digest]
	if !ok {
		s.mu.RUnlock()
		return nil, fmt.Errorf("l3kv: span %q: %w", digest, ErrNotFound)
	}

	// Fast path: if span is in active in-memory chunk buffer
	if loc.ChunkID == s.activeChunkID {
		if loc.Offset+loc.Length > uint64(len(s.activeBuf)) {
			s.mu.RUnlock()
			return nil, fmt.Errorf("l3kv: span %q bounds exceed active buffer (%d+%d > %d)", digest, loc.Offset, loc.Length, len(s.activeBuf))
		}
		res := make([]byte, loc.Length)
		copy(res, s.activeBuf[loc.Offset:loc.Offset+loc.Length])
		s.mu.RUnlock()
		return res, nil
	}
	s.mu.RUnlock()

	// Slow path: span is in committed chunk file on disk
	return s.readFromDisk(loc)
}

func (s *ChunkStore) readFromDisk(loc SpanLocation) ([]byte, error) {
	chunkPath := s.ChunkPath(loc.ChunkID)
	f, err := os.Open(chunkPath)
	if err != nil {
		return nil, fmt.Errorf("l3kv: open chunk file %s: %w", chunkPath, err)
	}
	defer f.Close()

	buf := make([]byte, loc.Length)
	if loc.Length == 0 {
		return buf, nil
	}

	n, err := f.ReadAt(buf, int64(loc.Offset))
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("l3kv: read chunk %d at offset %d: %w", loc.ChunkID, loc.Offset, err)
	}
	if uint64(n) != loc.Length {
		return nil, fmt.Errorf("l3kv: short read from chunk %d: expected %d bytes, got %d", loc.ChunkID, loc.Length, n)
	}
	return buf, nil
}

// flushActiveChunkLocked commits the active chunk buffer to disk, aligns file size,
// verifies alignment invariants, advances to the next chunk ID, and persists the manifest.
func (s *ChunkStore) flushActiveChunkLocked() error {
	if s.activeOffset == 0 {
		return s.persistManifestLocked()
	}

	targetSize := s.activeOffset
	if s.padToChunk {
		targetSize = s.chunkSize
	} else {
		targetSize = alignUp(s.activeOffset, s.alignment)
	}

	// Zero-pad unused buffer space up to targetSize
	if targetSize > s.activeOffset {
		padding := s.activeBuf[s.activeOffset:targetSize]
		clear(padding)
	}

	// Pre-commit alignment verification
	if s.alignment > 0 && targetSize%s.alignment != 0 {
		return fmt.Errorf("l3kv: target flush size %d is not aligned to %d", targetSize, s.alignment)
	}

	chunkFile := s.ChunkPath(s.activeChunkID)
	data := s.activeBuf[:targetSize]
	if err := atomicWrite(chunkFile, data); err != nil {
		return fmt.Errorf("l3kv: write chunk file %s: %w", chunkFile, err)
	}

	// Post-commit alignment verification against physical file on disk
	info, err := os.Stat(chunkFile)
	if err != nil {
		return fmt.Errorf("l3kv: stat committed chunk %s: %w", chunkFile, err)
	}
	if s.alignment > 0 && uint64(info.Size())%s.alignment != 0 {
		return fmt.Errorf("l3kv: committed chunk %s size %d not aligned to %d", chunkFile, info.Size(), s.alignment)
	}

	// Advance active chunk
	s.activeChunkID++
	s.activeOffset = 0
	clear(s.activeBuf)

	// Persist updated index manifest
	return s.persistManifestLocked()
}

func (s *ChunkStore) persistManifestLocked() error {
	m := chunkManifest{
		Version:     1,
		ChunkSize:   s.chunkSize,
		Alignment:   s.alignment,
		NextChunkID: s.activeChunkID,
		Spans:       s.spans,
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("l3kv: marshal manifest: %w", err)
	}
	manifestPath := filepath.Join(s.dir, manifestFileName)
	if err := atomicWrite(manifestPath, data); err != nil {
		return fmt.Errorf("l3kv: write manifest %s: %w", manifestPath, err)
	}
	return nil
}

func (s *ChunkStore) loadManifestLocked() error {
	manifestPath := filepath.Join(s.dir, manifestFileName)
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("l3kv: read manifest %s: %w", manifestPath, err)
	}
	var m chunkManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("l3kv: unmarshal manifest: %w", err)
	}
	if s.spans == nil {
		s.spans = make(map[string]SpanLocation, len(m.Spans))
	}
	for k, v := range m.Spans {
		s.spans[k] = v
	}
	s.activeChunkID = m.NextChunkID
	s.activeOffset = 0
	clear(s.activeBuf)
	return nil
}

// RecoverIndex reloads the index manifest from disk.
func (s *ChunkStore) RecoverIndex() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	return s.loadManifestLocked()
}

// Sync flushes the active chunk buffer to disk and persists the manifest.
func (s *ChunkStore) Sync() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	return s.flushActiveChunkLocked()
}

// Flush forces the active chunk buffer to be committed to disk.
func (s *ChunkStore) Flush() error {
	return s.Sync()
}

// Close flushes any pending writes to disk, persists the manifest, and marks the store closed.
func (s *ChunkStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if err := s.flushActiveChunkLocked(); err != nil {
		return err
	}
	s.closed = true
	s.activeBuf = nil
	return nil
}

// Location returns the ChunkID, Offset, and Length for the requested span digest if recorded.
func (s *ChunkStore) Location(digest string) (SpanLocation, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	loc, ok := s.spans[digest]
	return loc, ok
}

// Index returns a snapshot copy of the in-memory index manifest map.
func (s *ChunkStore) Index() map[string]SpanLocation {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cp := make(map[string]SpanLocation, len(s.spans))
	for k, v := range s.spans {
		cp[k] = v
	}
	return cp
}

// VerifyAlignment checks that the committed segment file for chunkID has a size that is
// an exact multiple of the configured alignment size.
func (s *ChunkStore) VerifyAlignment(chunkID uint64) error {
	chunkPath := s.ChunkPath(chunkID)
	info, err := os.Stat(chunkPath)
	if err != nil {
		return fmt.Errorf("l3kv: stat chunk %d (%s): %w", chunkID, chunkPath, err)
	}
	s.mu.RLock()
	align := s.alignment
	s.mu.RUnlock()

	if align > 0 && uint64(info.Size())%align != 0 {
		return fmt.Errorf("l3kv: chunk %d size %d is not a multiple of alignment %d", chunkID, info.Size(), align)
	}
	return nil
}

// VerifyAllAlignments checks that every committed chunk file on disk is an exact multiple
// of the configured alignment size.
func (s *ChunkStore) VerifyAllAlignments() error {
	s.mu.RLock()
	activeID := s.activeChunkID
	align := s.alignment
	s.mu.RUnlock()

	for id := uint64(0); id < activeID; id++ {
		chunkPath := s.ChunkPath(id)
		info, err := os.Stat(chunkPath)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return fmt.Errorf("l3kv: stat chunk %d (%s): %w", id, chunkPath, err)
		}
		if align > 0 && uint64(info.Size())%align != 0 {
			return fmt.Errorf("l3kv: chunk %d size %d is not aligned to %d", id, info.Size(), align)
		}
	}
	return nil
}

// GetSpan adapts Get to the (payload []byte, found bool, err error) return signature of l3kv.Store.
func (s *ChunkStore) GetSpan(ctx context.Context, digest string) ([]byte, bool, error) {
	b, err := s.Get(ctx, digest)
	if err != nil {
		if errors.Is(err, ErrNotFound) || os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	return b, true, nil
}

// AsStore returns a Store adapter wrapping this ChunkStore to satisfy the l3kv.Store interface.
func (s *ChunkStore) AsStore() Store {
	return &chunkStoreAdapter{cs: s}
}

type chunkStoreAdapter struct {
	cs *ChunkStore
}

func (a *chunkStoreAdapter) Put(ctx context.Context, key string, payload []byte) error {
	return a.cs.Put(ctx, key, payload)
}

func (a *chunkStoreAdapter) Get(ctx context.Context, key string) ([]byte, bool, error) {
	return a.cs.GetSpan(ctx, key)
}

var _ Store = (*chunkStoreAdapter)(nil)

func alignUp(n, align uint64) uint64 {
	if align == 0 || n%align == 0 {
		return n
	}
	return n + (align - n%align)
}
