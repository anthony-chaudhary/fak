package model

import (
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"
)

// V41EngramPackedRowBytes is the packed on-disk width of one V4.1 Engram row:
// 256 bytes of FP8 weights followed by 8 bytes of E8M0 scales.
const V41EngramPackedRowBytes = 264

// V41EngramStreamErrorKind classifies fail-closed Engram retrieval failures.
type V41EngramStreamErrorKind string

const (
	V41EngramStreamGeometry      V41EngramStreamErrorKind = "geometry"
	V41EngramStreamOutOfBounds   V41EngramStreamErrorKind = "out_of_bounds"
	V41EngramStreamShardMismatch V41EngramStreamErrorKind = "shard_mismatch"
	V41EngramStreamShortRead     V41EngramStreamErrorKind = "short_read"
)

// V41EngramStreamError is returned for malformed geometry, invalid row ranges,
// shard identity mismatches, and incomplete packed-row reads.
type V41EngramStreamError struct {
	Kind  V41EngramStreamErrorKind
	Row   int
	Shard string
	Err   error
}

func (e *V41EngramStreamError) Error() string {
	if e == nil {
		return "model: nil V4.1 Engram stream error"
	}
	where := ""
	if e.Row >= 0 {
		where += fmt.Sprintf(" row=%d", e.Row)
	}
	if e.Shard != "" {
		where += " shard=" + e.Shard
	}
	if e.Err != nil {
		return fmt.Sprintf("model: V4.1 Engram stream %s%s: %v", e.Kind, where, e.Err)
	}
	return fmt.Sprintf("model: V4.1 Engram stream %s%s", e.Kind, where)
}

func (e *V41EngramStreamError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// V41EngramShard describes the declared extent and identity of one packed row
// shard. ID is observed metadata; ExpectedID is the identity requested by the
// checkpoint index. Equality is checked, but no authentication is claimed.
type V41EngramShard struct {
	ID         string
	ExpectedID string
	Offset     int64
	Size       int64
	Rows       int
}

type v41PackedEngramRowSource struct {
	reader io.ReaderAt
	shard  V41EngramShard
}

// NewV41PackedEngramRowSource creates a packed 264-byte row reader over a
// declared shard extent.
func NewV41PackedEngramRowSource(reader io.ReaderAt, shard V41EngramShard) (V41EngramRowSource, error) {
	if reader == nil || shard.Rows <= 0 || shard.Offset < 0 || shard.Size <= 0 ||
		int64(shard.Rows) > math.MaxInt64/V41EngramPackedRowBytes ||
		shard.Size != int64(shard.Rows)*V41EngramPackedRowBytes ||
		shard.Offset > math.MaxInt64-shard.Size {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("invalid packed shard extent offset=%d size=%d rows=%d", shard.Offset, shard.Size, shard.Rows)}
	}
	if shard.ID == "" || shard.ExpectedID == "" || shard.ID != shard.ExpectedID {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamShardMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("observed identity %q does not match expected %q", shard.ID, shard.ExpectedID)}
	}
	return &v41PackedEngramRowSource{reader: reader, shard: shard}, nil
}

func (*v41PackedEngramRowSource) RowBytes() int { return V41EngramPackedRowBytes }

func (s *v41PackedEngramRowSource) ReadRows(start, count int, dst []byte) (int, error) {
	if start < 0 || count <= 0 || start > s.shard.Rows || count > s.shard.Rows-start {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamOutOfBounds, Row: start, Shard: s.shard.ID,
			Err: fmt.Errorf("row range [%d,%d) outside [0,%d)", start, start+count, s.shard.Rows)}
	}
	if count > math.MaxInt/V41EngramPackedRowBytes {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: start, Shard: s.shard.ID,
			Err: fmt.Errorf("row byte count overflows")}
	}
	want := count * V41EngramPackedRowBytes
	if len(dst) < want {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: start, Shard: s.shard.ID,
			Err: fmt.Errorf("destination bytes=%d want=%d", len(dst), want)}
	}
	offset := s.shard.Offset + int64(start)*V41EngramPackedRowBytes
	if offset < s.shard.Offset || int64(want) > s.shard.Offset+s.shard.Size-offset {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamOutOfBounds, Row: start, Shard: s.shard.ID,
			Err: fmt.Errorf("packed read exceeds declared shard extent")}
	}
	n, err := s.reader.ReadAt(dst[:want], offset)
	if n != want {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return n, &V41EngramStreamError{Kind: V41EngramStreamShortRead, Row: start, Shard: s.shard.ID,
			Err: fmt.Errorf("read bytes=%d want=%d: %w", n, want, err)}
	}
	if err != nil && err != io.EOF {
		return n, err
	}
	return n, nil
}

// V41EngramSetCacheOptions configures a bounded set-associative row cache.
// BudgetBytes bounds resident row payloads; zero means exactly Sets*Ways rows.
// Set metadata, in-flight reads, and caller-owned output buffers are separate.
type V41EngramSetCacheOptions struct {
	TableRows   int
	Sets        int
	Ways        int
	BudgetBytes int64
}

type v41EngramCacheSlot struct {
	row  int
	data []byte
	used bool
}

type v41EngramCacheSet struct {
	mu    sync.Mutex
	slots []v41EngramCacheSlot
	next  int
}

// V41EngramSetCache is a bounded set-associative V41EngramRowSource. Backing
// reads happen without a set lock, allowing colliding misses to overlap. Its
// budget covers retained row payload only, excluding metadata, in-flight reads,
// prefetch scratch, and caller-owned output.
type V41EngramSetCache struct {
	src        V41EngramRowSource
	tableRows  int
	rowBytes   int
	sets       []v41EngramCacheSet
	hits       atomic.Int64
	misses     atomic.Int64
	reads      atomic.Int64
	evictions  atomic.Int64
	prefetches atomic.Int64
	bytesRead  atomic.Int64
}

// V41EngramStreamStats is a concurrency-safe snapshot of cache activity.
type V41EngramStreamStats struct {
	Hits       int64 `json:"hits"`
	Misses     int64 `json:"misses"`
	Reads      int64 `json:"reads"`
	Evictions  int64 `json:"evictions"`
	Prefetches int64 `json:"prefetches"`
	BytesRead  int64 `json:"bytes_read"`
}

func NewV41EngramSetCache(src V41EngramRowSource, opts V41EngramSetCacheOptions) (*V41EngramSetCache, error) {
	if src == nil || opts.TableRows <= 0 || opts.Sets <= 0 || opts.Ways <= 0 || src.RowBytes() <= 0 ||
		opts.Sets > math.MaxInt/opts.Ways {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1,
			Err: fmt.Errorf("invalid cache geometry rows=%d sets=%d ways=%d", opts.TableRows, opts.Sets, opts.Ways)}
	}
	entries := opts.Sets * opts.Ways
	rowBytes := src.RowBytes()
	if int64(entries) > math.MaxInt64/int64(rowBytes) {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Err: fmt.Errorf("cache byte geometry overflows")}
	}
	required := int64(entries) * int64(rowBytes)
	if opts.BudgetBytes == 0 {
		opts.BudgetBytes = required
	}
	if opts.BudgetBytes < required {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1,
			Err: fmt.Errorf("cache budget=%d smaller than configured residency=%d", opts.BudgetBytes, required)}
	}
	c := &V41EngramSetCache{src: src, tableRows: opts.TableRows, rowBytes: rowBytes, sets: make([]v41EngramCacheSet, opts.Sets)}
	for i := range c.sets {
		c.sets[i].slots = make([]v41EngramCacheSlot, opts.Ways)
	}
	return c, nil
}

func (c *V41EngramSetCache) RowBytes() int { return c.rowBytes }

func (c *V41EngramSetCache) ReadRows(start, count int, dst []byte) (int, error) {
	if start < 0 || count <= 0 || start > c.tableRows || count > c.tableRows-start {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamOutOfBounds, Row: start,
			Err: fmt.Errorf("row range [%d,%d) outside [0,%d)", start, start+count, c.tableRows)}
	}
	if count > math.MaxInt/c.rowBytes || len(dst) < count*c.rowBytes {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: start,
			Err: fmt.Errorf("destination is too small for %d rows", count)}
	}
	written := 0
	for row := start; row < start+count; row++ {
		data, err := c.load(row, false)
		if err != nil {
			return written, err
		}
		written += copy(dst[written:], data)
	}
	return written, nil
}

// Prefetch resolves token-derived row addresses into the bounded cache without
// changing Engram hash state.
func (c *V41EngramSetCache) Prefetch(rows []uint32) error {
	seen := make(map[uint32]struct{}, len(rows))
	for _, row := range rows {
		if _, ok := seen[row]; ok {
			continue
		}
		seen[row] = struct{}{}
		if uint64(row) >= uint64(c.tableRows) {
			return &V41EngramStreamError{Kind: V41EngramStreamOutOfBounds, Row: int(row),
				Err: fmt.Errorf("prefetch row outside [0,%d)", c.tableRows)}
		}
		if _, err := c.load(int(row), true); err != nil {
			return err
		}
	}
	return nil
}

func (c *V41EngramSetCache) load(row int, prefetch bool) ([]byte, error) {
	set := &c.sets[row%len(c.sets)]
	set.mu.Lock()
	for i := range set.slots {
		if set.slots[i].used && set.slots[i].row == row {
			out := append([]byte(nil), set.slots[i].data...)
			set.mu.Unlock()
			if !prefetch {
				c.hits.Add(1)
			}
			return out, nil
		}
	}
	set.mu.Unlock()
	if !prefetch {
		c.misses.Add(1)
	}

	buf := make([]byte, c.rowBytes)
	n, err := c.src.ReadRows(row, 1, buf)
	c.reads.Add(1)
	if n > 0 {
		c.bytesRead.Add(int64(n))
	}
	if n != c.rowBytes {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return nil, &V41EngramStreamError{Kind: V41EngramStreamShortRead, Row: row,
			Err: fmt.Errorf("source returned bytes=%d want=%d: %w", n, c.rowBytes, err)}
	}
	if err != nil {
		return nil, err
	}
	if prefetch {
		c.prefetches.Add(1)
	}

	set.mu.Lock()
	for i := range set.slots {
		if set.slots[i].used && set.slots[i].row == row {
			out := append([]byte(nil), set.slots[i].data...)
			set.mu.Unlock()
			return out, nil
		}
	}
	victim := -1
	for i := range set.slots {
		if !set.slots[i].used {
			victim = i
			break
		}
	}
	if victim < 0 {
		victim = set.next
		set.next = (set.next + 1) % len(set.slots)
		c.evictions.Add(1)
	}
	set.slots[victim] = v41EngramCacheSlot{row: row, data: append([]byte(nil), buf...), used: true}
	set.mu.Unlock()
	return buf, nil
}

func (c *V41EngramSetCache) Stats() V41EngramStreamStats {
	return V41EngramStreamStats{
		Hits: c.hits.Load(), Misses: c.misses.Load(), Reads: c.reads.Load(),
		Evictions: c.evictions.Load(), Prefetches: c.prefetches.Load(), BytesRead: c.bytesRead.Load(),
	}
}

// V41EngramStream binds token hashing to packed-row prefetch and consume. The
// constructor takes a snapshot of hash; later caller mutations cannot cross
// session boundaries. This deliberately does not enable the guarded V4.1
// native forward path.
type V41EngramStream struct {
	mu      sync.Mutex
	hash    *V41EngramHashState
	caches  []*V41EngramSetCache
	gather  []*V41EngramRowCache
	columns int
}

func NewV41EngramStream(hash *V41EngramHashState, caches []*V41EngramSetCache) (*V41EngramStream, error) {
	if hash == nil || len(caches) == 0 || len(caches) != len(hash.layout.Rows) {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1,
			Err: fmt.Errorf("hash/cache layer geometry does not match")}
	}
	columns := (hash.layout.MaxNgramSize - 1) * hash.layout.HeadsPerNgram
	if columns <= 0 {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Err: fmt.Errorf("invalid hash column geometry")}
	}
	s := &V41EngramStream{hash: hash.Clone(), caches: append([]*V41EngramSetCache(nil), caches...), columns: columns}
	s.gather = make([]*V41EngramRowCache, len(caches))
	for layer, cache := range caches {
		if cache == nil || uint64(cache.tableRows) != uint64(hash.layout.Rows[layer]) {
			return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1,
				Err: fmt.Errorf("cache layer %d rows do not match hash layout", layer)}
		}
		wrapper, err := NewV41EngramRowCache(cache, V41EngramRowCacheOptions{
			TableRows: cache.tableRows, RowBytes: cache.rowBytes, BudgetBytes: int64(cache.rowBytes),
		})
		if err != nil {
			return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Err: err}
		}
		s.gather[layer] = wrapper
	}
	return s, nil
}

// Prefetch hashes on a snapshot and fills caches without advancing live state.
func (s *V41EngramStream) Prefetch(tokens []int, mask []bool) error {
	if s == nil {
		return &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Err: fmt.Errorf("nil stream")}
	}
	s.mu.Lock()
	candidate := s.hash.Clone()
	s.mu.Unlock()
	rows, err := candidate.Hash(tokens, mask)
	if err != nil {
		return err
	}
	return s.prefetchRows(rows)
}

func (s *V41EngramStream) prefetchRows(rows []uint32) error {
	stride := len(s.caches) * s.columns
	if stride <= 0 || len(rows)%stride != 0 {
		return &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Err: fmt.Errorf("invalid hashed row geometry")}
	}
	for layer, cache := range s.caches {
		layerRows := make([]uint32, 0, len(rows)/len(s.caches))
		for token := 0; token < len(rows)/stride; token++ {
			lo := token*stride + layer*s.columns
			layerRows = append(layerRows, rows[lo:lo+s.columns]...)
		}
		if err := cache.Prefetch(layerRows); err != nil {
			return fmt.Errorf("model: prefetch V4.1 Engram layer %d: %w", layer, err)
		}
	}
	return nil
}

// Consume hashes and gathers atomically: failed retrieval leaves live history
// unchanged, so a caller can retry the same token chunk safely.
func (s *V41EngramStream) Consume(tokens []int, mask []bool) ([][]byte, error) {
	if s == nil {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Err: fmt.Errorf("nil stream")}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	candidate := s.hash.Clone()
	rows, err := candidate.Hash(tokens, mask)
	if err != nil {
		return nil, err
	}
	out, err := GatherV41EngramRows(s.gather, rows, s.columns)
	if err != nil {
		return nil, err
	}
	*s.hash = *candidate
	return out, nil
}
