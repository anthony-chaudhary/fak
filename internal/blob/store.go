// Package blob provides an in-memory, content-addressed blob store backing abi.Ref
// resolution, region management, and page-out caching with byte-bounded LRU eviction.
package blob

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// InlineMax is the byte-size threshold below which payloads stay inline on a Ref.
const InlineMax = 256

// DefaultMaxBytes is the default 1 GiB ceiling for resident unpinned CAS blobs.
const DefaultMaxBytes = 1 << 30

// Store is a concurrency-safe, byte-bounded content-addressed memory blob store
// that supports refcounted pins to protect live entries from LRU eviction.
type Store struct {
	mu       sync.RWMutex
	blobs    map[string][]byte
	bytes    int64
	maxBytes int64
	pins     map[string]int
	lru      *list.List
	lruIndex map[string]*list.Element
	puts     int64
	hits     int64
	// resolv is accessed via sync/atomic under RLock.
	resolv  int64
	evicted int64
}

// New returns an empty store bounded by FAK_BLOB_MAX_BYTES (default DefaultMaxBytes).
func New() *Store {
	return newStore(MaxBytesFromEnv("FAK_BLOB_MAX_BYTES", DefaultMaxBytes))
}

// NewWithBudget returns an empty store with a custom resident byte ceiling.
func NewWithBudget(maxBytes int64) *Store {
	if maxBytes < 1 {
		maxBytes = DefaultMaxBytes
	}
	return newStore(maxBytes)
}

func newStore(maxBytes int64) *Store {
	return &Store{
		blobs:    map[string][]byte{},
		maxBytes: maxBytes,
		pins:     map[string]int{},
		lru:      list.New(),
		lruIndex: map[string]*list.Element{},
	}
}

// MaxBytesFromEnv parses an integer byte budget from the named environment variable.
func MaxBytesFromEnv(name string, def int64) int64 {
	if v, ok := os.LookupEnv(name); ok {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n >= 0 {
			return n
		}
	}
	return def
}

func (s *Store) storeLocked(d string, b []byte) {
	s.blobs[d] = append([]byte(nil), b...)
	s.bytes += int64(len(b))
	if s.pins[d] == 0 {
		s.lruIndex[d] = s.lru.PushFront(d)
	}
	s.evictLocked()
}

func (s *Store) evictLocked() {
	if s.maxBytes <= 0 {
		return
	}
	for s.bytes > s.maxBytes {
		el := s.lru.Back()
		if el == nil {
			return
		}
		d := el.Value.(string)
		s.lru.Remove(el)
		delete(s.lruIndex, d)
		if b, ok := s.blobs[d]; ok {
			s.bytes -= int64(len(b))
			delete(s.blobs, d)
		}
		s.evicted++
	}
}

// Pin increments the refcounted protection preventing eviction of digest.
func (s *Store) Pin(digest string) {
	if digest == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pins[digest]++
	if s.pins[digest] == 1 {
		if el, ok := s.lruIndex[digest]; ok {
			s.lru.Remove(el)
			delete(s.lruIndex, digest)
		}
	}
}

// Unpin decrements pin protection, restoring digest to evictable LRU status when zero.
func (s *Store) Unpin(digest string) {
	if digest == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	n := s.pins[digest]
	if n <= 0 {
		return
	}
	if n == 1 {
		delete(s.pins, digest)
		if _, ok := s.blobs[digest]; ok {
			s.lruIndex[digest] = s.lru.PushFront(digest)
		}
		s.evictLocked()
		return
	}
	s.pins[digest] = n - 1
}

// Digest returns the canonical sha256 content address for byte slice b.
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// PreparePut constructs an addressable Ref header, keeping payloads at or below InlineMax inline.
func PreparePut(b []byte) (r abi.Ref, inline bool) {
	r = abi.Ref{Digest: Digest(b), Len: int64(len(b)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}
	if len(b) <= InlineMax {
		r.Kind = abi.RefInline
		r.Inline = append([]byte(nil), b...)
		return r, true
	}
	r.Kind = abi.RefBlob
	return r, false
}

// PageIn resolves a paged-out handle Ref into an inline Ref via the provided Resolver.
func PageIn(ctx context.Context, res abi.Resolver, handle abi.Ref) (abi.Ref, error) {
	b, err := res.Resolve(ctx, handle)
	if err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefInline, Digest: handle.Digest, Inline: b, Len: int64(len(b)), Taint: handle.Taint, Scope: handle.Scope}, nil
}

// Put stores bytes in the CAS or inline and returns the corresponding addressable Ref.
func (s *Store) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	r, inline := PreparePut(b)
	if inline {
		return r, nil
	}
	d := r.Digest
	s.mu.Lock()
	defer s.mu.Unlock()
	s.puts++
	if _, ok := s.blobs[d]; ok {
		s.hits++
	} else {
		s.storeLocked(d, b)
	}
	return r, nil
}

// Resolve materializes the bytes a Ref points at.
func (s *Store) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), nil
	case abi.RefBlob, abi.RefRegion:
		s.mu.RLock()
		defer s.mu.RUnlock()
		atomic.AddInt64(&s.resolv, 1)
		b, ok := s.blobs[r.Digest]
		if !ok {
			return nil, fmt.Errorf("blob: unknown digest %s", r.Digest)
		}
		return append([]byte(nil), b...), nil
	default:
		return nil, fmt.Errorf("blob: unknown RefKind %d", r.Kind)
	}
}

// PageOut offloads inline Ref bytes into the CAS and returns a handle Ref.
func (s *Store) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	b, err := s.Resolve(ctx, r)
	if err != nil {
		return abi.Ref{}, err
	}
	d := Digest(b)
	s.mu.Lock()
	if _, ok := s.blobs[d]; !ok {
		s.storeLocked(d, b)
	}
	s.mu.Unlock()
	return abi.Ref{Kind: abi.RefBlob, Digest: d, Len: int64(len(b)), Taint: r.Taint, Scope: r.Scope}, nil
}

// PageIn re-materializes a paged-out handle Ref into an inline Ref.
func (s *Store) PageIn(ctx context.Context, handle abi.Ref) (abi.Ref, error) {
	return PageIn(ctx, s, handle)
}

// Stats reports lifetime store metrics for puts, deduplication hits, and resolves.
func (s *Store) Stats() (puts, dedupHits, resolves int64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.puts, s.hits, atomic.LoadInt64(&s.resolv)
}

// Len reports the number of distinct blobs resident in the CAS.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.blobs)
}

// Bytes returns the total payload bytes currently resident in the CAS.
func (s *Store) Bytes() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.bytes
}

// Reset purges all stored blobs and clears resident memory while retaining activity counters.
func (s *Store) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.blobs = map[string][]byte{}
	s.bytes = 0
	s.pins = map[string]int{}
	s.lru = list.New()
	s.lruIndex = map[string]*list.Element{}
}

// Evicted returns the count of unpinned blobs dropped to satisfy resident bounds.
func (s *Store) Evicted() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.evicted
}

// Resident returns a point-in-time snapshot of blob count, total bytes, and evicted count.
func (s *Store) Resident() (blobCount int, bytes, evicted int64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.blobs), s.bytes, s.evicted
}

// MaxBytes reports the configured resident-footprint ceiling (0 = unbounded).
func (s *Store) MaxBytes() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.maxBytes
}

// SetMaxBytes updates the resident byte limit and immediately evicts eligible unpinned blobs.
func (s *Store) SetMaxBytes(n int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.maxBytes = n
	s.evictLocked()
}

type backend struct{ s *Store }

// Resolver returns the underlying Store as the abi.Resolver.
func (b backend) Resolver() abi.Resolver { return b.s }

// Caps returns backend capabilities supported by the Store.
func (b backend) Caps() []abi.Capability { return nil }

// PageOut offloads a Ref payload into the underlying Store.
func (b backend) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return b.s.PageOut(ctx, r)
}

// PageIn resolves a paged handle Ref through the underlying Store.
func (b backend) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	return b.s.PageIn(ctx, h)
}

// Default is the process-wide Store instance registered with the ABI runtime.
var Default = New()

func init() {
	b := backend{Default}
	abi.RegisterRegionBackend(b)
	abi.RegisterPageOutBackend("blob", b)
}
