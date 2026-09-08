package ctxmmu

// paged_store.go — In-memory staging and read-through cache for paged-out context MMU refs (#10018).
//
// When an oversize tool result (e.g. from fak_read) is paged out to an opaque pointer stub
// ({"_paged":true,"ref":<digest>}), it is staged in this bounded in-memory store while
// asynchronous or background CAS writeback proceeds to disk.
//
// An immediate restore request (via fak_context_restore, ResolvePagedRef, or ResolvePaged)
// checks this in-memory staging store as a read-through fallback if the background CAS flush
// has not yet completed disk writeback. This guarantees that paged refs are immediately
// and deterministically restorable without racing background disk I/O, while bounded LRU
// eviction prevents unbounded in-memory retention.

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/numfmt"
)

const (
	// DefaultPagedStoreMaxBytes is the default byte ceiling for resident staged paged bodies (64 MiB).
	DefaultPagedStoreMaxBytes int64 = 64 << 20

	// DefaultPagedStoreMaxEntries bounds the maximum number of distinct staged paged bodies.
	DefaultPagedStoreMaxEntries = 1024
)

var (
	// ErrPagedRefNotFound is returned when a paged ref is not present in staging.
	ErrPagedRefNotFound = errors.New("ctxmmu: paged ref not found in staging store")

	// ErrPagedRefTampered is returned when staged bytes fail content address verification.
	ErrPagedRefTampered = errors.New("ctxmmu: paged ref content tampered or corrupted")
)

func cleanDigest(d string) string {
	d = strings.TrimSpace(strings.ToLower(d))
	d = strings.TrimPrefix(d, "sha256:")
	return d
}

type pagedEntry struct {
	digest  string
	body    []byte
	size    int64
	element *list.Element
	pinned  int
}

// PagedStore is a thread-safe, byte- and entry-bounded in-memory staging store
// for paged-out results awaiting or completing asynchronous CAS writeback.
type PagedStore struct {
	mu         sync.RWMutex
	entries    map[string]*pagedEntry
	lru        *list.List
	bytes      int64
	maxBytes   int64
	maxEntries int

	hits    int64
	misses  int64
	staged  int64
	evicted int64
}

// NewPagedStore constructs a new bounded PagedStore.
func NewPagedStore(maxBytes int64, maxEntries int) *PagedStore {
	if maxBytes <= 0 {
		maxBytes = DefaultPagedStoreMaxBytes
	}
	if maxEntries <= 0 {
		maxEntries = DefaultPagedStoreMaxEntries
	}
	return &PagedStore{
		entries:    make(map[string]*pagedEntry),
		lru:        list.New(),
		maxBytes:   maxBytes,
		maxEntries: maxEntries,
	}
}

// Stage saves a copy of body into in-memory staging keyed by digest.
func (s *PagedStore) Stage(digest string, body []byte) {
	if s == nil || len(body) == 0 {
		return
	}
	clean := cleanDigest(digest)
	if clean == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	owned := append([]byte(nil), body...)
	if existing, ok := s.entries[clean]; ok {
		s.bytes += int64(len(owned)) - existing.size
		existing.body = owned
		existing.size = int64(len(owned))
		s.lru.MoveToFront(existing.element)
		return
	}

	entry := &pagedEntry{
		digest: clean,
		body:   owned,
		size:   int64(len(owned)),
	}
	entry.element = s.lru.PushFront(entry)
	s.entries[clean] = entry
	s.bytes += entry.size
	s.staged++

	s.evictExcessLocked()
}

// Get returns the staged body for digest if present and intact.
func (s *PagedStore) Get(digest string) ([]byte, bool) {
	if s == nil {
		return nil, false
	}
	clean := cleanDigest(digest)
	if clean == "" {
		return nil, false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.entries[clean]
	if !ok {
		atomic.AddInt64(&s.misses, 1)
		return nil, false
	}

	// Verify content integrity if clean digest is a 64-hex SHA-256
	if len(clean) == 64 {
		sum := sha256.Sum256(entry.body)
		actual := hex.EncodeToString(sum[:])
		if actual != clean {
			s.removeLocked(clean)
			atomic.AddInt64(&s.misses, 1)
			return nil, false
		}
	}

	s.lru.MoveToFront(entry.element)
	atomic.AddInt64(&s.hits, 1)
	out := append([]byte(nil), entry.body...)
	return out, true
}

// Pin prevents an entry from being evicted while in active use.
func (s *PagedStore) Pin(digest string) {
	if s == nil {
		return
	}
	clean := cleanDigest(digest)
	if clean == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if entry, ok := s.entries[clean]; ok {
		entry.pinned++
	}
}

// Unpin releases one pin, making the entry evictable again when pin count drops to zero.
func (s *PagedStore) Unpin(digest string) {
	if s == nil {
		return
	}
	clean := cleanDigest(digest)
	if clean == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if entry, ok := s.entries[clean]; ok && entry.pinned > 0 {
		entry.pinned--
		if entry.pinned == 0 {
			s.evictExcessLocked()
		}
	}
}

// Remove deletes an entry from staging by digest.
func (s *PagedStore) Remove(digest string) {
	if s == nil {
		return
	}
	clean := cleanDigest(digest)
	if clean == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removeLocked(clean)
}

func (s *PagedStore) removeLocked(clean string) {
	if entry, ok := s.entries[clean]; ok {
		s.lru.Remove(entry.element)
		s.bytes -= entry.size
		delete(s.entries, clean)
		s.evicted++
	}
}

func (s *PagedStore) evictExcessLocked() {
	for (s.maxBytes > 0 && s.bytes > s.maxBytes) || (s.maxEntries > 0 && len(s.entries) > s.maxEntries) {
		// Find oldest unpinned entry from the back of the LRU
		var candidate *list.Element
		for curr := s.lru.Back(); curr != nil; curr = curr.Prev() {
			ent := curr.Value.(*pagedEntry)
			if ent.pinned == 0 {
				candidate = curr
				break
			}
		}
		if candidate == nil {
			// All resident entries are pinned (in active use)
			break
		}
		ent := candidate.Value.(*pagedEntry)
		s.lru.Remove(candidate)
		s.bytes -= ent.size
		delete(s.entries, ent.digest)
		s.evicted++
	}
}

// Len reports current entry count in staging.
func (s *PagedStore) Len() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// Bytes reports current resident staged bytes.
func (s *PagedStore) Bytes() int64 {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.bytes
}

// Stats returns hit, miss, stage, and eviction counters.
func (s *PagedStore) Stats() (hits, misses, staged, evicted int64) {
	if s == nil {
		return 0, 0, 0, 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return atomic.LoadInt64(&s.hits), atomic.LoadInt64(&s.misses), s.staged, s.evicted
}

// Clear removes all entries and resets byte accounting.
func (s *PagedStore) Clear() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = make(map[string]*pagedEntry)
	s.lru.Init()
	s.bytes = 0
}

var (
	defaultPagedStoreMu sync.RWMutex
	defaultPagedStore   = newDefaultPagedStore()
)

func envPositiveInt64(key string, def int64) int64 {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func newDefaultPagedStore() *PagedStore {
	maxBytes := envPositiveInt64("FAK_CTXMMU_STAGING_MAX_BYTES", DefaultPagedStoreMaxBytes)
	maxEntries := numfmt.EnvPositiveInt("FAK_CTXMMU_STAGING_MAX_ENTRIES", DefaultPagedStoreMaxEntries)
	return NewPagedStore(maxBytes, maxEntries)
}

// DefaultPagedStore returns the process-default in-memory staging store for paged MMU refs.
func DefaultPagedStore() *PagedStore {
	defaultPagedStoreMu.RLock()
	defer defaultPagedStoreMu.RUnlock()
	return defaultPagedStore
}

// StagePagedRef stages a paged payload into the process-default staging store.
func StagePagedRef(digest string, body []byte) {
	DefaultPagedStore().Stage(digest, body)
}

// GetStagedPagedRef looks up a paged payload from the process-default staging store.
func GetStagedPagedRef(digest string) ([]byte, bool) {
	return DefaultPagedStore().Get(digest)
}

// ResetPagedStoreForTest clears the process-default staging store for testing.
func ResetPagedStoreForTest() {
	defaultPagedStoreMu.Lock()
	defer defaultPagedStoreMu.Unlock()
	defaultPagedStore = newDefaultPagedStore()
}
