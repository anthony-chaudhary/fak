package tokenizer

import (
	"container/list"
	"crypto/sha256"
	"strconv"
	"strings"
	"sync"
)

// SegmentCache is a content-addressed, bounded LRU of ordinary-run encodings that
// many EncodeMemo instances (one per conversation) can share, so a conversation
// chunk is encoded once per process instead of once per conversation. Keys are the
// SHA-256 of the run text; a hit also compares the stored text, so a hash
// collision can never return another run's IDs. A cache binds to the first
// Tokenizer that uses it and bypasses any other, so IDs never cross vocabularies.
// Entries are immutable: callers receive the retained slice and must not modify it
// (the encoder only appends from it). Zero maxBytes/maxEntries select defaults.
// Scope is one process; a cross-process store is out of scope here.
type SegmentCache struct {
	mu         sync.Mutex
	tokenizer  *Tokenizer
	entries    map[[sha256.Size]byte]*list.Element
	lru        *list.List // front = most recently used
	bytes      int
	maxBytes   int
	maxEntries int
	hits       int
	misses     int
	evictions  int
}

type segmentEntry struct {
	key  [sha256.Size]byte
	text string
	ids  []int
}

const (
	defaultSegmentCacheBytes   = 64 << 20
	defaultSegmentCacheEntries = 65536
)

// SegmentCacheStats is a point-in-time snapshot of a SegmentCache.
type SegmentCacheStats struct {
	Hits, Misses, Evictions int
	Entries, PayloadBytes   int
}

// NewSegmentCache returns an empty shared cache; non-positive bounds use defaults.
func NewSegmentCache(maxBytes, maxEntries int) *SegmentCache {
	if maxBytes <= 0 {
		maxBytes = defaultSegmentCacheBytes
	}
	if maxEntries <= 0 {
		maxEntries = defaultSegmentCacheEntries
	}
	return &SegmentCache{
		entries:    make(map[[sha256.Size]byte]*list.Element),
		lru:        list.New(),
		maxBytes:   maxBytes,
		maxEntries: maxEntries,
	}
}

// Stats reports hit/miss/eviction counters and current residency.
func (c *SegmentCache) Stats() SegmentCacheStats {
	if c == nil {
		return SegmentCacheStats{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return SegmentCacheStats{Hits: c.hits, Misses: c.misses, Evictions: c.evictions, Entries: len(c.entries), PayloadBytes: c.bytes}
}

// bind attaches the cache to t on first use and reports whether t may use it.
func (c *SegmentCache) bind(t *Tokenizer) bool {
	if c.tokenizer == nil {
		c.tokenizer = t
	}
	return c.tokenizer == t
}

func (c *SegmentCache) get(t *Tokenizer, text string) ([]int, bool) {
	if c == nil {
		return nil, false
	}
	key := sha256.Sum256([]byte(text))
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.bind(t) {
		return nil, false
	}
	if el, ok := c.entries[key]; ok {
		if e := el.Value.(*segmentEntry); e.text == text {
			c.lru.MoveToFront(el)
			c.hits++
			return e.ids, true
		}
	}
	c.misses++
	return nil, false
}

func (c *SegmentCache) put(t *Tokenizer, text string, ids []int) {
	if c == nil {
		return
	}
	size := len(text) + len(ids)*(strconv.IntSize/8)
	key := sha256.Sum256([]byte(text))
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.bind(t) || size > c.maxBytes {
		return
	}
	if _, ok := c.entries[key]; ok {
		return
	}
	c.entries[key] = c.lru.PushFront(&segmentEntry{key: key, text: strings.Clone(text), ids: append([]int(nil), ids...)})
	c.bytes += size
	for c.bytes > c.maxBytes || len(c.entries) > c.maxEntries {
		oldest := c.lru.Back()
		if oldest == nil {
			break
		}
		e := oldest.Value.(*segmentEntry)
		c.lru.Remove(oldest)
		delete(c.entries, e.key)
		c.bytes -= len(e.text) + len(e.ids)*(strconv.IntSize/8)
		c.evictions++
	}
}
