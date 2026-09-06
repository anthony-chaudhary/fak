// Package kv provides core Key-Value storage management abstractions for
// LLM inference KV-caches, including page allocation, direct I/O block store
// integration, eviction policies (LRU, FIFO), multi-turn session indexing,
// token range lookup, and runtime access statistics.
package kv

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	directio "github.com/anthony-chaudhary/fak/internal/kv/direct_io"
)

// DefaultPageSize is the standard 4096-byte (4KB) memory and block page size.
const DefaultPageSize = directio.BlockSize

// Sentinel errors returned by KV storage operations.
var (
	// ErrKeyNotFound indicates the specified cache key does not exist in the store.
	ErrKeyNotFound = errors.New("kv: key not found")

	// ErrPageNotFound indicates the specified page identifier does not exist.
	ErrPageNotFound = errors.New("kv: page not found")

	// ErrStoreClosed indicates operations were attempted on a closed store.
	ErrStoreClosed = errors.New("kv: store is closed")

	// ErrCapacityExceeded indicates the store cannot allocate more pages and eviction cannot free space.
	ErrCapacityExceeded = errors.New("kv: storage capacity exceeded")

	// ErrInvalidConfig indicates the store configuration contains invalid or conflicting values.
	ErrInvalidConfig = errors.New("kv: invalid configuration")

	// ErrInvalidKey indicates the cache key fails basic validation requirements.
	ErrInvalidKey = errors.New("kv: invalid cache key")

	// ErrDataTooLarge indicates the payload exceeds the maximum configured page size.
	ErrDataTooLarge = errors.New("kv: data exceeds page size")

	// ErrPagePinned indicates a page cannot be evicted because it is actively pinned.
	ErrPagePinned = errors.New("kv: page is pinned and cannot be evicted")
)

// EvictionPolicy defines the replacement algorithm used when capacity is exhausted.
type EvictionPolicy string

const (
	// EvictionPolicyLRU evicts least-recently accessed unpinned pages first.
	EvictionPolicyLRU EvictionPolicy = "lru"

	// EvictionPolicyFIFO evicts oldest-allocated unpinned pages first.
	EvictionPolicyFIFO EvictionPolicy = "fifo"
)

// Config configures the capacity, page geometry, backing storage, and eviction policy of a KVStore.
type Config struct {
	// PageSize is the byte size of each storage page (must be > 0, defaults to 4096).
	PageSize int `json:"page_size"`

	// MaxPages is the maximum number of pages allowed (<= 0 implies unlimited memory).
	MaxPages int `json:"max_pages"`

	// EvictionPolicy specifies LRU or FIFO eviction when MaxPages is reached.
	EvictionPolicy EvictionPolicy `json:"eviction_policy"`

	// DirectIO enables write-through to a page-aligned direct I/O block file.
	DirectIO bool `json:"direct_io"`

	// BackingFile specifies the filesystem path for direct I/O block storage.
	// Required if DirectIO is true.
	BackingFile string `json:"backing_file"`
}

// DefaultConfig returns a recommended production configuration with 4KB pages and LRU eviction.
func DefaultConfig() Config {
	return Config{
		PageSize:       DefaultPageSize,
		MaxPages:       1024,
		EvictionPolicy: EvictionPolicyLRU,
		DirectIO:       false,
		BackingFile:    "",
	}
}

// Validate checks that configuration values are well-formed and consistent,
// ensuring positive page sizing, valid eviction policy selection, and backing paths.
func (c *Config) Validate() error {
	if c.PageSize <= 0 {
		return fmt.Errorf("%w: page size must be positive, got %d", ErrInvalidConfig, c.PageSize)
	}
	if c.EvictionPolicy == "" {
		c.EvictionPolicy = EvictionPolicyLRU
	}
	switch c.EvictionPolicy {
	case EvictionPolicyLRU, EvictionPolicyFIFO:
	default:
		return fmt.Errorf("%w: unknown eviction policy %q", ErrInvalidConfig, c.EvictionPolicy)
	}
	if c.DirectIO && strings.TrimSpace(c.BackingFile) == "" {
		return fmt.Errorf("%w: backing file path is required when DirectIO is enabled", ErrInvalidConfig)
	}
	return nil
}

// CacheKey uniquely addresses a KV cache tensor slice by session, turn, layer, and token range.
type CacheKey struct {
	// SessionID identifies the conversation or agent session.
	SessionID string `json:"session_id"`

	// Turn indicates the conversation turn number (0-indexed or 1-indexed).
	Turn int `json:"turn"`

	// Layer indicates the transformer layer index.
	Layer int `json:"layer"`

	// TokenOffset indicates the sequence start token offset.
	TokenOffset int `json:"token_offset"`

	// NumTokens indicates the number of tokens represented in this block.
	NumTokens int `json:"num_tokens"`

	// Tag is an optional classification string (e.g., "system", "prompt", "turn").
	Tag string `json:"tag,omitempty"`
}

// Validate checks that the cache key possesses a non-empty session identifier
// and non-negative turn, layer, offset, and token count coordinates.
func (k CacheKey) Validate() error {
	if strings.TrimSpace(k.SessionID) == "" {
		return fmt.Errorf("%w: session ID cannot be empty", ErrInvalidKey)
	}
	if k.Turn < 0 {
		return fmt.Errorf("%w: turn must be non-negative, got %d", ErrInvalidKey, k.Turn)
	}
	if k.Layer < 0 {
		return fmt.Errorf("%w: layer must be non-negative, got %d", ErrInvalidKey, k.Layer)
	}
	if k.TokenOffset < 0 {
		return fmt.Errorf("%w: token offset must be non-negative, got %d", ErrInvalidKey, k.TokenOffset)
	}
	if k.NumTokens < 0 {
		return fmt.Errorf("%w: num tokens must be non-negative, got %d", ErrInvalidKey, k.NumTokens)
	}
	return nil
}

// String returns a deterministic, canonical representation of the cache key.
func (k CacheKey) String() string {
	if k.Tag != "" {
		return fmt.Sprintf("%s/t%d/l%d/tok%d:%d#%s", k.SessionID, k.Turn, k.Layer, k.TokenOffset, k.TokenOffset+k.NumTokens, k.Tag)
	}
	return fmt.Sprintf("%s/t%d/l%d/tok%d:%d", k.SessionID, k.Turn, k.Layer, k.TokenOffset, k.TokenOffset+k.NumTokens)
}

// MatchesSession returns true if the key belongs to the given session identifier.
func (k CacheKey) MatchesSession(sessionID string) bool {
	return k.SessionID == sessionID
}

// Overlaps reports whether this key and another share the same session, turn, layer,
// and have intersecting token ranges.
func (k CacheKey) Overlaps(other CacheKey) bool {
	if k.SessionID != other.SessionID || k.Turn != other.Turn || k.Layer != other.Layer {
		return false
	}
	endA := k.TokenOffset + k.NumTokens
	endB := other.TokenOffset + other.NumTokens
	return k.TokenOffset < endB && endA > other.TokenOffset
}

// Page represents an allocated unit of KV cache memory and block storage.
type Page struct {
	// ID is the unique integer identifier assigned to this page.
	ID int `json:"id"`

	// Key is the multi-dimensional cache coordinate associated with this page.
	Key CacheKey `json:"key"`

	// Data is the raw byte buffer holding tensor representations.
	Data []byte `json:"-"`

	// BytesUsed is the count of valid data bytes written to this page.
	BytesUsed int `json:"bytes_used"`

	// NumTokens is the number of tokens represented by the data in this page.
	NumTokens int `json:"num_tokens"`

	// AllocatedAt records the creation time of this page.
	AllocatedAt time.Time `json:"allocated_at"`

	// LastAccessed records the most recent read or write operation timestamp.
	LastAccessed time.Time `json:"last_accessed"`

	// AccessCount counts the cumulative read/write operations against this page.
	AccessCount uint64 `json:"access_count"`

	// Pinned indicates whether this page is protected against automatic eviction.
	Pinned bool `json:"pinned"`

	// Dirty indicates that in-memory contents have not been synchronized.
	Dirty bool `json:"dirty"`

	// DirectIOBlockID is the backing block index if persisted via direct I/O, or -1.
	DirectIOBlockID int `json:"direct_io_block_id"`
}

// Touch updates the last access time to the current timestamp and increments the access count.
func (p *Page) Touch() {
	p.LastAccessed = time.Now()
	p.AccessCount++
}

// Pin marks the page as pinned, preventing it from being removed by eviction algorithms.
func (p *Page) Pin() {
	p.Pinned = true
}

// Unpin releases the eviction protection lock on the page.
func (p *Page) Unpin() {
	p.Pinned = false
}

// IsPinned returns true if the page is currently protected from eviction.
func (p *Page) IsPinned() bool {
	return p.Pinned
}

// Bytes returns a slice over the valid portion of the page data buffer.
func (p *Page) Bytes() []byte {
	if p.BytesUsed <= 0 || len(p.Data) == 0 {
		return nil
	}
	if p.BytesUsed > len(p.Data) {
		return p.Data
	}
	return p.Data[:p.BytesUsed]
}

// Clone creates an isolated, deep copy of the page and its underlying data buffer.
func (p *Page) Clone() *Page {
	if p == nil {
		return nil
	}
	cp := *p
	if len(p.Data) > 0 {
		cp.Data = make([]byte, len(p.Data))
		copy(cp.Data, p.Data)
	}
	return &cp
}

// Stats captures cumulative and point-in-time metrics for KV cache operations.
type Stats struct {
	// AllocatedPages is the current number of pages currently held in the store.
	AllocatedPages int `json:"allocated_pages"`

	// FreePages is the remaining capacity of pages before eviction is required.
	FreePages int `json:"free_pages"`

	// CapacityPages is the total maximum page capacity configured.
	CapacityPages int `json:"capacity_pages"`

	// PinnedPages is the count of pages currently protected by pin locks.
	PinnedPages int `json:"pinned_pages"`

	// BytesUsed is the cumulative memory currently allocated for active page payloads.
	BytesUsed int64 `json:"bytes_used"`

	// BytesCapacity is the theoretical maximum byte capacity.
	BytesCapacity int64 `json:"bytes_capacity"`

	// Puts is the cumulative count of Put operations executed.
	Puts uint64 `json:"puts"`

	// Gets is the cumulative count of Get operations executed.
	Gets uint64 `json:"gets"`

	// Hits is the cumulative count of successful lookups.
	Hits uint64 `json:"hits"`

	// Misses is the cumulative count of failed lookups.
	Misses uint64 `json:"misses"`

	// Evictions is the cumulative count of pages evicted due to capacity constraints.
	Evictions uint64 `json:"evictions"`

	// Frees is the cumulative count of pages explicitly freed by caller request.
	Frees uint64 `json:"frees"`
}

// HitRatio calculates the lookup cache hit ratio as a fraction between 0.0 and 1.0.
func (s Stats) HitRatio() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0.0
	}
	return float64(s.Hits) / float64(total)
}

// Snapshot returns a point-in-time copy of the statistics counters.
func (s Stats) Snapshot() Stats {
	return s
}

// Store defines the interface for KV cache storage management,
// supporting allocation, retrieval, eviction, session grouping, and block lookup.
type Store interface {
	// AllocatePage reserves a new page for the given key, evicting if necessary.
	AllocatePage(key CacheKey) (*Page, error)

	// AllocateBatch reserves a batch of pages atomically.
	AllocateBatch(keys []CacheKey) ([]*Page, error)

	// FreePage releases the specified page by ID, returning it to the pool.
	FreePage(pageID int) error

	// FreeSession frees all allocated pages belonging to the specified session.
	FreeSession(sessionID string) (int, error)

	// Put stores tensor data under the specified key, creating or updating the page.
	Put(key CacheKey, data []byte) (*Page, error)

	// PutPage inserts or updates a pre-allocated page.
	PutPage(page *Page) error

	// Get retrieves a copy of the page matching the key and updates access telemetry.
	Get(key CacheKey) (*Page, error)

	// GetByID retrieves a copy of the page by page ID and updates access telemetry.
	GetByID(pageID int) (*Page, error)

	// Evict evicts up to count unpinned pages using the configured eviction policy.
	Evict(count int) (int, error)

	// EvictOldest evicts up to count unpinned pages with the oldest access or allocation times.
	EvictOldest(count int) (int, error)

	// EvictSession evicts all unpinned pages associated with a specific session.
	EvictSession(sessionID string) (int, error)

	// PruneIdle removes all unpinned pages that have not been accessed within the duration.
	PruneIdle(olderThan time.Duration) (int, error)

	// Pin locks a page from eviction.
	Pin(pageID int) error

	// Unpin removes the eviction lock on a page.
	Unpin(pageID int) error

	// Lookup finds a page by key without mutating access telemetry or returning error on miss.
	Lookup(key CacheKey) (*Page, bool)

	// LookupSession returns all pages belonging to a session sorted by turn and token offset.
	LookupSession(sessionID string) []*Page

	// LookupRange returns pages for a session and turn overlapping the given token range.
	LookupRange(sessionID string, turn int, startToken, endToken int) []*Page

	// ListPages returns copies of all active pages in the store.
	ListPages() []*Page

	// Stats returns a snapshot of runtime operation statistics.
	Stats() Stats

	// ResetStats resets mutable operational metrics (puts, gets, hits, misses).
	ResetStats()

	// Config returns the immutable configuration used to initialize the store.
	Config() Config

	// Close flushes backing storage and frees all resources.
	Close() error
}

// KVStore implements the Store interface with thread-safe in-memory indexing
// and optional direct I/O block persistence.
type KVStore struct {
	mu           sync.RWMutex
	cfg          Config
	pages        map[int]*Page
	keyIndex     map[string]int
	sessionIndex map[string]map[int]struct{}
	nextPageID   int
	freeIDs      []int
	directStore  *directio.LoopbackKVStore
	stats        Stats
	closed       bool
}

// New creates and initializes a new KVStore instance conforming to the provided configuration,
// validating options and initializing any configured direct I/O backing store.
func New(cfg Config) (*KVStore, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	var dStore *directio.LoopbackKVStore
	if cfg.DirectIO || cfg.BackingFile != "" {
		if cfg.BackingFile != "" {
			ds, err := directio.OpenLoopbackKVStore(cfg.BackingFile, cfg.MaxPages)
			if err != nil {
				return nil, fmt.Errorf("kv: open direct I/O backing store: %w", err)
			}
			dStore = ds
		}
	}

	s := &KVStore{
		cfg:          cfg,
		pages:        make(map[int]*Page),
		keyIndex:     make(map[string]int),
		sessionIndex: make(map[string]map[int]struct{}),
		nextPageID:   0,
		freeIDs:      make([]int, 0),
		directStore:  dStore,
		stats: Stats{
			CapacityPages: cfg.MaxPages,
			BytesCapacity: int64(cfg.MaxPages) * int64(cfg.PageSize),
		},
	}
	s.updateCapacityStatsLocked()
	return s, nil
}

// NewStore is an alias for New, returning the Store interface.
func NewStore(cfg Config) (Store, error) {
	return New(cfg)
}

// DefaultStore constructs a KVStore with DefaultConfig settings.
func DefaultStore() (*KVStore, error) {
	return New(DefaultConfig())
}

// Config returns the configuration struct used to initialize this store.
func (s *KVStore) Config() Config {
	return s.cfg
}

// AllocatePage reserves a new page for the given key. If the store is at maximum
// capacity, it attempts to evict an unpinned page according to the configured policy.
func (s *KVStore) AllocatePage(key CacheKey) (*Page, error) {
	if err := key.Validate(); err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	keyStr := key.String()
	if existingID, exists := s.keyIndex[keyStr]; exists {
		if page, ok := s.pages[existingID]; ok {
			page.Touch()
			return page.Clone(), nil
		}
	}

	if err := s.ensureCapacityLocked(1); err != nil {
		return nil, err
	}

	return s.allocatePageInternalLocked(key)
}

// AllocateBatch allocates multiple pages atomically under a single store lock.
func (s *KVStore) AllocateBatch(keys []CacheKey) ([]*Page, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	for _, k := range keys {
		if err := k.Validate(); err != nil {
			return nil, err
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	if err := s.ensureCapacityLocked(len(keys)); err != nil {
		return nil, err
	}

	pages := make([]*Page, 0, len(keys))
	for _, k := range keys {
		p, err := s.allocatePageInternalLocked(k)
		if err != nil {
			return nil, err
		}
		pages = append(pages, p)
	}
	return pages, nil
}

func (s *KVStore) allocatePageInternalLocked(key CacheKey) (*Page, error) {
	var pageID int
	if len(s.freeIDs) > 0 {
		pageID = s.freeIDs[len(s.freeIDs)-1]
		s.freeIDs = s.freeIDs[:len(s.freeIDs)-1]
	} else {
		pageID = s.nextPageID
		s.nextPageID++
	}

	directBlockID := -1
	if s.directStore != nil {
		blockMeta := directio.BlockMetadata{
			SessionID:   key.SessionID,
			Turn:        key.Turn,
			TokenOffset: key.TokenOffset,
			NumTokens:   key.NumTokens,
			Layer:       key.Layer,
		}
		bid, err := s.directStore.AllocateBlock(blockMeta)
		if err != nil {
			s.freeIDs = append(s.freeIDs, pageID)
			return nil, fmt.Errorf("kv: direct I/O allocate block: %w", err)
		}
		directBlockID = bid
	}

	now := time.Now()
	page := &Page{
		ID:              pageID,
		Key:             key,
		Data:            directio.AlignedBuffer(s.cfg.PageSize),
		BytesUsed:       0,
		NumTokens:       key.NumTokens,
		AllocatedAt:     now,
		LastAccessed:    now,
		AccessCount:     1,
		Pinned:          false,
		Dirty:           false,
		DirectIOBlockID: directBlockID,
	}

	s.pages[pageID] = page
	s.keyIndex[key.String()] = pageID
	if _, ok := s.sessionIndex[key.SessionID]; !ok {
		s.sessionIndex[key.SessionID] = make(map[int]struct{})
	}
	s.sessionIndex[key.SessionID][pageID] = struct{}{}

	s.updateCapacityStatsLocked()
	return page.Clone(), nil
}

// FreePage releases a page and any associated backing blocks back to the pool.
func (s *KVStore) FreePage(pageID int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}

	page, ok := s.pages[pageID]
	if !ok {
		return ErrPageNotFound
	}

	s.freePageLocked(page)
	s.stats.Frees++
	s.updateCapacityStatsLocked()
	return nil
}

// FreeSession frees all pages associated with the given session identifier.
func (s *KVStore) FreeSession(sessionID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}

	pageIDsMap, ok := s.sessionIndex[sessionID]
	if !ok || len(pageIDsMap) == 0 {
		return 0, nil
	}

	ids := make([]int, 0, len(pageIDsMap))
	for id := range pageIDsMap {
		ids = append(ids, id)
	}

	freed := 0
	for _, id := range ids {
		if page, exists := s.pages[id]; exists {
			s.freePageLocked(page)
			freed++
		}
	}
	delete(s.sessionIndex, sessionID)

	s.stats.Frees += uint64(freed)
	s.updateCapacityStatsLocked()
	return freed, nil
}

func (s *KVStore) freePageLocked(page *Page) {
	if page.DirectIOBlockID >= 0 && s.directStore != nil {
		_ = s.directStore.FreeBlock(page.DirectIOBlockID)
	}

	delete(s.pages, page.ID)
	delete(s.keyIndex, page.Key.String())
	if sessMap, ok := s.sessionIndex[page.Key.SessionID]; ok {
		delete(sessMap, page.ID)
		if len(sessMap) == 0 {
			delete(s.sessionIndex, page.Key.SessionID)
		}
	}
	s.freeIDs = append(s.freeIDs, page.ID)
}

// Put writes tensor payload bytes into the page addressed by the key.
// If the page does not exist, it is allocated.
func (s *KVStore) Put(key CacheKey, data []byte) (*Page, error) {
	if err := key.Validate(); err != nil {
		return nil, err
	}
	if len(data) > s.cfg.PageSize {
		return nil, fmt.Errorf("%w: payload %d bytes exceeds page size %d", ErrDataTooLarge, len(data), s.cfg.PageSize)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	s.stats.Puts++

	keyStr := key.String()
	var page *Page
	if pageID, exists := s.keyIndex[keyStr]; exists {
		page = s.pages[pageID]
	} else {
		if err := s.ensureCapacityLocked(1); err != nil {
			return nil, err
		}
		p, err := s.allocatePageInternalLocked(key)
		if err != nil {
			return nil, err
		}
		page = s.pages[p.ID]
	}

	copy(page.Data, data)
	page.BytesUsed = len(data)
	page.Touch()
	page.Dirty = true

	if s.directStore != nil && page.DirectIOBlockID >= 0 {
		meta := directio.BlockMetadata{
			BlockID:     page.DirectIOBlockID,
			SessionID:   page.Key.SessionID,
			Turn:        page.Key.Turn,
			TokenOffset: page.Key.TokenOffset,
			NumTokens:   page.Key.NumTokens,
			Layer:       page.Key.Layer,
			BytesUsed:   page.BytesUsed,
		}
		if err := s.directStore.WriteBlockWithMeta(page.DirectIOBlockID, meta, data); err != nil {
			return nil, fmt.Errorf("kv: direct I/O write block: %w", err)
		}
		page.Dirty = false
	}

	s.updateCapacityStatsLocked()
	return page.Clone(), nil
}

// PutPage inserts or replaces a page directly into the store.
func (s *KVStore) PutPage(page *Page) error {
	if page == nil {
		return errors.New("kv: nil page")
	}
	if err := page.Key.Validate(); err != nil {
		return err
	}
	if len(page.Bytes()) > s.cfg.PageSize {
		return ErrDataTooLarge
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}

	s.stats.Puts++

	existing, exists := s.pages[page.ID]
	if exists {
		delete(s.keyIndex, existing.Key.String())
		if sessMap, ok := s.sessionIndex[existing.Key.SessionID]; ok {
			delete(sessMap, existing.ID)
		}
	} else {
		if err := s.ensureCapacityLocked(1); err != nil {
			return err
		}
	}

	cloned := page.Clone()
	if len(cloned.Data) < s.cfg.PageSize {
		buf := directio.AlignedBuffer(s.cfg.PageSize)
		copy(buf, cloned.Data)
		cloned.Data = buf
	}
	cloned.Touch()

	s.pages[cloned.ID] = cloned
	s.keyIndex[cloned.Key.String()] = cloned.ID
	if _, ok := s.sessionIndex[cloned.Key.SessionID]; !ok {
		s.sessionIndex[cloned.Key.SessionID] = make(map[int]struct{})
	}
	s.sessionIndex[cloned.Key.SessionID][cloned.ID] = struct{}{}

	if cloned.ID >= s.nextPageID {
		s.nextPageID = cloned.ID + 1
	}

	if s.directStore != nil && cloned.DirectIOBlockID >= 0 {
		meta := directio.BlockMetadata{
			BlockID:     cloned.DirectIOBlockID,
			SessionID:   cloned.Key.SessionID,
			Turn:        cloned.Key.Turn,
			TokenOffset: cloned.Key.TokenOffset,
			NumTokens:   cloned.Key.NumTokens,
			Layer:       cloned.Key.Layer,
			BytesUsed:   cloned.BytesUsed,
		}
		_ = s.directStore.WriteBlockWithMeta(cloned.DirectIOBlockID, meta, cloned.Bytes())
	}

	s.updateCapacityStatsLocked()
	return nil
}

// Get retrieves a page by key and records access statistics.
func (s *KVStore) Get(key CacheKey) (*Page, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	s.stats.Gets++

	pageID, ok := s.keyIndex[key.String()]
	if !ok {
		s.stats.Misses++
		return nil, ErrKeyNotFound
	}

	page, ok := s.pages[pageID]
	if !ok {
		s.stats.Misses++
		return nil, ErrKeyNotFound
	}

	s.stats.Hits++
	page.Touch()
	return page.Clone(), nil
}

// GetByID retrieves a page by its integer page ID and records access statistics.
func (s *KVStore) GetByID(pageID int) (*Page, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrStoreClosed
	}

	s.stats.Gets++

	page, ok := s.pages[pageID]
	if !ok {
		s.stats.Misses++
		return nil, ErrPageNotFound
	}

	s.stats.Hits++
	page.Touch()
	return page.Clone(), nil
}

// Lookup queries for a page by key without mutating access timestamps or stats.
func (s *KVStore) Lookup(key CacheKey) (*Page, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil, false
	}

	pageID, ok := s.keyIndex[key.String()]
	if !ok {
		return nil, false
	}
	page, ok := s.pages[pageID]
	if !ok {
		return nil, false
	}
	return page.Clone(), true
}

// LookupSession returns all pages belonging to the session, sorted by Turn and TokenOffset.
func (s *KVStore) LookupSession(sessionID string) []*Page {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil
	}

	pageIDs, ok := s.sessionIndex[sessionID]
	if !ok || len(pageIDs) == 0 {
		return nil
	}

	results := make([]*Page, 0, len(pageIDs))
	for id := range pageIDs {
		if page, exists := s.pages[id]; exists {
			results = append(results, page.Clone())
		}
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Key.Turn != results[j].Key.Turn {
			return results[i].Key.Turn < results[j].Key.Turn
		}
		if results[i].Key.TokenOffset != results[j].Key.TokenOffset {
			return results[i].Key.TokenOffset < results[j].Key.TokenOffset
		}
		return results[i].Key.Layer < results[j].Key.Layer
	})
	return results
}

// LookupRange returns all pages for a session and turn overlapping the given token span [startToken, endToken).
func (s *KVStore) LookupRange(sessionID string, turn int, startToken, endToken int) []*Page {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil
	}

	pageIDs, ok := s.sessionIndex[sessionID]
	if !ok || len(pageIDs) == 0 {
		return nil
	}

	results := make([]*Page, 0)
	for id := range pageIDs {
		page, exists := s.pages[id]
		if !exists || page.Key.Turn != turn {
			continue
		}
		pageStart := page.Key.TokenOffset
		pageEnd := page.Key.TokenOffset + page.NumTokens
		if pageEnd <= pageStart {
			pageEnd = pageStart + 1
		}

		overlaps := false
		if startToken < endToken {
			overlaps = pageStart < endToken && pageEnd > startToken
		} else {
			overlaps = pageStart <= startToken && startToken < pageEnd
		}

		if overlaps {
			results = append(results, page.Clone())
		}
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Key.TokenOffset != results[j].Key.TokenOffset {
			return results[i].Key.TokenOffset < results[j].Key.TokenOffset
		}
		return results[i].Key.Layer < results[j].Key.Layer
	})
	return results
}

// ListPages returns copies of all active pages sorted by page ID.
func (s *KVStore) ListPages() []*Page {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil
	}

	results := make([]*Page, 0, len(s.pages))
	for _, page := range s.pages {
		results = append(results, page.Clone())
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].ID < results[j].ID
	})
	return results
}

// Pin prevents a page from being evicted by automatic or explicit eviction passes.
func (s *KVStore) Pin(pageID int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}

	page, ok := s.pages[pageID]
	if !ok {
		return ErrPageNotFound
	}
	page.Pin()
	s.updateCapacityStatsLocked()
	return nil
}

// Unpin releases the eviction lock from the given page.
func (s *KVStore) Unpin(pageID int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}

	page, ok := s.pages[pageID]
	if !ok {
		return ErrPageNotFound
	}
	page.Unpin()
	s.updateCapacityStatsLocked()
	return nil
}

// Evict removes up to count unpinned pages according to the configured eviction policy.
func (s *KVStore) Evict(count int) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}
	return s.evictCandidatesLocked(count, s.cfg.EvictionPolicy)
}

// EvictOldest removes up to count unpinned pages with the oldest access times (LRU).
func (s *KVStore) EvictOldest(count int) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}
	return s.evictCandidatesLocked(count, EvictionPolicyLRU)
}

// EvictSession evicts all unpinned pages associated with the given session ID.
func (s *KVStore) EvictSession(sessionID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}

	pageIDsMap, ok := s.sessionIndex[sessionID]
	if !ok || len(pageIDsMap) == 0 {
		return 0, nil
	}

	var candidates []*Page
	for id := range pageIDsMap {
		if page, exists := s.pages[id]; exists && !page.IsPinned() {
			candidates = append(candidates, page)
		}
	}

	for _, page := range candidates {
		s.freePageLocked(page)
	}

	evicted := len(candidates)
	s.stats.Evictions += uint64(evicted)
	s.updateCapacityStatsLocked()
	return evicted, nil
}

// PruneIdle removes all unpinned pages that have not been accessed within olderThan duration.
func (s *KVStore) PruneIdle(olderThan time.Duration) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrStoreClosed
	}

	cutoff := time.Now().Add(-olderThan)
	var candidates []*Page
	for _, page := range s.pages {
		if !page.IsPinned() && !page.LastAccessed.After(cutoff) {
			candidates = append(candidates, page)
		}
	}

	for _, page := range candidates {
		s.freePageLocked(page)
	}

	evicted := len(candidates)
	s.stats.Evictions += uint64(evicted)
	s.updateCapacityStatsLocked()
	return evicted, nil
}

func (s *KVStore) evictCandidatesLocked(count int, policy EvictionPolicy) (int, error) {
	if count <= 0 || len(s.pages) == 0 {
		return 0, nil
	}

	candidates := make([]*Page, 0, len(s.pages))
	for _, page := range s.pages {
		if !page.IsPinned() {
			candidates = append(candidates, page)
		}
	}

	if len(candidates) == 0 {
		return 0, nil
	}

	switch policy {
	case EvictionPolicyFIFO:
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].AllocatedAt.Before(candidates[j].AllocatedAt)
		})
	case EvictionPolicyLRU:
		fallthrough
	default:
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].LastAccessed.Before(candidates[j].LastAccessed)
		})
	}

	toEvict := count
	if toEvict > len(candidates) {
		toEvict = len(candidates)
	}

	for i := 0; i < toEvict; i++ {
		s.freePageLocked(candidates[i])
	}

	s.stats.Evictions += uint64(toEvict)
	s.updateCapacityStatsLocked()
	return toEvict, nil
}

func (s *KVStore) ensureCapacityLocked(needed int) error {
	if s.cfg.MaxPages <= 0 {
		return nil
	}

	available := s.cfg.MaxPages - len(s.pages)
	if available >= needed {
		return nil
	}

	toFree := needed - available
	freed, err := s.evictCandidatesLocked(toFree, s.cfg.EvictionPolicy)
	if err != nil {
		return err
	}
	if freed < toFree {
		return ErrCapacityExceeded
	}
	return nil
}

func (s *KVStore) updateCapacityStatsLocked() {
	s.stats.AllocatedPages = len(s.pages)
	if s.cfg.MaxPages > 0 {
		s.stats.FreePages = s.cfg.MaxPages - len(s.pages)
		if s.stats.FreePages < 0 {
			s.stats.FreePages = 0
		}
	} else {
		s.stats.FreePages = 0
	}

	pinned := 0
	var bytesUsed int64
	for _, page := range s.pages {
		if page.IsPinned() {
			pinned++
		}
		bytesUsed += int64(page.BytesUsed)
	}
	s.stats.PinnedPages = pinned
	s.stats.BytesUsed = bytesUsed
}

// Stats returns a thread-safe snapshot of operation statistics and capacity.
func (s *KVStore) Stats() Stats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.stats.Snapshot()
}

// ResetStats resets mutable throughput and lookup metrics while preserving capacity counters.
func (s *KVStore) ResetStats() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats.Puts = 0
	s.stats.Gets = 0
	s.stats.Hits = 0
	s.stats.Misses = 0
	s.stats.Evictions = 0
	s.stats.Frees = 0
}

// Close closes the store, releases all pages, and closes backing direct I/O files.
func (s *KVStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true

	var err error
	if s.directStore != nil {
		err = s.directStore.Close()
		s.directStore = nil
	}

	s.pages = nil
	s.keyIndex = nil
	s.sessionIndex = nil
	s.freeIDs = nil
	return err
}
