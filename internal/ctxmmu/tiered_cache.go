package ctxmmu

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

var (
	// ErrPromptNotFound indicates the specified prompt sequence was not found in cache.
	ErrPromptNotFound = errors.New("ctxmmu: prompt not found in cache")

	// ErrCacheCapacityExceeded indicates cache capacity limits were breached without eviction recourse.
	ErrCacheCapacityExceeded = errors.New("ctxmmu: cache capacity exceeded")
)

// PromptRole categorizes cached prompt segments according to conversational role.
type PromptRole string

const (
	RoleSystem    PromptRole = "system"
	RoleUser      PromptRole = "user"
	RoleAssistant PromptRole = "assistant"
)

// IsValid checks if the prompt role is one of the recognized roles.
func (r PromptRole) IsValid() bool {
	switch r {
	case RoleSystem, RoleUser, RoleAssistant:
		return true
	default:
		return false
	}
}

// RetentionPriority returns the retention priority score (higher = retained longer).
// RoleSystem (3, retain longest) > RoleUser (2) > RoleAssistant (1, evict first).
func (r PromptRole) RetentionPriority() int {
	switch r {
	case RoleSystem:
		return 3
	case RoleUser:
		return 2
	case RoleAssistant:
		return 1
	default:
		return 0
	}
}

// EvictionPriority returns the eviction order rank (lower = evicted earlier).
// RoleAssistant (1, evict first) -> RoleUser (2, evict next) -> RoleSystem (3, evict last).
func (r PromptRole) EvictionPriority() int {
	switch r {
	case RoleAssistant:
		return 1
	case RoleUser:
		return 2
	case RoleSystem:
		return 3
	default:
		return 1
	}
}

// CacheBuffer represents an allocated KV cache buffer slab that supports in-place
// zero-allocation logical sequence trimming.
type CacheBuffer struct {
	mu          sync.RWMutex
	SequenceLen int    `json:"sequence_len"`
	Capacity    int    `json:"capacity"`
	BytesPerTok int    `json:"bytes_per_tok"`
	Data        []byte `json:"-"`
	Allocated   bool   `json:"allocated"`
	TrimCount   int    `json:"trim_count"`
}

// NewCacheBuffer creates a new CacheBuffer pre-allocated to capacity tokens.
func NewCacheBuffer(seqLen int, capacity int, bytesPerTok int) *CacheBuffer {
	if capacity < seqLen {
		capacity = seqLen
	}
	if bytesPerTok <= 0 {
		bytesPerTok = 2 // default fp16
	}
	totalBytes := capacity * bytesPerTok
	data := make([]byte, totalBytes)
	return &CacheBuffer{
		SequenceLen: seqLen,
		Capacity:    capacity,
		BytesPerTok: bytesPerTok,
		Data:        data,
		Allocated:   true,
		TrimCount:   0,
	}
}

// Trim adjusts logical sequence bounds in-place without allocating new memory.
func (b *CacheBuffer) Trim(targetLen int) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if targetLen < 0 || targetLen > b.SequenceLen {
		return ErrTokenOutOfBounds
	}
	b.SequenceLen = targetLen
	b.TrimCount++
	return nil
}

// Len returns the current logical sequence length in tokens.
func (b *CacheBuffer) Len() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.SequenceLen
}

// Cap returns the total allocated capacity in tokens.
func (b *CacheBuffer) Cap() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.Capacity
}

// ByteSize returns the byte footprint of the active sequence length.
func (b *CacheBuffer) ByteSize() int64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return int64(b.SequenceLen * b.BytesPerTok)
}

// PromptEntry represents a cached prompt sequence with associated role, tokens, and KV buffer.
type PromptEntry struct {
	ID          string       `json:"id"`
	Role        PromptRole   `json:"role"`
	Tokens      []int        `json:"tokens"`
	Buffer      *CacheBuffer `json:"buffer,omitempty"`
	CreatedAt   time.Time    `json:"created_at"`
	LastAccess  time.Time    `json:"last_access"`
	AccessCount int64        `json:"access_count"`
	node        *TrieNode    `json:"-"`
}

// Len returns the number of tokens in the prompt entry.
func (e *PromptEntry) Len() int {
	return len(e.Tokens)
}

// ByteSize returns the estimated or actual byte size of the cached prompt.
func (e *PromptEntry) ByteSize() int64 {
	if e.Buffer != nil {
		return e.Buffer.ByteSize()
	}
	return int64(len(e.Tokens) * 2)
}

// Trim adjusts the entry's token sequence and underlying buffer in-place to targetLen.
func (e *PromptEntry) Trim(targetLen int) error {
	if targetLen < 0 || targetLen > len(e.Tokens) {
		return ErrTokenOutOfBounds
	}
	e.Tokens = e.Tokens[:targetLen]
	if e.Buffer != nil {
		if err := e.Buffer.Trim(targetLen); err != nil {
			return err
		}
	}
	e.LastAccess = time.Now()
	return nil
}

// TieredCacheConfig configures capacity and eviction limits for TieredPromptCache.
type TieredCacheConfig struct {
	MaxTokens   int   `json:"max_tokens"`  // 0 = unbounded
	MaxEntries  int   `json:"max_entries"` // 0 = unbounded
	MaxBytes    int64 `json:"max_bytes"`   // 0 = unbounded
	BytesPerTok int   `json:"bytes_per_tok"`
}

// TieredCacheStats reports operational metrics for TieredPromptCache.
type TieredCacheStats struct {
	TotalEntries     int   `json:"total_entries"`
	TotalTokens      int   `json:"total_tokens"`
	TotalBytes       int64 `json:"total_bytes"`
	SystemEntries    int   `json:"system_entries"`
	UserEntries      int   `json:"user_entries"`
	AssistantEntries int   `json:"assistant_entries"`
	SystemTokens     int   `json:"system_tokens"`
	UserTokens       int   `json:"user_tokens"`
	AssistantTokens  int   `json:"assistant_tokens"`
	Hits             int64 `json:"hits"`
	Misses           int64 `json:"misses"`
	TrimHits         int64 `json:"trim_hits"`
	Evictions        int64 `json:"evictions"`
}

// TieredPromptCache implements role-tiered prompt cache eviction with trimmable prefix reuse.
// Eviction hierarchy: assistant (evict first) -> user (evict next) -> system (retain longest).
type TieredPromptCache struct {
	mu           sync.RWMutex
	trie         *PromptTrie
	config       TieredCacheConfig
	entries      map[string]*PromptEntry
	assistantLRU []string // IDs of assistant entries, oldest first
	userLRU      []string // IDs of user entries, oldest first
	systemLRU    []string // IDs of system entries, oldest first
	totalTokens  int
	totalBytes   int64
	hits         int64
	misses       int64
	evictions    int64
	trimHits     int64
	nextID       uint64
}

// NewTieredPromptCache constructs a new TieredPromptCache with the specified configuration.
func NewTieredPromptCache(cfg TieredCacheConfig) *TieredPromptCache {
	if cfg.BytesPerTok <= 0 {
		cfg.BytesPerTok = 2
	}
	return &TieredPromptCache{
		trie:         NewPromptTrie(),
		config:       cfg,
		entries:      make(map[string]*PromptEntry),
		assistantLRU: make([]string, 0),
		userLRU:      make([]string, 0),
		systemLRU:    make([]string, 0),
	}
}

// NewTieredPromptCacheWithMaxTokens constructs a cache constrained by a maximum token limit.
func NewTieredPromptCacheWithMaxTokens(maxTokens int) *TieredPromptCache {
	return NewTieredPromptCache(TieredCacheConfig{
		MaxTokens:   maxTokens,
		BytesPerTok: 2,
	})
}

// DefaultTieredPromptCache constructs a default-configured TieredPromptCache.
func DefaultTieredPromptCache() *TieredPromptCache {
	return NewTieredPromptCache(TieredCacheConfig{
		MaxTokens:   32768,
		BytesPerTok: 2,
	})
}

// TrimCache adjusts logical sequence bounds of a token slice without allocating new memory.
func TrimCache(tokens []int, targetLen int) ([]int, error) {
	if targetLen < 0 || targetLen > len(tokens) {
		return nil, ErrTokenOutOfBounds
	}
	return tokens[:targetLen], nil
}

// removeFromList removes an ID from an LRU slice.
func removeFromList(list []string, target string) []string {
	for i, id := range list {
		if id == target {
			return append(list[:i], list[i+1:]...)
		}
	}
	return list
}

// touchEntryLocked marks an entry as recently accessed and updates its tier LRU position.
func (c *TieredPromptCache) touchEntryLocked(entry *PromptEntry) {
	entry.LastAccess = time.Now()
	entry.AccessCount++

	switch entry.Role {
	case RoleAssistant:
		c.assistantLRU = append(removeFromList(c.assistantLRU, entry.ID), entry.ID)
	case RoleUser:
		c.userLRU = append(removeFromList(c.userLRU, entry.ID), entry.ID)
	case RoleSystem:
		c.systemLRU = append(removeFromList(c.systemLRU, entry.ID), entry.ID)
	}
}

// shouldEvict checks whether adding additional tokens/bytes triggers capacity limits.
func (c *TieredPromptCache) shouldEvict(additionalTokens int, additionalBytes int64) bool {
	if c.config.MaxTokens > 0 && c.totalTokens+additionalTokens > c.config.MaxTokens {
		return true
	}
	if c.config.MaxEntries > 0 && len(c.entries)+1 > c.config.MaxEntries {
		return true
	}
	if c.config.MaxBytes > 0 && c.totalBytes+additionalBytes > c.config.MaxBytes {
		return true
	}
	return false
}

// evictOne removes a single entry according to the tiered eviction hierarchy:
// assistant (evict first) -> user (evict next) -> system (retain longest).
// Within each tier, the least-recently-used (LRU) entry is evicted.
func (c *TieredPromptCache) evictOne() bool {
	var targetID string

	if len(c.assistantLRU) > 0 {
		targetID = c.assistantLRU[0]
	} else if len(c.userLRU) > 0 {
		targetID = c.userLRU[0]
	} else if len(c.systemLRU) > 0 {
		targetID = c.systemLRU[0]
	} else {
		return false
	}

	entry, found := c.entries[targetID]
	if !found {
		// Clean stale ID
		c.assistantLRU = removeFromList(c.assistantLRU, targetID)
		c.userLRU = removeFromList(c.userLRU, targetID)
		c.systemLRU = removeFromList(c.systemLRU, targetID)
		return false
	}

	c.deleteEntryLocked(entry)
	c.evictions++
	return true
}

// deleteEntryLocked removes an entry from maps, LRU queues, and prefix trie.
func (c *TieredPromptCache) deleteEntryLocked(entry *PromptEntry) {
	c.trie.Delete(entry.Tokens)
	delete(c.entries, entry.ID)
	c.totalTokens -= len(entry.Tokens)
	c.totalBytes -= entry.ByteSize()

	switch entry.Role {
	case RoleAssistant:
		c.assistantLRU = removeFromList(c.assistantLRU, entry.ID)
	case RoleUser:
		c.userLRU = removeFromList(c.userLRU, entry.ID)
	case RoleSystem:
		c.systemLRU = removeFromList(c.systemLRU, entry.ID)
	}
}

// evictUnderPressure evicts entries until capacity constraints are met.
func (c *TieredPromptCache) evictUnderPressure(additionalTokens int, additionalBytes int64) int {
	evictedCount := 0
	for c.shouldEvict(additionalTokens, additionalBytes) && len(c.entries) > 0 {
		if c.evictOne() {
			evictedCount++
		} else {
			break
		}
	}
	return evictedCount
}

// EvictUnderPressure explicitly forces eviction until requiredTokens can be accommodated.
func (c *TieredPromptCache) EvictUnderPressure(requiredTokens int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.evictUnderPressure(requiredTokens, int64(requiredTokens*c.config.BytesPerTok))
}

// Put inserts a prompt sequence into the cache with an auto-generated ID.
func (c *TieredPromptCache) Put(role PromptRole, tokens []int, buffer *CacheBuffer) (*PromptEntry, error) {
	return c.PutWithID("", role, tokens, buffer)
}

// PutWithID inserts a prompt sequence into the cache with a specified ID.
func (c *TieredPromptCache) PutWithID(id string, role PromptRole, tokens []int, buffer *CacheBuffer) (*PromptEntry, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(tokens) == 0 {
		return nil, errors.New("ctxmmu: cannot cache empty prompt tokens")
	}
	if !role.IsValid() {
		role = RoleAssistant
	}

	if id == "" {
		c.nextID++
		id = fmt.Sprintf("prompt-%d", c.nextID)
	}

	tokenCount := len(tokens)
	var byteSize int64
	if buffer != nil {
		byteSize = buffer.ByteSize()
	} else {
		byteSize = int64(tokenCount * c.config.BytesPerTok)
	}

	// Evict entries under pressure prior to insertion
	c.evictUnderPressure(tokenCount, byteSize)

	// If entry with same ID already exists, replace it
	if existing, found := c.entries[id]; found {
		c.deleteEntryLocked(existing)
	}

	tokensCopy := make([]int, len(tokens))
	copy(tokensCopy, tokens)

	entry := &PromptEntry{
		ID:          id,
		Role:        role,
		Tokens:      tokensCopy,
		Buffer:      buffer,
		CreatedAt:   time.Now(),
		LastAccess:  time.Now(),
		AccessCount: 1,
	}

	c.entries[id] = entry
	c.trie.Insert(entry.Tokens, entry)
	c.totalTokens += tokenCount
	c.totalBytes += byteSize

	switch role {
	case RoleAssistant:
		c.assistantLRU = append(c.assistantLRU, id)
	case RoleUser:
		c.userLRU = append(c.userLRU, id)
	case RoleSystem:
		c.systemLRU = append(c.systemLRU, id)
	}

	return entry, nil
}

// Get performs an exact match lookup for the given token sequence.
func (c *TieredPromptCache) Get(tokens []int) (*PromptEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry := c.trie.ExactMatch(tokens)
	if entry != nil {
		c.touchEntryLocked(entry)
		c.hits++
		return entry, true
	}
	c.misses++
	return nil, false
}

// GetByID looks up a cached entry by its identifier and updates its access recency.
func (c *TieredPromptCache) GetByID(id string) (*PromptEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, found := c.entries[id]
	if found {
		c.touchEntryLocked(entry)
		c.hits++
		return entry, true
	}
	c.misses++
	return nil, false
}

// PeekByID looks up a cached entry by ID without modifying its LRU access recency or count.
func (c *TieredPromptCache) PeekByID(id string) (*PromptEntry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, found := c.entries[id]
	return entry, found
}

// ContainsID reports whether an entry with the given ID exists in the cache without updating recency.
func (c *TieredPromptCache) ContainsID(id string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	_, found := c.entries[id]
	return found
}

// GetLongestPrefix searches for the longest cached prefix matching the given tokens.
func (c *TieredPromptCache) GetLongestPrefix(tokens []int) (*PromptEntry, int, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, matchLen := c.trie.LongestPrefix(tokens)
	if entry != nil && matchLen > 0 {
		c.touchEntryLocked(entry)
		c.hits++
		return entry, matchLen, true
	}
	c.misses++
	return nil, 0, false
}

// LookupTrimmable searches for an exact match, longest prefix match, or trimmable prefix match.
// If a trimmable match is found (trimLen > 0), the caller can reuse the first matchedLen tokens.
func (c *TieredPromptCache) LookupTrimmable(tokens []int) (entry *PromptEntry, matchedLen int, trimLen int, ok bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, matchedLen, trimLen, ok = c.trie.FindTrimmablePrefix(tokens)
	if ok && entry != nil {
		c.touchEntryLocked(entry)
		if trimLen > 0 {
			c.trimHits++
		} else {
			c.hits++
		}
		return entry, matchedLen, trimLen, true
	}
	c.misses++
	return nil, 0, 0, false
}

// TrimCache trims an existing cached sequence matching tokens down to targetLen in-place,
// updating its logical sequence bounds and trie index without allocating new memory.
func (c *TieredPromptCache) TrimCache(tokens []int, targetLen int) (*PromptEntry, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if targetLen < 0 {
		return nil, ErrTokenOutOfBounds
	}

	// Find the entry. Try exact match first.
	entry := c.trie.ExactMatch(tokens)
	if entry == nil {
		// Fall back to longest trimmable prefix match
		var ok bool
		entry, _, _, ok = c.trie.FindTrimmablePrefix(tokens)
		if !ok || entry == nil {
			return nil, ErrPromptNotFound
		}
	}

	if targetLen > len(entry.Tokens) {
		return nil, ErrTokenOutOfBounds
	}

	return c.trimEntryLocked(entry, targetLen)
}

// TrimEntry trims a specific prompt entry to targetLen in-place.
func (c *TieredPromptCache) TrimEntry(entry *PromptEntry, targetLen int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if entry == nil {
		return ErrPromptNotFound
	}
	if targetLen < 0 || targetLen > len(entry.Tokens) {
		return ErrTokenOutOfBounds
	}

	_, err := c.trimEntryLocked(entry, targetLen)
	return err
}

func (c *TieredPromptCache) trimEntryLocked(entry *PromptEntry, targetLen int) (*PromptEntry, error) {
	if targetLen == len(entry.Tokens) {
		return entry, nil
	}

	oldTokens := entry.Tokens
	deltaTokens := len(oldTokens) - targetLen
	oldBytes := entry.ByteSize()

	// Re-index in trie
	c.trie.Delete(oldTokens)
	if err := entry.Trim(targetLen); err != nil {
		// Attempt rollback into trie
		c.trie.Insert(oldTokens, entry)
		return nil, err
	}
	c.trie.Insert(entry.Tokens, entry)

	// Update accounting
	c.totalTokens -= deltaTokens
	newBytes := entry.ByteSize()
	c.totalBytes -= (oldBytes - newBytes)

	c.touchEntryLocked(entry)
	return entry, nil
}

// Delete removes an entry matching the token sequence from the cache.
func (c *TieredPromptCache) Delete(tokens []int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry := c.trie.ExactMatch(tokens)
	if entry == nil {
		return false
	}
	c.deleteEntryLocked(entry)
	return true
}

// DeleteByID removes an entry by ID from the cache.
func (c *TieredPromptCache) DeleteByID(id string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, found := c.entries[id]
	if !found {
		return false
	}
	c.deleteEntryLocked(entry)
	return true
}

// ContainsRole reports whether the cache currently holds any entries with the given role.
func (c *TieredPromptCache) ContainsRole(role PromptRole) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	switch role {
	case RoleAssistant:
		return len(c.assistantLRU) > 0
	case RoleUser:
		return len(c.userLRU) > 0
	case RoleSystem:
		return len(c.systemLRU) > 0
	default:
		return false
	}
}

// CountByRole returns the number of cached entries for the given role.
func (c *TieredPromptCache) CountByRole(role PromptRole) int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	switch role {
	case RoleAssistant:
		return len(c.assistantLRU)
	case RoleUser:
		return len(c.userLRU)
	case RoleSystem:
		return len(c.systemLRU)
	default:
		return 0
	}
}

// TokensByRole returns the total number of cached tokens for the given role.
func (c *TieredPromptCache) TokensByRole(role PromptRole) int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := 0
	for _, entry := range c.entries {
		if entry.Role == role {
			total += len(entry.Tokens)
		}
	}
	return total
}

// Len returns the current number of cached prompt entries.
func (c *TieredPromptCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// TotalTokens returns the total number of tokens currently cached.
func (c *TieredPromptCache) TotalTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalTokens
}

// TotalBytes returns the total memory bytes used by cached entries.
func (c *TieredPromptCache) TotalBytes() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalBytes
}

// Stats returns a snapshot of cache performance metrics and current occupancy.
func (c *TieredPromptCache) Stats() TieredCacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := TieredCacheStats{
		TotalEntries: len(c.entries),
		TotalTokens:  c.totalTokens,
		TotalBytes:   c.totalBytes,
		Hits:         c.hits,
		Misses:       c.misses,
		TrimHits:     c.trimHits,
		Evictions:    c.evictions,
	}

	for _, entry := range c.entries {
		switch entry.Role {
		case RoleSystem:
			stats.SystemEntries++
			stats.SystemTokens += len(entry.Tokens)
		case RoleUser:
			stats.UserEntries++
			stats.UserTokens += len(entry.Tokens)
		case RoleAssistant:
			stats.AssistantEntries++
			stats.AssistantTokens += len(entry.Tokens)
		}
	}

	return stats
}

// Clear purges all entries, resets the trie, and zeros all token and byte totals.
func (c *TieredPromptCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.trie = NewPromptTrie()
	c.entries = make(map[string]*PromptEntry)
	c.assistantLRU = make([]string, 0)
	c.userLRU = make([]string, 0)
	c.systemLRU = make([]string, 0)
	c.totalTokens = 0
	c.totalBytes = 0
}
