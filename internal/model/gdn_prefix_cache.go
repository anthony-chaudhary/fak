package model

import (
	"container/list"
	"errors"
	"fmt"
	"sync"
)

// DefaultGDNPrefixCacheCapacity matches the 4-session capacity from carloslfu/slotstream.
const DefaultGDNPrefixCacheCapacity = 4

var (
	// ErrBackwardRewind is returned when Put attempts to rewind or shrink the prefix length
	// for an irreversible linear recurrence state.
	ErrBackwardRewind = errors.New("gdn prefix cache: backward rewind refused by extend-only invariant")

	// ErrPrefixDivergence is returned when Put attempts to update state for a divergent token prefix.
	ErrPrefixDivergence = errors.New("gdn prefix cache: prefix divergence refused by extend-only invariant")

	// ErrEmptySessionID is returned when sessionID is empty.
	ErrEmptySessionID = errors.New("gdn prefix cache: session ID cannot be empty")

	// ErrEmptyTokens is returned when tokenIDs is empty.
	ErrEmptyTokens = errors.New("gdn prefix cache: token prefix cannot be empty")
)

// GDNLayerState holds the linear recurrent matrix state and 1D convolution tail buffer
// for a single Gated DeltaNet linear attention layer.
type GDNLayerState struct {
	Layer     int       `json:"layer"`
	Conv      []float32 `json:"conv,omitempty"`
	Recurrent []float32 `json:"recurrent,omitempty"`
}

// Clone returns an independent deep copy of the layer state.
func (l GDNLayerState) Clone() GDNLayerState {
	var conv, rec []float32
	if len(l.Conv) > 0 {
		conv = append([]float32(nil), l.Conv...)
	}
	if len(l.Recurrent) > 0 {
		rec = append([]float32(nil), l.Recurrent...)
	}
	return GDNLayerState{
		Layer:     l.Layer,
		Conv:      conv,
		Recurrent: rec,
	}
}

// StateSnapshot captures the complete multi-layer GDN recurrent and convolution state
// corresponding to an irreversible linear recurrence fold over a token prefix.
type StateSnapshot struct {
	SessionID string          `json:"session_id,omitempty"`
	TokenIDs  []int           `json:"token_ids,omitempty"`
	Layers    []GDNLayerState `json:"layers,omitempty"`
}

// Clone returns an atomic, independent deep copy of the snapshot to prevent reference leaks.
func (s StateSnapshot) Clone() StateSnapshot {
	var tokenIDs []int
	if len(s.TokenIDs) > 0 {
		tokenIDs = append([]int(nil), s.TokenIDs...)
	}
	var layers []GDNLayerState
	if len(s.Layers) > 0 {
		layers = make([]GDNLayerState, len(s.Layers))
		for i, l := range s.Layers {
			layers[i] = l.Clone()
		}
	}
	return StateSnapshot{
		SessionID: s.SessionID,
		TokenIDs:  tokenIDs,
		Layers:    layers,
	}
}

// Layer returns the state for the given layer index if present.
func (s StateSnapshot) Layer(layer int) (GDNLayerState, bool) {
	for _, l := range s.Layers {
		if l.Layer == layer {
			return l, true
		}
	}
	return GDNLayerState{}, false
}

// LayerCount returns the number of captured layers.
func (s StateSnapshot) LayerCount() int {
	return len(s.Layers)
}

// TotalBytes returns the total memory footprint in bytes across all layers and token IDs.
func (s StateSnapshot) TotalBytes() int64 {
	var bytes int64
	bytes += int64(len(s.TokenIDs) * 8)
	for _, l := range s.Layers {
		bytes += int64(len(l.Conv)*4 + len(l.Recurrent)*4)
	}
	return bytes
}

// FromQwen35GDNLayerSnapshots constructs a StateSnapshot from native Metal layer snapshots.
func FromQwen35GDNLayerSnapshots(sessionID string, tokenIDs []int, snaps []qwen35GDNLayerSnapshot) StateSnapshot {
	layers := make([]GDNLayerState, len(snaps))
	for i, s := range snaps {
		layers[i] = GDNLayerState{
			Layer:     s.layer,
			Conv:      append([]float32(nil), s.conv...),
			Recurrent: append([]float32(nil), s.recurrent...),
		}
	}
	var tok []int
	if len(tokenIDs) > 0 {
		tok = append([]int(nil), tokenIDs...)
	}
	return StateSnapshot{
		SessionID: sessionID,
		TokenIDs:  tok,
		Layers:    layers,
	}
}

// ToQwen35GDNLayerSnapshots converts StateSnapshot layers to native Metal layer snapshots.
func (s StateSnapshot) ToQwen35GDNLayerSnapshots() []qwen35GDNLayerSnapshot {
	out := make([]qwen35GDNLayerSnapshot, len(s.Layers))
	for i, l := range s.Layers {
		out[i] = qwen35GDNLayerSnapshot{
			layer:     l.Layer,
			conv:      append([]float32(nil), l.Conv...),
			recurrent: append([]float32(nil), l.Recurrent...),
		}
	}
	return out
}

// GDNPrefixCacheStats records telemetry for cache lookups, mutations, and invariant checks.
type GDNPrefixCacheStats struct {
	Hits      int64 `json:"hits"`
	Misses    int64 `json:"misses"`
	Puts      int64 `json:"puts"`
	Evictions int64 `json:"evictions"`
	Refusals  int64 `json:"refusals"`
}

type gdnCacheEntry struct {
	sessionID string
	tokenIDs  []int
	state     StateSnapshot
}

// GDNPrefixCache implements an extend-only prefix cache for Gated DeltaNet (GDN)
// linear recurrent states across multi-turn agent sessions.
//
// In Gated DeltaNet architectures (such as Qwen3.8), linear recurrence layers
// compute an accumulated state matrix that represents an irreversible fold over
// past tokens. Unlike softmax KV caches where tokens can be arbitrarily sliced or
// deleted with RoPE re-rotation, a linear recurrence cannot be rewound or partially
// spliced without full recomputation.
//
// GDNPrefixCache enforces:
//  1. Strict prefix matching: prompt.starts(with: heldIds).
//  2. Extend-only invariant: refuses backward rewinds or divergence on irreversible folds.
//  3. Bounded session retention: LRU eviction preserving up to N conversation sessions.
type GDNPrefixCache struct {
	mu       sync.RWMutex
	capacity int
	sessions map[string]*list.Element
	lru      *list.List
	stats    GDNPrefixCacheStats
}

// NewGDNPrefixCache creates an extend-only GDN prefix cache retaining up to capacity sessions.
// If capacity <= 0, DefaultGDNPrefixCacheCapacity (4) is used.
func NewGDNPrefixCache(capacity int) *GDNPrefixCache {
	if capacity <= 0 {
		capacity = DefaultGDNPrefixCacheCapacity
	}
	return &GDNPrefixCache{
		capacity: capacity,
		sessions: make(map[string]*list.Element),
		lru:      list.New(),
	}
}

// Capacity returns the maximum number of conversation sessions retained in the cache.
func (c *GDNPrefixCache) Capacity() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.capacity
}

// Len returns the current number of cached conversation sessions.
func (c *GDNPrefixCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.sessions)
}

// Stats returns a copy of cache telemetry.
func (c *GDNPrefixCache) Stats() GDNPrefixCacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.stats
}

// HeldTokens returns a copy of the held token sequence for sessionID if cached.
func (c *GDNPrefixCache) HeldTokens(sessionID string) ([]int, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	elem, ok := c.sessions[sessionID]
	if !ok {
		return nil, false
	}
	entry := elem.Value.(*gdnCacheEntry)
	return append([]int(nil), entry.tokenIDs...), true
}

// TotalBytes returns the total memory footprint in bytes across all cached sessions.
func (c *GDNPrefixCache) TotalBytes() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var total int64
	for _, elem := range c.sessions {
		total += elem.Value.(*gdnCacheEntry).state.TotalBytes()
	}
	return total
}

// Clear removes all sessions from the cache.
func (c *GDNPrefixCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.sessions = make(map[string]*list.Element)
	c.lru.Init()
}

// Get looks up a cached GDN state snapshot for sessionID that is a strict prefix of tokenIDs.
//
// Invariant check:
//   - Returns (snapshot, true) if and only if prompt starts with heldIds
//     (len(tokenIDs) >= len(held) && tokenIDs[:len(held)] == held).
//   - Returns (StateSnapshot{}, false) on miss, session not found, backward rewind
//     (len(tokenIDs) < len(held)), or divergent prefix.
//
// On a hit, the session is promoted to most-recently-used in the LRU order, and an
// independent deep copy of StateSnapshot is returned.
func (c *GDNPrefixCache) Get(sessionID string, tokenIDs []int) (StateSnapshot, bool) {
	if sessionID == "" || len(tokenIDs) == 0 {
		c.mu.Lock()
		c.stats.Misses++
		c.mu.Unlock()
		return StateSnapshot{}, false
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.sessions[sessionID]
	if !ok {
		c.stats.Misses++
		return StateSnapshot{}, false
	}

	entry := elem.Value.(*gdnCacheEntry)
	held := entry.tokenIDs

	// Prompt cannot be shorter than the irreversible linear recurrence fold.
	if len(tokenIDs) < len(held) {
		c.stats.Misses++
		return StateSnapshot{}, false
	}

	// Strict prefix matching: prompt must match all held tokens exactly.
	for i, tok := range held {
		if tokenIDs[i] != tok {
			c.stats.Misses++
			return StateSnapshot{}, false
		}
	}

	c.lru.MoveToFront(elem)
	c.stats.Hits++
	return entry.state.Clone(), true
}

// Put stores or extends the GDN recurrent state snapshot for sessionID.
//
// Invariant check:
//   - If sessionID already exists, tokenIDs must be an extension of the held prefix:
//     1. len(tokenIDs) >= len(held) (backward rewinds refused with ErrBackwardRewind).
//     2. tokenIDs[:len(held)] == held (divergence refused with ErrPrefixDivergence).
//   - If sessionID is new, it is inserted at the front of the LRU. If capacity is exceeded,
//     the least-recently-used session is evicted.
//
// Returns nil on success, or a typed error if the extend-only invariant is violated.
func (c *GDNPrefixCache) Put(sessionID string, tokenIDs []int, state StateSnapshot) error {
	if sessionID == "" {
		c.mu.Lock()
		c.stats.Refusals++
		c.mu.Unlock()
		return ErrEmptySessionID
	}
	if len(tokenIDs) == 0 {
		c.mu.Lock()
		c.stats.Refusals++
		c.mu.Unlock()
		return ErrEmptyTokens
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	clonedState := state.Clone()
	clonedState.SessionID = sessionID
	clonedState.TokenIDs = append([]int(nil), tokenIDs...)

	if elem, ok := c.sessions[sessionID]; ok {
		entry := elem.Value.(*gdnCacheEntry)
		held := entry.tokenIDs

		// Enforce extend-only: refuse backward rewinds on irreversible linear recurrence.
		if len(tokenIDs) < len(held) {
			c.stats.Refusals++
			return fmt.Errorf("%w: cannot rewind session %q from %d tokens to %d tokens",
				ErrBackwardRewind, sessionID, len(held), len(tokenIDs))
		}

		// Enforce extend-only: refuse prefix divergence on irreversible linear recurrence.
		for i, tok := range held {
			if tokenIDs[i] != tok {
				c.stats.Refusals++
				return fmt.Errorf("%w: token mismatch in session %q at position %d (held %d, got %d)",
					ErrPrefixDivergence, sessionID, i, tok, tokenIDs[i])
			}
		}

		// Valid extension or state update at current prefix length.
		entry.tokenIDs = append([]int(nil), tokenIDs...)
		entry.state = clonedState
		c.lru.MoveToFront(elem)
		c.stats.Puts++
		return nil
	}

	// New session: enforce bounded capacity via LRU eviction.
	if c.capacity > 0 && c.lru.Len() >= c.capacity {
		oldest := c.lru.Back()
		if oldest != nil {
			oldEntry := oldest.Value.(*gdnCacheEntry)
			delete(c.sessions, oldEntry.sessionID)
			c.lru.Remove(oldest)
			c.stats.Evictions++
		}
	}

	newEntry := &gdnCacheEntry{
		sessionID: sessionID,
		tokenIDs:  append([]int(nil), tokenIDs...),
		state:     clonedState,
	}
	elem := c.lru.PushFront(newEntry)
	c.sessions[sessionID] = elem
	c.stats.Puts++
	return nil
}

// Evict explicitly removes the sessionID from the cache, releasing its state.
// This is called when a session resets, forks, or completes.
func (c *GDNPrefixCache) Evict(sessionID string) {
	if sessionID == "" {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.sessions[sessionID]; ok {
		delete(c.sessions, sessionID)
		c.lru.Remove(elem)
		c.stats.Evictions++
	}
}
