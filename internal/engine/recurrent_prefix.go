package engine

import (
	"container/list"
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// DefaultRecurrentPrefixCapacity is the number of retained conversation
// boundaries, matching the 4-session capacity from carloslfu/slotstream and the
// model-layer DefaultGDNPrefixCacheCapacity.
const DefaultRecurrentPrefixCapacity = 4

var (
	// ErrRecurrentPrefixEmptyKey is returned when a cache key is empty.
	ErrRecurrentPrefixEmptyKey = errors.New("recurrent prefix cache: key cannot be empty")

	// ErrRecurrentPrefixEmptyTokens is returned when a token prefix is empty, or
	// when the caller supplies no snapshot to own.
	ErrRecurrentPrefixEmptyTokens = errors.New("recurrent prefix cache: token prefix cannot be empty")

	// ErrRecurrentPrefixBackwardRewind refuses shrinking the held prefix on an
	// irreversible linear-recurrence fold.
	ErrRecurrentPrefixBackwardRewind = errors.New("recurrent prefix cache: backward rewind refused by extend-only invariant")

	// ErrRecurrentPrefixDivergence refuses updating a prefix whose held tokens
	// do not match on an irreversible linear-recurrence fold.
	ErrRecurrentPrefixDivergence = errors.New("recurrent prefix cache: prefix divergence refused by extend-only invariant")
)

// RecurrentPrefixCacheStats records telemetry for lookups, mutations, and
// invariant checks of a RecurrentPrefixCache.
type RecurrentPrefixCacheStats struct {
	Hits      int64
	Misses    int64
	Puts      int64
	Evictions int64
	Refusals  int64
}

type recurrentPrefixEntry struct {
	key    string
	tokens []int
	snap   *model.PrefixSnapshot
}

// RecurrentPrefixCache is an extend-only, LRU-bounded cache of GDN recurrent
// prefix boundaries keyed by conversation identity.
//
// Why a stored snapshot is the only safe reuse unit: a Gated DeltaNet (GDN)
// linear-attention layer folds the whole token history into an accumulated
// state matrix. Unlike a softmax KV cache, that fold is irreversible — there is
// no RoPE re-rotation that lets a sub-span be sliced out or rewound. The only
// reuse unit that is mathematically valid is therefore a *whole-prefix
// boundary* captured at the token position that produced it. model.PrefixSnapshot
// carries exactly that: the full GDN linear/recurrent state (via KVCache.Clone →
// linear.clone()) plus the softmax KV, so restoring it and prefilling only the
// new suffix produces identical greedy token streams to a full cold prefill
// (witnessed on the synthetic GDN fixture).
//
// Invariants:
//  1. Strict prefix match: a lookup hits only when prompt extends held tokens.
//  2. Extend-only: a store refuses backward rewinds or prefix divergence.
//  3. Bounded retention: at most capacity conversations, LRU-evicted.
//
// Ownership. Lookup is a *take*: on a hit the entry is removed from the cache
// and the stored *model.PrefixSnapshot pointer transfers to the caller, who
// becomes responsible for restoring and/or closing it. Store is a *put*: the
// cache takes ownership of the supplied snapshot. A refused Store closes the
// snapshot so a rejected caller never leaks device ownership. Evict, Clear, and
// LRU eviction close the snapshots they drop, because the cache owns them until
// then.
type RecurrentPrefixCache struct {
	mu       sync.Mutex
	capacity int
	entries  map[string]*list.Element
	lru      *list.List
	stats    RecurrentPrefixCacheStats
}

// NewRecurrentPrefixCache creates an extend-only prefix cache retaining up to
// capacity conversations. If capacity <= 0, DefaultRecurrentPrefixCapacity (4)
// is used.
func NewRecurrentPrefixCache(capacity int) *RecurrentPrefixCache {
	if capacity <= 0 {
		capacity = DefaultRecurrentPrefixCapacity
	}
	return &RecurrentPrefixCache{
		capacity: capacity,
		entries:  make(map[string]*list.Element),
		lru:      list.New(),
	}
}

// Capacity returns the maximum number of conversations retained.
func (c *RecurrentPrefixCache) Capacity() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.capacity
}

// Len returns the current number of cached conversations.
func (c *RecurrentPrefixCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}

// Stats returns a copy of cache telemetry.
func (c *RecurrentPrefixCache) Stats() RecurrentPrefixCacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}

// Clear drops every conversation, closing each owned snapshot.
func (c *RecurrentPrefixCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, elem := range c.entries {
		elem.Value.(*recurrentPrefixEntry).snap.Close()
	}
	c.entries = make(map[string]*list.Element)
	c.lru.Init()
}

// Evict removes a single conversation, closing its owned snapshot.
func (c *RecurrentPrefixCache) Evict(key string) {
	if key == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if elem, ok := c.entries[key]; ok {
		entry := elem.Value.(*recurrentPrefixEntry)
		delete(c.entries, key)
		c.lru.Remove(elem)
		entry.snap.Close()
		c.stats.Evictions++
	}
}

// Lookup takes the cached snapshot for key when prompt strictly extends the
// held token prefix.
//
// It returns a hit if and only if an entry exists and
// len(prompt) > len(held) && prompt[:len(held)] == held — at least one new token
// must remain, so a pure re-issue of the exact same prompt is a miss (there is
// no suffix to prefill). On a hit the entry is removed (take semantics, no LRU
// promotion) and the *model.PrefixSnapshot pointer transfers to the caller;
// stats.Hits increments and matched is len(held). On a miss (including empty key
// or prompt) stats.Misses increments and the result is (0, nil, false).
func (c *RecurrentPrefixCache) Lookup(key string, prompt []int) (matched int, snap *model.PrefixSnapshot, hit bool) {
	if key == "" || len(prompt) == 0 {
		c.mu.Lock()
		c.stats.Misses++
		c.mu.Unlock()
		return 0, nil, false
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.entries[key]
	if !ok {
		c.stats.Misses++
		return 0, nil, false
	}
	entry := elem.Value.(*recurrentPrefixEntry)
	held := entry.tokens

	// Strict extend: the prompt must be strictly longer than the fold, so the
	// caller always has at least one token left to prefill.
	if len(prompt) <= len(held) {
		c.stats.Misses++
		return 0, nil, false
	}
	for i, tok := range held {
		if prompt[i] != tok {
			c.stats.Misses++
			return 0, nil, false
		}
	}

	// Take: remove and hand ownership of the snapshot to the caller.
	delete(c.entries, key)
	c.lru.Remove(elem)
	c.stats.Hits++
	return len(held), entry.snap, true
}

// Store puts (or extends) the recurrent boundary snapshot for key, taking
// ownership of snap.
//
// An existing key must extend the held prefix: a shorter token list is refused
// with ErrRecurrentPrefixBackwardRewind and a divergent prefix with
// ErrRecurrentPrefixDivergence. A new key enters at the front of the LRU; while
// the cache is at capacity the oldest entries are evicted (stats.Evictions).
// Empty key, empty tokens, or a nil snapshot are refused. Passing a snapshot the
// cache already owns under the same key is a no-op (already stored). On success
// stats.Puts increments; on any refusal stats.Refusals increments and snap is
// closed so the caller never leaks ownership.
func (c *RecurrentPrefixCache) Store(key string, tokens []int, snap *model.PrefixSnapshot) error {
	if key == "" {
		c.mu.Lock()
		c.stats.Refusals++
		c.mu.Unlock()
		if snap != nil {
			snap.Close()
		}
		return ErrRecurrentPrefixEmptyKey
	}
	if len(tokens) == 0 || snap == nil {
		c.mu.Lock()
		c.stats.Refusals++
		c.mu.Unlock()
		if snap != nil {
			snap.Close()
		}
		return ErrRecurrentPrefixEmptyTokens
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.entries[key]; ok {
		entry := elem.Value.(*recurrentPrefixEntry)
		held := entry.tokens

		// Idempotent same-pointer put: the cache already owns this snapshot under
		// this key, so there is nothing to extend or replace.
		if entry.snap == snap {
			return nil
		}

		// Extend-only: refuse backward rewinds on an irreversible fold.
		if len(tokens) < len(held) {
			c.stats.Refusals++
			snap.Close()
			return fmt.Errorf("%w: cannot rewind key %q from %d tokens to %d tokens",
				ErrRecurrentPrefixBackwardRewind, key, len(held), len(tokens))
		}
		// Extend-only: refuse prefix divergence on an irreversible fold.
		for i, tok := range held {
			if tokens[i] != tok {
				c.stats.Refusals++
				snap.Close()
				return fmt.Errorf("%w: token mismatch in key %q at position %d (held %d, got %d)",
					ErrRecurrentPrefixDivergence, key, i, tok, tokens[i])
			}
		}

		// Valid extension at the same or a longer prefix. The replaced snapshot
		// was owned by the cache, so close it before adopting the new owner.
		entry.snap.Close()
		entry.tokens = append([]int(nil), tokens...)
		entry.snap = snap
		c.lru.MoveToFront(elem)
		c.stats.Puts++
		return nil
	}

	// New key: LRU-evict oldest while at capacity.
	for c.capacity > 0 && len(c.entries) >= c.capacity {
		oldest := c.lru.Back()
		if oldest == nil {
			break
		}
		oldEntry := oldest.Value.(*recurrentPrefixEntry)
		delete(c.entries, oldEntry.key)
		c.lru.Remove(oldest)
		oldEntry.snap.Close()
		c.stats.Evictions++
	}

	entry := &recurrentPrefixEntry{
		key:    key,
		tokens: append([]int(nil), tokens...),
		snap:   snap,
	}
	c.entries[key] = c.lru.PushFront(entry)
	c.stats.Puts++
	return nil
}
