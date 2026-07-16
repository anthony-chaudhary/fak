package agent

import (
	"container/list"
	"crypto/sha256"
	"encoding/json"
	"sync"
)

const (
	schemaCacheKeyGemini = "gemini"
	schemaCacheKeyOpenAI = "openai"
)

// normalizedSchemaCacheCap bounds the process-global memoization table below. A single
// client with a stable tool set collapses to ~15-30 keys, but `fak serve` is a gateway
// fronting arbitrary heterogeneous clients / MCP tool sets, and the key is
// sha256(raw pre-normalization bytes): distinct tools, dynamically-generated defs, even
// whitespace/key-order variants each mint a permanent key. Without a bound that grows for
// the multi-day gateway lifetime (#3297). The cap is far above any realistic concurrent
// tool-set diversity, so the hot path never evicts a live client's entries; when it is
// reached the least-recently-used key is dropped. Eviction is always correctness-safe —
// this is pure memoization and a miss simply recomputes via the Compute path.
const normalizedSchemaCacheCap = 2048

type normalizedSchemaCacheKey struct {
	provider string
	root     bool
	sum      [sha256.Size]byte
}

// normalizedSchemaLRU is a size-capped, content-addressed memoization cache: a
// mutex-guarded map/recency-list pair, the same bounded-LRU idiom session.Table uses one
// rung over. The zero value is ready to use (lazily initialized under the lock).
type normalizedSchemaLRU struct {
	mu    sync.Mutex
	byKey map[normalizedSchemaCacheKey]*list.Element
	lru   *list.List
	cap   int
}

type normalizedSchemaEntry struct {
	key normalizedSchemaCacheKey
	val []byte
}

var normalizedSchemaCache normalizedSchemaLRU

func (c *normalizedSchemaLRU) ensureLocked() {
	if c.cap <= 0 {
		c.cap = normalizedSchemaCacheCap
	}
	if c.byKey == nil {
		c.byKey = map[normalizedSchemaCacheKey]*list.Element{}
	}
	if c.lru == nil {
		c.lru = list.New()
	}
}

func (c *normalizedSchemaLRU) load(key normalizedSchemaCacheKey) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ensureLocked()
	el, ok := c.byKey[key]
	if !ok {
		return nil, false
	}
	c.lru.MoveToFront(el)
	return el.Value.(*normalizedSchemaEntry).val, true
}

func (c *normalizedSchemaLRU) store(key normalizedSchemaCacheKey, val []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ensureLocked()
	if el, ok := c.byKey[key]; ok {
		el.Value.(*normalizedSchemaEntry).val = val
		c.lru.MoveToFront(el)
		return
	}
	c.byKey[key] = c.lru.PushFront(&normalizedSchemaEntry{key: key, val: val})
	for len(c.byKey) > c.cap {
		back := c.lru.Back()
		if back == nil {
			return
		}
		ent := back.Value.(*normalizedSchemaEntry)
		c.lru.Remove(back)
		delete(c.byKey, ent.key)
	}
}

// len reports the number of retained entries — the observable footprint that keeps the
// bound checkable (leakcheck.BoundedSize) and a growth metric visible.
func (c *normalizedSchemaLRU) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.byKey)
}

func loadNormalizedSchema(provider string, root bool, raw json.RawMessage) (json.RawMessage, bool) {
	b, ok := normalizedSchemaCache.load(normalizedSchemaKey(provider, root, raw))
	if !ok {
		return nil, false
	}
	out := make([]byte, len(b))
	copy(out, b)
	return json.RawMessage(out), true
}

func storeNormalizedSchema(provider string, root bool, raw, normalized json.RawMessage) {
	b := make([]byte, len(normalized))
	copy(b, normalized)
	normalizedSchemaCache.store(normalizedSchemaKey(provider, root, raw), b)
}

// normalizedSchemaCacheLen exposes the live entry count for observability and the
// bounded-growth regression proof.
func normalizedSchemaCacheLen() int { return normalizedSchemaCache.len() }

func normalizedSchemaKey(provider string, root bool, raw json.RawMessage) normalizedSchemaCacheKey {
	return normalizedSchemaCacheKey{
		provider: provider,
		root:     root,
		sum:      sha256.Sum256(raw),
	}
}
