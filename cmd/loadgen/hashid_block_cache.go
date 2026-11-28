// Hash-id block cache for trace-replay prompt synthesis.
//
// (PromptGenerator._build_token_sequence, sample_tokens_from_corpus,
// reject_separator_only_blocks, and the per-(trace_id, hash_id) reseed),
// pinned at
// in idiomatic Go over a plain token-ID corpus; no upstream code is copied.
// Study note: fak-private docs/research/STUDY-aiperf-2026-11-09.md, candidate
// aiperf-data-11.
//
// Why the seam exists: a KV-cache trace records each request as a list of
// hash ids, one per fixed-size block of prompt content. Replaying it must
// send byte-identical content for a repeated hash id so the server's prefix
// cache hits the way it did in the capture, and must send different content
// for the same hash id in a different trace so two traces never alias each
// other's cache entries.
//
// Departure from upstream: upstream keeps an unbounded dict and reseeds a
// shared generator before each miss. Here block content is a pure function of
// (seed, trace id, hash id), so the cache can be bounded: an evicted block
// regenerates to the same tokens. Tail-pad and unhashed draws walk a per-trace
// counter instead of a shared generator, so they repeat for a given trace and
// request order.
package main

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
)

// DefaultHashIDBlockCacheEntries is the entry cap applied when
// HashIDBlockCacheConfig.MaxEntries is zero.
const DefaultHashIDBlockCacheEntries = 65536

var (
	// ErrHashIDBlockLayout marks a request whose token count, hash-id count
	// and block size fit none of the three layouts, or a cache built from
	// unusable configuration.
	ErrHashIDBlockLayout = errors.New("loadgen/hashid: incompatible block layout")
	// ErrHashIDBlockSizeConflict marks a hash id requested at a size other
	// than the one it is cached at: a corrupt trace, or a block size that
	// disagrees with the recorded blocks.
	ErrHashIDBlockSizeConflict = errors.New("loadgen/hashid: hash id requested at two block sizes")
)

// HashIDLayout names how a request's hash ids tile its token count.
type HashIDLayout int

const (
	// HashIDLayoutUnhashed has no hash ids: every token is a fresh draw.
	HashIDLayoutUnhashed HashIDLayout = iota
	// HashIDLayoutExactTile has len(hashIDs)*blockSize == numTokens: every
	// hash id is a full block.
	HashIDLayoutExactTile
	// HashIDLayoutPartialLastBlock has the hash ids overshoot the token
	// count by less than one block: the final hash id maps to a short block.
	HashIDLayoutPartialLastBlock
	// HashIDLayoutPrefixOnly has the hash ids cover only a cached prefix:
	// every hash id is a full block and the tail is padded with uncached
	// draws.
	HashIDLayoutPrefixOnly
)

func (l HashIDLayout) String() string {
	switch l {
	case HashIDLayoutUnhashed:
		return "unhashed"
	case HashIDLayoutExactTile:
		return "exact_tile"
	case HashIDLayoutPartialLastBlock:
		return "partial_last_block"
	case HashIDLayoutPrefixOnly:
		return "prefix_only"
	default:
		return fmt.Sprintf("HashIDLayout(%d)", int(l))
	}
}

// HashIDBlockCacheConfig configures a HashIDBlockCache.
type HashIDBlockCacheConfig struct {
	// Corpus is the token-ID corpus blocks are cut from. Required, non-empty.
	// The cache reads it and never writes it.
	Corpus []int
	// Seed scopes every draw; two caches with the same Seed, corpus and
	// trace id produce identical blocks in any process.
	Seed uint64
	// SeparatorToken, when non-nil, is written at index 0 of every hashed
	// block and counts toward the block's size.
	SeparatorToken *int
	// MaxEntries caps resident blocks. Zero means
	// DefaultHashIDBlockCacheEntries; negative is rejected.
	MaxEntries int
}

type hashIDBlock struct {
	id     int64
	tokens []int
}

// HashIDBlockCache maps trace hash ids to fixed token blocks cut from a
// corpus and assembles per-request token sequences from them.
//
// Capacity and eviction: at most MaxEntries blocks are resident (default
// DefaultHashIDBlockCacheEntries). Inserting past the cap evicts the least
// recently used block; a hit or an insert both count as a use. Eviction
// never changes replayed content, because a block is regenerated from
// (seed, trace id, hash id) alone. It does narrow the size-conflict check to
// resident blocks: an evicted hash id re-requested at a different size is
// regenerated at the new size instead of being refused.
//
// A cache is not safe for concurrent use.
type HashIDBlockCache struct {
	corpus     []int
	seed       uint64
	sep        *int
	maxEntries int

	traceID   string
	tailDraws uint64
	blocks    map[int64]*list.Element
	lru       *list.List // front = most recently used
}

// NewHashIDBlockCache returns an empty cache scoped to the empty trace id.
func NewHashIDBlockCache(cfg HashIDBlockCacheConfig) (*HashIDBlockCache, error) {
	if len(cfg.Corpus) == 0 {
		return nil, fmt.Errorf("%w: empty corpus", ErrHashIDBlockLayout)
	}
	if cfg.MaxEntries < 0 {
		return nil, fmt.Errorf("%w: max entries %d is negative", ErrHashIDBlockLayout, cfg.MaxEntries)
	}
	maxEntries := cfg.MaxEntries
	if maxEntries == 0 {
		maxEntries = DefaultHashIDBlockCacheEntries
	}
	c := &HashIDBlockCache{
		corpus:     cfg.Corpus,
		seed:       cfg.Seed,
		maxEntries: maxEntries,
		blocks:     make(map[int64]*list.Element),
		lru:        list.New(),
	}
	if cfg.SeparatorToken != nil {
		sep := *cfg.SeparatorToken
		c.sep = &sep
	}
	return c, nil
}

// SetTrace rescopes the cache to one trace. Resident blocks are dropped and
// the tail-pad stream restarts, so the same hash id under a different trace
// id yields different tokens and re-entering a trace reproduces its blocks.
// Trace loaders call it once per trace file.
func (c *HashIDBlockCache) SetTrace(traceID string) {
	c.traceID = traceID
	c.tailDraws = 0
	c.blocks = make(map[int64]*list.Element)
	c.lru.Init()
}

// Len reports the number of resident blocks.
func (c *HashIDBlockCache) Len() int { return c.lru.Len() }

// Cap reports the entry cap.
func (c *HashIDBlockCache) Cap() int { return c.maxEntries }

// Contains reports whether hashID is resident, without counting as a use.
func (c *HashIDBlockCache) Contains(hashID int64) bool {
	_, ok := c.blocks[hashID]
	return ok
}

// Build assembles exactly numTokens token IDs for one request. Each hash id
// names a block of blockSize tokens: a resident block is reused verbatim, a
// missing one is cut from the corpus and cached. The layout follows from the
// arithmetic alone:
//
//   - exact tile: len(hashIDs)*blockSize == numTokens
//   - partial last block: (m-1)*blockSize < numTokens < m*blockSize; the
//     final hash id maps to numTokens-(m-1)*blockSize tokens
//   - prefix only: m*blockSize < numTokens; the un-hashed tail is padded
//     with uncached draws
//
// With no hash ids the whole request is one uncached draw. A hash id seen at
// two sizes while resident is refused with ErrHashIDBlockSizeConflict; the
// blocks cached before the refusal stay cached.
func (c *HashIDBlockCache) Build(numTokens int, hashIDs []int64, blockSize int) ([]int, HashIDLayout, error) {
	if numTokens <= 0 || blockSize <= 0 {
		return nil, 0, fmt.Errorf("%w: num tokens %d and block size %d must both be positive",
			ErrHashIDBlockLayout, numTokens, blockSize)
	}
	m := len(hashIDs)
	if m == 0 {
		return c.sampleTail(numTokens), HashIDLayoutUnhashed, nil
	}
	// A one-token block would be the separator alone, and adjacent
	// separator-only blocks can merge into one token on the wire.
	if blockSize == 1 && c.sep != nil {
		return nil, 0, fmt.Errorf("%w: a separator token needs block size >= 2", ErrHashIDBlockLayout)
	}

	// Checked by division so an absurd block size cannot wrap the products
	// below into a different layout.
	if m-1 > (numTokens-1)/blockSize {
		return nil, 0, fmt.Errorf("%w: %d hash ids at block size %d overshoot %d tokens by a whole block or more",
			ErrHashIDBlockLayout, m, blockSize, numTokens)
	}

	layout := HashIDLayoutExactTile
	finalSize := blockSize
	switch full := (m - 1) * blockSize; {
	case full+blockSize < numTokens:
		layout = HashIDLayoutPrefixOnly
	case full+blockSize > numTokens:
		layout = HashIDLayoutPartialLastBlock
		finalSize = numTokens - full
	}

	out := make([]int, 0, numTokens)
	for i, id := range hashIDs {
		size := blockSize
		if i == m-1 {
			size = finalSize
		}
		block, err := c.block(id, size)
		if err != nil {
			return nil, 0, err
		}
		out = append(out, block...)
	}
	if tail := numTokens - len(out); tail > 0 {
		out = append(out, c.sampleTail(tail)...)
	}
	return out, layout, nil
}

func (c *HashIDBlockCache) block(id int64, size int) ([]int, error) {
	if el, ok := c.blocks[id]; ok {
		cached := el.Value.(*hashIDBlock).tokens
		if len(cached) != size {
			return nil, fmt.Errorf("%w: hash id %d requested at %d tokens, cached at %d",
				ErrHashIDBlockSizeConflict, id, size, len(cached))
		}
		c.lru.MoveToFront(el)
		return cached, nil
	}
	tokens := make([]int, 0, size)
	body := size
	if c.sep != nil {
		tokens = append(tokens, *c.sep)
		body--
	}
	tokens = c.appendFromCorpus(tokens, c.offset(drawBlock, uint64(id)), body)
	c.blocks[id] = c.lru.PushFront(&hashIDBlock{id: id, tokens: tokens})
	for c.lru.Len() > c.maxEntries {
		oldest := c.lru.Back()
		c.lru.Remove(oldest)
		delete(c.blocks, oldest.Value.(*hashIDBlock).id)
	}
	return tokens, nil
}

// sampleTail draws n uncached tokens. Successive draws within a trace walk a
// counter, so tail content is fresh per request yet reproducible for a given
// request order.
func (c *HashIDBlockCache) sampleTail(n int) []int {
	start := c.offset(drawTail, c.tailDraws)
	c.tailDraws++
	return c.appendFromCorpus(make([]int, 0, n), start, n)
}

func (c *HashIDBlockCache) appendFromCorpus(dst []int, start, n int) []int {
	for i := 0; i < n; i++ {
		dst = append(dst, c.corpus[(start+i)%len(c.corpus)])
	}
	return dst
}

const (
	drawBlock byte = 'b'
	drawTail  byte = 't'
)

// offset is the reseed: the corpus start for a draw depends only on the
// seed, the draw kind, its id and the trace id. The fixed-width fields are
// hashed before the variable-length trace id so no two inputs share bytes.
func (c *HashIDBlockCache) offset(kind byte, id uint64) int {
	var head [17]byte
	binary.BigEndian.PutUint64(head[0:8], c.seed)
	head[8] = kind
	binary.BigEndian.PutUint64(head[9:17], id)
	h := sha256.New()
	h.Write(head[:])
	h.Write([]byte(c.traceID))
	sum := h.Sum(nil)
	return int(binary.BigEndian.Uint64(sum[:8]) % uint64(len(c.corpus)))
}
