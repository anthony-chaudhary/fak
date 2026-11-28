package ggufload

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// DefaultInkling2PrefixMaxEntries bounds resident prefix checkpoints kept in
	// host RAM when DefaultInkling2ResidentPrefixConfig is used.
	DefaultInkling2PrefixMaxEntries = 16
	// DefaultInkling2PrefixMaxBytes bounds host RAM across all cached resident
	// prefix checkpoints (512 MiB).
	DefaultInkling2PrefixMaxBytes int64 = 512 << 20
)

var (
	// ErrInkling2PrefixTooLarge indicates a single prefix checkpoint exceeds
	// the configured Inkling2ResidentPrefixConfig.MaxBytes budget.
	ErrInkling2PrefixTooLarge = errors.New("inkling2 prefix: checkpoint exceeds cache MaxBytes budget")
	// ErrInkling2PrefixConfigInvalid indicates invalid prefix cache bounds.
	ErrInkling2PrefixConfigInvalid = errors.New("inkling2 prefix: invalid cache config")
)

// Inkling2ResidentPrefixConfig sets explicit host-memory eviction bounds on an
// Inkling2ResidentPrefixCache. Either MaxEntries or MaxBytes must be positive,
// and neither may be negative.
type Inkling2ResidentPrefixConfig struct {
	MaxEntries int
	MaxBytes   int64
}

// DefaultInkling2ResidentPrefixConfig returns the bounded default host-memory
// prefix cache configuration.
func DefaultInkling2ResidentPrefixConfig() Inkling2ResidentPrefixConfig {
	return Inkling2ResidentPrefixConfig{
		MaxEntries: DefaultInkling2PrefixMaxEntries,
		MaxBytes:   DefaultInkling2PrefixMaxBytes,
	}
}

// Validate verifies the prefix cache bounds are explicit and non-negative.
func (c Inkling2ResidentPrefixConfig) Validate() error {
	if c.MaxEntries < 0 || c.MaxBytes < 0 || (c.MaxEntries == 0 && c.MaxBytes == 0) {
		return fmt.Errorf("%w: max_entries=%d max_bytes=%d", ErrInkling2PrefixConfigInvalid, c.MaxEntries, c.MaxBytes)
	}
	return nil
}

// Inkling2PrefixSlotState holds one cache slot's saved state (short-conv ring
// plus linear global KV or sliding-window ring KV) in host RAM.
type Inkling2PrefixSlotState struct {
	Local    bool
	FullRing bool
	Lo       int
	Cap      int
	Conv     []float32
	K, V     []float32
	KS, VS   []float32
}

// Inkling2PrefixDraftSlotState holds one optional attached draft layer ring.
type Inkling2PrefixDraftSlotState struct {
	Lo, Hi, Cap, Words int
	K, V               []float32
}

// Inkling2ResidentPrefixCheckpoint is an immutable-contract host snapshot of a
// resident Inkling-2 session at Pos == len(PrefixTokens).
type Inkling2ResidentPrefixCheckpoint struct {
	Scope        string
	PrefixTokens []int
	HParamsJSON  string
	Pos          int
	KVDType      Inkling2KVDType
	LocalKVDType Inkling2KVDType
	Slots        []Inkling2PrefixSlotState
	DraftSlots   []Inkling2PrefixDraftSlotState
	LastLogits   []float32
	byteSize     int64
}

// ByteSize returns the accounted host byte footprint of the checkpoint.
func (ck *Inkling2ResidentPrefixCheckpoint) ByteSize() int64 {
	if ck == nil {
		return 0
	}
	if ck.byteSize > 0 {
		return ck.byteSize
	}
	ck.byteSize = ck.computeByteSize()
	return ck.byteSize
}

func (ck *Inkling2ResidentPrefixCheckpoint) computeByteSize() int64 {
	var total int64
	total += int64(len(ck.Scope)) + int64(len(ck.HParamsJSON))
	total += int64(len(ck.PrefixTokens)) * 8
	total += int64(len(ck.LastLogits)) * 4
	for _, sc := range ck.Slots {
		total += int64(len(sc.Conv)+len(sc.K)+len(sc.V)+len(sc.KS)+len(sc.VS)) * 4
	}
	for _, ds := range ck.DraftSlots {
		total += int64(len(ds.K)+len(ds.V)) * 4
	}
	return total
}

func (ck *Inkling2ResidentPrefixCheckpoint) validate() error {
	if ck == nil {
		return ErrInkling2NoCheckpoint
	}
	if ck.Pos < 0 || ck.HParamsJSON == "" {
		return fmt.Errorf("%w: invalid checkpoint metadata pos=%d", model.ErrInkling2StateSnapshotInvalid, ck.Pos)
	}
	if len(ck.Slots) == 0 {
		return fmt.Errorf("%w: checkpoint has no slots", model.ErrInkling2StateSnapshotInvalid)
	}
	if len(ck.PrefixTokens) > 0 && len(ck.PrefixTokens) != ck.Pos {
		return fmt.Errorf("%w: prefix_tokens=%d mismatches pos=%d", model.ErrInkling2StateSnapshotInvalid, len(ck.PrefixTokens), ck.Pos)
	}
	return nil
}

// MatchesTokens reports whether ck captures the exact token sequence tokens.
func (ck *Inkling2ResidentPrefixCheckpoint) MatchesTokens(tokens []int) bool {
	if ck == nil || len(ck.PrefixTokens) != len(tokens) {
		return false
	}
	for i := range tokens {
		if ck.PrefixTokens[i] != tokens[i] {
			return false
		}
	}
	return true
}

// Clone returns an isolated deep copy of ck so callers can test mutation or
// reuse snapshots without aliasing cache storage.
func (ck *Inkling2ResidentPrefixCheckpoint) Clone() *Inkling2ResidentPrefixCheckpoint {
	if ck == nil {
		return nil
	}
	out := &Inkling2ResidentPrefixCheckpoint{
		Scope:        ck.Scope,
		PrefixTokens: append([]int(nil), ck.PrefixTokens...),
		HParamsJSON:  ck.HParamsJSON,
		Pos:          ck.Pos,
		KVDType:      ck.KVDType,
		LocalKVDType: ck.LocalKVDType,
		Slots:        make([]Inkling2PrefixSlotState, len(ck.Slots)),
		DraftSlots:   make([]Inkling2PrefixDraftSlotState, len(ck.DraftSlots)),
		LastLogits:   append([]float32(nil), ck.LastLogits...),
	}
	for i, sc := range ck.Slots {
		out.Slots[i] = Inkling2PrefixSlotState{
			Local:    sc.Local,
			FullRing: sc.FullRing,
			Lo:       sc.Lo,
			Cap:      sc.Cap,
			Conv:     append([]float32(nil), sc.Conv...),
			K:        append([]float32(nil), sc.K...),
			V:        append([]float32(nil), sc.V...),
			KS:       append([]float32(nil), sc.KS...),
			VS:       append([]float32(nil), sc.VS...),
		}
	}
	for i, ds := range ck.DraftSlots {
		out.DraftSlots[i] = Inkling2PrefixDraftSlotState{
			Lo:    ds.Lo,
			Hi:    ds.Hi,
			Cap:   ds.Cap,
			Words: ds.Words,
			K:     append([]float32(nil), ds.K...),
			V:     append([]float32(nil), ds.V...),
		}
	}
	out.byteSize = out.computeByteSize()
	return out
}

// Inkling2PrefixCacheStats reports bounded prefix-cache utilization and hit/eviction counters.
type Inkling2PrefixCacheStats struct {
	Entries    int
	Bytes      int64
	Hits       int64
	ExactHits  int64
	PrefixHits int64
	Misses     int64
	Evictions  int64
	Rejections int64
}

type ink2ResidentPrefixEntry struct {
	key string
	ck  *Inkling2ResidentPrefixCheckpoint
}

// Inkling2ResidentPrefixCache is a bounded LRU host-memory cache of resident
// prompt-prefix checkpoints keyed by (scope, kvDType, localKVDType, prefixTokens).
// When unused, newInkling2Resident behavior and memory allocation are unchanged.
type Inkling2ResidentPrefixCache struct {
	mu    sync.Mutex
	cfg   Inkling2ResidentPrefixConfig
	ll    *list.List
	byKey map[string]*list.Element
	stats Inkling2PrefixCacheStats
}

// NewInkling2ResidentPrefixCache constructs a bounded LRU prefix checkpoint cache.
func NewInkling2ResidentPrefixCache(cfg Inkling2ResidentPrefixConfig) (*Inkling2ResidentPrefixCache, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Inkling2ResidentPrefixCache{
		cfg:   cfg,
		ll:    list.New(),
		byKey: make(map[string]*list.Element),
	}, nil
}

// Inkling2PrefixTokensKey hashes scope, dtype pair, and exact token sequence.
func Inkling2PrefixTokensKey(scope string, kv, localKV Inkling2KVDType, tokens []int) string {
	h := sha256.New()
	var hdr [16]byte
	binary.LittleEndian.PutUint64(hdr[0:8], uint64(len(scope)))
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(len(tokens)))
	h.Write(hdr[:])
	h.Write([]byte(scope))
	h.Write([]byte{byte(kv), byte(localKV)})
	var tokBuf [8]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint64(tokBuf[:], uint64(t))
		h.Write(tokBuf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Put inserts or refreshes ck under LRU eviction bounds. Single entries exceeding
// cfg.MaxBytes fail cleanly with ErrInkling2PrefixTooLarge without evicting
// existing entries.
func (c *Inkling2ResidentPrefixCache) Put(ck *Inkling2ResidentPrefixCheckpoint) error {
	if c == nil {
		return fmt.Errorf("inkling2 prefix: nil cache")
	}
	if err := ck.validate(); err != nil {
		return err
	}
	if len(ck.PrefixTokens) == 0 {
		return fmt.Errorf("inkling2 prefix: empty PrefixTokens cannot be keyed")
	}
	cp := ck.Clone()
	sz := cp.ByteSize()

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cfg.MaxBytes > 0 && sz > c.cfg.MaxBytes {
		c.stats.Rejections++
		return fmt.Errorf("%w: entry=%d max=%d", ErrInkling2PrefixTooLarge, sz, c.cfg.MaxBytes)
	}
	key := Inkling2PrefixTokensKey(cp.Scope, cp.KVDType, cp.LocalKVDType, cp.PrefixTokens)
	if el, ok := c.byKey[key]; ok {
		old := el.Value.(*ink2ResidentPrefixEntry)
		c.stats.Bytes -= old.ck.ByteSize()
		old.ck = cp
		c.stats.Bytes += sz
		c.ll.MoveToFront(el)
	} else {
		el := c.ll.PushFront(&ink2ResidentPrefixEntry{key: key, ck: cp})
		c.byKey[key] = el
		c.stats.Entries++
		c.stats.Bytes += sz
	}
	c.evictLocked()
	return nil
}

func (c *Inkling2ResidentPrefixCache) evictLocked() {
	for {
		overEntries := c.cfg.MaxEntries > 0 && c.stats.Entries > c.cfg.MaxEntries
		overBytes := c.cfg.MaxBytes > 0 && c.stats.Bytes > c.cfg.MaxBytes
		if !overEntries && !overBytes {
			break
		}
		tail := c.ll.Back()
		if tail == nil {
			break
		}
		ent := c.ll.Remove(tail).(*ink2ResidentPrefixEntry)
		delete(c.byKey, ent.key)
		c.stats.Entries--
		c.stats.Bytes -= ent.ck.ByteSize()
		c.stats.Evictions++
	}
}

// GetExact looks up an exact-token checkpoint in c and promotes it on hit.
func (c *Inkling2ResidentPrefixCache) GetExact(scope string, kv, localKV Inkling2KVDType, tokens []int) (*Inkling2ResidentPrefixCheckpoint, bool) {
	if c == nil || len(tokens) == 0 {
		return nil, false
	}
	key := Inkling2PrefixTokensKey(scope, kv, localKV, tokens)
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.byKey[key]
	if !ok {
		c.stats.Misses++
		return nil, false
	}
	ent := el.Value.(*ink2ResidentPrefixEntry)
	if !ent.ck.MatchesTokens(tokens) || ent.ck.Scope != scope || ent.ck.KVDType != kv || ent.ck.LocalKVDType != localKV {
		c.stats.Misses++
		return nil, false
	}
	c.ll.MoveToFront(el)
	c.stats.Hits++
	c.stats.ExactHits++
	return ent.ck, true
}

// LookupLongestPrefix returns the longest cached prefix of tokens[:min(len(tokens), maxPrefixLen)].
func (c *Inkling2ResidentPrefixCache) LookupLongestPrefix(scope string, kv, localKV Inkling2KVDType, tokens []int, maxPrefixLen int) (*Inkling2ResidentPrefixCheckpoint, int, bool) {
	if c == nil || len(tokens) == 0 || maxPrefixLen <= 0 {
		return nil, 0, false
	}
	limit := min(len(tokens), maxPrefixLen)
	c.mu.Lock()
	defer c.mu.Unlock()
	for n := limit; n >= 1; n-- {
		prefix := tokens[:n]
		key := Inkling2PrefixTokensKey(scope, kv, localKV, prefix)
		el, ok := c.byKey[key]
		if !ok {
			continue
		}
		ent := el.Value.(*ink2ResidentPrefixEntry)
		if !ent.ck.MatchesTokens(prefix) || ent.ck.Scope != scope || ent.ck.KVDType != kv || ent.ck.LocalKVDType != localKV {
			continue
		}
		c.ll.MoveToFront(el)
		c.stats.Hits++
		if n == len(tokens) {
			c.stats.ExactHits++
		} else {
			c.stats.PrefixHits++
		}
		return ent.ck, n, true
	}
	c.stats.Misses++
	return nil, 0, false
}

// Stats returns a snapshot of current cache utilization and counters.
func (c *Inkling2ResidentPrefixCache) Stats() Inkling2PrefixCacheStats {
	if c == nil {
		return Inkling2PrefixCacheStats{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}

// Clear drops all cached prefix checkpoints while preserving cumulative hit/eviction counters.
func (c *Inkling2ResidentPrefixCache) Clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ll.Init()
	clear(c.byKey)
	c.stats.Entries = 0
	c.stats.Bytes = 0
}

// Inkling2LoopPrefixSession is an optional extension of Inkling2LoopSession for
// explicit host-memory prompt-prefix checkpoint capture, restore, and cache-backed prefill.
type Inkling2LoopPrefixSession interface {
	CheckpointPrefix(prefixTokens []int, scope string) (*Inkling2ResidentPrefixCheckpoint, error)
	RestorePrefix(ck *Inkling2ResidentPrefixCheckpoint) error
	PrefillWithPrefixCache(ctx context.Context, cache *Inkling2ResidentPrefixCache, scope string, prompt []int) (next int, matchedPrefix int, err error)
}

func (r *inkling2ResidentSession) CheckpointPrefix(prefixTokens []int, scope string) (*Inkling2ResidentPrefixCheckpoint, error) {
	if r == nil || r.s == nil {
		return nil, fmt.Errorf("inkling2 prefix: closed session")
	}
	r.own()
	if len(prefixTokens) != r.s.pos {
		return nil, fmt.Errorf("inkling2 prefix: prefixTokens len=%d mismatches session pos=%d", len(prefixTokens), r.s.pos)
	}
	return r.s.capturePrefixCheckpoint(prefixTokens, scope)
}

func (r *inkling2ResidentSession) RestorePrefix(ck *Inkling2ResidentPrefixCheckpoint) error {
	if r == nil || r.s == nil {
		return fmt.Errorf("inkling2 prefix: closed session")
	}
	r.own()
	if err := r.s.restorePrefixCheckpoint(ck); err != nil {
		return err
	}
	if r.ck != nil {
		r.ck.valid = false
	}
	return nil
}

// PrefillWithPrefixCache matches the longest cached prefix of prompt on cache,
// restores it if compatible, and prefills only the remaining suffix tokens. If no
// prefix hits, it runs a standard cold prefill starting from position 0.
func (r *inkling2ResidentSession) PrefillWithPrefixCache(ctx context.Context, cache *Inkling2ResidentPrefixCache, scope string, prompt []int) (int, int, error) {
	if r == nil || r.s == nil {
		return 0, 0, fmt.Errorf("inkling2 prefix: closed session")
	}
	if len(prompt) == 0 {
		return 0, 0, fmt.Errorf("inkling2_loop: empty prompt")
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	if cache != nil {
		ck, matched, ok := cache.LookupLongestPrefix(scope, r.s.kvDType(), r.s.localKVDtype, prompt, len(prompt))
		if ok && matched == len(prompt) && len(ck.LastLogits) != r.s.hp.VocabUnpadded {
			// Exact match saved without logits: try longest strictly shorter prefix so at least one token recomputes logits cleanly.
			ck, matched, ok = cache.LookupLongestPrefix(scope, r.s.kvDType(), r.s.localKVDtype, prompt, len(prompt)-1)
		}
		if ok && matched > 0 {
			if err := r.RestorePrefix(ck); err == nil {
				if matched == len(prompt) {
					return inkling2Argmax(ck.LastLogits), matched, nil
				}
				next, err := r.Prefill(ctx, prompt[matched:])
				return next, matched, err
			}
		}
	}
	if r.s.pos != 0 {
		if err := r.Reset(); err != nil {
			return 0, 0, err
		}
	}
	next, err := r.Prefill(ctx, prompt)
	return next, 0, err
}
