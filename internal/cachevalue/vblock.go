package cachevalue

// The engine-agnostic vBlock abstraction (issue #10948, child #1498 of the
// vCache epic). A "vBlock" is the vCache design's canonical unit of cache
// accounting: a prefix span identified by a stable key with a token extent —
// the thing warmth beliefs, P&L, and allocation all rank. Every engine names
// and paces it differently: the fak-native in-kernel engine's KV pages
// (internal/ctxmmu), vLLM's paged KV blocks, SGLang's RadixAttention tree
// units. An economics verdict computed on one engine's rows cannot be
// compared with — or reused across — another's without one canonical shape.
//
// VBlock is that shape. An engine adapter (EngineAdapter) maps its native
// block rows into it, stamping BlockTokens with the engine's native page
// size so cross-engine comparisons happen in TOKENS, not pages (a 16-token
// native page and a 32-token paged block are 16 and 32 tokens of warm prefix,
// not "1 and 1"). Rows carrying Engine "inkernel" are the native adapter's;
// "sglang" and "vllm" are the external adapters' names — the same identity
// strings ParseMLXMetrics and the parity dashboards already key on.
//
// The adapter supplies each native block identity. Translate preserves that
// identity inside a length-prefixed engine/session/native-ID tuple, which is
// unique for accounting but makes no content-identity claim. Translate
// verifies the invariants it can see — an exact non-empty ID list, unique
// canonical keys, non-negative token extents, and checked arithmetic — and
// rejects invalid input instead of scoring an economy on phantom blocks.
//
// Std-only and pure, like the rest of the package: no I/O, no clock, no
// engine call.

import (
	"fmt"
	"math"
	"sort"
)

// Engine identity tokens for the shipped adapters. The native in-kernel
// engine's session rows already stamp Engine "inkernel" (qwen38_runner.go's
// endpoint identity, health.Engine), and the external parity surfaces key on
// "vllm" / "sglang"; reusing those exact strings keeps one engine namespace
// across telemetry and the vBlock layer.
const (
	EngineNative = "inkernel"
	EngineVLLM   = "vllm"
	EngineSGLang = "sglang"
)

// EngineUnknown is the bucket a row without an engine stamp translates under
// rather than being dropped — the same fail-open grouping SplitOwnerPnL uses
// for an unstamped owner, so unattributed blocks stay visible.
const EngineUnknown = "unknown"

// VBlock is the engine-agnostic virtual block: one addressable prefix span
// a warmth belief or a P&L line attaches to, whatever engine holds it.
type VBlock struct {
	// Key is an injective accounting tuple of engine, session, and the
	// adapter-supplied native ID. It is not a content identity.
	Key string
	// Engine is the identity token of the engine whose native block row was
	// translated: one of the Engine* constants or an adapter's own token.
	Engine string
	// Tokens is the span's extent in tokens — the cross-engine currency.
	// Non-negative; the sum of a translation's members MUST equal the row's
	// total (exact-sum accounting).
	Tokens int64
	// EngineBlockID is the native block identifier in the SOURCE engine's
	// namespace, carried beside the canonical key so a translated verdict can
	// always be traced back to the engine object it was scored on.
	EngineBlockID string
}

// EngineBlockRow is one engine's native block record — the adapter's input.
// Every engine's counters flatten to these four fields.
type EngineBlockRow struct {
	// Engine is the identity token (EngineNative / EngineVLLM / EngineSGLang
	// / "" → EngineUnknown).
	Engine string
	// SessionKey is the session the blocks belong to; the same ledger key
	// SessionPnL scores, so a translated block set and its session's P&L join
	// on one coordinate.
	SessionKey string
	// Blocks is the engine's native block count for the session.
	Blocks int64
	// BlockTokens is the engine's native page size in tokens (e.g. 16 for the
	// in-kernel KV pages, 16 or 32 for vLLM's). TokensPerBlock in the §5
	// design's vocabulary. Must be >= 0.
	BlockTokens int64
	// Cached is how many of Blocks were served warm (a hit) rather than
	// written — the read/write split P&L prices.
	Cached int64
	// BlockIDs are the adapter-supplied native identities, one per block and
	// in the same order used by Cached. Translation preserves these IDs; it
	// does not pretend an input position is an engine-native identity.
	BlockIDs []string
}

// validate returns an error for a row Translate's invariants reject.
func (r EngineBlockRow) validate() error {
	if r.Blocks < 0 {
		return fmt.Errorf("cachevalue: engine %q session %q: negative block count %d", r.engineName(), r.SessionKey, r.Blocks)
	}
	if r.BlockTokens < 0 {
		return fmt.Errorf("cachevalue: engine %q session %q: negative block token size %d", r.engineName(), r.SessionKey, r.BlockTokens)
	}
	if r.Cached < 0 || r.Cached > r.Blocks {
		return fmt.Errorf("cachevalue: engine %q session %q: cached %d outside [0,%d]", r.engineName(), r.SessionKey, r.Cached, r.Blocks)
	}
	if r.Blocks > int64(len(r.BlockIDs)) || int64(len(r.BlockIDs)) != r.Blocks {
		return fmt.Errorf("cachevalue: engine %q session %q: %d block ids for %d blocks", r.engineName(), r.SessionKey, len(r.BlockIDs), r.Blocks)
	}
	seen := make(map[string]struct{}, len(r.BlockIDs))
	for _, id := range r.BlockIDs {
		if id == "" {
			return fmt.Errorf("cachevalue: engine %q session %q: empty block id", r.engineName(), r.SessionKey)
		}
		if _, ok := seen[id]; ok {
			return fmt.Errorf("cachevalue: engine %q session %q: duplicate block id %q", r.engineName(), r.SessionKey, id)
		}
		seen[id] = struct{}{}
	}
	if r.Blocks != 0 && r.BlockTokens > math.MaxInt64/r.Blocks {
		return fmt.Errorf("cachevalue: engine %q session %q: token extent overflows int64", r.engineName(), r.SessionKey)
	}
	return nil
}

// blockKey encodes the engine, session, and adapter-supplied native ID as a
// length-prefixed tuple. It is an accounting key, not a content identity.
func blockKey(engine, session, nativeID string) string {
	return fmt.Sprintf("%d:%s%d:%s%d:%s", len(engine), engine, len(session), session, len(nativeID), nativeID)
}

// engineName normalizes an empty engine stamp to EngineUnknown (a row the
// adapter stamped without one stays visible rather than vanishing into
// EngineNative).
func (r EngineBlockRow) engineName() string {
	if r.Engine == "" {
		return EngineUnknown
	}
	return r.Engine
}

// VBlockSet is the translated canonical form of one engine row: the
// per-session VBlocks plus the write/read split the session P&L prices.
// CachedBlocks and WrittenBlocks partition Blocks exactly.
type VBlockSet struct {
	// Engine and SessionKey are copied from the row.
	Engine     string
	SessionKey string
	// Blocks lists one VBlock per adapter-supplied native block ID. Its Key is
	// a length-prefixed accounting tuple; EngineBlockID preserves the native
	// identity verbatim.
	Blocks []VBlock
	// CachedTokens / WrittenTokens partition the row's total token extent
	// exactly (their sum equals Blocks×BlockTokens).
	CachedTokens  int64
	WrittenTokens int64
	// TotalBlocks / TotalTokens are the raw counts, kept so a report can show
	// the engine-native quantities beside the canonical ones.
	TotalBlocks int64
	TotalTokens int64
}

// TranslateRow maps ONE engine block row into the canonical VBlock set. The
// token extent per block is the engine's native BlockTokens stamped onto
// every VBlock, so all engines' outputs compare in tokens. A zero-block row
// translates to an empty (but valid, keyed) set — an idle session is honest
// silence, not an error.
func TranslateRow(r EngineBlockRow) (VBlockSet, error) {
	if err := r.validate(); err != nil {
		return VBlockSet{}, err
	}
	engine := r.engineName()
	set := VBlockSet{
		Engine:      engine,
		SessionKey:  r.SessionKey,
		TotalBlocks: r.Blocks,
		TotalTokens: r.Blocks * r.BlockTokens,
	}
	if r.Blocks == 0 {
		return set, nil
	}
	set.Blocks = make([]VBlock, len(r.BlockIDs))
	for i := int64(0); i < r.Blocks; i++ {
		cached := i < r.Cached
		if cached {
			set.CachedTokens += r.BlockTokens
		} else {
			set.WrittenTokens += r.BlockTokens
		}
		block := VBlock{
			Key:           blockKey(engine, r.SessionKey, r.BlockIDs[i]),
			Engine:        engine,
			Tokens:        r.BlockTokens,
			EngineBlockID: r.BlockIDs[i],
		}
		set.Blocks[i] = block
	}
	return set, nil
}

// TranslateRows maps every row, in input order, rejecting the whole
// translation on the first invariants-violating row (fail-closed: a
// mis-mapped adapter must not yield a partial economy). Rows from DIFFERENT
// engines over the SAME session are the normal multi-engine case and are all
// returned, in engine-then-input order — the "translated across multiple
// engine adapters" acceptance surface.
func TranslateRows(rows []EngineBlockRow) ([]VBlockSet, error) {
	out := make([]VBlockSet, 0, len(rows))
	keys := make(map[string]struct{})
	for _, r := range rows {
		set, err := TranslateRow(r)
		if err != nil {
			return nil, err
		}
		for _, block := range set.Blocks {
			if _, ok := keys[block.Key]; ok {
				return nil, fmt.Errorf("cachevalue: duplicate canonical block key %q", block.Key)
			}
			keys[block.Key] = struct{}{}
		}
		out = append(out, set)
	}
	return out, nil
}

// EngineAggregates folds translated sets into one per-engine total: blocks
// and tokens summed across its sessions, cached vs written split kept apart
// so an engine's warmth (and its write bill) is visible per engine. In
// ascending-engine order, deterministically.
type EngineAggregate struct {
	Engine        string `json:"engine"`
	Sessions      int    `json:"sessions"`
	TotalBlocks   int64  `json:"total_blocks"`
	TotalTokens   int64  `json:"total_tokens"`
	CachedTokens  int64  `json:"cached_tokens"`
	WrittenTokens int64  `json:"written_tokens"`
}

// AggregateByEngine folds taken VBlockSets into per-engine totals. Cached /
// written token splits are summed across sets; TotalTokens is asserted to
// equal the split sum by construction (Translate guarantees it per set).
func AggregateByEngine(sets []VBlockSet) ([]EngineAggregate, error) {
	idx := map[string]int{}
	out := make([]EngineAggregate, 0, 3)
	for _, s := range sets {
		i, ok := idx[s.Engine]
		if !ok {
			i = len(out)
			idx[s.Engine] = i
			out = append(out, EngineAggregate{Engine: s.Engine})
		}
		a := &out[i]
		if s.TotalBlocks < 0 || s.TotalTokens < 0 || s.CachedTokens < 0 || s.WrittenTokens < 0 ||
			a.TotalBlocks > math.MaxInt64-s.TotalBlocks || a.TotalTokens > math.MaxInt64-s.TotalTokens ||
			a.CachedTokens > math.MaxInt64-s.CachedTokens || a.WrittenTokens > math.MaxInt64-s.WrittenTokens {
			return nil, fmt.Errorf("cachevalue: engine %q aggregate overflows or contains negative counters", s.Engine)
		}
		a.Sessions++
		a.TotalBlocks += s.TotalBlocks
		a.TotalTokens += s.TotalTokens
		a.CachedTokens += s.CachedTokens
		a.WrittenTokens += s.WrittenTokens
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Engine < out[j].Engine })
	return out, nil
}
