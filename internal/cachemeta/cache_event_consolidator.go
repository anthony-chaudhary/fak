package cachemeta

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// CacheLogicalBlockKeyVersion is bumped whenever the fields participating in a
// cache-event block identity change. Persisted consolidator state refuses a
// different version instead of silently joining blocks under incompatible keys.
const CacheLogicalBlockKeyVersion = "fak.cache-logical-block/v1"

// CacheLogicalBlockKey is the engine-neutral identity of one logical KV block.
// SourceID is deliberately not part of the key: two workers publishing the same
// model/tokenizer/block are two sources for one visible logical block.
type CacheLogicalBlockKey struct {
	Version     string `json:"version"`
	ModelID     string `json:"model_id"`
	TokenizerID string `json:"tokenizer_id"`
	Digest      string `json:"digest"`
}

// NewCacheLogicalBlockKey returns the current version of the stable logical
// block key used by live cache-event producers.
func NewCacheLogicalBlockKey(modelID, tokenizerID, digest string) CacheLogicalBlockKey {
	return CacheLogicalBlockKey{
		Version:     CacheLogicalBlockKeyVersion,
		ModelID:     modelID,
		TokenizerID: tokenizerID,
		Digest:      digest,
	}
}

func (k CacheLogicalBlockKey) valid() bool {
	return k.Version == CacheLogicalBlockKeyVersion && strings.TrimSpace(k.Digest) != ""
}

func (k CacheLogicalBlockKey) less(other CacheLogicalBlockKey) bool {
	if k.Version != other.Version {
		return k.Version < other.Version
	}
	if k.ModelID != other.ModelID {
		return k.ModelID < other.ModelID
	}
	if k.TokenizerID != other.TokenizerID {
		return k.TokenizerID < other.TokenizerID
	}
	return k.Digest < other.Digest
}

// CacheSourceEventID gives a producer event both a stable deduplication identity
// and a source-local causal order. Sequence must increase within SourceID;
// EventID breaks ties deterministically and makes exact replays recognizable.
type CacheSourceEventID struct {
	SourceID string `json:"source_id"`
	EventID  string `json:"event_id"`
	Sequence uint64 `json:"sequence"`
}

func (id CacheSourceEventID) valid() bool {
	return strings.TrimSpace(id.SourceID) != "" && strings.TrimSpace(id.EventID) != ""
}

func compareCacheSourceEventID(a, b CacheSourceEventID) int {
	if a.Sequence < b.Sequence {
		return -1
	}
	if a.Sequence > b.Sequence {
		return 1
	}
	return strings.Compare(a.EventID, b.EventID)
}

// CacheVisibilityAction is the logical visibility transition represented by a
// source event. It is separate from KVTransferDirection: a STORE can arrive as a
// restore/migration, while REMOVE describes source departure rather than a byte
// transfer destination.
type CacheVisibilityAction string

const (
	CacheVisibilityStore  CacheVisibilityAction = "store"
	CacheVisibilityRemove CacheVisibilityAction = "remove"
)

// CacheVisibilityEvent is the minimal input to the source-aware consolidator.
type CacheVisibilityEvent struct {
	Identity CacheSourceEventID    `json:"identity"`
	Block    CacheLogicalBlockKey  `json:"block"`
	Action   CacheVisibilityAction `json:"action"`
}

// CacheEventSuppression explains why a source event was not published.
type CacheEventSuppression string

const (
	CacheEventNotSuppressed       CacheEventSuppression = ""
	CacheEventDuplicateReplay     CacheEventSuppression = "duplicate_replay"
	CacheEventReorderedReplay     CacheEventSuppression = "reordered_replay"
	CacheEventDuplicateProducer   CacheEventSuppression = "duplicate_producer"
	CacheEventSourceStillResident CacheEventSuppression = "source_still_resident"
	CacheEventUnknown             CacheEventSuppression = "unknown"
	CacheEventStateBound          CacheEventSuppression = "state_bound"
)

// CacheVisibilityDecision says whether the event is the one logical transition
// observers may publish. Store publishes only on 0->1 resident sources; remove
// publishes only on 1->0.
type CacheVisibilityDecision struct {
	Publish     bool                  `json:"publish"`
	Action      CacheVisibilityAction `json:"action"`
	Suppression CacheEventSuppression `json:"suppression,omitempty"`
}

// CacheEventConsolidatorCounters are cumulative and restartable. Unknown counts
// malformed events, impossible removes, and events rejected at the state bound.
type CacheEventConsolidatorCounters struct {
	PublishedStores        uint64 `json:"published_stores"`
	PublishedRemoves       uint64 `json:"published_removes"`
	DuplicateReplays       uint64 `json:"duplicate_replays"`
	ReorderedReplays       uint64 `json:"reordered_replays"`
	DuplicateProducers     uint64 `json:"duplicate_producers"`
	SuppressedRemoves      uint64 `json:"suppressed_removes"`
	UnknownEvents          uint64 `json:"unknown_events"`
	StateBoundSuppressions uint64 `json:"state_bound_suppressions"`
}

type cacheSourceVisibilityState struct {
	Identity CacheSourceEventID
	Present  bool
}

type cacheBlockVisibilityState struct {
	Sources map[string]cacheSourceVisibilityState
}

// CacheEventConsolidator is a bounded, source-aware visibility state machine.
// It retains final tombstones so an older replay cannot resurrect a removed
// block. Once the source-state bound is reached, a new (logical block, source)
// pair is reported unknown and suppressed; existing tracked state stays correct.
type CacheEventConsolidator struct {
	mu         sync.Mutex
	maxEntries int
	entries    int
	blocks     map[CacheLogicalBlockKey]*cacheBlockVisibilityState
	counters   CacheEventConsolidatorCounters
}

// NewCacheEventConsolidator constructs an empty consolidator. maxEntries bounds
// the total (logical block, source) states, including replay-safe tombstones.
func NewCacheEventConsolidator(maxEntries int) *CacheEventConsolidator {
	if maxEntries < 1 {
		maxEntries = 1
	}
	return &CacheEventConsolidator{
		maxEntries: maxEntries,
		blocks:     make(map[CacheLogicalBlockKey]*cacheBlockVisibilityState),
	}
}

// Consolidate applies one event. Source-local sequence ordering makes the final
// visibility state independent of replay arrival order; exact/current duplicates
// and causally older events never mutate state.
func (c *CacheEventConsolidator) Consolidate(ev CacheVisibilityEvent) CacheVisibilityDecision {
	decision := CacheVisibilityDecision{Action: ev.Action}
	if c == nil {
		decision.Publish = true
		return decision
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if !ev.Identity.valid() || !ev.Block.valid() || (ev.Action != CacheVisibilityStore && ev.Action != CacheVisibilityRemove) {
		c.counters.UnknownEvents++
		decision.Suppression = CacheEventUnknown
		return decision
	}
	block := c.blocks[ev.Block]
	previous, knownSource := cacheSourceVisibilityState{}, false
	if block != nil {
		previous, knownSource = block.Sources[ev.Identity.SourceID]
	}
	if knownSource {
		order := compareCacheSourceEventID(ev.Identity, previous.Identity)
		if order == 0 {
			c.counters.DuplicateReplays++
			decision.Suppression = CacheEventDuplicateReplay
			return decision
		}
		if order < 0 {
			c.counters.ReorderedReplays++
			decision.Suppression = CacheEventReorderedReplay
			return decision
		}
	}
	if !knownSource && c.entries >= c.maxEntries {
		c.counters.UnknownEvents++
		c.counters.StateBoundSuppressions++
		decision.Suppression = CacheEventStateBound
		return decision
	}
	if block == nil {
		block = &cacheBlockVisibilityState{Sources: make(map[string]cacheSourceVisibilityState)}
		c.blocks[ev.Block] = block
	}
	if !knownSource {
		c.entries++
	}

	presentBefore := cachePresentSourceCount(block)
	switch ev.Action {
	case CacheVisibilityStore:
		block.Sources[ev.Identity.SourceID] = cacheSourceVisibilityState{Identity: ev.Identity, Present: true}
		if knownSource && previous.Present {
			c.counters.DuplicateProducers++
			decision.Suppression = CacheEventDuplicateProducer
			return decision
		}
		if presentBefore > 0 {
			c.counters.DuplicateProducers++
			decision.Suppression = CacheEventDuplicateProducer
			return decision
		}
		c.counters.PublishedStores++
		decision.Publish = true
		return decision

	case CacheVisibilityRemove:
		block.Sources[ev.Identity.SourceID] = cacheSourceVisibilityState{Identity: ev.Identity, Present: false}
		if !knownSource || !previous.Present {
			c.counters.UnknownEvents++
			decision.Suppression = CacheEventUnknown
			return decision
		}
		if presentBefore > 1 {
			c.counters.SuppressedRemoves++
			decision.Suppression = CacheEventSourceStillResident
			return decision
		}
		c.counters.PublishedRemoves++
		decision.Publish = true
		return decision
	}
	panic("unreachable cache visibility action")
}

func cachePresentSourceCount(block *cacheBlockVisibilityState) int {
	n := 0
	for _, source := range block.Sources {
		if source.Present {
			n++
		}
	}
	return n
}

// CacheEventConsolidatorSourceState is the stable persisted form of one source's
// latest event for a logical block.
type CacheEventConsolidatorSourceState struct {
	Identity CacheSourceEventID `json:"identity"`
	Present  bool               `json:"present"`
}

// CacheEventConsolidatorBlockState is sorted by block key in SnapshotState.
type CacheEventConsolidatorBlockState struct {
	Block   CacheLogicalBlockKey                `json:"block"`
	Sources []CacheEventConsolidatorSourceState `json:"sources"`
}

// CacheEventConsolidatorState is a deterministic restart/bootstrap document.
// Blocks and Sources are emitted in stable lexical order.
type CacheEventConsolidatorState struct {
	Version   string                             `json:"version"`
	MaxBlocks int                                `json:"max_blocks"`
	Blocks    []CacheEventConsolidatorBlockState `json:"blocks"`
	Counters  CacheEventConsolidatorCounters     `json:"counters"`
}

const cacheEventConsolidatorStateVersion = "fak.cache-event-consolidator/v1"

// SnapshotState returns a deterministic bootstrap/restart representation.
func (c *CacheEventConsolidator) SnapshotState() CacheEventConsolidatorState {
	state := CacheEventConsolidatorState{Version: cacheEventConsolidatorStateVersion}
	if c == nil {
		return state
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	state.MaxBlocks = c.maxEntries
	state.Counters = c.counters
	state.Blocks = make([]CacheEventConsolidatorBlockState, 0, len(c.blocks))
	for key, block := range c.blocks {
		row := CacheEventConsolidatorBlockState{Block: key, Sources: make([]CacheEventConsolidatorSourceState, 0, len(block.Sources))}
		for _, source := range block.Sources {
			row.Sources = append(row.Sources, CacheEventConsolidatorSourceState{Identity: source.Identity, Present: source.Present})
		}
		sort.Slice(row.Sources, func(i, j int) bool {
			return row.Sources[i].Identity.SourceID < row.Sources[j].Identity.SourceID
		})
		state.Blocks = append(state.Blocks, row)
	}
	sort.Slice(state.Blocks, func(i, j int) bool { return state.Blocks[i].Block.less(state.Blocks[j].Block) })
	return state
}

// RestoreCacheEventConsolidator validates and restores a persisted state without
// depending on its serialized row order.
func RestoreCacheEventConsolidator(state CacheEventConsolidatorState) (*CacheEventConsolidator, error) {
	if state.Version != cacheEventConsolidatorStateVersion {
		return nil, fmt.Errorf("cache-event consolidator: state version %q, want %q", state.Version, cacheEventConsolidatorStateVersion)
	}
	if state.MaxBlocks < 1 {
		return nil, fmt.Errorf("cache-event consolidator: invalid state bound %d", state.MaxBlocks)
	}
	out := NewCacheEventConsolidator(state.MaxBlocks)
	out.counters = state.Counters
	for _, row := range state.Blocks {
		if !row.Block.valid() {
			return nil, fmt.Errorf("cache-event consolidator: invalid logical block key %+v", row.Block)
		}
		if _, exists := out.blocks[row.Block]; exists {
			return nil, fmt.Errorf("cache-event consolidator: duplicate logical block %+v", row.Block)
		}
		block := &cacheBlockVisibilityState{Sources: make(map[string]cacheSourceVisibilityState, len(row.Sources))}
		for _, source := range row.Sources {
			if !source.Identity.valid() {
				return nil, fmt.Errorf("cache-event consolidator: invalid source identity %+v", source.Identity)
			}
			if _, exists := block.Sources[source.Identity.SourceID]; exists {
				return nil, fmt.Errorf("cache-event consolidator: duplicate source %q", source.Identity.SourceID)
			}
			block.Sources[source.Identity.SourceID] = cacheSourceVisibilityState{Identity: source.Identity, Present: source.Present}
			out.entries++
			if out.entries > out.maxEntries {
				return nil, fmt.Errorf("cache-event consolidator: %d source states exceed bound %d", out.entries, out.maxEntries)
			}
		}
		out.blocks[row.Block] = block
	}
	return out, nil
}

// Counters returns an immutable copy of the current consolidation counters.
func (c *CacheEventConsolidator) Counters() CacheEventConsolidatorCounters {
	if c == nil {
		return CacheEventConsolidatorCounters{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.counters
}

// StateEntries reports the bounded (logical block, source) state cardinality.
func (c *CacheEventConsolidator) StateEntries() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.entries
}

// PresentBlocksForSource returns the logical blocks currently held by source in
// stable key order. A source-wide clear can therefore lower to ordinary per-block
// REMOVE decisions without hiding another source's residency.
func (c *CacheEventConsolidator) PresentBlocksForSource(sourceID string) []CacheLogicalBlockKey {
	if c == nil || strings.TrimSpace(sourceID) == "" {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]CacheLogicalBlockKey, 0)
	for key, block := range c.blocks {
		if source, ok := block.Sources[sourceID]; ok && source.Present {
			out = append(out, key)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].less(out[j]) })
	return out
}
