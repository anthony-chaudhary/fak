package cachemeta

import (
	"reflect"
	"testing"
)

func TestCacheEventConsolidatorPublishesFirstStoreAndFinalRemove(t *testing.T) {
	c := NewCacheEventConsolidator(4)
	block := NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-a")

	firstStore := c.Consolidate(cacheVisibilityEvent(block, "source-a", "a-store", 1, CacheVisibilityStore))
	if !firstStore.Publish || firstStore.Action != CacheVisibilityStore || firstStore.Suppression != CacheEventNotSuppressed {
		t.Fatalf("first STORE decision = %+v, want published STORE", firstStore)
	}

	secondStore := c.Consolidate(cacheVisibilityEvent(block, "source-b", "b-store", 1, CacheVisibilityStore))
	if secondStore.Publish || secondStore.Suppression != CacheEventDuplicateProducer {
		t.Fatalf("second-source STORE decision = %+v, want duplicate-producer suppression", secondStore)
	}

	firstRemove := c.Consolidate(cacheVisibilityEvent(block, "source-a", "a-remove", 2, CacheVisibilityRemove))
	if firstRemove.Publish || firstRemove.Suppression != CacheEventSourceStillResident {
		t.Fatalf("first-source REMOVE decision = %+v, want source-still-resident suppression", firstRemove)
	}

	finalRemove := c.Consolidate(cacheVisibilityEvent(block, "source-b", "b-remove", 2, CacheVisibilityRemove))
	if !finalRemove.Publish || finalRemove.Action != CacheVisibilityRemove || finalRemove.Suppression != CacheEventNotSuppressed {
		t.Fatalf("final REMOVE decision = %+v, want published REMOVE", finalRemove)
	}

	want := CacheEventConsolidatorCounters{
		PublishedStores:    1,
		PublishedRemoves:   1,
		DuplicateProducers: 1,
		SuppressedRemoves:  1,
	}
	if got := c.Counters(); !reflect.DeepEqual(got, want) {
		t.Fatalf("counters = %+v, want %+v", got, want)
	}
}

func TestCacheEventConsolidatorSuppressesDuplicateAndReorderedReplay(t *testing.T) {
	c := NewCacheEventConsolidator(2)
	block := NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-replay")
	store := cacheVisibilityEvent(block, "source-a", "store-10", 10, CacheVisibilityStore)

	if got := c.Consolidate(store); !got.Publish {
		t.Fatalf("initial STORE = %+v, want published", got)
	}
	if got := c.Consolidate(store); got.Publish || got.Suppression != CacheEventDuplicateReplay {
		t.Fatalf("duplicate STORE replay = %+v, want duplicate-replay suppression", got)
	}

	remove := cacheVisibilityEvent(block, "source-a", "remove-12", 12, CacheVisibilityRemove)
	if got := c.Consolidate(remove); !got.Publish {
		t.Fatalf("newer REMOVE = %+v, want published", got)
	}
	if got := c.Consolidate(cacheVisibilityEvent(block, "source-a", "store-11", 11, CacheVisibilityStore)); got.Publish || got.Suppression != CacheEventReorderedReplay {
		t.Fatalf("reordered STORE replay = %+v, want reordered-replay suppression", got)
	}

	want := CacheEventConsolidatorCounters{
		PublishedStores:  1,
		PublishedRemoves: 1,
		DuplicateReplays: 1,
		ReorderedReplays: 1,
	}
	if got := c.Counters(); !reflect.DeepEqual(got, want) {
		t.Fatalf("counters = %+v, want %+v", got, want)
	}
	state := c.SnapshotState()
	if len(state.Blocks) != 1 || len(state.Blocks[0].Sources) != 1 || state.Blocks[0].Sources[0].Present {
		t.Fatalf("reordered replay resurrected removed block: %+v", state)
	}
}

func TestCacheEventConsolidatorCountsUnknownRemove(t *testing.T) {
	c := NewCacheEventConsolidator(2)
	block := NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-unknown")

	got := c.Consolidate(cacheVisibilityEvent(block, "source-a", "remove-1", 1, CacheVisibilityRemove))
	if got.Publish || got.Suppression != CacheEventUnknown {
		t.Fatalf("unknown REMOVE decision = %+v, want unknown suppression", got)
	}
	want := CacheEventConsolidatorCounters{UnknownEvents: 1}
	if counters := c.Counters(); !reflect.DeepEqual(counters, want) {
		t.Fatalf("counters = %+v, want %+v", counters, want)
	}
}

func TestCacheEventConsolidatorBoundsNewBlockState(t *testing.T) {
	c := NewCacheEventConsolidator(1)
	first := NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-first")
	overflow := NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-overflow")

	if got := c.Consolidate(cacheVisibilityEvent(first, "source-a", "store-1", 1, CacheVisibilityStore)); !got.Publish {
		t.Fatalf("first block STORE = %+v, want published", got)
	}
	if got := c.Consolidate(cacheVisibilityEvent(overflow, "source-b", "store-1", 1, CacheVisibilityStore)); got.Publish || got.Suppression != CacheEventStateBound {
		t.Fatalf("overflow block STORE = %+v, want state-bound suppression", got)
	}
	if got := c.StateEntries(); got != 1 {
		t.Fatalf("StateEntries = %d, want bounded cardinality 1", got)
	}

	// Hitting the bound must not break transitions for the block already tracked.
	if got := c.Consolidate(cacheVisibilityEvent(first, "source-a", "remove-2", 2, CacheVisibilityRemove)); !got.Publish {
		t.Fatalf("tracked block REMOVE after overflow = %+v, want published", got)
	}
	want := CacheEventConsolidatorCounters{
		PublishedStores:        1,
		PublishedRemoves:       1,
		UnknownEvents:          1,
		StateBoundSuppressions: 1,
	}
	if got := c.Counters(); !reflect.DeepEqual(got, want) {
		t.Fatalf("counters = %+v, want %+v", got, want)
	}
}

func TestCacheEventConsolidatorSnapshotBootstrap(t *testing.T) {
	c := NewCacheEventConsolidator(4)
	blockA := NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-a")
	blockB := NewCacheLogicalBlockKey("Qwen3.8", "qwen-tokenizer", "block-b")

	c.Consolidate(cacheVisibilityEvent(blockB, "source-z", "z-store-7", 7, CacheVisibilityStore))
	c.Consolidate(cacheVisibilityEvent(blockA, "source-b", "b-store-2", 2, CacheVisibilityStore))
	c.Consolidate(cacheVisibilityEvent(blockA, "source-a", "a-store-1", 1, CacheVisibilityStore))
	c.Consolidate(cacheVisibilityEvent(blockA, "source-b", "b-remove-3", 3, CacheVisibilityRemove))

	snapshot := c.SnapshotState()
	if snapshot.Version != cacheEventConsolidatorStateVersion || snapshot.MaxBlocks != 4 {
		t.Fatalf("snapshot header = %+v, want current version and bound 4", snapshot)
	}
	if len(snapshot.Blocks) != 2 || snapshot.Blocks[0].Block != blockA || snapshot.Blocks[1].Block != blockB {
		t.Fatalf("snapshot block order = %+v, want block-a then block-b", snapshot.Blocks)
	}
	if got := snapshot.Blocks[0].Sources; len(got) != 2 ||
		got[0].Identity.SourceID != "source-a" ||
		got[1].Identity.SourceID != "source-b" {
		t.Fatalf("snapshot source order = %+v, want source-a then source-b", got)
	}

	restored, err := RestoreCacheEventConsolidator(snapshot)
	if err != nil {
		t.Fatalf("RestoreCacheEventConsolidator: %v", err)
	}
	if got := restored.SnapshotState(); !reflect.DeepEqual(got, snapshot) {
		t.Fatalf("restored snapshot = %+v, want %+v", got, snapshot)
	}

	// The restored tombstone must continue to reject an older replay, while the
	// remaining resident source may still publish the final REMOVE.
	if got := restored.Consolidate(cacheVisibilityEvent(blockA, "source-b", "b-store-replay", 2, CacheVisibilityStore)); got.Publish || got.Suppression != CacheEventReorderedReplay {
		t.Fatalf("post-bootstrap reordered replay = %+v, want suppression", got)
	}
	if got := restored.Consolidate(cacheVisibilityEvent(blockA, "source-a", "a-remove-2", 2, CacheVisibilityRemove)); !got.Publish {
		t.Fatalf("post-bootstrap final REMOVE = %+v, want published", got)
	}
}

func cacheVisibilityEvent(block CacheLogicalBlockKey, sourceID, eventID string, sequence uint64, action CacheVisibilityAction) CacheVisibilityEvent {
	return CacheVisibilityEvent{
		Identity: CacheSourceEventID{
			SourceID: sourceID,
			EventID:  eventID,
			Sequence: sequence,
		},
		Block:  block,
		Action: action,
	}
}
