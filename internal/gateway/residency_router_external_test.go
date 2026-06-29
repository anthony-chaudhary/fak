package gateway

import "testing"

func TestPrefixResidencyIndexFoldsExternalVLLMBlockEvents(t *testing.T) {
	idx := NewPrefixResidencyIndex(8)
	batch := ExternalKVCacheEventBatch{
		Worker: "vllm-a",
		Events: []ExternalKVCacheEvent{
			{Type: "BlockStored", BlockHashes: []string{"h1", "h2", "h3"}},
		},
	}
	if got := idx.IngestExternalKVCacheEvents(batch); got != 3 {
		t.Fatalf("stored mutations = %d, want 3 cumulative prefix runs", got)
	}
	if got := idx.Overlap("vllm-a", []string{"h1", "h2", "h3"}); got != 3 {
		t.Fatalf("full vLLM block prefix overlap = %d, want 3", got)
	}
	if got := idx.Overlap("vllm-a", []string{"h1", "h2", "h3", "next"}); got != 3 {
		t.Fatalf("extended request should reuse stored leading prefix, overlap = %d, want 3", got)
	}

	// Removing a middle block must invalidate every resident prefix that depended on
	// it. The worker may still hold h1, but it no longer holds the full h1/h2/h3 run.
	if got := idx.IngestExternalKVCacheEvents(ExternalKVCacheEventBatch{
		Worker: "vllm-a",
		Events: []ExternalKVCacheEvent{{Type: "BlockRemoved", BlockHashes: []string{"h2"}}},
	}); got != 2 {
		t.Fatalf("removed mutations = %d, want 2 prefixes containing h2", got)
	}
	if got := idx.Overlap("vllm-a", []string{"h1", "h2", "h3"}); got != 1 {
		t.Fatalf("stale full-prefix hit survived block eviction, overlap = %d, want 1", got)
	}

	if got := idx.IngestExternalKVCacheEvents(ExternalKVCacheEventBatch{
		Worker: "vllm-a",
		Events: []ExternalKVCacheEvent{{Type: "AllBlocksCleared"}},
	}); got != 1 {
		t.Fatalf("clear mutations = %d, want 1 remaining prefix", got)
	}
	if got := idx.Overlap("vllm-a", []string{"h1"}); got != 0 {
		t.Fatalf("worker still appeared resident after AllBlocksCleared, overlap = %d", got)
	}
}
