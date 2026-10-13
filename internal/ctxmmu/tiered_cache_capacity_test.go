package ctxmmu

import (
	"errors"
	"slices"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestTieredPromptCache_AdmissionCapacity(t *testing.T) {
	t.Parallel()
	type testCase struct {
		name        string
		config      TieredCacheConfig
		tokenCount  int
		buffer      *CacheBuffer
		replace     bool
		reject      bool
		wantBytes   int64
		wantEvicted bool
	}
	tests := []testCase{
		{name: "token limit", config: TieredCacheConfig{MaxTokens: 4}, tokenCount: 8, reject: true},
		{name: "estimated byte limit", config: TieredCacheConfig{MaxBytes: 8, BytesPerTok: 4}, tokenCount: 3, reject: true},
		{name: "active buffer byte limit", config: TieredCacheConfig{MaxBytes: 8}, tokenCount: 3, buffer: NewCacheBuffer(3, 8, 4), reject: true},
		{name: "oversized same ID", config: TieredCacheConfig{MaxTokens: 4}, tokenCount: 8, replace: true, reject: true},
		{name: "exact token fit evicts resident", config: TieredCacheConfig{MaxTokens: 4}, tokenCount: 4, wantBytes: 8, wantEvicted: true},
		{name: "exact active bytes ignore allocated capacity", config: TieredCacheConfig{MaxBytes: 8, BytesPerTok: 16}, tokenCount: 2, buffer: NewCacheBuffer(2, 16, 4), wantBytes: 8, wantEvicted: true},
		{name: "zero limits", tokenCount: 8, wantBytes: 16},
		{name: "negative limits", config: TieredCacheConfig{MaxTokens: -1, MaxEntries: -1, MaxBytes: -1, BytesPerTok: -1}, tokenCount: 8, wantBytes: 16},
		{name: "successful same ID", config: TieredCacheConfig{MaxTokens: 8, MaxBytes: 16}, tokenCount: 4, replace: true, wantBytes: 8},
	}
	if strconv.IntSize == 64 {
		maxInt := int(^uint(0) >> 1)
		tests = append(tests,
			testCase{name: "unbounded estimated bytes overflow", config: TieredCacheConfig{BytesPerTok: maxInt}, tokenCount: 2, reject: true},
			testCase{name: "unbounded active bytes overflow", tokenCount: 2, buffer: &CacheBuffer{SequenceLen: 2, Capacity: 2, BytesPerTok: maxInt}, reject: true},
		)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := NewTieredPromptCache(tt.config)
			residentTokens := []int{1, 2}
			resident, err := cache.Put(RoleSystem, residentTokens, NewCacheBuffer(2, 2, 2))
			if err != nil {
				t.Fatalf("seed resident: %v", err)
			}
			before, nextID := cache.Stats(), cache.nextID
			tokens := make([]int, tt.tokenCount)
			for i := range tokens {
				tokens[i] = 10 + i
			}
			callerTokens := slices.Clone(tokens)
			var callerData []byte
			var callerLen, callerTrimCount int
			if tt.buffer != nil {
				callerData = slices.Clone(tt.buffer.Data)
				callerLen, callerTrimCount = tt.buffer.Len(), tt.buffer.TrimCount
			}
			id := ""
			if tt.replace {
				id = resident.ID
			}
			entry, err := cache.PutWithID(id, RoleUser, tokens, tt.buffer)
			if !slices.Equal(tokens, callerTokens) {
				t.Fatal("admission changed caller tokens")
			}
			if tt.buffer != nil && (tt.buffer.Len() != callerLen || tt.buffer.TrimCount != callerTrimCount || !slices.Equal(tt.buffer.Data, callerData)) {
				t.Fatal("admission changed caller buffer")
			}
			if tt.reject {
				if entry != nil || !errors.Is(err, ErrCacheCapacityExceeded) {
					t.Fatalf("PutWithID = (%v, %v), want (nil, ErrCacheCapacityExceeded)", entry, err)
				}
				if got := cache.Stats(); got != before || cache.nextID != nextID {
					t.Fatalf("refusal changed occupancy, eviction accounting, or automatic ID: stats=%+v, nextID=%d", got, cache.nextID)
				}
				if got, ok := cache.PeekByID(resident.ID); !ok || got != resident {
					t.Fatal("refusal replaced or evicted the resident")
				}
				if cache.trie.ExactMatch(residentTokens) != resident || cache.trie.ExactMatch(tokens) != nil {
					t.Fatal("refusal changed trie membership")
				}
				return
			}
			if err != nil || entry == nil {
				t.Fatalf("admissible entry = (%v, %v)", entry, err)
			}
			if entry.Buffer != tt.buffer || &entry.Tokens[0] == &tokens[0] {
				t.Fatal("successful admission changed buffer alias or token-copy ownership")
			}
			wantID := "prompt-2"
			if tt.replace {
				wantID = resident.ID
			}
			if entry.ID != wantID || cache.trie.ExactMatch(tokens) != entry {
				t.Fatalf("successful admission ID/trie mismatch: ID=%q, want %q", entry.ID, wantID)
			}
			wantEntries, wantTokens, wantBytes := 1, tt.tokenCount, tt.wantBytes
			var wantResident *PromptEntry
			if !tt.replace && !tt.wantEvicted {
				wantEntries++
				wantTokens += len(residentTokens)
				wantBytes += 4
				wantResident = resident
			}
			var wantEvictions int64
			if tt.wantEvicted {
				wantEvictions = 1
			}
			got := cache.Stats()
			if got.TotalEntries != wantEntries || got.TotalTokens != wantTokens || got.TotalBytes != wantBytes || got.Evictions != wantEvictions {
				t.Fatalf("admission accounting = %+v; want entries=%d tokens=%d bytes=%d evictions=%d", got, wantEntries, wantTokens, wantBytes, wantEvictions)
			}
			if cache.trie.ExactMatch(residentTokens) != wantResident {
				t.Fatal("admission changed normal resident eviction/replacement policy")
			}
		})
	}
}
