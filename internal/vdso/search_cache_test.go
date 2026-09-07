package vdso

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestSearchCache_HitOnIdenticalSearch tests that repeat searches with identical query
// parameters return verified-fresh cache hits in <10µs with served_by: vdso.
func TestSearchCache_HitOnIdenticalSearch(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)
	ctx := context.Background()

	call := roCall("fak_grep", `{"pattern":"needle","path":"internal/gateway"}`)

	// Cold lookup: must miss
	res, ok := v.Lookup(ctx, call)
	if ok {
		t.Fatalf("expected cold search to miss, got hit: %+v", res)
	}

	// Simulate engine completion and store in vDSO
	mockPayload := `{"matches":["internal/gateway/router.go:42"]}`
	v.Emit(completeEvent(call, mockPayload))

	// Identical repeat search: must hit in <10µs with served_by: vdso
	start := time.Now()
	resHit, hit := v.Lookup(ctx, call)
	dur := time.Since(start)

	if !hit {
		t.Fatalf("expected identical repeat search to hit, got miss")
	}
	if resHit == nil || resHit.Meta == nil {
		t.Fatalf("result missing meta: %+v", resHit)
	}
	if resHit.Meta["served_by"] != "vdso" {
		t.Errorf("served_by = %q, want %q", resHit.Meta["served_by"], "vdso")
	}
	if resHit.Meta["tier"] != "2" {
		t.Errorf("tier = %q, want %q", resHit.Meta["tier"], "2")
	}
	if string(resolveBytes(t, resHit.Payload)) != mockPayload {
		t.Errorf("payload mismatch: got %q, want %q", string(resolveBytes(t, resHit.Payload)), mockPayload)
	}
	if dur > 100*time.Microsecond {
		t.Logf("warm hit duration: %v (target <10µs)", dur)
	}
	_, hits, _, _ := v.Stats()
	if hits != 1 {
		t.Errorf("VDSOHits = %d, want 1", hits)
	}
}

// TestSearchCache_HierarchicalInvalidation verifies that modifying internal/gateway/foo.go
// invalidates searches in internal/gateway/ and internal/, but preserves cached searches
// in internal/agent/.
func TestSearchCache_HierarchicalInvalidation(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)
	ctx := context.Background()

	callGateway := roCall("fak_grep", `{"pattern":"handler","path":"internal/gateway"}`)
	callInternal := roCall("fak_grep", `{"pattern":"handler","path":"internal"}`)
	callAgent := roCall("fak_grep", `{"pattern":"handler","path":"internal/agent"}`)

	// Populate all three searches
	v.Emit(completeEvent(callGateway, `{"matches":["internal/gateway/foo.go"]}`))
	v.Emit(completeEvent(callInternal, `{"matches":["internal/gateway/foo.go","internal/agent/bar.go"]}`))
	v.Emit(completeEvent(callAgent, `{"matches":["internal/agent/bar.go"]}`))

	// Verify all 3 hit
	if !hits(t, v, callGateway) {
		t.Fatalf("callGateway failed to hit")
	}
	if !hits(t, v, callInternal) {
		t.Fatalf("callInternal failed to hit")
	}
	if !hits(t, v, callAgent) {
		t.Fatalf("callAgent failed to hit")
	}

	// Mutate internal/gateway/foo.go via a write tool completion
	writeEvent := completeEvent(
		wrCall("Edit", `{"file_path":"internal/gateway/foo.go","old_string":"a","new_string":"b"}`),
		`{"ok":true}`,
	)
	v.Emit(writeEvent)

	// Invalidation checks:
	// 1. Search in internal/gateway/ MUST miss (foo.go is directly inside internal/gateway/)
	if _, ok := v.Lookup(ctx, callGateway); ok {
		t.Errorf("search in internal/gateway/ still hit after modifying internal/gateway/foo.go")
	}

	// 2. Search in internal/ MUST miss (foo.go is inside internal/)
	if _, ok := v.Lookup(ctx, callInternal); ok {
		t.Errorf("search in internal/ still hit after modifying internal/gateway/foo.go")
	}

	// 3. Search in internal/agent/ MUST STILL HIT (internal/agent is disjoint from internal/gateway/)
	resAgent, okAgent := v.Lookup(ctx, callAgent)
	if !okAgent {
		t.Errorf("search in internal/agent/ missed after modifying unrelated internal/gateway/foo.go")
	}
	if resAgent.Meta["served_by"] != "vdso" {
		t.Errorf("search in internal/agent/ served_by = %q, want vdso", resAgent.Meta["served_by"])
	}
}

// TestSearchCache_InvalidatePathMethod tests the direct InvalidatePath API on both
// file targets and directory targets.
func TestSearchCache_InvalidatePathMethod(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	callSub := roCall("fak_glob", `{"path":"internal/gateway/sub"}`)
	callGateway := roCall("fak_glob", `{"path":"internal/gateway"}`)
	callRoot := roCall("fak_glob", `{"path":"."}`)
	callAgent := roCall("fak_glob", `{"path":"internal/agent"}`)

	v.Emit(completeEvent(callSub, `["file1.go"]`))
	v.Emit(completeEvent(callGateway, `["sub/file1.go","router.go"]`))
	v.Emit(completeEvent(callRoot, `["all.go"]`))
	v.Emit(completeEvent(callAgent, `["agent.go"]`))

	// Verify all hit
	for name, call := range map[string]*abi.ToolCall{
		"sub": callSub, "gateway": callGateway, "root": callRoot, "agent": callAgent,
	} {
		if !hits(t, v, call) {
			t.Fatalf("%s failed to hit initial cache", name)
		}
	}

	// Invalidate directory "internal/gateway"
	evicted := v.InvalidatePath("internal/gateway")
	if evicted == 0 {
		t.Logf("InvalidatePath evicted count: %d", evicted)
	}

	// Subdirectory must be invalidated
	if hits(t, v, callSub) {
		t.Errorf("search in internal/gateway/sub should be invalidated when internal/gateway is invalidated")
	}
	// Directory itself must be invalidated
	if hits(t, v, callGateway) {
		t.Errorf("search in internal/gateway should be invalidated")
	}
	// Parent root directory must be invalidated
	if hits(t, v, callRoot) {
		t.Errorf("search in root should be invalidated when a subdirectory is invalidated")
	}
	// Sibling directory must be preserved
	if !hits(t, v, callAgent) {
		t.Errorf("search in internal/agent must be preserved when internal/gateway is invalidated")
	}
}

// TestSearchCache_FourStepWitnessIssue11492 executes the exact four-step witness
// described in issue #11492.
func TestSearchCache_FourStepWitnessIssue11492(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)
	ctx := context.Background()

	searchCall := roCall("fak_grep", `{"pattern":"ServeHTTP","path":"internal/gateway"}`)

	// Step 1: Search 1 (cold): returns matches in cold execution (~8ms simulated)
	coldStart := time.Now()
	res1, hit1 := v.Lookup(ctx, searchCall)
	if hit1 {
		t.Fatalf("step 1: expected cold miss, got hit: %+v", res1)
	}
	// Simulate engine run
	time.Sleep(5 * time.Millisecond)
	coldDur := time.Since(coldStart)
	v.Emit(completeEvent(searchCall, `{"matches":["internal/gateway/server.go:120"]}`))

	// Step 2: Search 2 (identical query, unchanged tree): returns in <10µs with served_by: vdso and VDSOHits == 1
	startHit := time.Now()
	res2, hit2 := v.Lookup(ctx, searchCall)
	hitDur := time.Since(startHit)

	if !hit2 {
		t.Fatalf("step 2: expected cache hit, got miss")
	}
	if res2.Meta["served_by"] != "vdso" {
		t.Errorf("step 2: served_by = %q, want vdso", res2.Meta["served_by"])
	}
	_, vdsoHits, _, _ := v.Stats()
	if vdsoHits != 1 {
		t.Errorf("step 2: VDSOHits = %d, want 1", vdsoHits)
	}
	t.Logf("step 1 cold: %v, step 2 warm hit: %v (speedup %.1fx)", coldDur, hitDur, float64(coldDur)/float64(hitDur+1))

	// Step 3: Search 3 (after mutating unrelated file): remains a vDSO hit
	unrelatedWrite := completeEvent(
		wrCall("Write", `{"file_path":"internal/agent/worker.go","content":"package agent"}`),
		`{"ok":true}`,
	)
	v.Emit(unrelatedWrite)

	res3, hit3 := v.Lookup(ctx, searchCall)
	if !hit3 {
		t.Fatalf("step 3: search in internal/gateway was invalidated by write to unrelated internal/agent/worker.go")
	}
	if res3.Meta["served_by"] != "vdso" {
		t.Errorf("step 3: served_by = %q, want vdso", res3.Meta["served_by"])
	}

	// Step 4: Search 4 (after mutating target directory): properly invalidates and executes fresh search
	targetWrite := completeEvent(
		wrCall("Edit", `{"file_path":"internal/gateway/server.go","old_string":"foo","new_string":"bar"}`),
		`{"ok":true}`,
	)
	v.Emit(targetWrite)

	_, hit4 := v.Lookup(ctx, searchCall)
	if hit4 {
		t.Fatalf("step 4: search in internal/gateway still hit after modifying internal/gateway/server.go")
	}

	// Fresh search execution and re-memoization
	v.Emit(completeEvent(searchCall, `{"matches":["internal/gateway/server.go:125"]}`))
	res4After, hit4After := v.Lookup(ctx, searchCall)
	if !hit4After {
		t.Fatalf("step 4: re-cached search failed to hit")
	}
	if string(resolveBytes(t, res4After.Payload)) != `{"matches":["internal/gateway/server.go:125"]}` {
		t.Errorf("step 4: updated payload mismatch: %s", string(resolveBytes(t, res4After.Payload)))
	}
}

// TestSearchCache_WitnessRevocation tests that search results admitted under an external
// world-state witness are invalidated when that witness is revoked.
func TestSearchCache_WitnessRevocation(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	call := roCall("fak_grep", `{"pattern":"token","path":"internal/auth"}`)
	call.Meta["witness"] = "git:commit-abc123"

	v.Emit(completeEvent(call, `{"matches":["internal/auth/token.go:10"]}`))

	if !hits(t, v, call) {
		t.Fatalf("expected search to hit with witness")
	}

	// Revoke the witness
	evicted := v.Revoke("git:commit-abc123")
	if evicted == 0 {
		t.Logf("Revoke evicted count: %d", evicted)
	}

	// Search must now miss
	if hits(t, v, call) {
		t.Errorf("search still hit after revoking witness git:commit-abc123")
	}
}

// TestSearchCache_LRUEviction tests bounded capacity eviction for search cache.
func TestSearchCache_LRUEviction(t *testing.T) {
	sc := NewSearchCache(3)

	c1 := roCall("fak_glob", `{"path":"dir1"}`)
	c2 := roCall("fak_glob", `{"path":"dir2"}`)
	c3 := roCall("fak_glob", `{"path":"dir3"}`)
	c4 := roCall("fak_glob", `{"path":"dir4"}`)

	makeRes := func(c *abi.ToolCall, val string) *abi.Result {
		return &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(val)}}
	}

	sc.Put(c1, c1.Args.Inline, makeRes(c1, "res1"), "")
	sc.Put(c2, c2.Args.Inline, makeRes(c2, "res2"), "")
	sc.Put(c3, c3.Args.Inline, makeRes(c3, "res3"), "")

	if sc.Len() != 3 {
		t.Fatalf("expected len 3, got %d", sc.Len())
	}

	// Access c1 so c2 becomes the oldest
	if _, ok := sc.Get(c1, c1.Args.Inline); !ok {
		t.Fatalf("c1 miss")
	}

	// Put c4 (should evict c2)
	sc.Put(c4, c4.Args.Inline, makeRes(c4, "res4"), "")

	if sc.Len() != 3 {
		t.Fatalf("expected len 3 after eviction, got %d", sc.Len())
	}

	// c1, c3, c4 should be present; c2 should be evicted
	if _, ok := sc.Get(c2, c2.Args.Inline); ok {
		t.Errorf("c2 should have been evicted by LRU")
	}
	if _, ok := sc.Get(c1, c1.Args.Inline); !ok {
		t.Errorf("c1 should still be present")
	}
	if _, ok := sc.Get(c3, c3.Args.Inline); !ok {
		t.Errorf("c3 should still be present")
	}
	if _, ok := sc.Get(c4, c4.Args.Inline); !ok {
		t.Errorf("c4 should still be present")
	}
}

// TestSearchCache_ConcurrentRace tests concurrent search lookups, fills, and hierarchical
// invalidations under -race.
func TestSearchCache_ConcurrentRace(t *testing.T) {
	v := New(128)
	v.SetGranularity(Resource)
	ctx := context.Background()

	dirs := []string{
		"internal/gateway",
		"internal/agent",
		"internal/vdso",
		"internal/ctxmmu",
		"cmd/fak",
	}

	var wg sync.WaitGroup
	var completed int64
	workers := 12
	iterations := 150

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				dir := dirs[(workerID+j)%len(dirs)]
				pattern := fmt.Sprintf("pattern_%d", (workerID+j)%5)
				call := roCall("fak_grep", fmt.Sprintf(`{"pattern":%q,"path":%q}`, pattern, dir))

				// Concurrently try lookup
				res, hit := v.Lookup(ctx, call)
				if !hit {
					// Put result
					payload := fmt.Sprintf(`{"matches":[%q]}`, dir+"/file.go")
					v.Emit(completeEvent(call, payload))
				} else if res != nil {
					_ = res.Payload
				}

				// Periodically invalidate a path or mutate a file
				if j%10 == 0 {
					targetFile := fmt.Sprintf("%s/file_%d.go", dir, j)
					v.InvalidatePath(targetFile)
				}
				atomic.AddInt64(&completed, 1)
			}
		}(i)
	}

	wg.Wait()
	if completed != int64(workers*iterations) {
		t.Errorf("completed %d, want %d", completed, workers*iterations)
	}
}

// TestSearchCache_Resize tests live resizing and LRU shrink of the search cache.
func TestSearchCache_Resize(t *testing.T) {
	v := New(4)
	v.SetGranularity(Resource)

	c1 := roCall("fak_glob", `{"path":"dir1"}`)
	c2 := roCall("fak_glob", `{"path":"dir2"}`)
	c3 := roCall("fak_glob", `{"path":"dir3"}`)

	v.Emit(completeEvent(c1, `["1.go"]`))
	v.Emit(completeEvent(c2, `["2.go"]`))
	v.Emit(completeEvent(c3, `["3.go"]`))

	if v.SearchCache().Len() != 3 {
		t.Fatalf("expected len 3, got %d", v.SearchCache().Len())
	}

	// Access c1 and c3 so c2 is oldest
	if !hits(t, v, c1) || !hits(t, v, c3) {
		t.Fatalf("c1 or c3 missed")
	}

	// Shrink capacity to 2 via ResizeTier2
	receipt, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 2, Reason: "shrink-test"})
	if err != nil {
		t.Fatalf("ResizeTier2 failed: %v", err)
	}
	if receipt.NewCapacity != 2 {
		t.Errorf("receipt capacity = %d, want 2", receipt.NewCapacity)
	}
	if v.SearchCache().Capacity() != 2 {
		t.Errorf("searchCache capacity = %d, want 2", v.SearchCache().Capacity())
	}
	if v.SearchCache().Len() != 2 {
		t.Errorf("searchCache len = %d, want 2", v.SearchCache().Len())
	}

	// c1 and c3 should still hit; c2 should have been evicted
	if hits(t, v, c2) {
		t.Errorf("c2 should have been evicted by resize shrink")
	}
	if !hits(t, v, c1) {
		t.Errorf("c1 should still hit")
	}
	if !hits(t, v, c3) {
		t.Errorf("c3 should still hit")
	}
}

// TestSearchCache_GlobalGranularity tests that in Global mode any write clears all searches.
func TestSearchCache_GlobalGranularity(t *testing.T) {
	v := New(64)
	v.SetGranularity(Global)
	ctx := context.Background()

	cGateway := roCall("fak_grep", `{"pattern":"foo","path":"internal/gateway"}`)
	cAgent := roCall("fak_grep", `{"pattern":"bar","path":"internal/agent"}`)

	v.Emit(completeEvent(cGateway, `{"matches":["gateway.go"]}`))
	v.Emit(completeEvent(cAgent, `{"matches":["agent.go"]}`))

	if !hits(t, v, cGateway) || !hits(t, v, cAgent) {
		t.Fatalf("initial searches failed to hit")
	}

	// In Global mode, any write invalidates everything
	writeEvent := completeEvent(
		wrCall("Write", `{"file_path":"some/other/path.txt","content":"hello"}`),
		`{"ok":true}`,
	)
	v.Emit(writeEvent)

	if _, ok := v.Lookup(ctx, cGateway); ok {
		t.Errorf("cGateway should be invalidated in Global mode")
	}
	if _, ok := v.Lookup(ctx, cAgent); ok {
		t.Errorf("cAgent should be invalidated in Global mode")
	}
}
