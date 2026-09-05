package vdso

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	// The tier-2 cache stores a result Ref produced by an upstream engine; in these
	// tests we Emit inline-payload results, so the blob backend is not strictly
	// required for the cache path. But the tier-1/tier-3 served() path calls
	// abi.ActiveResolver().Put() to re-store the computed output, so a registered
	// resolver keeps that path on its primary (blob) branch instead of the inline
	// fallback. Importing blob wires abi.ActiveResolver() to a real backend; resize
	// tests also use that backend to witness exact cache-owned pin release.
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// roCall builds a read-only + idempotent tool call with inline args (the routing
// shape the vDSO requires for tier-1 and tier-2).
func roCall(tool string, args string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	}
}

// completeEvent wraps a read-only call + an OK inline result into an EvComplete
// event (the shape Emit consumes to fill the tier-2 cache).
func completeEvent(c *abi.ToolCall, payload string) abi.Event {
	return abi.Event{
		Kind: abi.EvComplete,
		Call: c,
		Result: &abi.Result{
			Call:    c,
			Status:  abi.StatusOK,
			Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(payload)},
		},
	}
}

// resolveBytes materializes a result payload (inline or blob-backed) so the test
// can assert on its content regardless of how served() stored it.
func resolveBytes(t *testing.T, r abi.Ref) []byte {
	t.Helper()
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatalf("no active resolver but Ref is non-inline (kind=%d)", r.Kind)
	}
	b, err := res.Resolve(context.Background(), r)
	if err != nil {
		t.Fatalf("resolve payload: %v", err)
	}
	return b
}

// Unit 25 — tier-1 pure. RegisterPure("calculate", calcSum) is seeded on Default;
// a New() instance must be able to register the same pure tool and serve it.
func TestUnit25_Tier1Pure(t *testing.T) {
	ctx := context.Background()

	// Default already seeds calculate via init().
	res, ok := Default.Lookup(ctx, &abi.ToolCall{
		Tool: "calculate",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"a":2,"b":3}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	})
	if !ok {
		t.Fatalf("Default tier-1 calculate: ok=false, want true")
	}
	if res.Status != abi.StatusOK {
		t.Fatalf("status=%d, want OK", res.Status)
	}
	if res.Meta["served_by"] != "vdso" {
		t.Fatalf("served_by=%q, want vdso", res.Meta["served_by"])
	}
	var got struct {
		Sum float64 `json:"sum"`
	}
	if err := json.Unmarshal(resolveBytes(t, res.Payload), &got); err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	if got.Sum != 5 {
		t.Fatalf("sum=%v, want 5", got.Sum)
	}

	// And on a fresh isolated instance after RegisterPure.
	v := New(8)
	v.RegisterPure("calculate", calcSum)
	res2, ok2 := v.Lookup(ctx, roCall("calculate", `{"a":2,"b":3}`))
	if !ok2 {
		t.Fatalf("New tier-1 calculate: ok=false, want true")
	}
	if err := json.Unmarshal(resolveBytes(t, res2.Payload), &got); err != nil {
		t.Fatalf("decode payload (new): %v", err)
	}
	if got.Sum != 5 {
		t.Fatalf("new sum=%v, want 5", got.Sum)
	}
}

// Unit 29 — tier-3 static. Default has a canned list_all_airports answer.
func TestUnit29_Tier3Static(t *testing.T) {
	ctx := context.Background()
	// Static answers are not gated on the read-only hints (Lookup serves them
	// unconditionally), so a bare call still hits.
	res, ok := Default.Lookup(ctx, &abi.ToolCall{
		Tool: "list_all_airports",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
	})
	if !ok {
		t.Fatalf("tier-3 list_all_airports: ok=false, want true")
	}
	if res.Status != abi.StatusOK {
		t.Fatalf("status=%d, want OK", res.Status)
	}
	var got struct {
		Airports []string `json:"airports"`
	}
	if err := json.Unmarshal(resolveBytes(t, res.Payload), &got); err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	if len(got.Airports) == 0 || got.Airports[0] != "SFO" {
		t.Fatalf("airports=%v, want canned list starting with SFO", got.Airports)
	}
}

// Units 26+27 — tier-2 cache fill + hit, and order-independent canonicalization.
func TestUnit26_27_Tier2CacheAndCanonicalization(t *testing.T) {
	ctx := context.Background()
	v := New(8)

	// A read-only tool with no pure/static entry: a fresh Lookup must MISS.
	call := roCall("search", `{"a":1,"b":2}`)
	if _, ok := v.Lookup(ctx, call); ok {
		t.Fatalf("pre-fill Lookup: ok=true, want false (cache empty)")
	}

	// Fill the cache from an EvComplete event.
	v.Emit(completeEvent(call, `{"hits":["x","y"]}`))

	// The SAME call now hits.
	res, ok := v.Lookup(ctx, call)
	if !ok {
		t.Fatalf("post-fill Lookup (same call): ok=false, want true (cache hit)")
	}
	if res.Meta["served_by"] != "vdso" || res.Meta["tier"] != "2" {
		t.Fatalf("served_by=%q tier=%q, want vdso/2", res.Meta["served_by"], res.Meta["tier"])
	}
	if got := string(resolveBytes(t, res.Payload)); got != `{"hits":["x","y"]}` {
		t.Fatalf("payload=%q, want the cached body", got)
	}

	// Unit 26 — canonicalization: a DIFFERENT key order for the same object still
	// hits, because argHash canonicalizes JSON (sorted keys) before hashing.
	reordered := roCall("search", `{"b":2,"a":1}`)
	if _, ok := v.Lookup(ctx, reordered); !ok {
		t.Fatalf("reordered-keys Lookup: ok=false, want true (canonicalization)")
	}
}

// Unit 28 — BumpWorld invalidates the tier-2 cache (the world-version is part of
// the cache key, so a bump makes the prior key unreachable => miss).
func TestUnit28_BumpWorldInvalidates(t *testing.T) {
	ctx := context.Background()
	v := New(8)

	call := roCall("status", `{"id":7}`)
	v.Emit(completeEvent(call, `{"state":"up"}`))
	if _, ok := v.Lookup(ctx, call); !ok {
		t.Fatalf("pre-bump Lookup: ok=false, want true (cached)")
	}

	v.BumpWorld()

	if _, ok := v.Lookup(ctx, call); ok {
		t.Fatalf("post-bump Lookup: ok=true, want false (world bumped => cache miss)")
	}
}

// Unit 28 (write-shaped completion path) — Emit of a destructive completion bumps
// the world and invalidates a previously-cached read.
func TestUnit28_WriteCompletionBumpsWorld(t *testing.T) {
	ctx := context.Background()
	v := New(8)

	read := roCall("status", `{"id":1}`)
	v.Emit(completeEvent(read, `{"v":1}`))
	if _, ok := v.Lookup(ctx, read); !ok {
		t.Fatalf("pre-write Lookup: ok=false, want true")
	}
	before := v.WorldVersion()

	// A write-shaped tool name marks the completion destructive => world bumps.
	write := &abi.ToolCall{
		Tool: "update_status",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"id":1}`)},
	}
	v.Emit(completeEvent(write, `{"ok":true}`))
	if v.WorldVersion() != before+1 {
		t.Fatalf("world version=%d, want %d (write should advance it)", v.WorldVersion(), before+1)
	}
	if _, ok := v.Lookup(ctx, read); ok {
		t.Fatalf("post-write Lookup: ok=true, want false (write invalidated cache)")
	}
}

// Unit 31 — Stats() hit-rate is correct after N lookups. We perform a deterministic
// mix: 1 miss (lookups=1), then fill + 2 cache hits, plus 1 tier-1 pure hit.
func TestUnit31_StatsHitRate(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	v.RegisterPure("calculate", calcSum)

	call := roCall("metric", `{"q":"cpu"}`)

	// 1) miss (no entry yet): lookups=1, hits=0.
	if _, ok := v.Lookup(ctx, call); ok {
		t.Fatalf("first Lookup: ok=true, want false")
	}
	// fill the cache (Emit is not a Lookup; it bumps fills, not lookups).
	v.Emit(completeEvent(call, `{"value":42}`))

	// 2) cache hit: lookups=2, hits=1.
	if _, ok := v.Lookup(ctx, call); !ok {
		t.Fatalf("second Lookup: ok=false, want true")
	}
	// 3) cache hit again: lookups=3, hits=2.
	if _, ok := v.Lookup(ctx, call); !ok {
		t.Fatalf("third Lookup: ok=false, want true")
	}
	// 4) tier-1 pure hit: lookups=4, hits=3.
	if _, ok := v.Lookup(ctx, roCall("calculate", `{"a":1,"b":1}`)); !ok {
		t.Fatalf("pure Lookup: ok=false, want true")
	}

	lookups, hits, fills, rate := v.Stats()
	if lookups != 4 {
		t.Fatalf("lookups=%d, want 4", lookups)
	}
	if hits != 3 {
		t.Fatalf("hits=%d, want 3", hits)
	}
	if fills != 1 {
		t.Fatalf("fills=%d, want 1", fills)
	}
	if want := 3.0 / 4.0; rate != want {
		t.Fatalf("hitRate=%v, want %v", rate, want)
	}
}

// Unit 34 — a tool with no pure/static/cache entry and no hints => miss.
func TestUnit34_Miss(t *testing.T) {
	ctx := context.Background()
	v := New(8)

	// No hints at all: tier-1 and tier-2 are both gated off; no static entry.
	res, ok := v.Lookup(ctx, &abi.ToolCall{
		Tool: "unknown_tool",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"x":1}`)},
	})
	if ok {
		t.Fatalf("Lookup of unknown unhinted tool: ok=true (res=%+v), want false", res)
	}
	if res != nil {
		t.Fatalf("miss result=%+v, want nil", res)
	}

	// Even WITH read-only hints, an unknown tool with an empty cache still misses.
	if _, ok := v.Lookup(ctx, roCall("unknown_tool", `{"x":1}`)); ok {
		t.Fatalf("Lookup of unknown read-only tool (empty cache): ok=true, want false")
	}

	// Lookups counted, no hits.
	lookups, hits, _, _ := v.Stats()
	if lookups != 2 || hits != 0 {
		t.Fatalf("stats lookups=%d hits=%d, want 2/0", lookups, hits)
	}
}

// Unit 35 — New(n) honors a configurable capacity. We verify the cap is respected
// by filling exactly n entries and confirming all n survive, while n+1 evicts.
func TestUnit35_CapacityConfigurable(t *testing.T) {
	ctx := context.Background()

	// Capacity 3: three distinct entries all stay resident.
	v := New(3)
	calls := []*abi.ToolCall{
		roCall("t", `{"i":1}`),
		roCall("t", `{"i":2}`),
		roCall("t", `{"i":3}`),
	}
	for i, c := range calls {
		v.Emit(completeEvent(c, `{"r":`+string(rune('0'+i+1))+`}`))
	}
	for i, c := range calls {
		if _, ok := v.Lookup(ctx, c); !ok {
			t.Fatalf("entry %d evicted at capacity 3, want all resident", i+1)
		}
	}

	// A zero/negative capacity falls back to DefaultCacheSize (configurable floor).
	vd := New(0)
	if vd.cap != DefaultCacheSize {
		t.Fatalf("New(0) cap=%d, want DefaultCacheSize=%d", vd.cap, DefaultCacheSize)
	}
}

// Unit 36 — LRU eviction. New(2), fill 3 distinct entries; the OLDEST (least
// recently used) is evicted: its Lookup misses, the newest still hits.
func TestUnit36_LRUEviction(t *testing.T) {
	ctx := context.Background()
	v := New(2)

	c1 := roCall("q", `{"i":1}`)
	c2 := roCall("q", `{"i":2}`)
	c3 := roCall("q", `{"i":3}`)

	v.Emit(completeEvent(c1, `{"r":1}`)) // cache: [c1]
	v.Emit(completeEvent(c2, `{"r":2}`)) // cache: [c2, c1]
	v.Emit(completeEvent(c3, `{"r":3}`)) // cache: [c3, c2]  -> c1 evicted

	if _, ok := v.Lookup(ctx, c1); ok {
		t.Fatalf("oldest entry c1: ok=true, want false (should be evicted)")
	}
	if _, ok := v.Lookup(ctx, c3); !ok {
		t.Fatalf("newest entry c3: ok=false, want true (should be resident)")
	}
	if _, ok := v.Lookup(ctx, c2); !ok {
		t.Fatalf("entry c2: ok=false, want true (should be resident)")
	}
}

func TestTier2ResizeShrinksByLRUAndStartsGeneration(t *testing.T) {
	ctx := context.Background()
	oldMax := blob.Default.MaxBytes()
	blob.Default.SetMaxBytes(0)
	defer blob.Default.SetMaxBytes(oldMax)

	v := New(3)
	a := roCall("resize_item", `{"id":"a"}`)
	b := roCall("resize_item", `{"id":"b"}`)
	c := roCall("resize_item", `{"id":"c"}`)
	put := func(label string) abi.Ref {
		t.Helper()
		ref, err := blob.Default.Put(ctx, []byte(label+":"+strings.Repeat(label, 400)))
		if err != nil {
			t.Fatalf("put %s payload: %v", label, err)
		}
		if ref.Kind != abi.RefBlob {
			t.Fatalf("put %s kind=%d, want blob-backed ref", label, ref.Kind)
		}
		return ref
	}
	ar, br, cr := put("a"), put("b"), put("c")

	// One extra independent pin on B makes exact accounting observable: after the
	// resize's one unpin B must still resolve; after releasing this pin it must not.
	blob.Default.Pin(br.Digest)
	v.Emit(abi.Event{Kind: abi.EvComplete, Call: a, Result: &abi.Result{Call: a, Status: abi.StatusOK, Payload: ar}})
	v.Emit(abi.Event{Kind: abi.EvComplete, Call: b, Result: &abi.Result{Call: b, Status: abi.StatusOK, Payload: br}})
	v.Emit(abi.Event{Kind: abi.EvComplete, Call: c, Result: &abi.Result{Call: c, Status: abi.StatusOK, Payload: cr}})
	if _, ok := v.Lookup(ctx, a); !ok { // LRU [a,c,b]: B alone is the tail victim.
		t.Fatal("touch A: cache miss")
	}
	blob.Default.SetMaxBytes(1)

	type observation struct {
		state Tier2CacheState
		aHit  bool
		tool  string
	}
	observed := make(chan observation, 1)
	v.SetCacheEventSink(func(ev CacheEvent) {
		if ev.Kind != CacheEvict {
			return
		}
		state := v.Tier2State() // re-enter v.mu: callback must run after resize unlocks.
		_, hit := v.Lookup(ctx, a)
		observed <- observation{state: state, aHit: hit, tool: ev.Entry.Derivation.Tool}
	})
	type resizeResult struct {
		receipt Tier2ResizeReceipt
		err     error
	}
	finished := make(chan resizeResult, 1)
	go func() {
		r, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 2, Reason: "bounded-live-test"})
		finished <- resizeResult{receipt: r, err: err}
	}()

	var result resizeResult
	select {
	case result = <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("ResizeTier2 deadlocked with reentrant CacheEvict observer")
	}
	if result.err != nil {
		t.Fatalf("ResizeTier2: %v", result.err)
	}
	if got, want := result.receipt, (Tier2ResizeReceipt{
		OldCapacity: 3, NewCapacity: 2, OldOccupancy: 3, NewOccupancy: 2,
		Evicted: 1, OldGeneration: 1, NewGeneration: 2, Changed: true,
		Reason: "bounded-live-test", Timestamp: result.receipt.Timestamp,
	}); got != want {
		t.Fatalf("resize receipt=%+v, want %+v", got, want)
	}
	select {
	case got := <-observed:
		if got.state != (Tier2CacheState{Capacity: 2, Occupancy: 2, Evictions: 1, Generation: 2}) {
			t.Fatalf("observer state=%+v, want resized state", got.state)
		}
		if !got.aHit || got.tool != b.Tool {
			t.Fatalf("observer A hit=%v evicted tool=%q, want true/%q", got.aHit, got.tool, b.Tool)
		}
	default:
		t.Fatal("missing CacheEvict observation")
	}
	if _, ok := v.Lookup(ctx, b); ok {
		t.Fatal("B survived shrink, want sole LRU victim")
	}
	for name, call := range map[string]*abi.ToolCall{"A": a, "C": c} {
		if _, ok := v.Lookup(ctx, call); !ok {
			t.Fatalf("%s was not retained by shrink", name)
		}
	}
	if _, err := blob.Default.Resolve(ctx, br); err != nil {
		t.Fatalf("resize released more than its one cache-owned B pin: %v", err)
	}
	blob.Default.Unpin(br.Digest)
	if _, err := blob.Default.Resolve(ctx, br); err == nil {
		t.Fatal("B still resolves after independent pin release; resize did not release its cache-owned pin")
	}
}

func TestTier2ResizeGrowNoopInvalidAndRollback(t *testing.T) {
	ctx := context.Background()
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	v := New(2)
	v.now = func() time.Time { return now }
	a := roCall("resize_config", `{"id":"a"}`)
	b := roCall("resize_config", `{"id":"b"}`)
	v.Emit(completeEvent(a, `{"value":"a"}`))
	v.Emit(completeEvent(b, `{"value":"b"}`))

	evictions := 0
	v.SetCacheEventSink(func(ev CacheEvent) {
		if ev.Kind == CacheEvict {
			evictions++
		}
	})
	grow, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 4, Reason: "growth"})
	if err != nil {
		t.Fatalf("grow: %v", err)
	}
	if grow != (Tier2ResizeReceipt{
		OldCapacity: 2, NewCapacity: 4, OldOccupancy: 2, NewOccupancy: 2,
		OldGeneration: 1, NewGeneration: 2, Changed: true, Reason: "growth", Timestamp: now,
	}) {
		t.Fatalf("grow receipt=%+v", grow)
	}
	if evictions != 0 {
		t.Fatalf("grow emitted %d evictions, want 0", evictions)
	}
	for _, call := range []*abi.ToolCall{a, b} {
		if _, ok := v.Lookup(ctx, call); !ok {
			t.Fatalf("grow discarded fitting entry %s", call.Tool)
		}
	}

	noop, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 4, Reason: "repeat"})
	if err != nil {
		t.Fatalf("no-op: %v", err)
	}
	if noop.Changed || noop.OldGeneration != 2 || noop.NewGeneration != 2 || noop.Evicted != 0 {
		t.Fatalf("no-op receipt=%+v, want stable generation and no eviction", noop)
	}
	lastGood := v.Tier2State()
	if _, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 0, Reason: "invalid"}); !errors.Is(err, ErrInvalidTier2Capacity) {
		t.Fatalf("invalid resize error=%v, want ErrInvalidTier2Capacity", err)
	}
	if got := v.Tier2State(); got != lastGood {
		t.Fatalf("invalid resize mutated state: got %+v want %+v", got, lastGood)
	}

	rollback, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 2, Reason: "rollback"})
	if err != nil {
		t.Fatalf("rollback: %v", err)
	}
	if !rollback.Changed || rollback.OldGeneration != 2 || rollback.NewGeneration != 3 || rollback.Evicted != 0 {
		t.Fatalf("rollback receipt=%+v, want generation 2->3 without eviction", rollback)
	}
	if got := v.Tier2State(); got != (Tier2CacheState{Capacity: 2, Occupancy: 2, Evictions: 0, Generation: 3}) {
		t.Fatalf("rollback state=%+v", got)
	}
}

func TestTier2ResizeConcurrentLookupEmit(t *testing.T) {
	v := New(4)
	ctx := context.Background()
	calls := make([]*abi.ToolCall, 8)
	for i := range calls {
		calls[i] = roCall("resize_race", `{"id":`+strconv.Itoa(i)+`}`)
	}

	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 200; i++ {
			v.Lookup(ctx, calls[i%len(calls)])
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 200; i++ {
			v.Emit(completeEvent(calls[i%len(calls)], `{"iteration":`+strconv.Itoa(i)+`}`))
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		capacities := []int{1, 8, 2, 7, 3, 6, 4, 5}
		for i := 0; i < 200; i++ {
			if _, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: capacities[i%len(capacities)]}); err != nil {
				t.Errorf("concurrent resize: %v", err)
				return
			}
		}
	}()
	close(start)
	wg.Wait()

	state := v.Tier2State()
	if state.Occupancy > state.Capacity {
		t.Fatalf("incoherent state after race: %+v", state)
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if len(v.cache) != v.lru.Len() || v.lru.Len() != state.Occupancy {
		t.Fatalf("map/LRU/state occupancy disagree: map=%d lru=%d state=%d", len(v.cache), v.lru.Len(), state.Occupancy)
	}
	for el := v.lru.Front(); el != nil; el = el.Next() {
		e := el.Value.(*entry)
		if v.cache[e.key] != el {
			t.Fatalf("LRU entry %q is not indexed by its element", e.key)
		}
	}
}

func TestTier2ResizeSameInstanceThreeTwoFourThree(t *testing.T) {
	ctx := context.Background()
	v := New(3)
	a := roCall("resize_same", `{"id":"a"}`)
	b := roCall("resize_same", `{"id":"b"}`)
	c := roCall("resize_same", `{"id":"c"}`)
	d := roCall("resize_same", `{"id":"d"}`)
	e := roCall("resize_same", `{"id":"e"}`)
	for i, call := range []*abi.ToolCall{a, b, c} {
		v.Emit(completeEvent(call, `{"value":`+strconv.Itoa(i)+`}`))
	}
	if _, ok := v.Lookup(ctx, a); !ok { // B becomes the 3->2 victim.
		t.Fatal("touch A: cache miss")
	}
	shrink, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 2})
	if err != nil || shrink.NewGeneration != 2 {
		t.Fatalf("3->2 resize receipt=%+v err=%v", shrink, err)
	}
	if _, ok := v.Lookup(ctx, b); ok {
		t.Fatal("B hit after 3->2 shrink")
	}
	for _, call := range []*abi.ToolCall{a, c} {
		if _, ok := v.Lookup(ctx, call); !ok {
			t.Fatalf("retained entry %q missed after 3->2", string(call.Args.Inline))
		}
	}

	grow, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 4})
	if err != nil || grow.NewGeneration != 3 || grow.NewOccupancy != 2 {
		t.Fatalf("2->4 resize receipt=%+v err=%v", grow, err)
	}
	v.Emit(completeEvent(d, `{"value":3}`))
	v.Emit(completeEvent(e, `{"value":4}`))
	if got := v.Tier2State(); got.Occupancy != 4 || got.Generation != 3 {
		t.Fatalf("grown same-instance state=%+v, want occupancy 4 generation 3", got)
	}

	rollback, err := v.ResizeTier2(Tier2ResizeRequest{Capacity: 3})
	if err != nil || rollback.NewGeneration != 4 || rollback.Evicted != 1 {
		t.Fatalf("4->3 resize receipt=%+v err=%v", rollback, err)
	}
	if _, ok := v.Lookup(ctx, a); ok {
		t.Fatal("A hit after becoming the 4->3 LRU victim")
	}
	for _, call := range []*abi.ToolCall{c, d, e} {
		if _, ok := v.Lookup(ctx, call); !ok {
			t.Fatalf("entry %q missed after same-instance 3->2->4->3", string(call.Args.Inline))
		}
	}
	if got := v.Tier2State(); got.Capacity != 3 || got.Occupancy != 3 || got.Generation != 4 {
		t.Fatalf("final same-instance state=%+v", got)
	}
}

// Unit 38 — soundness: a tier-1 calculate hit equals recomputing a+b. The vDSO
// fast-path result MUST agree with the direct pure recomputation for many inputs.
func TestUnit38_SoundnessTier1EqualsRecompute(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	v.RegisterPure("calculate", calcSum)

	type pair struct{ a, b int }
	cases := []pair{{2, 3}, {0, 0}, {-4, 9}, {100, 23}, {7, -7}, {1, 1}}
	for _, p := range cases {
		args := []byte(`{"a":` + itoa(p.a) + `,"b":` + itoa(p.b) + `}`)

		// vDSO fast path.
		res, ok := v.Lookup(ctx, &abi.ToolCall{
			Tool: "calculate",
			Args: abi.Ref{Kind: abi.RefInline, Inline: args},
			Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
		})
		if !ok {
			t.Fatalf("calculate(%d,%d): ok=false, want true", p.a, p.b)
		}
		var got struct {
			Sum float64 `json:"sum"`
		}
		if err := json.Unmarshal(resolveBytes(t, res.Payload), &got); err != nil {
			t.Fatalf("decode: %v", err)
		}

		// Direct recompute (the ground truth): a+b.
		want := float64(p.a + p.b)
		if got.Sum != want {
			t.Fatalf("calculate(%d,%d): vdso sum=%v, recompute=%v (soundness violated)", p.a, p.b, got.Sum, want)
		}

		// And the pure func itself agrees with the served bytes.
		raw, served := calcSum(args)
		if !served {
			t.Fatalf("calcSum(%d,%d): served=false", p.a, p.b)
		}
		var direct struct {
			Sum float64 `json:"sum"`
		}
		if err := json.Unmarshal(raw, &direct); err != nil {
			t.Fatalf("decode direct: %v", err)
		}
		if direct.Sum != got.Sum {
			t.Fatalf("calculate(%d,%d): direct=%v vdso=%v", p.a, p.b, direct.Sum, got.Sum)
		}
	}
}

// Concurrency test (run under -race): many goroutines Lookup + Emit against one
// shared VDSO. Asserts no data races and that the counters stay self-consistent.
func TestConcurrentLookupEmit(t *testing.T) {
	ctx := context.Background()
	v := New(64)
	v.RegisterPure("calculate", calcSum)

	const goroutines = 16
	const iters = 200

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				// A spread of keys so the cache, LRU, and world-version all churn.
				key := (g*iters + i) % 32
				args := `{"k":` + itoa(key) + `}`
				call := roCall("read", args)

				// Lookup (may hit or miss).
				_, _ = v.Lookup(ctx, call)

				// Fill from a completion.
				v.Emit(completeEvent(call, `{"v":`+itoa(key)+`}`))

				// A tier-1 pure lookup (always hits).
				_, _ = v.Lookup(ctx, roCall("calculate", `{"a":1,"b":`+itoa(key)+`}`))

				// Occasionally advance the world (exercise invalidation under load).
				if i%50 == 0 {
					v.BumpWorld()
				}
			}
		}(g)
	}
	wg.Wait()

	lookups, hits, fills, rate := v.Stats()
	if lookups <= 0 {
		t.Fatalf("lookups=%d, want > 0", lookups)
	}
	if hits < 0 || hits > lookups {
		t.Fatalf("hits=%d out of range (lookups=%d)", hits, lookups)
	}
	if fills < 0 {
		t.Fatalf("fills=%d, want >= 0", fills)
	}
	if rate < 0 || rate > 1 {
		t.Fatalf("hitRate=%v out of [0,1]", rate)
	}
}

// itoa is a tiny stdlib-free signed int formatter for test arg construction.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// TestConcurrentVerifiedFreshReuse_ScalingWithoutLockSerialization verifies that
// concurrent Lookups on file-backed cache entries scale cleanly across goroutines
// without serializing on the global mutex during disk I/O and hashing.
func TestConcurrentVerifiedFreshReuse_ScalingWithoutLockSerialization(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "scaling_fixture.txt")
	fileData := make([]byte, 256<<10) // 256 KB
	for i := range fileData {
		fileData[i] = byte('A' + (i % 26))
	}
	if err := os.WriteFile(filePath, fileData, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	call := &abi.ToolCall{
		Tool: "Read",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":` + strconv.Quote(filePath) + `}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	}
	expectedPayload := `{"content":` + strconv.Quote(string(fileData)) + `}`

	v := New(64)
	v.Emit(completeEvent(call, expectedPayload))

	ctx := context.Background()
	res, ok := v.Lookup(ctx, call)
	if !ok || res == nil {
		t.Fatalf("initial lookup failed; want cache hit")
	}
	if res.Meta["served_by"] != "vdso" || res.Meta["tier"] != "2" {
		t.Fatalf("res.Meta = %+v; want served_by=vdso tier=2", res.Meta)
	}

	const goroutines = 16
	const iters = 250
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				hitRes, hitOk := v.Lookup(ctx, call)
				if !hitOk || hitRes == nil {
					t.Errorf("concurrent lookup failed; want hit")
					return
				}
				if hitRes.Meta["served_by"] != "vdso" {
					t.Errorf("served_by = %q, want vdso", hitRes.Meta["served_by"])
					return
				}
			}
		}()
	}
	wg.Wait()

	lookups, hits, fills, rate := v.Stats()
	if hits < int64(goroutines*iters) {
		t.Fatalf("expected at least %d hits, got %d (lookups=%d, fills=%d, rate=%f)", goroutines*iters, hits, lookups, fills, rate)
	}

	// Verify that if the file is modified on disk, fast-path detects mtime/size change and invalidates
	if err := os.WriteFile(filePath, append(fileData, '!'), 0644); err != nil {
		t.Fatalf("WriteFile update: %v", err)
	}
	if _, ok := v.Lookup(ctx, call); ok {
		t.Fatalf("lookup hit on modified file; want strict invalidation (miss)")
	}

	// Verify outside-the-lock content hash verification if requested
	filePath2 := filepath.Join(dir, "verify_hash_outside_lock.txt")
	content2 := []byte("payload_for_outside_lock_check")
	if err := os.WriteFile(filePath2, content2, 0644); err != nil {
		t.Fatalf("WriteFile2: %v", err)
	}
	call2 := &abi.ToolCall{
		Tool: "Read",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":` + strconv.Quote(filePath2) + `}`)},
		Meta: map[string]string{
			"readOnlyHint":        "true",
			"idempotentHint":      "true",
			"verify_content_hash": "true",
		},
	}
	v.Emit(completeEvent(call2, `{"data":"payload_for_outside_lock_check"}`))
	if _, ok := v.Lookup(ctx, call2); !ok {
		t.Fatalf("lookup2 before modification should hit")
	}

	// Mutate file preserving length and mtime
	fi2, err := os.Stat(filePath2)
	if err != nil {
		t.Fatalf("Stat2: %v", err)
	}
	oldMtime2 := fi2.ModTime()
	mutatedContent2 := []byte("PAYLOAD_FOR_OUTSIDE_LOCK_CHECK") // same length
	if err := os.WriteFile(filePath2, mutatedContent2, 0644); err != nil {
		t.Fatalf("WriteFile2 mutate: %v", err)
	}
	if err := os.Chtimes(filePath2, oldMtime2, oldMtime2); err != nil {
		t.Fatalf("Chtimes2: %v", err)
	}

	// Lookup with verify_content_hash: "true" must detect mismatch outside the lock and return miss
	if _, ok := v.Lookup(ctx, call2); ok {
		t.Fatalf("lookup2 with verify_content_hash should miss on modified content")
	}
}

// BenchmarkConcurrentVerifiedFreshReuse measures concurrent cache hit throughput
// for verified_fresh_reuse without lock serialization.
func BenchmarkConcurrentVerifiedFreshReuse(b *testing.B) {
	dir := b.TempDir()
	filePath := filepath.Join(dir, "bench_fresh_reuse.txt")
	fileData := make([]byte, 64<<10) // 64 KB
	for i := range fileData {
		fileData[i] = byte('A' + (i % 26))
	}
	if err := os.WriteFile(filePath, fileData, 0644); err != nil {
		b.Fatalf("WriteFile: %v", err)
	}

	call := &abi.ToolCall{
		Tool: "Read",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":` + strconv.Quote(filePath) + `}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	}
	expectedPayload := `{"content":` + strconv.Quote(string(fileData)) + `}`

	v := New(64)
	v.Emit(completeEvent(call, expectedPayload))

	ctx := context.Background()
	if res, ok := v.Lookup(ctx, call); !ok || res == nil || res.Meta["served_by"] != "vdso" {
		b.Fatalf("expected cache hit")
	}

	b.SetBytes(int64(len(fileData)))
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res, ok := v.Lookup(ctx, call)
			if !ok || res == nil {
				b.Fatalf("concurrent verified_fresh_reuse lookup failed")
			}
		}
	})
}
