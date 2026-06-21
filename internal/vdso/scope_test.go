package vdso

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"

	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// wrCall builds a write-shaped tool call (destructive is decided by the tool NAME,
// so no hints are needed). Inline args so entityOf can extract the route/entity.
func wrCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

// fillAndExpectHit fills the cache from a read's completion and asserts it then hits.
func fillAndExpectHit(t *testing.T, v *VDSO, c *abi.ToolCall, payload string) {
	t.Helper()
	v.Emit(completeEvent(c, payload))
	if _, ok := v.Lookup(context.Background(), c); !ok {
		t.Fatalf("post-fill Lookup(%s): miss, want hit", c.Tool)
	}
}

func hits(t *testing.T, v *VDSO, c *abi.ToolCall) bool {
	t.Helper()
	_, ok := v.Lookup(context.Background(), c)
	return ok
}

const (
	sfoJFK = `{"origin":"SFO","destination":"JFK","date":"2026-07-01"}`
	laxORD = `{"origin":"LAX","destination":"ORD","date":"2026-07-01"}`
)

// Namespace granularity: a flights write invalidates flights reads but leaves a
// different-namespace (users) read warm. The cross-namespace win.
func TestScope_Namespace_SparesOtherNamespace(t *testing.T) {
	v := New(32)
	v.SetGranularity(Namespace)

	flight := roCall("search_direct_flight", sfoJFK)
	user := roCall("get_user_details", `{"user_id":"u1"}`)
	fillAndExpectHit(t, v, flight, `{"flights":["AA1"]}`)
	fillAndExpectHit(t, v, user, `{"name":"Ada"}`)

	// A booking bumps only the "flights" namespace epoch.
	v.Emit(completeEvent(wrCall("book_flight", `{"user_id":"u1","flight_id":"AA1","origin":"SFO","destination":"JFK"}`), `{"ok":true}`))

	if hits(t, v, flight) {
		t.Errorf("flights read still hits after a flights write (namespace not invalidated)")
	}
	if !hits(t, v, user) {
		t.Errorf("users read MISSED after a flights write — a flight booking must not erase user reads")
	}
}

// Resource granularity: a write on route SFO-JFK invalidates that route's read but
// leaves a DIFFERENT route (LAX-ORD) in the same namespace warm. The within-namespace
// win — the "finer eraser" that pushes the write-rate crossover out by ~1/pool.
func TestScope_Resource_SparesOtherRoute(t *testing.T) {
	v := New(32)
	v.SetGranularity(Resource)

	r1 := roCall("search_direct_flight", sfoJFK)
	r2 := roCall("search_direct_flight", laxORD)
	fillAndExpectHit(t, v, r1, `{"flights":["AA1"]}`)
	fillAndExpectHit(t, v, r2, `{"flights":["UA9"]}`)

	// Book SFO-JFK: bumps only "flights:SFO-JFK".
	v.Emit(completeEvent(wrCall("book_flight", `{"user_id":"u1","flight_id":"AA1","origin":"SFO","destination":"JFK"}`), `{"ok":true}`))

	if hits(t, v, r1) {
		t.Errorf("booked route SFO-JFK still hits after its booking (SOUNDNESS: must be invalidated)")
	}
	if !hits(t, v, r2) {
		t.Errorf("unbooked route LAX-ORD MISSED after an SFO-JFK booking — the finer eraser must spare it")
	}
}

// Soundness: a write on a route ALWAYS invalidates that same route's read at every
// granularity — the finer eraser never under-invalidates the entry it must clear.
func TestScope_Soundness_SameRouteAlwaysInvalidated(t *testing.T) {
	for _, g := range []Granularity{Global, Namespace, Resource} {
		v := New(32)
		v.SetGranularity(g)
		r := roCall("search_direct_flight", sfoJFK)
		fillAndExpectHit(t, v, r, `{"flights":["AA1"]}`)
		v.Emit(completeEvent(wrCall("book_flight", `{"flight_id":"AA1","origin":"SFO","destination":"JFK"}`), `{"ok":true}`))
		if hits(t, v, r) {
			t.Errorf("[%s] booked route still hits after its own booking — stale serve (soundness violated)", g)
		}
	}
}

// Soundness: a write that CANNOT name its row (no origin/destination) falls back UP
// the chain to the namespace, so it still invalidates a fine read bound at the leaf.
// A coarse write must catch fine reads — the ancestor/subtree guarantee.
func TestScope_Soundness_CoarseWriteCatchesFineRead(t *testing.T) {
	v := New(32)
	v.SetGranularity(Resource)

	r := roCall("search_direct_flight", sfoJFK) // bound at [*, flights, flights:SFO-JFK]
	fillAndExpectHit(t, v, r, `{"flights":["AA1"]}`)

	// A booking with no origin/destination: writeTags can only name "flights".
	v.Emit(completeEvent(wrCall("book_flight", `{"flight_id":"AA1"}`), `{"ok":true}`))

	if hits(t, v, r) {
		t.Errorf("fine read survived a coarse (namespace-only) write — ancestor invalidation failed (stale serve)")
	}
}

// Soundness: a write to an UNKNOWN namespace degrades to a full flush (bumps root),
// so it invalidates every read regardless of namespace. Never silently under-scoped.
func TestScope_Soundness_UnknownWriteFlushesAll(t *testing.T) {
	v := New(32)
	v.SetGranularity(Resource)

	flight := roCall("search_direct_flight", sfoJFK)
	user := roCall("get_user_details", `{"user_id":"u1"}`)
	fillAndExpectHit(t, v, flight, `{"flights":["AA1"]}`)
	fillAndExpectHit(t, v, user, `{"name":"Ada"}`)

	// send_email maps to no known namespace => writeTags == ["*"] => global flush.
	v.Emit(completeEvent(wrCall("send_email", `{"to":"x@y.z"}`), `{"ok":true}`))

	if hits(t, v, flight) || hits(t, v, user) {
		t.Errorf("an unknown-namespace write must flush ALL reads (degrade to global), but something survived")
	}
}

// Regression (review blocker): a single-endpoint flights read names no full route, so
// entityOf returns "" and the resource-mode cacheability gate refuses to tier-2 it. A
// prefix leaf "flights:SFO" would be a SIBLING of the route leaf "flights:SFO-JFK"
// (neither an ancestor of the other), so a booking on SFO-JFK would NOT bump it — a
// stale serve. Not caching it closes that hole soundly: the read always reaches the
// engine (fresh). At namespace granularity the same read DOES cache and is correctly
// invalidated by a flights write (it binds [*, flights], which the write bumps).
func TestScope_Soundness_SingleEndpointReadNotCachedResource(t *testing.T) {
	v := New(32)
	v.SetGranularity(Resource)
	originOnly := roCall("search_direct_flight", `{"origin":"SFO"}`)

	// A fill attempt must NOT store an un-nameable-entity read.
	v.Emit(completeEvent(originOnly, `{"flights":["AA1","AA2"]}`))
	if hits(t, v, originOnly) {
		t.Fatalf("single-endpoint read was tier-2 cached at resource granularity — it must not be (would be stale after a route booking)")
	}
	// And after a booking on a route through SFO it still must not serve a stale body.
	v.Emit(completeEvent(wrCall("book_flight", `{"origin":"SFO","destination":"JFK"}`), `{"ok":true}`))
	if hits(t, v, originOnly) {
		t.Errorf("single-endpoint read served from cache after an SFO-JFK booking (stale serve)")
	}

	// Namespace granularity DOES cache it (binds [*, flights]) and a flights write
	// correctly invalidates it — the gate is resource-specific, not a blanket refusal.
	v.SetGranularity(Namespace)
	v.Emit(completeEvent(originOnly, `{"flights":["AA1"]}`))
	if !hits(t, v, originOnly) {
		t.Fatalf("namespace granularity should cache a single-endpoint flights read")
	}
	v.Emit(completeEvent(wrCall("book_flight", `{"origin":"SFO","destination":"JFK"}`), `{"ok":true}`))
	if hits(t, v, originOnly) {
		t.Errorf("a flights write must invalidate the namespace-bound single-endpoint read")
	}
}

// Regression: Global granularity is byte-for-byte the v0.1 behavior — ANY write
// strands EVERY entry, even across unrelated namespaces.
func TestScope_Global_AnyWriteFlushesEverything(t *testing.T) {
	v := New(32) // default granularity is Global

	flight := roCall("search_direct_flight", sfoJFK)
	user := roCall("get_user_details", `{"user_id":"u1"}`)
	fillAndExpectHit(t, v, flight, `{"flights":["AA1"]}`)
	fillAndExpectHit(t, v, user, `{"name":"Ada"}`)

	v.Emit(completeEvent(wrCall("book_flight", `{"flight_id":"AA1","origin":"SFO","destination":"JFK"}`), `{"ok":true}`))

	if hits(t, v, flight) || hits(t, v, user) {
		t.Errorf("Global mode must flush everything on any write (v0.1 behavior), but a read survived")
	}
}

// BumpWorld remains the panic button at every granularity: it bumps the root epoch,
// which every read binds, so it invalidates the whole cache (the trial-isolation
// reset the fleet benchmark relies on).
func TestScope_BumpWorldFlushesAtEveryGranularity(t *testing.T) {
	for _, g := range []Granularity{Global, Namespace, Resource} {
		v := New(32)
		v.SetGranularity(g)
		r1 := roCall("search_direct_flight", sfoJFK)
		r2 := roCall("search_direct_flight", laxORD)
		fillAndExpectHit(t, v, r1, `{"flights":["AA1"]}`)
		fillAndExpectHit(t, v, r2, `{"flights":["UA9"]}`)
		v.BumpWorld()
		if hits(t, v, r1) || hits(t, v, r2) {
			t.Errorf("[%s] BumpWorld must invalidate every entry (root epoch bump), but a read survived", g)
		}
	}
}

// The coherence bus: a write-shaped completion publishes one typed Mutation naming
// exactly the tags that scoped the invalidation; cancel unsubscribes.
func TestScope_CoherenceBus_PublishesMutation(t *testing.T) {
	v := New(32)
	v.SetGranularity(Resource)

	var got []Mutation
	cancel := v.Subscribe(func(m Mutation) { got = append(got, m) })

	v.Emit(completeEvent(wrCall("book_flight", `{"flight_id":"AA1","origin":"SFO","destination":"JFK"}`), `{"ok":true}`))

	if len(got) != 1 {
		t.Fatalf("subscriber received %d mutations, want 1", len(got))
	}
	m := got[0]
	if m.Tool != "book_flight" {
		t.Errorf("mutation tool=%q, want book_flight", m.Tool)
	}
	if len(m.Tags) != 1 || m.Tags[0] != "flights:SFO-JFK" {
		t.Errorf("mutation tags=%v, want [flights:SFO-JFK] (the invalidation scope)", m.Tags)
	}
	if m.Seq != 1 {
		t.Errorf("mutation seq=%d, want 1 (first mutation)", m.Seq)
	}

	// After cancel, no further deliveries.
	cancel()
	v.Emit(completeEvent(wrCall("book_flight", `{"flight_id":"BB2","origin":"LAX","destination":"ORD"}`), `{"ok":true}`))
	if len(got) != 1 {
		t.Errorf("subscriber kept receiving after cancel: got %d, want 1", len(got))
	}

	// The mutation counter still advances (the bus observed both writes).
	if v.Mutations() != 2 {
		t.Errorf("Mutations()=%d, want 2 (both writes observed)", v.Mutations())
	}
}

// A namespace write must publish the namespace tag, not a row tag (the feed names
// the true invalidation scope so observers can react at the right granularity).
func TestScope_CoherenceBus_NamespaceScope(t *testing.T) {
	v := New(32)
	v.SetGranularity(Namespace)
	var tags []string
	v.Subscribe(func(m Mutation) { tags = m.Tags })
	v.Emit(completeEvent(wrCall("book_flight", `{"origin":"SFO","destination":"JFK"}`), `{"ok":true}`))
	if len(tags) != 1 || tags[0] != "flights" {
		t.Errorf("namespace-mode mutation tags=%v, want [flights]", tags)
	}
}

func TestScope_NodeEpochTableBoundedAndFlushesOnTrim(t *testing.T) {
	v := New(32)
	v.SetGranularity(Resource)
	v.SetNodeEpochLimit(2)
	if v.NodeEpochLimit() != 2 {
		t.Fatalf("NodeEpochLimit=%d, want 2", v.NodeEpochLimit())
	}

	user := roCall("get_user_details", `{"user_id":"u1"}`)
	fillAndExpectHit(t, v, user, `{"name":"Ada"}`)
	before := v.WorldVersion()

	var got []Mutation
	v.Subscribe(func(m Mutation) { got = append(got, m) })

	v.Emit(completeEvent(wrCall("book_flight", `{"origin":"SFO","destination":"JFK"}`), `{"ok":true}`))
	v.Emit(completeEvent(wrCall("book_flight", `{"origin":"LAX","destination":"ORD"}`), `{"ok":true}`))
	if got := v.NodeEpochs(); got != 2 {
		t.Fatalf("NodeEpochs before overflow = %d, want 2", got)
	}
	if !hits(t, v, user) {
		t.Fatalf("unrelated user read missed before node-table overflow")
	}

	v.Emit(completeEvent(wrCall("book_flight", `{"origin":"BOS","destination":"SEA"}`), `{"ok":true}`))
	if got := v.NodeEpochs(); got != 2 {
		t.Fatalf("NodeEpochs after overflow = %d, want cap 2", got)
	}
	if v.WorldVersion() != before+1 {
		t.Fatalf("WorldVersion after node trim = %d, want %d", v.WorldVersion(), before+1)
	}
	if hits(t, v, user) {
		t.Fatalf("old user read still hit after node-table trim; root fallback flush did not strand old keys")
	}
	if len(got) != 3 || !intersects(got[2].Tags, []string{rootTag}) {
		t.Fatalf("overflow mutation tags = %v, want root fallback tag", got)
	}
}

// Concurrency under -race: many goroutines lookup/fill/write across routes while a
// subscriber drains the bus. Asserts no data race and the mutation count is exact.
func TestScope_ConcurrentScopedRace(t *testing.T) {
	v := New(256)
	v.SetGranularity(Resource)
	var mu sync.Mutex
	seen := 0
	v.Subscribe(func(m Mutation) { mu.Lock(); seen++; mu.Unlock() })

	const goroutines, iters = 8, 100
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			ctx := context.Background()
			routes := []string{sfoJFK, laxORD, `{"origin":"BOS","destination":"SEA"}`}
			for i := 0; i < iters; i++ {
				rd := roCall("search_direct_flight", routes[(g+i)%len(routes)])
				_, _ = v.Lookup(ctx, rd)
				v.Emit(completeEvent(rd, `{"flights":["X"]}`))
				if i%10 == 0 {
					v.Emit(completeEvent(wrCall("book_flight", routes[(g+i)%len(routes)]), `{"ok":true}`))
				}
			}
		}(g)
	}
	wg.Wait()

	wantWrites := int64(goroutines * (iters/10 + boolToInt(iters%10 != 0)))
	if v.Mutations() != wantWrites {
		t.Fatalf("Mutations()=%d, want %d", v.Mutations(), wantWrites)
	}
	mu.Lock()
	defer mu.Unlock()
	if int64(seen) != wantWrites {
		t.Errorf("subscriber saw %d mutations, want %d (bus dropped events)", seen, wantWrites)
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func intersects(a, b []string) bool {
	for _, x := range a {
		for _, y := range b {
			if x == y {
				return true
			}
		}
	}
	return false
}

// The over-approximation invariant, asserted directly on the tag functions (not via
// the cache): for every (read, write) pair that touches the SAME resource, the write's
// bumped tag set must INTERSECT the read's bound chain at every finer granularity —
// otherwise a write could fail to invalidate a read it changed (a silent stale serve).
// And a join (multi-namespace) or unknown tool must degrade to root-only scoping.
func TestScope_OverApproximationInvariant(t *testing.T) {
	v := New(8)
	type pair struct{ readTool, readArgs, writeTool, writeArgs string }
	pairs := []pair{
		{"search_direct_flight", sfoJFK, "book_flight", `{"origin":"SFO","destination":"JFK"}`},
		{"search_direct_flight", sfoJFK, "cancel_flight", `{"origin":"SFO","destination":"JFK"}`},
		{"get_user_details", `{"user_id":"u1"}`, "update_user", `{"user_id":"u1"}`},
	}
	for _, g := range []Granularity{Namespace, Resource} {
		v.SetGranularity(g)
		for _, p := range pairs {
			rc := v.readChain(&abi.ToolCall{Tool: p.readTool}, []byte(p.readArgs))
			wt := v.writeTags(&abi.ToolCall{Tool: p.writeTool}, []byte(p.writeArgs))
			if !intersects(rc, wt) {
				t.Errorf("[%s] write %s%s shares no tag with read %s%s (chain=%v tags=%v): the write could not invalidate the read",
					g, p.writeTool, p.writeArgs, p.readTool, p.readArgs, rc, wt)
			}
		}
	}

	// Join / unknown tools must collapse to the root tag at every finer granularity
	// (a single chain cannot soundly express a multi-resource dependency).
	v.SetGranularity(Resource)
	for _, tool := range []string{"price_flight_in_currency", "frobnicate_widget", "send_email"} {
		rc := v.readChain(&abi.ToolCall{Tool: tool}, []byte(`{}`))
		if len(rc) != 1 || rc[0] != rootTag {
			t.Errorf("read chain for join/unknown tool %q = %v, want [*] (sound degrade)", tool, rc)
		}
		wt := v.writeTags(&abi.ToolCall{Tool: tool}, []byte(`{}`))
		if len(wt) != 1 || wt[0] != rootTag {
			t.Errorf("write tags for join/unknown tool %q = %v, want [*] (sound degrade)", tool, wt)
		}
	}
}
