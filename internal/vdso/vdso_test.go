package vdso

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"

	// The tier-2 cache stores a result Ref produced by an upstream engine; in these
	// tests we Emit inline-payload results, so the blob backend is not strictly
	// required for the cache path. But the tier-1/tier-3 served() path calls
	// abi.ActiveResolver().Put() to re-store the computed output, so a registered
	// resolver keeps that path on its primary (blob) branch instead of the inline
	// fallback. Blank-import wires abi.ActiveResolver() to a real backend.
	_ "github.com/anthony-chaudhary/fak/internal/blob"
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
