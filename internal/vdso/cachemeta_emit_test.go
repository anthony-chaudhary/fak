package vdso

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"

	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// collectEvents installs a sink that appends every tier-2 lifecycle event (under a
// mutex, since the sink contract allows re-entry / concurrent dispatch).
func collectEvents(v *VDSO) (*[]CacheEvent, *sync.Mutex) {
	var mu sync.Mutex
	out := []CacheEvent{}
	v.SetCacheEventSink(func(ev CacheEvent) {
		mu.Lock()
		out = append(out, ev)
		mu.Unlock()
	})
	return &out, &mu
}

// §2.5: vDSO tier-2 fill/hit/eviction/revocation must emit first-class cachemeta
// entries in the same stream as tool/context entries.
func TestCacheEmission_FillHitEvictRevoke(t *testing.T) {
	v := New(2) // small cap so a third fill evicts one
	events, mu := collectEvents(v)
	ctx := context.Background()

	a := roCallW("get_doc", `{"id":"a"}`, "etag:A")
	b := roCallW("get_doc", `{"id":"b"}`, "etag:B")
	c := roCallW("get_doc", `{"id":"c"}`, "etag:C")

	v.Emit(completeEvent(a, `{"body":"A"}`)) // fill A
	v.Emit(completeEvent(b, `{"body":"B"}`)) // fill B
	if _, ok := v.Lookup(ctx, a); !ok {      // hit A
		t.Fatalf("expected A to hit")
	}
	v.Emit(completeEvent(c, `{"body":"C"}`)) // fill C -> evicts the LRU (A)

	mu.Lock()
	got := append([]CacheEvent(nil), *events...)
	mu.Unlock()

	count := func(k CacheEventKind) int {
		n := 0
		for _, e := range got {
			if e.Kind == k {
				n++
			}
		}
		return n
	}
	if count(CacheFill) != 3 {
		t.Fatalf("expected 3 fill events, got %d (%+v)", count(CacheFill), got)
	}
	if count(CacheHit) != 1 {
		t.Fatalf("expected 1 hit event, got %d", count(CacheHit))
	}
	if count(CacheEvict) != 1 {
		t.Fatalf("expected 1 evict event (cap=2, third fill), got %d", count(CacheEvict))
	}

	// Every emitted entry must be a cachemeta tool_result entry carrying the tool
	// and (for witnessed fills) the external witness + refutation invalidation.
	for _, e := range got {
		if e.Entry.Plane != cachemeta.PlaneToolResult {
			t.Fatalf("emitted entry not on tool_result plane: %s", e.Entry.Plane)
		}
		if e.Entry.Derivation.Tool != "get_doc" {
			t.Fatalf("emitted entry tool wrong: %q", e.Entry.Derivation.Tool)
		}
	}
	var fillA *CacheEvent
	for i := range got {
		if got[i].Kind == CacheFill && got[i].Entry.Validity.Witness == "etag:A" {
			fillA = &got[i]
		}
	}
	if fillA == nil {
		t.Fatalf("no fill event carried witness etag:A")
	}
	if fillA.Entry.Coherence.InvalidationMode != cachemeta.InvalidationExternalRefutation {
		t.Fatalf("witnessed fill should invalidate on refutation, got %q", fillA.Entry.Coherence.InvalidationMode)
	}

	// After fills A,B + hit A + fill C at cap=2, the resident set is {C, A} (B was
	// the LRU victim of C's fill). Revoking a still-resident witness (etag:A) must
	// emit exactly one CacheRevoke event for it.
	before := count(CacheRevoke)
	if ev := v.Revoke("etag:A"); ev != 1 {
		t.Fatalf("Revoke(etag:A) evicted=%d want 1", ev)
	}
	mu.Lock()
	got = append([]CacheEvent(nil), *events...)
	mu.Unlock()
	if c := countOf(got, CacheRevoke); c-before != 1 {
		t.Fatalf("revoke should emit exactly one CacheRevoke event, delta=%d", c-before)
	}
}

func countOf(events []CacheEvent, k CacheEventKind) int {
	n := 0
	for _, e := range events {
		if e.Kind == k {
			n++
		}
	}
	return n
}

// §2.5: tier-3 (static-table) serves must ALSO emit a first-class cachemeta hit —
// "tier-2 (and tier-3)" — attributed to the consuming agent/turn. A static answer is
// args/epoch-independent and never evicted, so HIT is its only lifecycle event.
func TestCacheEmission_Tier3StaticHit(t *testing.T) {
	v := New(8)
	events, mu := collectEvents(v)
	ctx := context.Background()

	v.RegisterStatic("list_airports", []byte(`{"airports":["SFO"]}`))

	// A consuming call names its agent/turn/trace — the tier-3 hit must attribute it.
	c := roCall("list_airports", `{}`)
	c.TraceID = "trace-t3"
	c.Meta[MetaAgentID] = "agent-Z"
	c.Meta[MetaTurn] = "turn-9"
	if _, ok := v.Lookup(ctx, c); !ok {
		t.Fatalf("expected the tier-3 static answer to serve")
	}

	mu.Lock()
	got := append([]CacheEvent(nil), *events...)
	mu.Unlock()

	var hit *CacheEvent
	for i := range got {
		if got[i].Kind == CacheHit {
			hit = &got[i]
		}
	}
	if hit == nil {
		t.Fatalf("tier-3 serve emitted no CacheHit event, got %+v", got)
	}
	if hit.Entry.Plane != cachemeta.PlaneToolResult || hit.Entry.Derivation.Tool != "list_airports" {
		t.Fatalf("tier-3 hit entry mis-shaped: plane=%s tool=%q", hit.Entry.Plane, hit.Entry.Derivation.Tool)
	}
	// A static answer is args/epoch-independent: no ArgsDigest, not write-epoch invalidated.
	if hit.Entry.Derivation.ArgsDigest != "" {
		t.Fatalf("tier-3 entry should carry no args digest, got %q", hit.Entry.Derivation.ArgsDigest)
	}
	if hit.Entry.Coherence.InvalidationMode != cachemeta.InvalidationNone {
		t.Fatalf("tier-3 entry should not be write-epoch invalidated, got %q", hit.Entry.Coherence.InvalidationMode)
	}
	cons := hit.Entry.Coherence.Consumers
	if len(cons) != 1 || cons[0].AgentID != "agent-Z" || cons[0].ID != "turn-9" || cons[0].TraceID != "trace-t3" {
		t.Fatalf("tier-3 hit consumer mis-attributed: %+v", cons)
	}

	// An ANONYMOUS tier-3 serve attaches no empty consumer.
	if _, ok := v.Lookup(ctx, roCall("list_airports", `{}`)); !ok {
		t.Fatalf("expected the static answer to serve on the anonymous lookup too")
	}
	mu.Lock()
	got = append([]CacheEvent(nil), *events...)
	mu.Unlock()
	var last *CacheEvent
	for i := range got {
		if got[i].Kind == CacheHit {
			last = &got[i]
		}
	}
	if last == nil || len(last.Entry.Coherence.Consumers) != 0 {
		t.Fatalf("anonymous tier-3 hit should carry no consumer, got %+v", last)
	}
}

// §2.5: a fast-path MISS must ALSO emit a first-class cachemeta event — the acceptance
// is "admissions/evictions/hits/misses", so the cache event stream is complete and a low
// hit rate is explainable from the SAME stream, by reason and by the consumer that
// experienced it. A served hit must emit no miss.
func TestCacheEmission_Miss(t *testing.T) {
	v := New(8)
	events, mu := collectEvents(v)
	ctx := context.Background()

	// (1) A cacheable read nothing has filled -> NOT_CACHED, attributed to its agent/turn.
	c := roCall("read_doc", `{"id":"q"}`)
	c.TraceID = "trace-m"
	c.Meta[MetaAgentID] = "agent-M"
	c.Meta[MetaTurn] = "turn-1"
	if _, ok := v.Lookup(ctx, c); ok {
		t.Fatalf("an uncached read must miss")
	}
	// (2) A write-shaped tool is never fast-path eligible -> DESTRUCTIVE.
	if _, ok := v.Lookup(ctx, roCall("delete_doc", `{"id":"q"}`)); ok {
		t.Fatalf("a write-shaped call must miss")
	}
	// (3) A hint-less call cannot be proven cacheable -> MISSING_HINTS.
	if _, ok := v.Lookup(ctx, &abi.ToolCall{
		Tool: "read_doc",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte("{}")},
	}); ok {
		t.Fatalf("a hint-less call must miss")
	}

	mu.Lock()
	got := append([]CacheEvent(nil), *events...)
	mu.Unlock()

	// Index the miss events by their lossless vDSO reason label.
	misses := map[string]*CacheEvent{}
	for i := range got {
		if got[i].Kind == CacheMiss {
			misses[got[i].Entry.Labels["vdso_miss"]] = &got[i]
		}
	}
	for _, want := range []string{MissNotCached, MissDestructive, MissMissingHints} {
		if misses[want] == nil {
			t.Fatalf("no CacheMiss event for reason %q, got %+v", want, got)
		}
	}

	// A NOT_CACHED miss is a tool_result-plane entry naming the looked-up tool, carrying
	// the cachemeta-native reason and the consumer that experienced it.
	nc := misses[MissNotCached]
	if nc.Entry.Plane != cachemeta.PlaneToolResult || nc.Entry.Derivation.Tool != "read_doc" {
		t.Fatalf("miss entry mis-shaped: plane=%s tool=%q", nc.Entry.Plane, nc.Entry.Derivation.Tool)
	}
	if nc.Reason != cachemeta.ReasonAbsent {
		t.Fatalf("NOT_CACHED should map to ReasonAbsent, got %q", nc.Reason)
	}
	// A miss has no payload, so it carries no content digest.
	if nc.Entry.ID.Digest != "" {
		t.Fatalf("a miss entry must carry no payload digest, got %q", nc.Entry.ID.Digest)
	}
	cons := nc.Entry.Coherence.Consumers
	if len(cons) != 1 || cons[0].AgentID != "agent-M" || cons[0].ID != "turn-1" || cons[0].TraceID != "trace-m" {
		t.Fatalf("miss consumer mis-attributed: %+v", cons)
	}
	if misses[MissDestructive].Reason != cachemeta.ReasonScopeDenied {
		t.Fatalf("DESTRUCTIVE should map to ReasonScopeDenied, got %q", misses[MissDestructive].Reason)
	}
	if misses[MissMissingHints].Reason != cachemeta.ReasonIncompleteBinding {
		t.Fatalf("MISSING_HINTS should map to ReasonIncompleteBinding, got %q", misses[MissMissingHints].Reason)
	}

	// A served hit must NOT emit a miss event (the stream stays exactly 3 misses).
	v.RegisterStatic("ping", []byte("pong"))
	if _, ok := v.Lookup(ctx, roCall("ping", `{}`)); !ok {
		t.Fatalf("static tool should hit")
	}
	mu.Lock()
	got = append([]CacheEvent(nil), *events...)
	mu.Unlock()
	if n := countOf(got, CacheMiss); n != 3 {
		t.Fatalf("a hit must not emit a miss; CacheMiss count = %d, want 3", n)
	}
}

// #1939: emitCache silently dropped the event when the tier-2 key failed to parse,
// with no signal — a key-format regression would read as "less cache activity"
// rather than "the emitter is dropping events". The drop must be counted (and a
// well-formed key must not bump that counter).
func TestCacheEmission_DropsMalformedKeyAreCounted(t *testing.T) {
	v := New(8)
	events, mu := collectEvents(v)

	if got := v.EmitDropped(); got != 0 {
		t.Fatalf("EmitDropped() = %d, want 0 on a fresh vDSO", got)
	}

	ref := abi.Ref{Kind: abi.RefInline, Inline: []byte("x")}
	v.emitCache(CacheFill, "malformed-key-no-colons", ref, "")
	if got := v.EmitDropped(); got != 1 {
		t.Fatalf("EmitDropped() = %d, want 1 after one malformed-key emit", got)
	}
	mu.Lock()
	n := len(*events)
	mu.Unlock()
	if n != 0 {
		t.Fatalf("a dropped emit must not reach the sink, got %d events", n)
	}

	// A well-formed key must NOT bump the drop counter.
	v.emitCache(CacheFill, "tool:argsdigest:epoch1", ref, "")
	if got := v.EmitDropped(); got != 1 {
		t.Fatalf("EmitDropped() = %d, want unchanged at 1 after a well-formed emit", got)
	}
	mu.Lock()
	n = len(*events)
	mu.Unlock()
	if n != 1 {
		t.Fatalf("the well-formed emit should reach the sink, got %d events", n)
	}
}

// §2.5: a per-tool witness adapter governs admission instead of the internal epoch.
// Registering one for get_doc makes a fill's cachemeta entry carry the adapter's
// witness, and revoking that witness evicts the entry.
func TestCacheEmission_PerToolWitnessAdapter(t *testing.T) {
	v := New(8)
	events, mu := collectEvents(v)
	ctx := context.Background()

	// The adapter derives a per-call witness (e.g. an etag from the entity id).
	v.RegisterWitness("get_doc", func(c *abi.ToolCall, r *abi.Result) string {
		return "etag:adapter:" + string(c.Args.Inline) // deterministic per-args witness
	})

	call := roCall("get_doc", `{"id":"x"}`) // NOTE: no meta witness — the adapter supplies it
	v.Emit(completeEvent(call, `{"body":"X"}`))

	if _, ok := v.Lookup(ctx, call); !ok {
		t.Fatalf("adapter-witnessed entry should hit")
	}
	mu.Lock()
	fills := []CacheEvent{}
	for _, e := range *events {
		if e.Kind == CacheFill {
			fills = append(fills, e)
		}
	}
	mu.Unlock()
	if len(fills) != 1 || fills[0].Entry.Validity.Witness != "etag:adapter:"+`{"id":"x"}` {
		t.Fatalf("adapter witness not on the fill event: %+v", fills)
	}

	// Revoking the adapter-supplied witness evicts the entry (soundness: the adapter,
	// not the epoch, now governs freshness).
	if ev := v.Revoke("etag:adapter:" + `{"id":"x"}`); ev != 1 {
		t.Fatalf("Revoke(adapter-witness) evicted=%d want 1", ev)
	}
	if _, ok := v.Lookup(ctx, call); ok {
		t.Fatalf("adapter-witnessed entry should NOT hit after its witness was refuted")
	}
}

// §2.5: consumer tracking — a tier-2 HIT must name the agent/turn that reused the
// cached result, so a shared entry carries the causal consumer graph cachemeta needs.
// The producing fill is anonymous (no consumer); only the consuming lookup attributes.
func TestCacheEmission_ConsumerTracking(t *testing.T) {
	v := New(8)
	events, mu := collectEvents(v)
	ctx := context.Background()

	// Fill from a producing call (no consumer identity — it admitted, did not reuse).
	producer := roCall("get_doc", `{"id":"y"}`)
	v.Emit(completeEvent(producer, `{"body":"Y"}`))

	// A consuming call names its agent/turn/session in the OPEN meta + a TraceID.
	consumer := roCall("get_doc", `{"id":"y"}`)
	consumer.TraceID = "trace-7"
	consumer.Meta[MetaAgentID] = "agent-A"
	consumer.Meta[MetaTurn] = "turn-3"
	if _, ok := v.Lookup(ctx, consumer); !ok {
		t.Fatalf("expected the cached entry to hit")
	}

	mu.Lock()
	got := append([]CacheEvent(nil), *events...)
	mu.Unlock()

	var fill, hit *CacheEvent
	for i := range got {
		switch got[i].Kind {
		case CacheFill:
			fill = &got[i]
		case CacheHit:
			hit = &got[i]
		}
	}
	if fill == nil || hit == nil {
		t.Fatalf("want one fill and one hit event, got %+v", got)
	}
	// The producing fill records NO consumer (it was an admission, not a reuse).
	if len(fill.Entry.Coherence.Consumers) != 0 {
		t.Fatalf("fill event should carry no consumer, got %+v", fill.Entry.Coherence.Consumers)
	}
	// The hit names exactly the agent/turn/trace that reused the result.
	cons := hit.Entry.Coherence.Consumers
	if len(cons) != 1 {
		t.Fatalf("hit event should carry exactly one consumer, got %+v", cons)
	}
	if cons[0].AgentID != "agent-A" || cons[0].ID != "turn-3" || cons[0].TraceID != "trace-7" {
		t.Fatalf("consumer mis-attributed: %+v", cons[0])
	}

	// An ANONYMOUS lookup (no TraceID, no meta identity) attaches no empty consumer.
	anon := roCall("get_doc", `{"id":"y"}`)
	if _, ok := v.Lookup(ctx, anon); !ok {
		t.Fatalf("expected the cached entry to hit on the anonymous lookup too")
	}
	mu.Lock()
	got = append([]CacheEvent(nil), *events...)
	mu.Unlock()
	var lastHit *CacheEvent
	for i := range got {
		if got[i].Kind == CacheHit {
			lastHit = &got[i]
		}
	}
	if lastHit == nil || len(lastHit.Entry.Coherence.Consumers) != 0 {
		t.Fatalf("anonymous hit should carry no consumer, got %+v", lastHit)
	}
}
