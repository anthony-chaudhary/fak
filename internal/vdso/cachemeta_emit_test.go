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
