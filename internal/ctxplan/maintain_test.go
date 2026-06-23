package ctxplan

import (
	"reflect"
	"testing"
)

// maintainFinalSpans is the post-session span set used by the equivalence witness: a mix of
// durable/session/turn spans, with one TOMBSTONED and one SEALED — the two flag mutations a
// span's life can undergo (content is immutable + content-addressed, so nothing else changes).
func maintainFinalSpans() []Span {
	return []Span{
		{ID: "s0", Step: 0, Role: "user", Descriptor: "deploy from the release branch", Digest: "d0", Bytes: 30, Durability: DurabilityDurable},
		{ID: "s1", Step: 1, Role: "user", Descriptor: "rotate the auth token now", Digest: "d1", Bytes: 25, Durability: DurabilitySession},
		{ID: "s2", Step: 2, Role: "tool", Descriptor: "auth token rotation runbook revoke", Digest: "d2", Bytes: 40, Durability: DurabilityDurable, Tombstoned: true},
		{ID: "s3", Step: 3, Role: "tool", Descriptor: "auth token billing scope note", Digest: "d3", Bytes: 35, Durability: DurabilitySession, Sealed: true},
		{ID: "s4", Step: 4, Role: "bash", Descriptor: "build log noise compiled files", Digest: "d4", Bytes: 20, Durability: DurabilityTurn},
	}
}

// incrementallyMaintained builds an Index the way the LIVE LOOP would: Add each span as its
// turn arrives (with the trust/suppression flag cleared), then flip the flag afterward via
// SetTombstoned / SetSealed — the order a real session takes (a span is recorded, then later
// quarantined or suppressed).
func incrementallyMaintained(final []Span) *Index {
	ix := NewIndex()
	for _, s := range final {
		if s.Tombstoned || s.Sealed {
			clean := s
			clean.Tombstoned = false
			clean.Sealed = false
			ix.Add(clean)
			if s.Tombstoned {
				ix.SetTombstoned(s.ID)
			}
			if s.Sealed {
				ix.SetSealed(s.ID)
			}
		} else {
			ix.Add(s)
		}
	}
	return ix
}

// TestIncrementalEqualsBatch is THE maintenance witness (issue #558): an index maintained
// incrementally across turns (Add + SetTombstoned/SetSealed) is STRUCTURALLY IDENTICAL to a
// fresh BuildIndex over the same final span set (for a store honoring the unique-id
// addressing contract — every shipped store does). This is what makes the Θ(c·N) compute
// flatten REAL on the live loop — the loop maintains one index (O(tokens) per turn) instead
// of rebuilding it (O(N) per turn, Θ(N²) cumulative, which would defeat the index entirely).
func TestIncrementalEqualsBatch(t *testing.T) {
	final := maintainFinalSpans()
	batch := BuildIndex(final)
	incr := incrementallyMaintained(final)

	// The strongest equivalence: the WHOLE index — span table, inverted posting lists,
	// durable set, and id index — is structurally identical (reflect.DeepEqual over the
	// unexported fields, which this in-package test can reach). Incremental maintenance is not
	// merely behavior-equivalent; it reconstructs the exact structure a rebuild would.
	if !reflect.DeepEqual(incr, batch) {
		t.Fatalf("incremental index != batch index structurally:\n incr=%+v\n batch=%+v", incr, batch)
	}
	// And the public span-table image agrees (same flags, same order).
	if !reflect.DeepEqual(incr.Spans(), batch.Spans()) {
		t.Fatalf("incremental span table != batch:\n incr=%+v\n batch=%+v", incr.Spans(), batch.Spans())
	}
	// And every probe must be identical, across several forecasts.
	for _, f := range []Forecast{
		{Intents: []string{"auth token rotation"}},
		{Intents: []string{"auth token"}, Pins: []string{"s0"}},
		{Intents: nil, Pins: []string{"s1"}},
		{Intents: []string{"runbook revoke billing"}},
	} {
		a := incr.Probe(f, ProbeOptions{})
		b := batch.Probe(f, ProbeOptions{})
		if !reflect.DeepEqual(a, b) {
			t.Errorf("incremental probe != batch for forecast %+v:\n incr=%v\n batch=%v", f, ids(a), ids(b))
		}
	}
	// And the index-bounded plans must be identical too (the end-to-end equivalence).
	pa := incr.PlanCells(Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"s0", "s1"}}, Budget{Tokens: 40}, nil, ProbeOptions{})
	pb := batch.PlanCells(Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"s0", "s1"}}, Budget{Tokens: 40}, nil, ProbeOptions{})
	if !reflect.DeepEqual(selectedIDs(pa), selectedIDs(pb)) {
		t.Errorf("incremental plan != batch plan: %v vs %v", selectedIDs(pa), selectedIDs(pb))
	}
}

// TestSetTombstonedSuppresses proves a tombstoned span, once probed, is elided tombstoned by
// the index-bounded plan and never selected — suppression is enforced at scoring + the gate,
// not by erasing the span from the index (so it still appears in the plan's audit).
func TestSetTombstonedSuppresses(t *testing.T) {
	ix := BuildIndex([]Span{
		{ID: "a", Step: 0, Role: "tool", Descriptor: "auth token rotation runbook", Digest: "da", Bytes: 30, Durability: DurabilitySession},
	})
	if !ix.SetTombstoned("a") {
		t.Fatal("SetTombstoned should find span a")
	}
	f := Forecast{Intents: []string{"auth token"}}
	if !probeIDset(ix.Probe(f, ProbeOptions{}))["a"] {
		t.Fatal("a tombstoned span must still be PROBED so the plan can record it elided-tombstoned")
	}
	p := ix.PlanCells(f, Budget{Tokens: 999}, nil, ProbeOptions{})
	if selectedIDs(p)["a"] {
		t.Fatal("a tombstoned span must never be selected into the resident view")
	}
	elidedTomb := false
	for _, e := range p.Elided {
		if e.ID == "a" && e.Reason == ElideTombstoned {
			elidedTomb = true
		}
	}
	if !elidedTomb {
		t.Errorf("a tombstoned span must be elided with reason %q; elided=%+v", ElideTombstoned, p.Elided)
	}
}

// TestSetSealedSuppresses is the seal twin of TestSetTombstonedSuppresses.
func TestSetSealedSuppresses(t *testing.T) {
	ix := BuildIndex([]Span{
		{ID: "a", Step: 0, Role: "tool", Descriptor: "auth token rotation runbook", Digest: "da", Bytes: 30, Durability: DurabilitySession},
	})
	if !ix.SetSealed("a") {
		t.Fatal("SetSealed should find span a")
	}
	p := ix.PlanCells(Forecast{Intents: []string{"auth token"}}, Budget{Tokens: 999}, nil, ProbeOptions{})
	if selectedIDs(p)["a"] {
		t.Fatal("INVARIANT VIOLATED: a sealed span entered the index-bounded resident view")
	}
	elidedSealed := false
	for _, e := range p.Elided {
		if e.ID == "a" && e.Reason == ElideSealed {
			elidedSealed = true
		}
	}
	if !elidedSealed {
		t.Errorf("a sealed span must be elided with reason %q; elided=%+v", ElideSealed, p.Elided)
	}
}

// TestSetFlagUnknownIdIsNoop proves a flag mutation on an absent id is a reported no-op (it
// returns false and changes nothing), so a stale/fabricated id cannot perturb the index.
func TestSetFlagUnknownIdIsNoop(t *testing.T) {
	final := maintainFinalSpans()
	ix := BuildIndex(final)
	before := ix.Spans()
	if ix.SetTombstoned("ghost") || ix.SetSealed("ghost") {
		t.Fatal("a flag mutation on an unknown id must return false")
	}
	if !reflect.DeepEqual(ix.Spans(), before) {
		t.Error("a no-op flag mutation must not change the span table")
	}
}

// TestSetFlagIdempotent proves a second SetTombstoned/SetSealed is a no-op (the flag is
// already set), so a defensive double-call is safe.
func TestSetFlagIdempotent(t *testing.T) {
	ix := BuildIndex([]Span{{ID: "a", Step: 0, Role: "tool", Descriptor: "x", Digest: "da", Bytes: 4}})
	ix.SetTombstoned("a")
	once := ix.Spans()
	ix.SetTombstoned("a")
	if !reflect.DeepEqual(ix.Spans(), once) {
		t.Error("SetTombstoned must be idempotent")
	}
}

// TestAddClonesAttrsDefendsImmutability proves the content-immutability contract is
// STRUCTURALLY defended for the one reference-type field: Add clones Attrs so a caller that
// mutates its own map after Add cannot reach into the index's stored metadata (Attrs feeds
// Benefit, so a shared map would silently change a span's score with no flag flip), and
// Spans() returns a defensive copy so a consumer cannot corrupt the index through the result.
func TestAddClonesAttrsDefendsImmutability(t *testing.T) {
	attrs := map[string]string{"utility": "4"}
	ix := BuildIndex([]Span{{ID: "a", Step: 0, Role: "tool", Descriptor: "auth token", Digest: "da", Bytes: 10, Durability: DurabilitySession, Attrs: attrs}})

	attrs["utility"] = "0" // mutate the CALLER's map after Add — the index must not see it
	if got := ix.Spans()[0].Attrs["utility"]; got != "4" {
		t.Errorf("Add must clone Attrs so a post-Add caller mutation cannot change the index; got utility=%q", got)
	}

	out := ix.Spans()
	out[0].Attrs["utility"] = "1" // mutate the Spans() result — the index must not be corrupted
	if got := ix.Spans()[0].Attrs["utility"]; got != "4" {
		t.Errorf("Spans() must return a defensive Attrs copy; index corrupted to utility=%q", got)
	}
}

// ids is a tiny helper returning the ordered id list of a span slice (for readable failures).
func ids(spans []Span) []string {
	out := make([]string, len(spans))
	for i, s := range spans {
		out[i] = s.ID
	}
	return out
}
