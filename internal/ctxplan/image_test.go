package ctxplan

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// TestIndexImageRoundTripIsIdentity is THE persistence witness (issue #558, half a): an
// index serialized to its image and restored is STRUCTURALLY IDENTICAL to the original —
// reflect.DeepEqual over the whole index (span table, posting lists, durable set, id index),
// the same strength maintain_test.go's incremental==batch witness uses. Re-attaching a
// persisted index is therefore never a behavior change, only a cost one: the loader rebuilds
// the derived posting lists + durable set from the stored span table and lands exactly where
// the live index was.
func TestIndexImageRoundTripIsIdentity(t *testing.T) {
	ix := incrementallyMaintained(maintainFinalSpans()) // a real index with sealed + tombstoned spans

	restored, err := RestoreIndex(ix.Image())
	if err != nil {
		t.Fatalf("RestoreIndex: %v", err)
	}
	if !reflect.DeepEqual(restored, ix) {
		t.Fatalf("restored index != original structurally:\n restored=%+v\n original=%+v", restored, ix)
	}
}

// TestIndexImageJSONRoundTrip proves the on-disk form survives: marshal the image to JSON,
// unmarshal, restore — and the result is still structurally identical AND probes identically.
// This is the property recall.PersistIndex/LoadIndex rely on (they write/read exactly these
// bytes alongside the core image).
func TestIndexImageJSONRoundTrip(t *testing.T) {
	ix := incrementallyMaintained(maintainFinalSpans())

	b, err := MarshalIndexImage(ix)
	if err != nil {
		t.Fatalf("MarshalIndexImage: %v", err)
	}
	restored, err := UnmarshalIndexImage(b)
	if err != nil {
		t.Fatalf("UnmarshalIndexImage: %v", err)
	}
	if !reflect.DeepEqual(restored, ix) {
		t.Fatalf("JSON round-trip changed the index:\n restored=%+v\n original=%+v", restored, ix)
	}
	// And a direct json.Marshal of Image() restores the same way (the convenience wrappers
	// are not doing anything the documented Image()+RestoreIndex pair would not).
	raw, _ := json.Marshal(ix.Image())
	var img IndexImage
	if err := json.Unmarshal(raw, &img); err != nil {
		t.Fatalf("json.Unmarshal(Image): %v", err)
	}
	restored2, err := RestoreIndex(img)
	if err != nil {
		t.Fatalf("RestoreIndex(unmarshaled): %v", err)
	}
	if !reflect.DeepEqual(restored2, ix) {
		t.Fatalf("Image()->json->RestoreIndex changed the index")
	}
}

// TestRestoreEqualsRebuildAcrossProbes is the "re-attach == rebuild" witness stated on the
// observable surface: a restored index Probes and PLANS byte-identically to a fresh
// BuildIndex over the same spans, across several forecasts — so a resumed session that loads
// its image makes the exact same per-turn decisions a session that rebuilt the index from
// the store would, for a fraction of the cost.
func TestRestoreEqualsRebuildAcrossProbes(t *testing.T) {
	final := maintainFinalSpans()
	rebuilt := BuildIndex(final)
	restored, err := RestoreIndex(rebuilt.Image())
	if err != nil {
		t.Fatalf("RestoreIndex: %v", err)
	}
	for _, f := range []Forecast{
		{Intents: []string{"auth token rotation"}},
		{Intents: []string{"auth token"}, Pins: []string{"s0"}},
		{Intents: nil, Pins: []string{"s1"}},
		{Intents: []string{"runbook revoke billing"}},
	} {
		if !reflect.DeepEqual(restored.Probe(f, ProbeOptions{}), rebuilt.Probe(f, ProbeOptions{})) {
			t.Errorf("restored probe != rebuild for forecast %+v", f)
		}
		pa := restored.PlanCells(f, Budget{Tokens: 40}, nil, ProbeOptions{})
		pb := rebuilt.PlanCells(f, Budget{Tokens: 40}, nil, ProbeOptions{})
		if !reflect.DeepEqual(selectedIDs(pa), selectedIDs(pb)) {
			t.Errorf("restored plan != rebuild plan for forecast %+v", f)
		}
	}
}

// TestRestoreEqualsRebuildUnderPruning is the LOAD-BEARING variant of the re-attach==rebuild
// witness: over a 500-span store under a tight recency window where the bounded probe ACTUALLY
// prunes (the probe is a strict subset of N), a restored index probes and plans byte-identically
// to a fresh BuildIndex. This exercises the image's posting-list rederivation exactly where it
// matters — when the index-bounded access path is selecting a small candidate set out of many,
// not only on the tiny inputs where every span trivially fits the candidate bound and a broken
// rederivation could hide.
func TestRestoreEqualsRebuildUnderPruning(t *testing.T) {
	ctx := context.Background()
	spans, _ := goodPlusNoiseStore(500).Spans(ctx)
	rebuilt := BuildIndex(spans)
	restored, err := RestoreIndex(rebuilt.Image())
	if err != nil {
		t.Fatalf("RestoreIndex: %v", err)
	}

	f := Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0", "span:1"}}
	opts := ProbeOptions{RecencyWindow: 8} // tight window so the probe is a strict subset of N
	probe := restored.Probe(f, opts)
	if len(probe) >= len(spans) {
		t.Fatalf("probe (%d) did not prune below N (%d) — the test would be vacuous", len(probe), len(spans))
	}
	if !reflect.DeepEqual(probe, rebuilt.Probe(f, opts)) {
		t.Fatal("restored probe != rebuild probe under pruning — the image rederivation diverged")
	}
	pa := restored.PlanCells(f, Budget{Tokens: 48}, nil, opts)
	pb := rebuilt.PlanCells(f, Budget{Tokens: 48}, nil, opts)
	if !reflect.DeepEqual(selectedIDs(pa), selectedIDs(pb)) {
		t.Fatalf("restored plan != rebuild plan under pruning: %v vs %v", selectedIDs(pa), selectedIDs(pb))
	}
}

// TestRestoreIndexRejectsBadVersion is the fail-closed guard: an image whose version the
// loader does not recognize is REFUSED, not silently rebuilt — so a forward-incompatible
// Span-shape change surfaces as a load error instead of a wrong index.
func TestRestoreIndexRejectsBadVersion(t *testing.T) {
	bad := IndexImage{Version: "ctxplan-index-v0-bogus", Spans: maintainFinalSpans()}
	if _, err := RestoreIndex(bad); err == nil {
		t.Fatal("RestoreIndex must refuse an unrecognized image version")
	}
	// A correctly-versioned but empty image restores to an empty index (the NewIndex base
	// case), never an error.
	empty, err := RestoreIndex(NewIndex().Image())
	if err != nil {
		t.Fatalf("RestoreIndex of an empty image must not error: %v", err)
	}
	if empty.Len() != 0 {
		t.Errorf("empty image must restore to an empty index, got Len=%d", empty.Len())
	}
}

// TestIndexImageIsSafeMetadataOnly is the trust witness: a SEALED span persists with its
// Sealed flag intact and its sealed-safe descriptor — the image carries SAFE metadata, and the
// trust invariant (a sealed span is never selected) survives a save/load. Critically it forces
// the sealed span in through the INVERTED INDEX (relevance) path, NOT recency: the sealed span
// is buried under fresh benign noise (outside a tight recency window) and is durability=turn, so
// the only access path that reaches it is the relevance match on its descriptor tokens — the
// same way a sealed recall page surfaces. Proving suppression on the index path (not just on an
// explicit pin) is what makes "the trust gate survives persistence" load-bearing, since the
// inverted index is where a restored posting-list bug would actually surface a sealed span.
func TestIndexImageIsSafeMetadataOnly(t *testing.T) {
	ix := NewIndex()
	ix.Add(Span{ID: "s0", Step: 0, Role: "user", Descriptor: "rotate the auth token", Digest: "d0", Bytes: 20, Durability: DurabilitySession})
	// The sealed span: its sealed-safe descriptor still carries its TOOL-NAME tokens (exactly as
	// recall seals a page — role preserved, content gone), so an intent matching "refund policy"
	// reaches it via the inverted index.
	ix.Add(Span{ID: "s1", Step: 1, Role: "read_refund_policy", Descriptor: "read_refund_policy: [sealed: 64 bytes]", Digest: "d1", Bytes: 64, Durability: DurabilityTurn})
	ix.SetSealed("s1")
	// Fresh benign noise AFTER the sealed span, pushing it OUT of any tight recency tail.
	for i := 2; i < 12; i++ {
		ix.Add(Span{ID: "s" + itoaTest(i), Step: i, Role: "Bash", Descriptor: "build log line compiled files", Digest: "dg" + itoaTest(i), Bytes: 10, Durability: DurabilityTurn})
	}

	b, err := MarshalIndexImage(ix)
	if err != nil {
		t.Fatalf("MarshalIndexImage: %v", err)
	}
	// The image is metadata: it carries the sealed-safe descriptor, never resolved bytes.
	if strings.Contains(string(b), "ignore previous instructions") {
		t.Fatal("the persisted index image must never carry sealed content")
	}

	restored, err := UnmarshalIndexImage(b)
	if err != nil {
		t.Fatalf("UnmarshalIndexImage: %v", err)
	}
	// A tight recency window so the sealed span (buried at step 1, durability=turn) is reachable
	// ONLY via the inverted index — not recency, not durability.
	f := Forecast{Intents: []string{"refund policy"}}
	opts := ProbeOptions{RecencyWindow: 2}
	if !probeIDset(restored.Probe(f, opts))["s1"] {
		t.Fatal("the sealed span must be reached via the inverted index (relevance) after restore, proving suppression on the index path")
	}
	p := restored.PlanCells(f, Budget{Tokens: 999}, nil, opts)
	if selectedIDs(p)["s1"] {
		t.Fatal("INVARIANT VIOLATED: a sealed span reached via the inverted index entered a restored index's resident view")
	}
	sealedElided := false
	for _, e := range p.Elided {
		if e.ID == "s1" && e.Reason == ElideSealed {
			sealedElided = true
		}
	}
	if !sealedElided {
		t.Errorf("the sealed span must be elided sealed; elided=%+v", p.Elided)
	}
}
