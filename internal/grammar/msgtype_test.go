package grammar

import (
	"reflect"
	"testing"
)

// foldResultFields is a representative typed payload an agent hands a peer: a
// fold/gather result with a required claim + count and an optional note.
func foldResultFields() []Param {
	return []Param{
		{Name: "claim", Type: "string", Required: true},
		{Name: "count", Type: "number", Required: true},
		{Name: "note", Type: "string", Required: false},
	}
}

// Register installs a typed payload and LookupByName/LookupByDigest resolve it;
// the returned MessageType carries the derived content address.
func TestRegisterAndLookup(t *testing.T) {
	tr := NewTypeRegistry()
	if got := tr.UniqueTypeCount(); got != 0 {
		t.Fatalf("fresh registry UniqueTypeCount = %d, want 0", got)
	}

	mt := tr.Register("fold_result", foldResultFields())
	if mt.Name != "fold_result" {
		t.Fatalf("registered Name = %q, want fold_result", mt.Name)
	}
	if mt.Digest == "" {
		t.Fatalf("registered MessageType has empty Digest (content address not derived)")
	}
	if got := tr.UniqueTypeCount(); got != 1 {
		t.Fatalf("UniqueTypeCount = %d, want 1 after one Register", got)
	}

	byName, ok := tr.LookupByName("fold_result")
	if !ok {
		t.Fatalf("LookupByName(fold_result) missing after Register")
	}
	if byName.Digest != mt.Digest {
		t.Fatalf("LookupByName digest = %q, want %q", byName.Digest, mt.Digest)
	}

	byDigest, ok := tr.LookupByDigest(mt.Digest)
	if !ok {
		t.Fatalf("LookupByDigest(%q) missing", mt.Digest)
	}
	if byDigest.Name != "fold_result" {
		t.Fatalf("LookupByDigest Name = %q, want fold_result", byDigest.Name)
	}
}

// Content-addressing: two names registered with the SAME field set (even in a
// different declaration order) collapse to one canonical digest entry — the
// byDigest dedup of unit 57, reused on the peer-output side.
func TestRegisterDedupByDigest(t *testing.T) {
	tr := NewTypeRegistry()

	a := tr.Register("fold_result", foldResultFields())

	// Same fields, shuffled order: structural identity must be order-invariant.
	shuffled := []Param{
		{Name: "note", Type: "string", Required: false},
		{Name: "count", Type: "number", Required: true},
		{Name: "claim", Type: "string", Required: true},
	}
	b := tr.Register("gather_item", shuffled)

	if a.Digest != b.Digest {
		t.Fatalf("digests differ for identical field sets: %q vs %q (order should not matter)", a.Digest, b.Digest)
	}
	if got := tr.UniqueTypeCount(); got != 1 {
		t.Fatalf("UniqueTypeCount = %d, want 1 (identical field sets dedupe)", got)
	}

	// Both names resolve to the one canonical type.
	for _, name := range []string{"fold_result", "gather_item"} {
		mt, ok := tr.LookupByName(name)
		if !ok {
			t.Fatalf("LookupByName(%q) missing", name)
		}
		if mt.Digest != a.Digest {
			t.Fatalf("name %q digest = %q, want shared %q", name, mt.Digest, a.Digest)
		}
	}

	// A genuinely different field set is a distinct content address.
	c := tr.Register("witness_claim", []Param{{Name: "sha", Type: "string", Required: true}})
	if c.Digest == a.Digest {
		t.Fatalf("distinct field set collided on digest %q", c.Digest)
	}
	if got := tr.UniqueTypeCount(); got != 2 {
		t.Fatalf("UniqueTypeCount = %d, want 2 after a distinct type", got)
	}
}

// Validate accepts a structurally-complete peer payload (every required field
// present, "named:" prefix and optional-field-absence both honored) and rejects
// one missing a required field — the structural repair-rung shape over the
// output-to-peer half, reusing countMissing.
func TestValidateAcceptAndReject(t *testing.T) {
	tr := NewTypeRegistry()
	tr.Register("fold_result", foldResultFields())

	// Complete: required claim+count present (note optional, absent is fine).
	missing, known := tr.Validate("fold_result", map[string]any{"claim": "shipped", "count": 3})
	if !known {
		t.Fatalf("Validate known = false for a registered type")
	}
	if len(missing) != 0 {
		t.Fatalf("Validate missing = %v, want none for a complete payload", missing)
	}

	// The "named:" prefix satisfies a field the same way the tool-input rung does.
	missing, _ = tr.Validate("fold_result", map[string]any{"named:claim": "x", "named:count": 1})
	if len(missing) != 0 {
		t.Fatalf("Validate missing = %v, want none with named: prefixed keys", missing)
	}

	// Reject: required "count" absent.
	missing, known = tr.Validate("fold_result", map[string]any{"claim": "shipped"})
	if !known {
		t.Fatalf("Validate known = false for a registered type")
	}
	if !reflect.DeepEqual(missing, []string{"count"}) {
		t.Fatalf("Validate missing = %v, want [count]", missing)
	}
}

// An unregistered type FAILS CLOSED on the peer-output side: known=false, so a
// caller cannot vouch for a payload whose declared type is unknown.
func TestValidateUnknownTypeFailsClosed(t *testing.T) {
	tr := NewTypeRegistry()
	missing, known := tr.Validate("never_registered", map[string]any{"x": 1})
	if known {
		t.Fatalf("Validate known = true for an unregistered type (should fail closed)")
	}
	if missing != nil {
		t.Fatalf("Validate missing = %v, want nil for unknown type", missing)
	}
}

// Roundtrip: Pack a valid typed payload to a self-describing envelope and Unpack
// it back to the same (type, payload) — the MPI_Pack analogue closing the loop.
func TestPackUnpackRoundtrip(t *testing.T) {
	tr := NewTypeRegistry()
	tr.Register("fold_result", foldResultFields())

	payload := map[string]any{"claim": "shipped", "count": float64(3), "note": "ok"}
	b, ok := tr.Pack("fold_result", payload)
	if !ok {
		t.Fatalf("Pack a valid payload returned ok=false")
	}

	// Determinism: the same (type, payload) packs byte-identically.
	b2, ok := tr.Pack("fold_result", payload)
	if !ok || !reflect.DeepEqual(b, b2) {
		t.Fatalf("Pack is not deterministic: %q vs %q", b, b2)
	}

	gotType, gotPayload, ok := tr.Unpack(b)
	if !ok {
		t.Fatalf("Unpack a valid envelope returned ok=false")
	}
	if gotType != "fold_result" {
		t.Fatalf("Unpack type = %q, want fold_result", gotType)
	}
	if !reflect.DeepEqual(gotPayload, payload) {
		t.Fatalf("roundtrip payload = %v, want %v", gotPayload, payload)
	}
}

// Pack REFUSES an invalid payload (required field missing) — a peer never
// receives a message that doesn't match its declared type.
func TestPackRejectsInvalidPayload(t *testing.T) {
	tr := NewTypeRegistry()
	tr.Register("fold_result", foldResultFields())

	if _, ok := tr.Pack("fold_result", map[string]any{"claim": "x"}); ok {
		t.Fatalf("Pack accepted a payload missing the required count field")
	}
	if _, ok := tr.Pack("never_registered", map[string]any{"x": 1}); ok {
		t.Fatalf("Pack accepted an unregistered type")
	}
}

// Unpack rejects an envelope whose declared digest does not match the registered
// type's content address (a forged/stale type claim) and malformed JSON.
func TestUnpackRejectsForgedDigestAndGarbage(t *testing.T) {
	tr := NewTypeRegistry()
	tr.Register("fold_result", foldResultFields())

	forged := []byte(`{"type":"fold_result","digest":"deadbeef","payload":{"claim":"x","count":1}}`)
	if _, _, ok := tr.Unpack(forged); ok {
		t.Fatalf("Unpack accepted an envelope with a forged digest")
	}
	if _, _, ok := tr.Unpack([]byte(`{not json`)); ok {
		t.Fatalf("Unpack accepted malformed JSON")
	}
}
