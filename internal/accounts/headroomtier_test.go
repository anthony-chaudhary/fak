package accounts

import "testing"

// TestClassifyBands pins the single sign→tier boundary rule (< 0 walled, == 0 unknown, > 0
// offerable), including the band anchors and points just inside each band, so a future retune
// of the producer bases cannot silently move a boundary the interpreters read.
func TestClassifyBands(t *testing.T) {
	cases := []struct {
		name  string
		score float64
		want  Tier
	}{
		{"walled base", WalledBase, TierWalled},
		{"walled interior (base+soonness)", WalledBase + 0.9, TierWalled},
		{"just below zero", -0.0001, TierWalled},
		{"unknown", UnknownScore, TierUnknown},
		{"just above zero", 0.0001, TierOfferable},
		{"offerable base", OfferableBase, TierOfferable},
		{"offerable interior (base+load)", OfferableBase + 0.5, TierOfferable},
	}
	for _, c := range cases {
		if got := Classify(c.score); got != c.want {
			t.Fatalf("Classify(%v) [%s] = %d, want %d", c.score, c.name, got, c.want)
		}
	}
}

// TestBandAnchorsSigns is the invariant the whole subsystem leans on: the anchors sit in the
// tiers their names claim, so producer arithmetic (base + within-tier bonus) never crosses a
// boundary. If someone edits a base to the wrong sign, this fails before any consumer misreads it.
func TestBandAnchorsSigns(t *testing.T) {
	if !(WalledBase < 0) {
		t.Fatalf("WalledBase must be < 0, got %v", WalledBase)
	}
	if UnknownScore != 0 {
		t.Fatalf("UnknownScore must be 0, got %v", UnknownScore)
	}
	if !(OfferableBase > 0) {
		t.Fatalf("OfferableBase must be > 0, got %v", OfferableBase)
	}
}

// TestHeadroomLabel pins the sign→word mapping in its single home, one word per tier.
func TestHeadroomLabel(t *testing.T) {
	cases := []struct {
		score float64
		want  string
	}{
		{OfferableBase + 0.5, "room"},
		{WalledBase, "walled"},
		{UnknownScore, "unknown"},
	}
	for _, c := range cases {
		if got := HeadroomLabel(c.score); got != c.want {
			t.Fatalf("HeadroomLabel(%v) = %q, want %q", c.score, got, c.want)
		}
	}
}

// TestUUIDBucketKeyAgreesWithAccountKey enforces the overlay-alignment invariant by code rather
// than a comment: the hand-callable UUIDBucketKey and Identity.AccountKey() must build the SAME
// bucket string for a login UUID, so the cooldown overlay (keyed by AccountKey) and the runtime
// headroom fold (keyed by UUIDBucketKey) can never wall the wrong bucket.
func TestUUIDBucketKeyAgreesWithAccountKey(t *testing.T) {
	uuid := "abc-123"
	if got, want := UUIDBucketKey(uuid), (Identity{AccountUUID: uuid}).AccountKey(); got != want {
		t.Fatalf("UUIDBucketKey(%q)=%q but AccountKey=%q — bucket keys must match", uuid, got, want)
	}
	if UUIDBucketKey(uuid) != "uuid:"+uuid {
		t.Fatalf("UUIDBucketKey drifted from the canonical prefix, got %q", UUIDBucketKey(uuid))
	}
}
