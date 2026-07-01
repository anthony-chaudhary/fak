package memq

import (
	"strings"
	"testing"
)

// TestMemoryPromotionAuditRecord pins the #1595 shape: promoting a cell past
// turn-class captures a PromotionRecord carrying the source span, durability class,
// consent, producer, and expiry — and a turn-class Add mints NO record at all (it was
// never promoted; it is pure context per CONTEXT-IS-NOT-MEMORY.md).
func TestMemoryPromotionAuditRecord(t *testing.T) {
	m := NewMemStore()

	// A turn-class observation ("it's 3pm"-shaped) must NOT earn a promotion record.
	turnCell := m.Add("clock", "system", DurabilityTurn, []byte("It is 3:47pm."), false)
	if _, ok := m.Promotions().For(turnCell.ID); ok {
		t.Fatalf("turn-class cell %s must not have a promotion record (it was never promoted to memory)", turnCell.ID)
	}

	// A durable promotion with explicit consent and a named producer.
	durCell := m.AddPromoted("user", "user", DurabilityDurable,
		[]byte("I prefer concise answers."), false,
		PromotionMeta{Consent: ConsentExplicit, Producer: "user", Reason: "user stated a standing preference"})

	recs, ok := m.Promotions().For(durCell.ID)
	if !ok || len(recs) != 1 {
		t.Fatalf("durable cell %s must have exactly one promotion record, got %d (ok=%v)", durCell.ID, len(recs), ok)
	}
	rec := recs[0]
	if rec.Durability != DurabilityDurable {
		t.Errorf("Durability = %q, want %q", rec.Durability, DurabilityDurable)
	}
	if rec.Consent != ConsentExplicit {
		t.Errorf("Consent = %q, want %q", rec.Consent, ConsentExplicit)
	}
	if rec.Producer != "user" {
		t.Errorf("Producer = %q, want %q", rec.Producer, "user")
	}
	if rec.SourceSpan.Step != durCell.Step || rec.SourceSpan.Role != "user" {
		t.Errorf("SourceSpan = %+v, does not match the source cell (step=%d role=user)", rec.SourceSpan, durCell.Step)
	}
	if rec.SourceSpan.Digest != durCell.Digest {
		t.Errorf("SourceSpan.Digest = %q, want the cell digest %q", rec.SourceSpan.Digest, durCell.Digest)
	}
	if rec.Expiry != "" {
		t.Errorf("Expiry = %q, want empty for an unbounded durable promotion", rec.Expiry)
	}

	// A bounded promotion with an expiry.
	boundedCell := m.AddPromoted("scheduler", "system", DurabilityBounded,
		[]byte("On vacation until 2026-07-15."), false,
		PromotionMeta{Consent: ConsentExplicit, Producer: "user", Expiry: "2026-07-15"})
	brecs, ok := m.Promotions().For(boundedCell.ID)
	if !ok || len(brecs) != 1 {
		t.Fatalf("bounded cell %s must have a promotion record", boundedCell.ID)
	}
	if brecs[0].Expiry != "2026-07-15" {
		t.Errorf("Expiry = %q, want %q", brecs[0].Expiry, "2026-07-15")
	}

	// The unclassified/default path (plain Add, no PromotionMeta) fails closed to
	// ConsentInferred and a non-empty producer — never a silent blank.
	inferredCell := m.Add("get_user_details", "tool_result", DurabilitySession,
		[]byte(`{"tier":"gold"}`), false)
	irecs, ok := m.Promotions().For(inferredCell.ID)
	if !ok || len(irecs) != 1 {
		t.Fatalf("session-class cell %s must have a promotion record", inferredCell.ID)
	}
	if irecs[0].Consent != ConsentInferred {
		t.Errorf("default Consent = %q, want %q (fail-closed default)", irecs[0].Consent, ConsentInferred)
	}
	if irecs[0].Producer == "" {
		t.Error("Producer must never be empty, even on the default path")
	}

	// The ledger accumulates every promotion in insertion order (Seq is monotonic and
	// distinguishes records — the append-only, no-hard-delete posture doc.go promises
	// elsewhere in memq).
	all := m.Promotions().All()
	if len(all) != 3 {
		t.Fatalf("ledger.All() = %d records, want 3 (turn-class must not appear)", len(all))
	}
	for i, r := range all {
		if r.Seq != i {
			t.Errorf("record %d has Seq=%d, want monotonic sequence", i, r.Seq)
		}
	}
}

// TestMemoryExplainUsesAuditNotNarration is the done-condition witness for #1595:
// PromotionLedger.Explain must answer "why is this fact in memory" using ONLY the
// structured record fields — never a model call, never free-form summarization of the
// cell body. It asserts both that a promoted cell's explanation is built purely from
// its record fields, and that an unpromoted/unknown cell is reported as such rather
// than guessed at.
func TestMemoryExplainUsesAuditNotNarration(t *testing.T) {
	m := NewMemStore()
	cell := m.AddPromoted("user", "user", DurabilityDurable,
		[]byte("I always want a confirmation before deletes."), false,
		PromotionMeta{Consent: ConsentExplicit, Producer: "user", Reason: "user stated a standing preference"})

	exp := m.Promotions().Explain(cell.ID)
	if !exp.Found {
		t.Fatalf("Explain(%s).Found = false, want true", cell.ID)
	}
	// Every field in the explanation must trace back to the record, not to the cell
	// body or any re-derivation of it.
	if exp.Durability != DurabilityDurable {
		t.Errorf("Durability = %q, want %q", exp.Durability, DurabilityDurable)
	}
	if exp.Consent != ConsentExplicit {
		t.Errorf("Consent = %q, want %q", exp.Consent, ConsentExplicit)
	}
	if exp.Producer != "user" {
		t.Errorf("Producer = %q, want %q", exp.Producer, "user")
	}
	if exp.Reason != "user stated a standing preference" {
		t.Errorf("Reason = %q, want the recorded reason", exp.Reason)
	}
	// The narrative is a fixed-template sentence over the record fields: every
	// structured value must appear verbatim in it (proof it was assembled from the
	// record, not from a separate narration pass that could drift from the evidence).
	for _, want := range []string{cell.ID, DurabilityDurable, ConsentExplicit, "user", "user stated a standing preference"} {
		if !strings.Contains(exp.Narrative, want) {
			t.Errorf("Narrative %q must contain %q (the explanation must be built from the record, not narrated)", exp.Narrative, want)
		}
	}

	// A cell with no promotion record (e.g. a turn-class one, or an unknown ID) is
	// reported honestly as unexplained — never backfilled with a guess.
	turnCell := m.Add("clock", "system", DurabilityTurn, []byte("It is 3:47pm."), false)
	turnExp := m.Promotions().Explain(turnCell.ID)
	if turnExp.Found {
		t.Errorf("Explain(%s).Found = true for a turn-class cell that was never promoted", turnCell.ID)
	}
	if turnExp.Narrative == "" {
		t.Error("an unfound explanation must still carry a narrative saying why, not an empty string")
	}

	unknownExp := m.Promotions().Explain("cell:does-not-exist")
	if unknownExp.Found {
		t.Error("Explain on an unknown cell ID must report Found=false")
	}
}

// TestMemoryPromotionRecordNormalizesUnknownVocabulary pins the fail-closed posture:
// an unrecognized durability/consent string is normalized to the safe default rather
// than stored verbatim (mirrors NormDurability's existing contract in memq.go).
func TestMemoryPromotionRecordNormalizesUnknownVocabulary(t *testing.T) {
	if got := NormConsent("yolo"); got != ConsentUnknown {
		t.Errorf("NormConsent(bogus) = %q, want %q", got, ConsentUnknown)
	}
	if got := NormConsent(""); got != ConsentUnknown {
		t.Errorf("NormConsent(\"\") = %q, want %q", got, ConsentUnknown)
	}
	if got := NormProducer(""); got != "unknown" {
		t.Errorf("NormProducer(\"\") = %q, want %q", got, "unknown")
	}

	l := NewPromotionLedger()
	// An unrecognized durability string normalizes to Turn (NormDurability's existing
	// fail-closed contract) — and Record treats a Turn-class record as "never
	// promoted," so it must not be retained at all: garbage-in never manufactures a
	// durable-looking promotion.
	l.Record(PromotionRecord{CellID: "cell:0", Durability: "forever", Consent: "yolo"})
	if _, ok := l.For("cell:0"); ok {
		t.Error("a record whose durability normalizes to turn-class must not be retained (it is not a promotion)")
	}

	// An unrecognized consent string on an otherwise-real promotion IS retained, but
	// normalized to ConsentUnknown rather than stored verbatim.
	l.Record(PromotionRecord{CellID: "cell:1", Durability: DurabilityDurable, Consent: "yolo"})
	rec, ok := l.Latest("cell:1")
	if !ok {
		t.Fatal("a durable-class record must be retained")
	}
	if rec.Consent != ConsentUnknown {
		t.Errorf("unknown consent normalizes to %q, want %q (fail-closed)", rec.Consent, ConsentUnknown)
	}
	if rec.Producer != "unknown" {
		t.Errorf("empty producer normalizes to %q, want %q", rec.Producer, "unknown")
	}
}
