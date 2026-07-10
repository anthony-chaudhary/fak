package memq

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// reclassifyEffect returns the single OpReclassify effect a run recorded, failing the
// test if none is present.
func reclassifyEffect(t *testing.T, r Result) Effect {
	t.Helper()
	for _, e := range r.Effects {
		if e.Kind == OpReclassify {
			return e
		}
	}
	t.Fatal("no reclassify effect recorded")
	return Effect{}
}

// cellDurability reads a cell's CURRENT persisted durability class straight off the
// backend's page table — the ground truth a re-scan/recall would return, never the
// executor's in-flight working set.
func cellDurability(t *testing.T, m *MemStore, id string) string {
	t.Helper()
	cells, err := m.Cells(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cells {
		if c.ID == id {
			return NormDurability(c.Durability)
		}
	}
	t.Fatalf("cell %s not found", id)
	return ""
}

// demoteQuery reclassifies every cell currently at `from` down to `to`.
func demoteQuery(from, to string) Query {
	return Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpFilter, Pred: &Pred{Op: PredEq, Field: "durability", Value: from}},
		{Kind: OpReclassify, By: to},
	}}
}

// roBackend is a minimal read-only Backend — it implements neither Reclassifier nor any
// other mutation seam, so it is the safe-floor case every effect must fall back to
// (proposal-only) when the backend cannot apply.
type roBackend struct{ cells []Cell }

func (b *roBackend) Cells(context.Context) ([]Cell, error) {
	out := make([]Cell, len(b.cells))
	copy(out, b.cells)
	return out, nil
}

func (b *roBackend) Materialize(context.Context, string) ([]byte, error) {
	return nil, fmt.Errorf("roBackend: page-in unsupported")
}

// TestReclassifyAppliesDemotionUnderCaps is the #4147 done-condition witness: with a
// Caps grant and a Reclassifier backend, a demotion is PERSISTED to the store AND minted
// as a demotion record on the promotion ledger — so a subsequent scan returns the lowered
// class and the class change is auditable + reversible.
func TestReclassifyAppliesDemotionUnderCaps(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	dur := m.AddPromoted("user", "user", DurabilityDurable,
		[]byte("I prefer concise answers."), false,
		PromotionMeta{Consent: ConsentExplicit, Producer: "user", Reason: "user stated a standing preference"})

	// EXPLAIN must honestly flag reclassify as a caps-gated mutation before anything runs.
	plan := Explain(demoteQuery(DurabilityDurable, DurabilitySession))
	if !plan.Valid {
		t.Fatalf("plan invalid: %s", plan.Error)
	}
	foundMut := false
	for _, mk := range plan.Mutations {
		if mk == OpReclassify {
			foundMut = true
		}
	}
	if !foundMut {
		t.Errorf("EXPLAIN did not flag reclassify as a mutation: %v", plan.Mutations)
	}

	r, err := Run(ctx, m, demoteQuery(DurabilityDurable, DurabilitySession), AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	e := reclassifyEffect(t, r)
	if !e.Applied {
		t.Fatalf("reclassify Applied=false under caps + Reclassifier backend; note=%q", e.Note)
	}

	// The persisted class was lowered (a re-scan returns the demoted class).
	if got := cellDurability(t, m, dur.ID); got != DurabilitySession {
		t.Fatalf("cell durability after demotion = %q, want %q (the write-back did not persist)", got, DurabilitySession)
	}

	// A demotion audit record was minted, layered over the original promotion (append-only:
	// the history is kept, so the change is auditable and reversible).
	recs, ok := m.Promotions().For(dur.ID)
	if !ok || len(recs) != 2 {
		t.Fatalf("promotion ledger For(%s) = %d records (ok=%v), want 2 (original promotion + demotion)", dur.ID, len(recs), ok)
	}
	last := recs[len(recs)-1]
	if last.Durability != DurabilitySession {
		t.Errorf("demotion record Durability = %q, want %q", last.Durability, DurabilitySession)
	}
	if !strings.Contains(last.Reason, "demote") || !strings.Contains(last.Reason, DurabilityDurable) || !strings.Contains(last.Reason, DurabilitySession) {
		t.Errorf("demotion record Reason = %q, want it to name the durable->session transition", last.Reason)
	}
	if last.Producer != "memq" {
		t.Errorf("demotion record Producer = %q, want %q (the reclassify op's producer)", last.Producer, "memq")
	}
}

// TestReclassifyProposalOnlyWithoutCaps pins the fail-closed default: without a Caps
// grant the demotion is proposed but NOT applied — the store keeps the original class and
// mints no demotion record.
func TestReclassifyProposalOnlyWithoutCaps(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	dur := m.AddPromoted("user", "user", DurabilityDurable, []byte("I prefer concise answers."), false,
		PromotionMeta{Consent: ConsentExplicit, Producer: "user"})

	r, err := Run(ctx, m, demoteQuery(DurabilityDurable, DurabilitySession), Caps{})
	if err != nil {
		t.Fatal(err)
	}
	e := reclassifyEffect(t, r)
	if e.Applied {
		t.Error("reclassify Applied=true without a Caps grant (must be proposal-only)")
	}
	if !strings.Contains(e.Note, "proposal only") {
		t.Errorf("note = %q, want it to state the demotion is proposal-only", e.Note)
	}
	if got := cellDurability(t, m, dur.ID); got != DurabilityDurable {
		t.Errorf("cell durability = %q, want %q unchanged (no caps must not mutate)", got, DurabilityDurable)
	}
	if recs, _ := m.Promotions().For(dur.ID); len(recs) != 1 {
		t.Errorf("ledger For(%s) = %d records, want 1 (a proposed-only demotion mints no audit record)", dur.ID, len(recs))
	}
}

// TestReclassifyProposalOnlyWithoutReclassifierBackend pins the safe floor: even with
// AllowAll caps, a backend that does not implement Reclassifier can only propose — it is
// never mutated.
func TestReclassifyProposalOnlyWithoutReclassifierBackend(t *testing.T) {
	ctx := context.Background()
	be := &roBackend{cells: []Cell{{
		ID: "cell:0", Step: 0, Role: "user", Kind: "user",
		Descriptor: "user: I prefer concise answers.", Durability: DurabilityDurable,
	}}}
	if _, isRC := any(be).(Reclassifier); isRC {
		t.Fatal("test precondition failed: roBackend must NOT implement Reclassifier")
	}

	r, err := Run(ctx, be, demoteQuery(DurabilityDurable, DurabilitySession), AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	e := reclassifyEffect(t, r)
	if e.Applied {
		t.Error("reclassify Applied=true against a backend with no Reclassifier (must be proposal-only)")
	}
	if !strings.Contains(e.Note, "does not support reclassify") {
		t.Errorf("note = %q, want it to state the backend does not support reclassify", e.Note)
	}
}

// TestReclassifyNeverPromotesEvenUnderCaps pins the demote-only invariant under the new
// apply path: a request to PROMOTE (toward a longer-lived class) is refused, capped at the
// current class, even with AllowAll caps and a Reclassifier backend — the store is not
// mutated and no promoting record is minted.
func TestReclassifyNeverPromotesEvenUnderCaps(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	sess := m.Add("task", "system", DurabilitySession, []byte("Working on the refund-fee bug today."), false)

	r, err := Run(ctx, m, demoteQuery(DurabilitySession, DurabilityDurable), AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	e := reclassifyEffect(t, r)
	if e.Applied {
		t.Error("reclassify Applied=true for a PROMOTION request (promotion must be refused even under caps)")
	}
	if !strings.Contains(e.Note, "promotion") || !strings.Contains(e.Note, "refused") {
		t.Errorf("note = %q, want it to record the refused promotion", e.Note)
	}
	if got := cellDurability(t, m, sess.ID); got != DurabilitySession {
		t.Errorf("cell durability = %q, want %q unchanged (a promotion must never persist)", got, DurabilitySession)
	}
	if recs, _ := m.Promotions().For(sess.ID); len(recs) != 1 {
		t.Errorf("ledger For(%s) = %d records, want 1 (a refused promotion mints no record)", sess.ID, len(recs))
	}
}
