package ctxplan

import (
	"context"
	"fmt"
	"testing"
)

// issue #548: the Materialize witness must RECONCILE with the rendered+refused outcome
// (not just audit the plan), and it must PIN the Span.Bytes cost contract — the bytes the
// gate pages in must equal the Span.Bytes the planner priced, or the budget is fictional.

// TestMaterializeWitnessHoldsOnHappyPath is the baseline: a clean MemStore materializes to
// a View whose witness is faithful, reconciled, and cost-contract-holding, with no
// divergence. Every downstream test breaks exactly one of those and expects the others to
// hold, so each property is witnessed independently.
func TestMaterializeWitnessHoldsOnHappyPath(t *testing.T) {
	store := NewMemStore()
	store.Add("user", DurabilitySession, []byte("rotate the auth token now"), false)
	store.Add("WebSearch", DurabilityDurable, []byte("auth token rotation runbook"), false)
	f := Forecast{Intents: []string{"auth token rotation"}}
	v, err := Materialize(context.Background(), store, f, Budget{Tokens: 64}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !v.Witness.Faithful || !v.Witness.Reconciled || !v.Witness.CostContract {
		t.Fatalf("happy path must be faithful+reconciled+cost-contract: %+v", v.Witness)
	}
	if len(v.Witness.CostDiverged) != 0 {
		t.Errorf("no divergence on the happy path, got %v", v.Witness.CostDiverged)
	}
	// The reconciled witness mirrors the view's actual rendered/refused counts (not the
	// plan's selected count alone) — that is the reconciliation.
	if v.Witness.Rendered != len(v.Rendered) || v.Witness.Refused != len(v.Refused) {
		t.Errorf("witness rendered/refused must mirror the view: rendered=%d/%d refused=%d/%d",
			v.Witness.Rendered, len(v.Rendered), v.Witness.Refused, len(v.Refused))
	}
	// ResidentTokens is the REALIZED resident size (what the gate paged in), not the plan's
	// selected cost — so on the clean path it equals the view's RenderedTokens exactly.
	if v.Witness.ResidentTokens != v.RenderedTokens() {
		t.Errorf("ResidentTokens must match RenderedTokens on the happy path: %d != %d",
			v.Witness.ResidentTokens, v.RenderedTokens())
	}
}

// refuseStore wraps a MemStore that refuses a named span at page-in — simulating a seal
// that lands BETWEEN selection and render (the race the Refused set exists to report). The
// span stays SELECTED (it was benign at planning time) but never renders.
type refuseStore struct {
	*MemStore
	refuse string
}

func (r *refuseStore) Materialize(ctx context.Context, id string) ([]byte, error) {
	if id == r.refuse {
		return nil, fmt.Errorf("%w: %s", ErrSealed, id)
	}
	return r.MemStore.Materialize(ctx, id)
}

// TestMaterializeWitnessReconcilesRefused is the rendered+refused reconciliation witness:
// a span the planner SELECTED (pinned) but the gate REFUSED at page-in is accounted for —
// Rendered+Refused == Selected, so no selected span silently vanished between plan and
// render. The plan stays faithful (the refusal is a page-in event, not a plan defect), and
// a refused span carries no body so it cannot diverge on the cost contract.
func TestMaterializeWitnessReconcilesRefused(t *testing.T) {
	store := &refuseStore{MemStore: NewMemStore(), refuse: "span:0"}
	store.Add("user", DurabilitySession, []byte("rotate the auth token now"), false)        // span:0, pinned -> refused
	store.Add("WebSearch", DurabilityDurable, []byte("auth token rotation runbook"), false) // span:1
	f := Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0"}}
	v, err := Materialize(context.Background(), store, f, Budget{Tokens: 64}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(v.Refused) == 0 || v.Refused[0].ID != "span:0" {
		t.Fatalf("expected span:0 (pinned, then sealed at page-in) in Refused, got %+v", v.Refused)
	}
	if !v.Witness.Reconciled {
		t.Errorf("refusing a selected span must still Reconcile (rendered+refused==selected): %+v", v.Witness)
	}
	if !v.Witness.Faithful {
		t.Errorf("a page-in refusal must not make the plan unfaithful: %+v", v.Witness)
	}
	if len(v.Witness.CostDiverged) != 0 {
		t.Errorf("a refused span has no body to diverge, got CostDiverged=%v", v.Witness.CostDiverged)
	}
	// The spans that DID page in must still honor the cost contract.
	if !v.Witness.CostContract {
		t.Errorf("rendered spans must still honor the Span.Bytes cost contract: %+v", v.Witness)
	}
	if v.Witness.Refused != 1 {
		t.Errorf("witness must report exactly 1 refused span, got %d", v.Witness.Refused)
	}
	// The refused span was SELECTED (so the plan's ResidentTokens counted it) but never
	// rendered. The reconciled witness must NOT still count it resident: ResidentTokens
	// drops to the tokens actually paged in (== RenderedTokens), the exact bug #548 names.
	if v.Witness.ResidentTokens != v.RenderedTokens() {
		t.Errorf("a refused span must not count as resident: ResidentTokens=%d != RenderedTokens=%d",
			v.Witness.ResidentTokens, v.RenderedTokens())
	}
	if v.Witness.ResidentTokens == v.Plan.CostUsed {
		t.Errorf("ResidentTokens must diverge from the plan's CostUsed once a selected span is refused: both=%d",
			v.Witness.ResidentTokens)
	}
}

// mismatchStore wraps a MemStore whose page-in hands back MORE bytes than Span.Bytes
// declares — the Span.Bytes cost-contract break. The planner charged ceil(Span.Bytes/4);
// the render realized ceil(len(body)/4) of a larger body, so the budget accounting is
// fictional and the witness must say so.
type mismatchStore struct {
	*MemStore
}

func (m *mismatchStore) Materialize(ctx context.Context, id string) ([]byte, error) {
	b, err := m.MemStore.Materialize(ctx, id)
	if err != nil {
		return nil, err
	}
	return append(b, []byte(" PADDING: THE STORE LIED ABOUT SIZE")...), nil
}

// TestMaterializeWitnessCatchesSpanBytesDivergence is the cost-contract pin: when the bytes
// paged in differ from Span.Bytes, CostContract falls false and CostDiverged names the
// offending span. The plan is still faithful and the page-in still reconciled — only the
// cost basis lied, so the three properties are genuinely orthogonal.
func TestMaterializeWitnessCatchesSpanBytesDivergence(t *testing.T) {
	store := &mismatchStore{MemStore: NewMemStore()}
	store.Add("WebSearch", DurabilityDurable, []byte("auth token rotation runbook"), false) // span:0
	f := Forecast{Intents: []string{"auth token rotation"}}
	v, err := Materialize(context.Background(), store, f, Budget{Tokens: 64}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if v.Witness.CostContract {
		t.Errorf("a store returning bytes != Span.Bytes must break the cost contract: %+v", v.Witness)
	}
	if len(v.Witness.CostDiverged) == 0 || v.Witness.CostDiverged[0] != "span:0" {
		t.Errorf("the divergent span must be named in CostDiverged, got %v", v.Witness.CostDiverged)
	}
	if !v.Witness.Faithful || !v.Witness.Reconciled {
		t.Errorf("a cost-contract break must not corrupt faithfulness/reconciliation: %+v", v.Witness)
	}
}

// TestReconcilePure witnesses the contract logic directly (no store): the count-based
// reconciliation, the byte-equality cost contract, and the named divergence — each broken
// in isolation so the property under test is the only thing that moves.
func TestReconcilePure(t *testing.T) {
	p := Plan{
		Candidates: 2,
		Selected:   []Selection{{ID: "a", Step: 0, Cost: 1}, {ID: "b", Step: 1, Cost: 1}},
	}
	pw := Audit(p)
	declared := map[string]int64{"a": 10, "b": 20}

	// Baseline: both rendered, bytes match -> everything holds.
	w := Reconcile(p, pw, []Rendered{{ID: "a", Bytes: 10, Tokens: 3}, {ID: "b", Bytes: 20, Tokens: 5}}, nil, declared)
	if !w.Reconciled || !w.CostContract || !w.Faithful {
		t.Fatalf("baseline reconcile must hold: %+v", w)
	}
	// ResidentTokens is reset to the realized rendered tokens (3+5), not the plan's cost.
	if w.ResidentTokens != 8 {
		t.Errorf("ResidentTokens must be the realized rendered tokens (3+5=8), got %d", w.ResidentTokens)
	}

	// A selected span that neither rendered nor refused -> not reconciled (silently dropped).
	wDrop := Reconcile(p, pw, []Rendered{{ID: "a", Bytes: 10}}, nil, declared)
	if wDrop.Reconciled {
		t.Error("dropping a selected span's render must fail Reconciled")
	}

	// A refused span accounts for the dropped one -> reconciled again.
	wRef := Reconcile(p, pw, []Rendered{{ID: "a", Bytes: 10}}, []Refusal{{ID: "b"}}, declared)
	if !wRef.Reconciled {
		t.Error("rendered+refused must reconcile the selected set")
	}

	// Bytes disagree -> cost contract breaks, divergence named, reconciliation unaffected.
	wBad := Reconcile(p, pw, []Rendered{{ID: "a", Bytes: 10}, {ID: "b", Bytes: 999}}, nil, declared)
	if wBad.CostContract {
		t.Error("a byte mismatch must break the cost contract")
	}
	if len(wBad.CostDiverged) != 1 || wBad.CostDiverged[0] != "b" {
		t.Errorf("divergence must name b, got %v", wBad.CostDiverged)
	}
	if !wBad.Reconciled {
		t.Error("a cost-contract break must not fail reconciliation (counts still balance)")
	}

	// A rendered span missing from declared is itself a contract break (the store rendered a
	// span it never reported to the planner).
	wUnk := Reconcile(p, pw, []Rendered{{ID: "a", Bytes: 10}, {ID: "ghost", Bytes: 5}}, nil, declared)
	if wUnk.CostContract {
		t.Error("a rendered span absent from declared must break the cost contract")
	}
}
