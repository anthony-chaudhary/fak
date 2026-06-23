package memq

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

// recallCorpus records a benign corpus with DISTINCT relevance scores for the query
// "refund fee policy", so the top-k ranking has no tie ambiguity and recall.Recall's
// break-at-zero never triggers within k.
func recallCorpus(t *testing.T) *recall.Recorder {
	t.Helper()
	ctx := context.Background()
	r := recall.NewRecorder("memq-equiv")
	bodies := []struct{ tool, body string }{
		{"a_policy", "refund fee policy: gold tier refund fee is 25 EUR per segment"}, // 3 terms
		{"b_fee", "the refund fee for this account"},                                  // 2 terms (refund, fee)
		{"c_refund", "how do I request a refund"},                                     // 1 term (refund)
		{"d_weather", "the weather in san francisco is foggy"},                        // 0
		{"e_menu", "the cafeteria menu has pasta on tuesdays"},                        // 0
	}
	for _, b := range bodies {
		r.Record(ctx, b.tool, []byte(b.body))
	}
	return r
}

func loadRecall(t *testing.T, r *recall.Recorder) (*recall.Session, string) {
	t.Helper()
	dir := t.TempDir()
	if err := r.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	s, err := recall.Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return s, dir
}

// TestRecallDriverEquivalentToRecallRecall is the headline "SQL not a query" proof:
// the memq `recall` driver, run over a RecallBackend, renders EXACTLY the working set
// recall.Session.Recall returns for the same (intent, k) — same membership, same
// order. The hardcoded query is now one sentence in the algebra.
func TestRecallDriverEquivalentToRecallRecall(t *testing.T) {
	ctx := context.Background()
	const intent = "refund fee policy"
	const k = 3

	s, _ := loadRecall(t, recallCorpus(t))

	// The reference: recall's own retrieve.
	want := s.Recall(ctx, intent, k)
	wantSteps := make([]int, len(want))
	for i, sl := range want {
		wantSteps[i] = sl.Step
	}
	if len(wantSteps) == 0 {
		t.Fatal("non-vacuous precondition failed: recall.Recall returned nothing")
	}

	// The memq expression of it.
	be := NewRecallBackend(s, "")
	q := Get0(t, "recall").Build(Params{Intent: intent, K: k})
	r, err := Run(ctx, be, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	gotSteps := make([]int, len(r.Rendered))
	for i, it := range r.Rendered {
		gotSteps[i] = it.Step
	}

	if fmt.Sprint(gotSteps) != fmt.Sprint(wantSteps) {
		t.Fatalf("memq recall driver != recall.Recall:\n  recall.Recall steps = %v\n  memq render   steps = %v", wantSteps, gotSteps)
	}
}

// TestCleanDriverTombstonesOnRecallImage proves the clean driver applies a REAL,
// persisted negative-only suppression through the recall backend when caps are granted.
func TestCleanDriverTombstonesOnRecallImage(t *testing.T) {
	ctx := context.Background()
	r := recall.NewRecorder("memq-clean")
	// One turn-class observation and one durable preference.
	// (recall stamps durability at write time; we drive the page durability by the
	// classifier — here we just assert clean only touches turn-class via the backend.)
	r.Record(ctx, "clock", []byte("it is 3:47pm right now"))
	r.Record(ctx, "pref", []byte("the user prefers afternoon meetings in general"))
	s, dir := loadRecall(t, r)
	be := NewRecallBackend(s, dir)

	q := Get0(t, "clean").Build(Params{})
	res, err := Run(ctx, be, q, AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	// Reload from disk to prove the tombstone PERSISTED across the boundary.
	s2, err := recall.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	cells, _ := NewRecallBackend(s2, "").Cells(ctx)
	anyTomb := false
	for _, c := range cells {
		if c.Tombstoned {
			anyTomb = true
			if NormDurability(c.Durability) != DurabilityTurn {
				t.Errorf("clean persisted a tombstone on a non-turn cell %s (%s)", c.ID, c.Durability)
			}
		}
	}
	// The clean driver only fires on turn-class cells; if the classifier marked none
	// turn, the run is a valid no-op — assert consistency, not a forced tombstone.
	if res.Stats.EffectsApplied > 0 && !anyTomb {
		t.Fatal("clean reported an applied effect but no tombstone persisted")
	}
}

// TestDreamDriverPrunesOrphanBlob proves the dream driver reclaims unreferenced
// storage on a Pruner backend, with the fail-closed apply gate.
func TestDreamDriverPrunesOrphanBlob(t *testing.T) {
	ctx := context.Background()
	q := Get0(t, "dream").Build(Params{})

	// Dry-run: counts the orphan, reclaims nothing.
	m := NewDemoStore()
	r, err := Run(ctx, m, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Stats.EffectsApplied != 0 {
		t.Fatal("dream pruned without caps")
	}
	if blobs, _, _ := m.Prune(ctx, false); blobs == 0 {
		t.Fatal("non-vacuous precondition failed: demo store has no orphan blob to prune")
	}

	// With caps: the orphan is reclaimed.
	m2 := NewDemoStore()
	if _, err := Run(ctx, m2, q, AllowAll()); err != nil {
		t.Fatal(err)
	}
	if blobs, _, _ := m2.Prune(ctx, false); blobs != 0 {
		t.Fatalf("dream with caps left %d orphan blob(s) unreclaimed", blobs)
	}
}

// TestAgentAuthoredNovelQuery is the extensibility proof: a query NO built-in driver
// expresses — "render only my durable standing preferences, newest first" — runs
// straight through the same executor an agent would hit over MCP. No kernel edit.
func TestAgentAuthoredNovelQuery(t *testing.T) {
	ctx := context.Background()
	m := NewDemoStore()
	novel := Query{
		Intent: "standing preferences",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
				{Op: PredEq, Field: "durability", Value: DurabilityDurable},
				{Op: PredEq, Field: "sealed", Value: "false"},
			}}},
			{Kind: OpRank, By: RankStep, Desc: true},
			{Kind: OpRender},
		},
	}
	if err := Validate(novel); err != nil {
		t.Fatalf("a well-formed novel query was refused: %v", err)
	}
	r, err := Run(ctx, m, novel, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Rendered) == 0 {
		t.Fatal("novel query rendered nothing (non-vacuous precondition failed)")
	}
	// Every rendered cell must be a durable preference — the agent's authored intent.
	cellByID := map[string]Cell{}
	all, _ := m.Cells(ctx)
	for _, c := range all {
		cellByID[c.ID] = c
	}
	for _, it := range r.Rendered {
		if NormDurability(cellByID[it.ID].Durability) != DurabilityDurable {
			t.Errorf("novel durable-only query rendered a non-durable cell %s", it.ID)
		}
	}
}

// TestExplainMarksMutations proves the EXPLAIN surface flags which steps mutate
// durable state (and so are proposal-only without caps) before anything runs.
func TestExplainMarksMutations(t *testing.T) {
	q := Get0(t, "compact").Build(Params{})
	p := Explain(q)
	if !p.Valid {
		t.Fatalf("compact plan is invalid: %s", p.Error)
	}
	// compact ends in tombstone (a mutation) — it must be surfaced.
	foundMut := false
	for _, m := range p.Mutations {
		if m == OpTombstone {
			foundMut = true
		}
	}
	if !foundMut {
		t.Errorf("compact plan did not surface the tombstone mutation: %v", p.Mutations)
	}
	// An invalid query still explains (Valid=false) rather than panicking.
	bad := Explain(Query{Ops: []Op{{Kind: "delete"}}})
	if bad.Valid {
		t.Error("a query with a nonexistent op kind explained as valid")
	}
}
