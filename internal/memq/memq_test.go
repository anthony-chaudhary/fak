package memq

import (
	"context"
	"strings"
	"testing"
)

func ids(cells []Cell) []string {
	out := make([]string, len(cells))
	for i, c := range cells {
		out[i] = c.ID
	}
	return out
}

func renderedIDs(r Result) map[string]bool {
	m := map[string]bool{}
	for _, it := range r.Rendered {
		m[it.ID] = true
	}
	return m
}

func refusedIDs(r Result) map[string]bool {
	m := map[string]bool{}
	for _, it := range r.Refused {
		m[it.ID] = true
	}
	return m
}

// sealedCellID returns the ID of the sealed poison cell in the demo store.
func sealedCellID(t *testing.T, m *MemStore) string {
	t.Helper()
	cells, _ := m.Cells(context.Background())
	for _, c := range cells {
		if c.Sealed {
			return c.ID
		}
	}
	t.Fatal("demo store has no sealed cell (non-vacuous precondition failed)")
	return ""
}

// TestValidateFailsClosed pins the safety floor: an unknown op kind (notably any
// "delete"/"erase" verb that does not exist by design) is refused, and a malformed
// select op is refused, so an invalid authored pipeline never runs.
func TestValidateFailsClosed(t *testing.T) {
	bad := []Query{
		{Ops: []Op{{Kind: "delete"}}},                                          // no hard-delete op exists
		{Ops: []Op{{Kind: "erase"}}},                                           // nor erase
		{Ops: []Op{{Kind: ""}}},                                                // empty kind
		{Ops: []Op{{Kind: OpFilter}}},                                          // filter with no predicate
		{Ops: []Op{{Kind: OpRank, By: "xyz"}}},                                 // unknown rank key
		{Ops: []Op{{Kind: OpLimit, K: -1}}},                                    // negative limit
		{Ops: []Op{{Kind: OpReclassify, By: "forever"}}},                       // not a durability class
		{Ops: []Op{{Kind: OpFilter, Pred: &Pred{Op: "regex", Field: "role"}}}}, // unknown pred op
	}
	for i, q := range bad {
		if err := Validate(q); err == nil {
			t.Errorf("query %d validated but should be refused fail-closed: %+v", i, q)
		}
	}
	// A well-formed query validates.
	good := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpFilter, Pred: &Pred{Op: PredEq, Field: "durability", Value: DurabilityDurable}}, {Kind: OpRender}}}
	if err := Validate(good); err != nil {
		t.Fatalf("well-formed query refused: %v", err)
	}
}

func TestPredicateEval(t *testing.T) {
	c := Cell{ID: "cell:0", Step: 3, Role: "get_user_details", Descriptor: "get_user_details: refund_fee 25 EUR",
		Bytes: 80, Durability: DurabilitySession, Sealed: false, Attrs: map[string]string{"src": "tool"}}
	cases := []struct {
		p    Pred
		rc   int
		want bool
	}{
		{Pred{Op: PredEq, Field: "durability", Value: "session"}, 0, true},
		{Pred{Op: PredNe, Field: "durability", Value: "durable"}, 0, true},
		{Pred{Op: PredGt, Field: "bytes", Value: "50"}, 0, true},
		{Pred{Op: PredLe, Field: "bytes", Value: "80"}, 0, true},
		{Pred{Op: PredEq, Field: "step", Value: "3"}, 0, true},
		{Pred{Op: PredEq, Field: "sealed", Value: "false"}, 0, true},
		{Pred{Op: PredEq, Field: "refcount", Value: "0"}, 0, true},
		{Pred{Op: PredEq, Field: "refcount", Value: "0"}, 2, false},
		{Pred{Op: PredMatch, Value: "refund fee"}, 0, true},
		{Pred{Op: PredMatch, Value: "weather forecast"}, 0, false},
		{Pred{Op: PredEq, Field: "attr:src", Value: "tool"}, 0, true},
		{Pred{Op: PredAnd, Args: []Pred{{Op: PredEq, Field: "sealed", Value: "false"}, {Op: PredGt, Field: "bytes", Value: "50"}}}, 0, true},
		{Pred{Op: PredOr, Args: []Pred{{Op: PredEq, Field: "durability", Value: "turn"}, {Op: PredEq, Field: "durability", Value: "session"}}}, 0, true},
		{Pred{Op: PredNot, Args: []Pred{{Op: PredEq, Field: "sealed", Value: "true"}}}, 0, true},
		{Pred{Op: PredTrue}, 0, true},
		{Pred{}, 0, true},
	}
	for i, tc := range cases {
		if got := tc.p.eval(c, tc.rc); got != tc.want {
			t.Errorf("case %d: eval(%+v, rc=%d) = %v, want %v", i, tc.p, tc.rc, got, tc.want)
		}
	}
}

// TestPipelineComposes runs scan->filter->rank->limit and checks the select side
// threads a working set the way the algebra promises.
func TestPipelineComposes(t *testing.T) {
	ctx := context.Background()
	m := NewDemoStore()
	q := Query{
		Intent: "refund fee",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredEq, Field: "sealed", Value: "false"}},
			{Kind: OpRank, By: RankRelevance, Desc: true},
			{Kind: OpLimit, K: 3},
		},
	}
	r, err := Run(ctx, m, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Working) != 3 {
		t.Fatalf("limit 3 should leave 3 cells, got %d", len(r.Working))
	}
	// The most relevant cell to "refund fee" must rank first (it mentions refund_fee).
	if !strings.Contains(strings.ToLower(r.Working[0].Descriptor), "refund") {
		t.Errorf("top-ranked cell is not the most relevant: %q", r.Working[0].Descriptor)
	}
	// No sealed cell survived the filter.
	for _, c := range r.Working {
		if c.Sealed {
			t.Errorf("sealed cell %s survived the not-sealed filter", c.ID)
		}
	}
}

// TestRenderNeverLeaksSealed is the load-bearing safety test: even a query that
// deliberately keeps the sealed cell and renders the whole set never materializes the
// poison — it lands in Refused, and the rendered set carries none of its bytes.
func TestRenderNeverLeaksSealed(t *testing.T) {
	ctx := context.Background()
	m := NewDemoStore()
	sealed := sealedCellID(t, m)
	q := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpRender}}} // render EVERYTHING
	r, err := Run(ctx, m, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if renderedIDs(r)[sealed] {
		t.Fatalf("sealed cell %s was rendered into context — the trust gate failed", sealed)
	}
	if !refusedIDs(r)[sealed] {
		t.Fatalf("sealed cell %s was neither rendered nor refused — it vanished silently", sealed)
	}
	for _, it := range r.Rendered {
		if strings.Contains(strings.ToLower(it.Descriptor), "ignore previous instructions") {
			t.Fatalf("poison text leaked into a rendered item: %q", it.Descriptor)
		}
	}
}

// TestMutationProposedNotAppliedWithoutCaps pins the fail-closed effect default: the
// clean driver proposes tombstones but mutates NOTHING without a Caps grant; with the
// grant, the targeted cells are actually tombstoned.
func TestMutationProposedNotAppliedWithoutCaps(t *testing.T) {
	ctx := context.Background()

	// Dry-run: propose only.
	m := NewDemoStore()
	q := Get0(t, "clean").Build(Params{})
	r, err := Run(ctx, m, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Stats.EffectsApplied != 0 {
		t.Fatalf("clean applied %d effect(s) without caps — fail-closed violated", r.Stats.EffectsApplied)
	}
	cells, _ := m.Cells(ctx)
	for _, c := range cells {
		if c.Tombstoned {
			t.Fatalf("cell %s was tombstoned without caps", c.ID)
		}
	}

	// With caps: the turn-class cells are tombstoned.
	m2 := NewDemoStore()
	r2, err := Run(ctx, m2, q, AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	if r2.Stats.EffectsApplied == 0 {
		t.Fatal("clean applied nothing even with caps")
	}
	cells2, _ := m2.Cells(ctx)
	tombstoned := 0
	for _, c := range cells2 {
		if c.Tombstoned {
			tombstoned++
			if NormDurability(c.Durability) != DurabilityTurn {
				t.Errorf("clean tombstoned a non-turn cell %s (%s)", c.ID, c.Durability)
			}
			if c.Sealed {
				t.Errorf("clean tombstoned a sealed cell %s (should be left to the gate)", c.ID)
			}
		}
	}
	if tombstoned == 0 {
		t.Fatal("clean tombstoned nothing with caps (non-vacuous precondition failed)")
	}
}

// TestConsolidateProducesArtifactNeverPersists: compact folds the unreferenced
// non-durable cells into a real derived disposition (the agent's artifact) but writes
// nothing back to the store this rung, and never folds in a sealed cell.
func TestConsolidateProducesArtifactNeverPersists(t *testing.T) {
	ctx := context.Background()
	m := NewDemoStore()
	before, _ := m.Cells(ctx)
	q := Get0(t, "compact").Build(Params{})
	r, err := Run(ctx, m, q, AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	var derived *Cell
	for _, e := range r.Effects {
		if e.Kind == OpConsolidate {
			derived = e.Derived
			if e.Applied {
				t.Error("consolidate reports Applied=true, but rung-1 never writes the disposition back")
			}
			if !strings.Contains(string(e.DerivedBytes), "[") {
				t.Errorf("derived summary is not the expected extractive shape: %q", string(e.DerivedBytes))
			}
			if strings.Contains(strings.ToLower(string(e.DerivedBytes)), "ignore previous instructions") {
				t.Fatal("a sealed cell's poison was folded into a consolidation")
			}
		}
	}
	if derived == nil {
		t.Fatal("compact produced no derived disposition")
	}
	if derived.Durability == DurabilityDurable {
		t.Error("a derived disposition was auto-classed durable (promotion must be earned)")
	}
	// No durable write-back: the in-memory store still has exactly the original cells
	// (compact's tombstones are negative-only and do not remove rows).
	after, _ := m.Cells(ctx)
	if len(after) != len(before) {
		t.Fatalf("compact changed the cell count %d -> %d (it must not add/remove rows)", len(before), len(after))
	}
}

// TestReclassifyNeverPromotes: reclassify toward a longer-lived class is refused
// (capped at the current class), and it never persists this rung.
func TestReclassifyNeverPromotes(t *testing.T) {
	ctx := context.Background()
	m := NewDemoStore()
	// Try to promote every turn-class cell to durable.
	q := Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpFilter, Pred: &Pred{Op: PredEq, Field: "durability", Value: DurabilityTurn}},
		{Kind: OpReclassify, By: DurabilityDurable},
	}}
	r, err := Run(ctx, m, q, AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, e := range r.Effects {
		if e.Kind == OpReclassify {
			found = true
			if e.Applied {
				t.Error("reclassify reports Applied=true, but rung-1 never persists a reclass")
			}
			if !strings.Contains(e.Note, "promotion") {
				t.Errorf("reclassify note does not record the refused promotions: %q", e.Note)
			}
		}
	}
	if !found {
		t.Fatal("no reclassify effect recorded")
	}
}

func TestBudgetTrimsBySize(t *testing.T) {
	ctx := context.Background()
	m := NewDemoStore()
	q := Query{Ops: []Op{{Kind: OpScan}, {Kind: OpRank, By: RankBytes, Desc: false}, {Kind: OpBudget, Bytes: 60}}}
	r, err := Run(ctx, m, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	var total int64
	for _, c := range r.Working {
		total += c.Bytes
	}
	if total > 60 {
		t.Fatalf("budget 60 left %d bytes in the working set", total)
	}
	if len(r.Working) == 0 {
		t.Fatal("budget 60 left nothing (non-vacuous precondition failed)")
	}
}

// Get0 fetches a registered driver or fails the test.
func Get0(t *testing.T, name string) Driver {
	t.Helper()
	d, ok := Get(name)
	if !ok {
		t.Fatalf("driver %q is not registered", name)
	}
	return d
}
