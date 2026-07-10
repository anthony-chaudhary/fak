package memq

import (
	"bytes"
	"context"
	"reflect"
	"testing"
)

// exemptFixtureStore builds a store with three near-duplicate session-class notes
// (redundant fold fodder with heavy shared vocabulary) and ONE lexically disjoint
// outlier — the cell a lossy consolidation would approximate worst. All four are
// unsealed, unreferenced (refcount 0), and non-durable, so all four are compact
// candidates.
func exemptFixtureStore() (*MemStore, Cell, []byte) {
	m := NewMemStore()
	m.Add("build", "tool_result", DurabilitySession, []byte("the build server acme runs in the east region and serves the build farm"), false)
	m.Add("build", "tool_result", DurabilitySession, []byte("the build server named acme is in the east region serving the build farm"), false)
	m.Add("build", "tool_result", DurabilitySession, []byte("acme the build server sits in the east region for the build farm"), false)
	outlierBody := []byte("wombat kayak zephyr: quarterly ledger reconciliation formula uses column q17")
	outlier := m.Add("ledger", "tool_result", DurabilitySession, outlierBody, false)
	return m, outlier, outlierBody
}

// TestCompactExemptsDivergentOutlier is the issue's witness: with RetentionCount set,
// the lone divergence outlier survives compaction BIT-EXACT (not folded, not
// tombstoned, bytes unchanged) while its redundant neighbors fold and tombstone.
func TestCompactExemptsDivergentOutlier(t *testing.T) {
	ctx := context.Background()
	m, outlier, outlierBody := exemptFixtureStore()

	q := Get0(t, "compact").Build(Params{RetentionCount: 1})
	if err := Validate(q); err != nil {
		t.Fatalf("compact+retention query refused: %v", err)
	}
	res, err := Run(ctx, m, q, AllowAll())
	if err != nil {
		t.Fatal(err)
	}

	var exemptEff, consEff, tombEff *Effect
	for i := range res.Effects {
		switch res.Effects[i].Kind {
		case OpExempt:
			exemptEff = &res.Effects[i]
		case OpConsolidate:
			consEff = &res.Effects[i]
		case OpTombstone:
			tombEff = &res.Effects[i]
		}
	}
	if exemptEff == nil || consEff == nil || tombEff == nil {
		t.Fatalf("compact with RetentionCount must record exempt+consolidate+tombstone effects; got %+v", res.Effects)
	}

	// (a) the carve names exactly the divergent outlier.
	if len(exemptEff.Cells) != 1 || exemptEff.Cells[0] != outlier.ID {
		t.Fatalf("exempt effect = %v, want exactly the outlier %s", exemptEff.Cells, outlier.ID)
	}
	// (b) the fold excludes the outlier and still folds the 3 redundant neighbors.
	for _, id := range consEff.Cells {
		if id == outlier.ID {
			t.Fatalf("outlier %s was folded into the consolidation despite the exemption", outlier.ID)
		}
	}
	if len(consEff.Cells) != 3 {
		t.Fatalf("consolidate folded %d cell(s), want the 3 redundant neighbors", len(consEff.Cells))
	}
	// (c) the tombstone pass never touches the exempted cell, and the store agrees.
	for _, id := range tombEff.Cells {
		if id == outlier.ID {
			t.Fatalf("outlier %s was tombstoned despite the exemption", outlier.ID)
		}
	}
	cells, _ := m.Cells(ctx)
	tombstoned := 0
	for _, c := range cells {
		if c.Tombstoned {
			tombstoned++
		}
		if c.ID == outlier.ID {
			if c.Tombstoned {
				t.Fatal("outlier survived the fold but was tombstoned")
			}
			if c.Digest != outlier.Digest {
				t.Fatalf("outlier digest changed %s -> %s", outlier.Digest, c.Digest)
			}
		}
	}
	if tombstoned != 3 {
		t.Fatalf("redundant neighbors tombstoned = %d, want 3", tombstoned)
	}
	// (d) ... and the outlier's bytes survive BIT-EXACT.
	got, err := m.Materialize(ctx, outlier.ID)
	if err != nil {
		t.Fatalf("outlier no longer materializes: %v", err)
	}
	if !bytes.Equal(got, outlierBody) {
		t.Fatalf("outlier bytes changed: got %q want %q", got, outlierBody)
	}
}

// TestRankDivergenceOrdersOutlierFirst exercises the ranker half (#4018): an
// agent-authored query can sort by divergence directly, and desc puts the lexical
// outlier first.
func TestRankDivergenceOrdersOutlierFirst(t *testing.T) {
	ctx := context.Background()
	m, outlier, _ := exemptFixtureStore()
	q := Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpRank, By: RankDivergence, Desc: true},
	}}
	if err := Validate(q); err != nil {
		t.Fatalf("divergence rank refused: %v", err)
	}
	res, err := Run(ctx, m, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Working) != 4 || res.Working[0].ID != outlier.ID {
		t.Fatalf("divergence desc must order the outlier first; got %v", ids(res.Working))
	}
}

// TestCompactDefaultOffBytesIdentical pins the default-off contract: with
// RetentionCount unset the compact driver builds EXACTLY the pre-#4018 pipeline, and
// a run carries no exemption — every candidate (outlier included) folds as today.
func TestCompactDefaultOffBytesIdentical(t *testing.T) {
	ctx := context.Background()

	got := Get0(t, "compact").Build(Params{})
	want := Query{
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
				{Op: PredEq, Field: "sealed", Value: "false"},
				{Op: PredEq, Field: "tombstoned", Value: "false"},
				{Op: PredNe, Field: "durability", Value: DurabilityDurable},
				{Op: PredEq, Field: "refcount", Value: "0"},
			}}},
			{Kind: OpRank, By: RankBytes, Desc: true},
			{Kind: OpBudget, Bytes: 0},
			{Kind: OpConsolidate},
			{Kind: OpTombstone, Reason: "compacted into a derived disposition"},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("default compact query changed:\n got %+v\nwant %+v", got, want)
	}

	m, outlier, _ := exemptFixtureStore()
	res, err := Run(ctx, m, got, AllowAll())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range res.Effects {
		if e.Kind == OpExempt {
			t.Fatalf("default (RetentionCount unset) run recorded an exempt effect: %+v", e)
		}
	}
	var consEff *Effect
	for i := range res.Effects {
		if res.Effects[i].Kind == OpConsolidate {
			consEff = &res.Effects[i]
		}
	}
	if consEff == nil || len(consEff.Cells) != 4 {
		t.Fatalf("default compact must fold all 4 candidates (outlier %s included); got %+v", outlier.ID, consEff)
	}
}
