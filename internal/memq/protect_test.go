package memq

import (
	"bytes"
	"context"
	"reflect"
	"testing"
)

// protectSeed is the shared pressure fixture: two durable cells that rank LAST
// by relevance against the "alpha beta gamma" intent, and two ephemeral cells
// that rank first. Total 400 bytes.
func protectSeed() []Cell {
	return []Cell{
		{ID: "dur1", Step: 1, Role: "user", Descriptor: "standing preference: tabs", Bytes: 100, Durability: DurabilityDurable},
		{ID: "dur2", Step: 2, Role: "user", Descriptor: "standing preference: no color", Bytes: 100, Durability: DurabilityDurable},
		{ID: "eph1", Step: 10, Role: "tool", Descriptor: "alpha beta gamma notes", Bytes: 100, Durability: DurabilityTurn},
		{ID: "eph2", Step: 11, Role: "tool", Descriptor: "alpha beta findings", Bytes: 100, Durability: DurabilityTurn},
	}
}

// TestProtectedFloorDurableSurvivesBudget is witness (1) for #4017: under a
// byte budget smaller than the working set, every durable cell survives the
// protected budget even though it ranks LAST by relevance — the floor is
// charged against the cap before any scored keep (the NACL/Scissorhands
// split). It also pins the overflow-degrade rule: a floor that alone exceeds
// the cap keeps the most-recent protected cells that fit.
func TestProtectedFloorDurableSurvivesBudget(t *testing.T) {
	ctx := context.Background()
	q := Query{Intent: "alpha beta gamma", Ops: []Op{
		{Kind: OpScan},
		{Kind: OpRank, By: RankRelevance, Desc: true},
		{Kind: OpBudget, Bytes: 300, Protect: true},
		{Kind: OpRender},
	}}
	res, err := Run(ctx, fixedBackend{cells: protectSeed()}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	// Ranked order is eph1(3), eph2(2), dur1(0), dur2(0); the plain budget keeps
	// eph1,eph2,dur1 and tail-drops dur2. The floor charges dur1+dur2 first
	// (200), leaving 100 headroom: eph1 fits, eph2 overflows.
	if got := ids(res.Working); !reflect.DeepEqual(got, []string{"eph1", "dur1", "dur2"}) {
		t.Fatalf("protected working set = %v, want [eph1 dur1 dur2]", got)
	}
	if res.Overflow == nil || len(res.Overflow.Dropped) != 1 || res.Overflow.Dropped[0].ID != "eph2" {
		t.Fatalf("overflow = %+v, want exactly eph2 dropped", res.Overflow)
	}
	// Both durable cells actually rendered (survived selection AND page-in).
	rendered := renderedIDs(res)
	if !rendered["dur1"] || !rendered["dur2"] {
		t.Fatalf("durable cells not rendered: %+v", res.Rendered)
	}

	// Degrade rule: cap 150 < the 200-byte floor — the most-recent durable
	// (dur2, Step 2) survives; dur1 no longer fits; no ephemeral fits either.
	q.Ops[2] = Op{Kind: OpBudget, Bytes: 150, Protect: true}
	res, err = Run(ctx, fixedBackend{cells: protectSeed()}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(res.Working); !reflect.DeepEqual(got, []string{"dur2"}) {
		t.Fatalf("degraded floor = %v, want [dur2] (most-recent protected that fits)", got)
	}
}

// TestRecentWindowByteExactThroughCleanCompact is witness (2) for #4017: the
// top-N most-recent turn-class cells survive a protected budget under pressure
// at relevance 0 (eviction), and their bytes stay byte-identical after the
// clean+compact drivers run with full caps (compaction) — the byte-exact
// recent-window invariant, memq's form of ctxplan's exact recent area.
func TestRecentWindowByteExactThroughCleanCompact(t *testing.T) {
	ctx := context.Background()
	store := NewMemStore()
	dur := store.Add("user", "user", DurabilityDurable, []byte("standing preference: tabs not spaces"), false)
	store.Add("tool", "tool_result", DurabilityTurn, []byte("old scratch observation one"), false)
	store.Add("tool", "tool_result", DurabilityTurn, []byte("old scratch observation two"), false)
	w1 := store.Add("tool", "tool_result", DurabilityTurn, []byte("recent window body A, keep byte-exact"), false)
	w2 := store.Add("user", "user", DurabilityTurn, []byte("recent window body B, keep byte-exact"), false)

	before := map[string][]byte{}
	for _, c := range []Cell{w1, w2} {
		b, err := store.Materialize(ctx, c.ID)
		if err != nil {
			t.Fatalf("materialize %s before: %v", c.ID, err)
		}
		before[c.ID] = b
	}

	// Eviction half: a protected budget sized to exactly (durable + window)
	// keeps the top-2 recent turn-class cells even though nothing matches the
	// intent (every relevance score is 0).
	q := Query{Intent: "unrelated query terms", Ops: []Op{
		{Kind: OpScan},
		{Kind: OpRank, By: RankRelevance, Desc: true},
		{Kind: OpBudget, Bytes: dur.Bytes + w1.Bytes + w2.Bytes, Protect: true, Recent: 2},
	}}
	res, err := Run(ctx, store, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	kept := map[string]bool{}
	for _, c := range res.Working {
		kept[c.ID] = true
	}
	for _, id := range []string{dur.ID, w1.ID, w2.ID} {
		if !kept[id] {
			t.Fatalf("protected cell %s evicted; working=%v", id, ids(res.Working))
		}
	}

	// Compaction half: clean tombstones the turn class, compact folds+tombstones
	// what is left — and neither may touch the window's BYTES (negative-only
	// effects) nor fold the window into a lossy consolidation.
	for _, name := range []string{"clean", "compact"} {
		d, ok := Get(name)
		if !ok {
			t.Fatalf("driver %s not registered", name)
		}
		dres, err := Run(ctx, store, d.Build(Params{}), AllowAll())
		if err != nil {
			t.Fatalf("driver %s: %v", name, err)
		}
		for _, e := range dres.Effects {
			if e.Kind != OpConsolidate {
				continue
			}
			for _, id := range e.Cells {
				if id == w1.ID || id == w2.ID {
					t.Fatalf("recent-window cell %s folded into a consolidation", id)
				}
			}
		}
	}
	for _, c := range []Cell{w1, w2} {
		after, err := store.Materialize(ctx, c.ID)
		if err != nil {
			t.Fatalf("materialize %s after clean+compact: %v", c.ID, err)
		}
		if !bytes.Equal(before[c.ID], after) {
			t.Fatalf("recent-window cell %s bytes changed across clean+compact:\n before %q\n after  %q", c.ID, before[c.ID], after)
		}
		if got := Digest(after); got != c.Digest {
			t.Fatalf("recent-window cell %s digest drifted: %s -> %s", c.ID, c.Digest, got)
		}
	}
}

// TestProtectedBudgetDefaultOffByteIdentical pins the default-off contract for
// #4017: with Protect/Recent unset the budget op takes the applyBudget branch
// untouched (the low-relevance durable cell still tail-drops exactly as today),
// the render driver with the new Params fields unset builds a query DeepEqual
// to the pre-knob literal shape, and Validate refuses recent-without-protect.
func TestProtectedBudgetDefaultOffByteIdentical(t *testing.T) {
	ctx := context.Background()
	q := Query{Intent: "alpha beta gamma", Ops: []Op{
		{Kind: OpScan},
		{Kind: OpRank, By: RankRelevance, Desc: true},
		{Kind: OpBudget, Bytes: 300},
	}}
	res, err := Run(ctx, fixedBackend{cells: protectSeed()}, q, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	// Today's plain prefix budget over the ranked order: dur2 tail-drops.
	if got := ids(res.Working); !reflect.DeepEqual(got, []string{"eph1", "eph2", "dur1"}) {
		t.Fatalf("unset-path working set = %v, want [eph1 eph2 dur1] (unchanged tail-drop)", got)
	}
	if res.Overflow == nil || len(res.Overflow.Dropped) != 1 || res.Overflow.Dropped[0].ID != "dur2" {
		t.Fatalf("unset-path overflow = %+v, want exactly dur2 dropped", res.Overflow)
	}

	// The render driver with the knobs unset compiles to the exact legacy shape.
	d, ok := Get("render")
	if !ok {
		t.Fatal("render driver not registered")
	}
	got := d.Build(Params{Intent: "alpha", Budget: 300})
	want := Query{
		Intent: "alpha",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
				{Op: PredEq, Field: "sealed", Value: "false"},
				{Op: PredEq, Field: "tombstoned", Value: "false"},
				{Op: PredOr, Args: []Pred{
					{Op: PredMatch, Value: "alpha"},
					{Op: PredEq, Field: "durability", Value: DurabilityDurable},
				}},
			}}},
			{Kind: OpDedup},
			{Kind: OpRank, By: RankRelevance, Desc: true},
			{Kind: OpBudget, Bytes: 300},
			{Kind: OpRender},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("render driver with knobs unset drifted from the legacy query:\n got %+v\nwant %+v", got, want)
	}

	// Fail-closed: a recent window without the protect gate never runs.
	bad := Query{Ops: []Op{{Kind: OpBudget, Bytes: 10, Recent: 2}}}
	if err := Validate(bad); err == nil {
		t.Fatal("Validate accepted recent without protect")
	}
}
