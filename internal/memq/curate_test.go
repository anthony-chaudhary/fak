package memq

import (
	"context"
	"reflect"
	"testing"
)

// tombBackend is a deterministic Backend that also records applied tombstones, so the
// caps-gated ApplyCurate path can be witnessed end to end.
type tombBackend struct {
	cells  []Cell
	tombed map[string]string // id -> reason
}

func newTombBackend(cells []Cell) *tombBackend {
	return &tombBackend{cells: cells, tombed: map[string]string{}}
}
func (b *tombBackend) Cells(context.Context) ([]Cell, error) { return b.cells, nil }
func (b *tombBackend) Materialize(_ context.Context, id string) ([]byte, error) {
	for _, c := range b.cells {
		if c.ID == id {
			return []byte(c.Descriptor), nil
		}
	}
	return nil, ErrSealed
}
func (b *tombBackend) Tombstone(_ context.Context, id, reason, _ string) (bool, error) {
	b.tombed[id] = reason
	return true, nil
}

func evIDs(evs []Eviction) []string {
	out := make([]string, len(evs))
	for i, e := range evs {
		out[i] = e.ID
	}
	return out
}

// TestBudgetCurateRanksByValueNotSize is the core #3908 witness: given EQUAL-size cells,
// eviction under a hard cap removes the LOWEST witnessed-value cells first — value, never
// size or capability presence (memvaluescore score.go:21-23), drives the cut.
func TestBudgetCurateRanksByValueNotSize(t *testing.T) {
	// Four equal-size (100B) ephemeral cells; a 200B cap keeps exactly two. Value is the
	// only discriminator, so the two LOWEST-value cells must be the ones evicted.
	cells := []Cell{
		{ID: "hi", Step: 1, Descriptor: "high value", Bytes: 100, Durability: "session"},
		{ID: "lo", Step: 2, Descriptor: "low value", Bytes: 100, Durability: "session"},
		{ID: "mid", Step: 3, Descriptor: "mid value", Bytes: 100, Durability: "session"},
		{ID: "zero", Step: 4, Descriptor: "no value", Bytes: 100, Durability: "session"},
	}
	value := map[string]int{"hi": 16, "mid": 4, "lo": 2, "zero": 0}

	rep := BudgetCurate(cells, 200, value)
	if rep.Reason != CurateReason {
		t.Fatalf("reason = %q, want %q", rep.Reason, CurateReason)
	}
	if rep.Kept != 2 {
		t.Fatalf("kept = %d, want 2", rep.Kept)
	}
	// The two lowest-value cells evict, reported worst-first (value asc).
	got := evIDs(rep.Evicted)
	want := []string{"zero", "lo"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("evicted = %v, want %v (lowest-value-first)", got, want)
	}
	if rep.Evicted[0].Value != 0 || rep.Evicted[1].Value != 2 {
		t.Fatalf("evicted values = %d,%d, want 0,2", rep.Evicted[0].Value, rep.Evicted[1].Value)
	}
}

// TestBudgetCurateValueBeatsSize proves size is only the COST axis: a small HIGH-value
// cell survives while a large LOW-value cell is evicted, even though evicting the small
// one would "save more per cell" under a size-ranked policy.
func TestBudgetCurateValueBeatsSize(t *testing.T) {
	cells := []Cell{
		{ID: "small_hi", Step: 1, Descriptor: "small but witnessed", Bytes: 20, Durability: "session"},
		{ID: "big_lo", Step: 2, Descriptor: "big but never recalled", Bytes: 100, Durability: "session"},
	}
	value := map[string]int{"small_hi": 8, "big_lo": 0}

	// Cap 100 cannot hold both (120B). Value keeps small_hi, evicts big_lo.
	rep := BudgetCurate(cells, 100, value)
	if got := evIDs(rep.Evicted); !reflect.DeepEqual(got, []string{"big_lo"}) {
		t.Fatalf("evicted = %v, want [big_lo] (value beats size)", got)
	}
	if rep.Kept != 1 || rep.KeptBytes != 20 {
		t.Fatalf("kept = %d (%dB), want 1 (20B)", rep.Kept, rep.KeptBytes)
	}
}

// TestBudgetCurateProtectsFloor pins the fail-closed floor (#3908 DoD 2): a durable, a
// referenced, and an intentional-floor cell are NEVER evicted, even at zero value and
// even when that forces an ephemeral cell out to make room.
func TestBudgetCurateProtectsFloor(t *testing.T) {
	cells := []Cell{
		{ID: "durable", Step: 1, Descriptor: "standing pref", Bytes: 100, Durability: "durable"},
		{ID: "floored", Step: 2, Descriptor: "operator guard", Bytes: 100, Durability: "session",
			Attrs: map[string]string{FloorAttr: "1"}},
		{ID: "target", Step: 3, Descriptor: "referenced target", Bytes: 100, Durability: "session", Digest: "d1"},
		{ID: "referrer", Step: 4, Descriptor: "points at target", Bytes: 100, Durability: "session", Refs: []string{"d1"}},
		{ID: "ephemeral", Step: 5, Descriptor: "plain ephemeral", Bytes: 100, Durability: "session"},
	}
	// The floor is durable ∪ floored ∪ the REFERENCED target (referrer points at it) =
	// 300B; the two unprotected candidates are referrer and ephemeral. Cap 400 leaves
	// 100B headroom for exactly one, so value must decide: referrer is witnessed, the
	// plain ephemeral is not, so ephemeral is the single eviction — no floor cell ever.
	value := map[string]int{"referrer": 8, "ephemeral": 0}
	rep := BudgetCurate(cells, 400, value)
	if got := evIDs(rep.Evicted); !reflect.DeepEqual(got, []string{"ephemeral"}) {
		t.Fatalf("evicted = %v, want [ephemeral] (floor never evicts, value ranks the rest)", got)
	}
	wantProtected := []string{"durable", "floored", "target"}
	if !reflect.DeepEqual(rep.Protected, wantProtected) {
		t.Fatalf("protected = %v, want %v", rep.Protected, wantProtected)
	}
}

// TestBudgetCurateFloorOverBudget is the fail-closed extreme: when the protected floor
// alone exceeds the cap, NO protected cell is dropped and the breach is typed.
func TestBudgetCurateFloorOverBudget(t *testing.T) {
	cells := []Cell{
		{ID: "d1", Step: 1, Descriptor: "durable one", Bytes: 100, Durability: "durable"},
		{ID: "d2", Step: 2, Descriptor: "durable two", Bytes: 100, Durability: "durable"},
		{ID: "eph", Step: 3, Descriptor: "ephemeral", Bytes: 100, Durability: "session"},
	}
	rep := BudgetCurate(cells, 50, nil) // floor is 200B, cap is 50B
	if !rep.FloorOverBudget {
		t.Fatal("floor (200B) exceeds cap (50B) but FloorOverBudget is false")
	}
	// The durables survive regardless; only the unprotected ephemeral is evicted.
	if got := evIDs(rep.Evicted); !reflect.DeepEqual(got, []string{"eph"}) {
		t.Fatalf("evicted = %v, want [eph] (protected floor never dropped)", got)
	}
	if rep.Kept != 2 {
		t.Fatalf("kept = %d, want 2 (both durables)", rep.Kept)
	}
}

// TestBudgetCurateUnboundedKeepsAll pins budget<=0 as unbounded: no eviction.
func TestBudgetCurateUnboundedKeepsAll(t *testing.T) {
	cells := []Cell{
		{ID: "a", Bytes: 100, Durability: "session"},
		{ID: "b", Bytes: 100, Durability: "session"},
	}
	rep := BudgetCurate(cells, 0, nil)
	if len(rep.Evicted) != 0 || rep.Kept != 2 {
		t.Fatalf("unbounded budget evicted %d / kept %d, want 0 / 2", len(rep.Evicted), rep.Kept)
	}
}

// TestBudgetCurateDeterministic proves the pass is byte-identical across runs (the memq
// determinism contract) — same (cells, budget, value) yields an equal report.
func TestBudgetCurateDeterministic(t *testing.T) {
	cells := []Cell{
		{ID: "a", Step: 1, Descriptor: "a", Bytes: 100, Durability: "session"},
		{ID: "b", Step: 2, Descriptor: "b", Bytes: 100, Durability: "session"},
		{ID: "c", Step: 3, Descriptor: "c", Bytes: 100, Durability: "session"},
	}
	value := map[string]int{"a": 1, "b": 1, "c": 0} // a,b tie — the tiebreak must be stable
	r1 := BudgetCurate(cells, 100, value)
	r2 := BudgetCurate(cells, 100, value)
	if !reflect.DeepEqual(r1, r2) {
		t.Fatalf("non-deterministic report:\n r1=%+v\n r2=%+v", r1, r2)
	}
}

// TestEvictionRegretReverts is the keep-bit witness (#3908 DoD 3): when a later recall
// needs an evicted cell, regret rises over the no-eviction baseline (0) and the policy
// REVERTS; the gross/net split accounts what forgetting actually cost.
func TestEvictionRegretReverts(t *testing.T) {
	cells := []Cell{
		{ID: "keep", Step: 1, Descriptor: "kept", Bytes: 100, Durability: "session"},
		{ID: "drop_a", Step: 2, Descriptor: "evicted a", Bytes: 100, Durability: "session"},
		{ID: "drop_b", Step: 3, Descriptor: "evicted b", Bytes: 100, Durability: "session"},
	}
	value := map[string]int{"keep": 8, "drop_a": 0, "drop_b": 0}
	rep := BudgetCurate(cells, 100, value)
	if len(rep.Evicted) != 2 {
		t.Fatalf("evicted = %d, want 2", len(rep.Evicted))
	}

	// A later recall needed drop_a (evicted) and keep (still present): one regret.
	reg := CurateRegret(rep, []string{"keep", "drop_a"})
	if reg.Reason != RegretReason {
		t.Fatalf("regret reason = %q, want %q", reg.Reason, RegretReason)
	}
	if reg.Regret != 1 || !reflect.DeepEqual(reg.Regretted, []string{"drop_a"}) {
		t.Fatalf("regret = %d (%v), want 1 ([drop_a])", reg.Regret, reg.Regretted)
	}
	if reg.Baseline != 0 || !reg.Reverts {
		t.Fatalf("baseline=%d reverts=%v, want 0/true (any regret exceeds the no-eviction baseline)", reg.Baseline, reg.Reverts)
	}
	// gross = 200B reclaimed; regret cost = 100B (drop_a had to be recovered); net = 100B.
	if reg.GrossReclaimed != 200 || reg.RegretCost != 100 || reg.NetReclaimed != 100 {
		t.Fatalf("gross/cost/net = %d/%d/%d, want 200/100/100", reg.GrossReclaimed, reg.RegretCost, reg.NetReclaimed)
	}
}

// TestEvictionRegretClean pins the net-positive case: a later recall that never touches an
// evicted cell accrues zero regret, so the policy is kept (net == gross).
func TestEvictionRegretClean(t *testing.T) {
	cells := []Cell{
		{ID: "keep", Step: 1, Descriptor: "kept", Bytes: 100, Durability: "session"},
		{ID: "drop", Step: 2, Descriptor: "evicted", Bytes: 100, Durability: "session"},
	}
	rep := BudgetCurate(cells, 100, map[string]int{"keep": 4, "drop": 0})
	reg := CurateRegret(rep, []string{"keep"})
	if reg.Regret != 0 || reg.Reverts {
		t.Fatalf("regret=%d reverts=%v, want 0/false", reg.Regret, reg.Reverts)
	}
	if reg.NetReclaimed != reg.GrossReclaimed || reg.NetReclaimed != 100 {
		t.Fatalf("net=%d gross=%d, want equal 100 (no regret cost)", reg.NetReclaimed, reg.GrossReclaimed)
	}
}

// TestApplyCurateFailClosed proves the caps gate: without a grant the eviction is
// PROPOSED and the backend is untouched; with a grant + a Tombstoner it is applied.
func TestApplyCurateFailClosed(t *testing.T) {
	cells := []Cell{
		{ID: "keep", Step: 1, Descriptor: "kept", Bytes: 100, Durability: "session"},
		{ID: "drop", Step: 2, Descriptor: "evicted", Bytes: 100, Durability: "session"},
	}
	rep := BudgetCurate(cells, 100, map[string]int{"keep": 4, "drop": 0})
	if len(rep.Evicted) != 1 {
		t.Fatalf("evicted = %d, want 1", len(rep.Evicted))
	}
	ctx := context.Background()

	// No caps: proposal only, nothing tombstoned.
	be := newTombBackend(cells)
	eff := ApplyCurate(ctx, be, rep, Caps{})
	if eff.Applied || len(be.tombed) != 0 {
		t.Fatalf("no-caps ApplyCurate applied=%v tombed=%v, want false / none", eff.Applied, be.tombed)
	}
	if len(eff.Cells) != 1 || eff.Cells[0] != "drop" {
		t.Fatalf("proposed cells = %v, want [drop]", eff.Cells)
	}

	// Caps granted + a Tombstoner backend: the eviction is applied and audited.
	be2 := newTombBackend(cells)
	eff2 := ApplyCurate(ctx, be2, rep, AllowAll())
	if !eff2.Applied {
		t.Fatal("caps-granted ApplyCurate did not apply")
	}
	if r, ok := be2.tombed["drop"]; !ok || r != CurateReason {
		t.Fatalf("drop tombstone reason = %q (present=%v), want %q", r, ok, CurateReason)
	}
	if _, ok := be2.tombed["keep"]; ok {
		t.Fatal("kept cell was tombstoned — eviction touched a survivor")
	}
}
