package model

import (
	"errors"
	"reflect"
	"testing"
)

// TestActiveRankMaskExcludesDeadRanks is the issue's first checkable step: a 4-rank group with
// rank 2 marked inactive. The mask must exclude rank 2 from the live subset, and the masked
// combine must equal the exact reduction over ranks {0,1,3} — partial-but-valid — rather than
// failing closed the way the un-masked collective would on a rank hole.
func TestActiveRankMaskExcludesDeadRanks(t *testing.T) {
	mask, err := NewActiveRankMask(4, []int{0, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	if mask.WorldSize() != 4 || mask.ActiveCount() != 3 {
		t.Fatalf("world=%d active=%d, want 4 and 3", mask.WorldSize(), mask.ActiveCount())
	}
	if mask.Active(2) {
		t.Fatal("rank 2 must be masked out")
	}
	if !mask.Active(0) || !mask.Active(1) || !mask.Active(3) {
		t.Fatal("ranks 0,1,3 must be live")
	}
	if got := mask.ActiveRanks(); !reflect.DeepEqual(got, []int{0, 1, 3}) {
		t.Fatalf("ActiveRanks=%v, want [0 1 3]", got)
	}
	if got := mask.DeadRanks(); !reflect.DeepEqual(got, []int{2}) {
		t.Fatalf("DeadRanks=%v, want [2]", got)
	}

	// Rank 2's partial is present but must NOT be summed — the combine skips it.
	partials := [][]float32{
		{1, 2, 3},       // rank 0 live
		{10, 20, 30},    // rank 1 live
		{99, 99, 99},    // rank 2 DEAD — must be ignored
		{100, 200, 300}, // rank 3 live
	}
	got, contributed, err := CombineActivePartials(partials, mask)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{111, 222, 333} // 0 + 1 + 3, rank 2 excluded
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("masked combine=%v, want exact reduction over {0,1,3}=%v", got, want)
	}
	if contributed != 3 {
		t.Fatalf("contributed=%d, want 3", contributed)
	}

	// The all-active mask over the same partials sums ALL four ranks — proof rank 2's exclusion
	// is the mask's doing, not a dropped term.
	all, err := AllActive(4)
	if err != nil {
		t.Fatal(err)
	}
	full, _, err := CombineActivePartials(partials, all)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(full, want) {
		t.Fatalf("all-active combine %v must differ from masked %v", full, want)
	}
	if !reflect.DeepEqual(full, []float32{210, 321, 432}) {
		t.Fatalf("all-active combine=%v, want [210 321 432]", full)
	}
}

// TestMaskDispatchDropsOrphanedExpertsFailClosed asserts the dispatch-side route-around: traffic
// bound for a dead rank is dropped fail-closed and its experts reported as orphaned, while live
// ranks keep their work verbatim — no traffic is ever routed to a masked-out rank.
func TestMaskDispatchDropsOrphanedExpertsFailClosed(t *testing.T) {
	mask, err := NewActiveRankMask(4, []int{0, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	dispatch := map[int][]V4ExpertDispatch{
		0: {{Rank: 0, Expert: 0, Weight: .5}},
		1: {{Rank: 1, Expert: 5, Weight: .25}},
		2: {{Rank: 2, Expert: 9, Weight: .1}, {Rank: 2, Expert: 8, Weight: .2}, {Rank: 2, Expert: 9, Weight: .3}}, // DEAD owner
		3: {{Rank: 3, Expert: 12, Weight: .4}},
	}
	live, orphaned, err := MaskDispatch(dispatch, mask)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := live[2]; ok {
		t.Fatal("no traffic may survive for masked-out rank 2")
	}
	if len(live) != 3 {
		t.Fatalf("live ranks=%d, want 3", len(live))
	}
	if !reflect.DeepEqual(live[0], dispatch[0]) || !reflect.DeepEqual(live[1], dispatch[1]) || !reflect.DeepEqual(live[3], dispatch[3]) {
		t.Fatalf("live dispatch mangled: %v", live)
	}
	// Experts 8 and 9 owned by dead rank 2 are orphaned, deduped and ascending.
	if !reflect.DeepEqual(orphaned, []int{8, 9}) {
		t.Fatalf("orphaned=%v, want [8 9]", orphaned)
	}
}

// TestCombineActivePartialsFailsClosedOnLiveWidthHole asserts a width mismatch AMONG LIVE ranks is
// a real inconsistency that fails closed, whereas a dead rank's odd-width partial is ignored.
func TestCombineActivePartialsFailsClosedOnLiveWidthHole(t *testing.T) {
	mask, err := NewActiveRankMask(3, []int{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = CombineActivePartials([][]float32{{1, 2}, {3}, {4, 5}}, mask)
	if !errors.Is(err, ErrActiveRankMembership) {
		t.Fatalf("live width hole err=%v, want ErrActiveRankMembership", err)
	}

	// The SAME odd-width vector on a DEAD rank is fine — it is never read for width or summed.
	deadMask, err := NewActiveRankMask(3, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	got, contributed, err := CombineActivePartials([][]float32{{1, 2}, {3}, {4, 5}}, deadMask)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []float32{5, 7}) || contributed != 2 {
		t.Fatalf("dead odd-width combine=%v contributed=%d, want [5 7] and 2", got, contributed)
	}
}

// TestActiveRankMaskConstructorGuards pins the fail-closed edges: empty membership, out-of-range
// member, bad world size, live-rank-with-no-partial subset, and a dispatch key beyond the world.
func TestActiveRankMaskConstructorGuards(t *testing.T) {
	if _, err := NewActiveRankMask(4, nil); !errors.Is(err, ErrActiveRankMembership) {
		t.Fatalf("empty membership err=%v", err)
	}
	if _, err := NewActiveRankMask(4, []int{0, 4}); !errors.Is(err, ErrActiveRankMembership) {
		t.Fatalf("out-of-range member err=%v", err)
	}
	if _, err := AllActive(0); !errors.Is(err, ErrActiveRankMembership) {
		t.Fatalf("zero world err=%v", err)
	}

	mask, err := NewActiveRankMask(2, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	// Wrong partial count for the world.
	if _, _, err := CombineActivePartials([][]float32{{1}}, mask); !errors.Is(err, ErrActiveRankMembership) {
		t.Fatalf("partial-count mismatch err=%v", err)
	}
	// No live rank carries a partial → empty live subset.
	if _, _, err := CombineActivePartials([][]float32{nil, nil}, mask); !errors.Is(err, ErrActiveRankMembership) {
		t.Fatalf("empty live subset err=%v", err)
	}
	// A dispatch key beyond the world fails closed.
	if _, _, err := MaskDispatch(map[int][]V4ExpertDispatch{5: {{Rank: 5, Expert: 1}}}, mask); !errors.Is(err, ErrActiveRankMembership) {
		t.Fatalf("out-of-world dispatch key err=%v", err)
	}

	// Duplicate members are idempotent — same live subset, no double count.
	dup, err := NewActiveRankMask(3, []int{1, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if dup.ActiveCount() != 2 || !dup.Active(1) || !dup.Active(2) || dup.Active(0) {
		t.Fatalf("dup membership=%v active=%d", dup.ActiveRanks(), dup.ActiveCount())
	}
}
