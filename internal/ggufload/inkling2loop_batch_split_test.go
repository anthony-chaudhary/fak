package ggufload

import (
	"errors"
	"reflect"
	"sort"
	"testing"
)

// ink2SplitPrompts returns n prompts of the given lengths, each row's tokens distinct.
func ink2SplitPrompts(lengths []int) [][]int {
	prompts := make([][]int, len(lengths))
	for i, n := range lengths {
		prompts[i] = make([]int, n)
		for j := range prompts[i] {
			prompts[i][j] = 1000*i + j
		}
	}
	return prompts
}

// ink2SplitRun runs rows of prompts as one cohort on bt and returns each row's tokens.
func ink2SplitRun(t *testing.T, bt *Inkling2LoopCohorts, prompts [][]int, rows []int, maxNew int, got [][]int) {
	t.Helper()
	reqs := make([]Inkling2LoopBatchRequest, len(rows))
	for k, row := range rows {
		reqs[k] = Inkling2LoopBatchRequest{Prompt: prompts[row], MaxTokens: maxNew,
			Emit: func(tok int) bool { got[row] = append(got[row], tok); return true }}
	}
	results, err := bt.Generate(reqs)
	if err != nil {
		t.Fatalf("cohort of rows %v: %v", rows, err)
	}
	for k, r := range results {
		if r.Err != nil || r.Generated != maxNew {
			t.Fatalf("row %d: %+v", rows[k], r)
		}
	}
}

func ink2SplitPositions(prompts [][]int, maxNew int) []int {
	pos := make([]int, len(prompts))
	for i, p := range prompts {
		pos[i] = len(p) + maxNew
	}
	return pos
}

// ink2SplitBudget caps d so a C x ctx state fits beside what is live now, and no more.
func ink2SplitBudget(t *testing.T, bt *Inkling2LoopCohorts, d *fakeInk2Cohort, C, ctx int) {
	t.Helper()
	if err := bt.bind(); err != nil {
		t.Fatal(err)
	}
	sh, err := bt.s.batchShape(C, ctx)
	if err != nil {
		t.Fatal(err)
	}
	d.capFloats = d.live + int((sh.need+inkling2BatchHeadroom+3)/4)
}

// A cohort whose state does not fit free device memory is split into sub-cohorts
// that each fit, and every row completes with the tokens a cohort with room produces.
func TestInkling2CohortsSplitOverBudgetEveryRowCompletes(t *testing.T) {
	const maxNew = 5
	lengths := make([]int, Inkling2LoopMaxBatch)
	for i := range lengths {
		lengths[i] = 3 + i%7
	}
	prompts := ink2SplitPrompts(lengths)
	rows := make([]int, len(prompts))
	for i := range rows {
		rows[i] = i
	}

	want := make([][]int, len(prompts))
	roomy := NewInkling2LoopCohorts(fakeInk2CohortModel(), newFakeInk2Cohort())
	ink2SplitRun(t, roomy, prompts, rows, maxNew, want)
	roomy.Close()

	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	ink2SplitBudget(t, bt, d, Inkling2LoopMaxBatch/2, 64)

	// the whole cohort is refused before any row runs, and the binding survives it
	_, err := bt.Generate(ink2SplitReqs(prompts, maxNew))
	var refused *Inkling2BatchMemoryError
	if !errors.As(err, &refused) {
		t.Fatalf("over-budget cohort: err %v, want an Inkling2BatchMemoryError", err)
	}
	if bt.s == nil {
		t.Fatal("a refused cohort dropped the weight binding")
	}

	groups, err := bt.Split(ink2SplitPositions(prompts, maxNew))
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) < 2 {
		t.Fatalf("over-budget cohort split into %d groups", len(groups))
	}
	var seen []int
	for _, g := range groups {
		if len(g) < 2 || len(g) > Inkling2LoopMaxBatch/2 {
			t.Fatalf("group %v outside 2..%d rows", g, Inkling2LoopMaxBatch/2)
		}
		seen = append(seen, g...)
	}
	sort.Ints(seen)
	if !reflect.DeepEqual(seen, rows) {
		t.Fatalf("groups %v do not cover every row once", groups)
	}

	got := make([][]int, len(prompts))
	for _, g := range groups {
		ink2SplitRun(t, bt, prompts, g, maxNew, got)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("split rows emitted %v, one roomy cohort %v", got, want)
	}
}

func ink2SplitReqs(prompts [][]int, maxNew int) []Inkling2LoopBatchRequest {
	reqs := make([]Inkling2LoopBatchRequest, len(prompts))
	for i, p := range prompts {
		reqs[i] = Inkling2LoopBatchRequest{Prompt: p, MaxTokens: maxNew, Emit: func(int) bool { return true }}
	}
	return reqs
}

// Fifteen short rows beside one long row that cannot share a state with them: the
// short rows run as one cohort sized for their own length, and the long row is left
// alone for the serialized engine.
func TestInkling2CohortsSplitKeepsLongRowOffShortRows(t *testing.T) {
	const maxNew = 8
	longLen := min(3000, Inkling2LoopResidentContext-maxNew)
	lengths := make([]int, Inkling2LoopMaxBatch)
	for i := range lengths {
		lengths[i] = 19
	}
	const long = 6
	lengths[long] = longLen
	prompts := ink2SplitPrompts(lengths)

	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	ink2SplitBudget(t, bt, d, Inkling2LoopMaxBatch-1, 64)

	groups, err := bt.Split(ink2SplitPositions(prompts, maxNew))
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || len(groups[0]) != Inkling2LoopMaxBatch-1 || !reflect.DeepEqual(groups[1], []int{long}) {
		t.Fatalf("groups %v, want the 15 short rows then [%d]", groups, long)
	}
	got := make([][]int, len(prompts))
	ink2SplitRun(t, bt, prompts, groups[0], maxNew, got)
	if bt.b.capC != Inkling2LoopMaxBatch-1 || bt.b.capCtx != 64 {
		t.Fatalf("short rows ran on a %dx%d state, want %dx64", bt.b.capC, bt.b.capCtx, Inkling2LoopMaxBatch-1)
	}
}

// After a wide cohort, a narrower one runs on a state sized for its own width and
// length, never on the wide state or the union of both shapes; a cohort close to the
// cached shape still reuses it.
func TestInkling2CohortsShrinkAfterWideCohort(t *testing.T) {
	const maxNew = 2
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	run := func(width, promptLen int) {
		t.Helper()
		lengths := make([]int, width)
		for i := range lengths {
			lengths[i] = promptLen
		}
		rows := make([]int, width)
		for i := range rows {
			rows[i] = i
		}
		ink2SplitRun(t, bt, ink2SplitPrompts(lengths), rows, maxNew, make([][]int, width))
	}

	run(Inkling2LoopMaxBatch, 100) // 16 x 128
	wide := d.live
	allocs := d.allocs
	run(Inkling2LoopMaxBatch, 100)
	run(Inkling2LoopMaxBatch-1, 100)
	if d.allocs != allocs {
		t.Fatalf("a cohort near the cached shape allocated again: %d -> %d allocs", allocs, d.allocs)
	}

	run(Inkling2LoopMaxBatch/2, 10) // 8 x 64
	if bt.b.capC != Inkling2LoopMaxBatch/2 || bt.b.capCtx != 64 {
		t.Fatalf("8x64 cohort after 16x128 ran on a %dx%d state", bt.b.capC, bt.b.capCtx)
	}
	if d.live >= wide {
		t.Fatalf("narrow state holds %d floats, the wide one held %d", d.live, wide)
	}

	run(Inkling2LoopMaxBatch, 10) // 16 x 64: wider than the cached 8, not the 128 seen before
	if bt.b.capC != Inkling2LoopMaxBatch || bt.b.capCtx != 64 {
		t.Fatalf("16x64 cohort after 8x64 ran on a %dx%d state, want 16x64", bt.b.capC, bt.b.capCtx)
	}
	bt.Close()
	if d.live != 0 || d.frees != d.allocs {
		t.Fatalf("after Close: %d floats live, %d frees of %d allocs", d.live, d.frees, d.allocs)
	}
}
