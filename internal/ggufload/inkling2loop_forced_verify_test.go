package ggufload

import (
	"context"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// countingInk2Cohort wraps fakeInk2CohortWithCaps to count serial MatVec
// calls (used by Step and final-token output projection) vs batched PRMS rows
// (used by Prefill).
type countingInk2Cohort struct {
	fakeInk2CohortWithCaps
	matVecCalls int
	pRMSRows    []int
}

func (c *countingInk2Cohort) Ink2MatVec(w compute.Tensor, x, out compute.Ink2Ref) error {
	c.matVecCalls++
	return c.fakeInk2CohortWithCaps.Ink2MatVec(w, x, out)
}

func (c *countingInk2Cohort) Ink2PRMS(x, w, y compute.Ink2Ref, rows, n int, eps float32) error {
	c.pRMSRows = append(c.pRMSRows, rows)
	return c.fakeInk2CohortWithCaps.Ink2PRMS(x, w, y, rows, n, eps)
}

type recordingSpecDrafter struct {
	observed [][]int
	keeps    []int
}

func (d *recordingSpecDrafter) Draft([]int, int) []int { return nil }

func (d *recordingSpecDrafter) Observe(_ *inkling2Resident, fed []int, keep int) error {
	d.observed = append(d.observed, append([]int(nil), fed...))
	d.keeps = append(d.keeps, keep)
	return nil
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2ForcedVerifyMatchesSerialStepAndBatches(t *testing.T) {
	m := fakeInk2CohortModel()
	prompt := []int{2, 4, 6}
	forced := []int{8, 10, 12, 14}
	ctx := context.Background()

	// 1. Serial baseline: Prefill(prompt), then Step(prevToken) and Step(forced[i]) one by one.
	serialDev := &countingInk2Cohort{fakeInk2CohortWithCaps: fakeInk2CohortWithCaps{newFakeInk2Cohort()}}
	serialRes, err := newInkling2ResidentSized(m, serialDev, 64, 1024)
	if err != nil {
		t.Fatalf("serial newInkling2ResidentSized: %v", err)
	}
	serialSess := &inkling2ResidentSession{s: serialRes}
	defer serialSess.Close()

	serialPrev, err := serialSess.Prefill(ctx, prompt)
	if err != nil {
		t.Fatalf("serial Prefill: %v", err)
	}
	serialMatVecBefore := serialDev.matVecCalls
	serialNext, err := serialSess.Step(serialPrev)
	if err != nil {
		t.Fatalf("serial Step(prev): %v", err)
	}
	for _, tok := range forced {
		serialNext, err = serialSess.Step(tok)
		if err != nil {
			t.Fatalf("serial Step(%d): %v", tok, err)
		}
	}
	serialStepMatVecs := serialDev.matVecCalls - serialMatVecBefore
	serialTail1, err := serialSess.Step(serialNext)
	if err != nil {
		t.Fatalf("serial tail Step: %v", err)
	}
	serialPos := serialSess.Pos()

	// 2. Batched CommitForcedSpan: Prefill(prompt), then CommitForcedSpan(prev, forced).
	batchDev := &countingInk2Cohort{fakeInk2CohortWithCaps: fakeInk2CohortWithCaps{newFakeInk2Cohort()}}
	batchRes, err := newInkling2ResidentSized(m, batchDev, 64, 1024)
	if err != nil {
		t.Fatalf("batch newInkling2ResidentSized: %v", err)
	}
	batchSess := &inkling2ResidentSession{s: batchRes}
	defer batchSess.Close()

	batchPrev, err := batchSess.Prefill(ctx, prompt)
	if err != nil {
		t.Fatalf("batch Prefill: %v", err)
	}
	if batchPrev != serialPrev {
		t.Fatalf("initial Prefill next=%d, want %d", batchPrev, serialPrev)
	}
	matVecBefore := batchDev.matVecCalls
	pRMSBefore := len(batchDev.pRMSRows)
	batchNext, batched, err := batchSess.CommitForcedSpan(ctx, batchPrev, forced)
	if err != nil {
		t.Fatalf("CommitForcedSpan: %v", err)
	}
	if !batched {
		t.Fatal("CommitForcedSpan returned batched=false on batched-capable resident session")
	}
	batchMatVecs := batchDev.matVecCalls - matVecBefore
	if serialStepMatVecs != 1+len(forced) {
		t.Fatalf("serial Step baseline ran %d MatVec calls, want %d", serialStepMatVecs, 1+len(forced))
	}
	if batchMatVecs != 1 {
		t.Fatalf("CommitForcedSpan ran %d output MatVec calls, want 1 (single batched pass)", batchMatVecs)
	}
	if !slices.Contains(batchDev.pRMSRows[pRMSBefore:], 1+len(forced)) {
		t.Fatalf("CommitForcedSpan PRMS rows=%v, want width %d", batchDev.pRMSRows[pRMSBefore:], 1+len(forced))
	}
	if batchNext != serialNext {
		t.Fatalf("CommitForcedSpan next=%d, want %d", batchNext, serialNext)
	}
	batchTail1, err := batchSess.Step(batchNext)
	if err != nil {
		t.Fatalf("batch tail Step: %v", err)
	}
	if batchTail1 != serialTail1 {
		t.Fatalf("post-forced Step=%d, want %d", batchTail1, serialTail1)
	}
	if batchSess.Pos() != serialPos {
		t.Fatalf("Pos=%d, want %d", batchSess.Pos(), serialPos)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2ForcedVerifyEmptySpanAndEdgePaths(t *testing.T) {
	if _, _, err := (*inkling2ResidentSession)(nil).CommitForcedSpan(context.Background(), 1, []int{2}); err == nil {
		t.Fatal("expected error on nil session")
	}

	m := fakeInk2CohortModel()
	prompt := []int{1, 3, 5}
	ctx := context.Background()

	serialDev := fakeInk2CohortWithCaps{newFakeInk2Cohort()}
	serialRes, err := newInkling2ResidentSized(m, serialDev, 64, 1024)
	if err != nil {
		t.Fatalf("serial newInkling2ResidentSized: %v", err)
	}
	serialSess := &inkling2ResidentSession{s: serialRes}
	defer serialSess.Close()

	serialPrev, err := serialSess.Prefill(ctx, prompt)
	if err != nil {
		t.Fatalf("serial Prefill: %v", err)
	}
	serialNext, err := serialSess.Step(serialPrev)
	if err != nil {
		t.Fatalf("serial Step: %v", err)
	}

	emptyDev := &countingInk2Cohort{fakeInk2CohortWithCaps: fakeInk2CohortWithCaps{newFakeInk2Cohort()}}
	emptyRes, err := newInkling2ResidentSized(m, emptyDev, 64, 1024)
	if err != nil {
		t.Fatalf("empty newInkling2ResidentSized: %v", err)
	}
	emptySess := &inkling2ResidentSession{s: emptyRes}
	defer emptySess.Close()

	var _ Inkling2LoopForcedVerifySession = emptySess

	emptyPrev, err := emptySess.Prefill(ctx, prompt)
	if err != nil {
		t.Fatalf("empty Prefill: %v", err)
	}
	pRMSBefore := len(emptyDev.pRMSRows)
	emptyNext, batched, err := emptySess.CommitForcedSpan(ctx, emptyPrev, nil)
	if err != nil {
		t.Fatalf("CommitForcedSpan empty: %v", err)
	}
	if batched {
		t.Fatal("expected batched=false for empty forced span")
	}
	if len(emptyDev.pRMSRows) != pRMSBefore {
		t.Fatalf("expected no batched PRMS call for empty forced span, got %v", emptyDev.pRMSRows[pRMSBefore:])
	}
	if emptyNext != serialNext || emptySess.Pos() != serialSess.Pos() {
		t.Fatalf("empty span next=%d pos=%d, want next=%d pos=%d", emptyNext, emptySess.Pos(), serialNext, serialSess.Pos())
	}

	nonBatchRes, err := newInkling2ResidentSized(m, fakeInk2CohortWithCaps{newFakeInk2Cohort()}, 64, 1024)
	if err != nil {
		t.Fatalf("non-batch newInkling2ResidentSized: %v", err)
	}
	nonBatchRes.batched = false
	nonBatchSess := &inkling2ResidentSession{s: nonBatchRes}
	defer nonBatchSess.Close()

	nbPrev, err := nonBatchSess.Prefill(ctx, prompt)
	if err != nil {
		t.Fatalf("non-batch Prefill: %v", err)
	}
	if _, nbBatched, err := nonBatchSess.CommitForcedSpan(ctx, nbPrev, []int{7, 9}); err != nil {
		t.Fatalf("CommitForcedSpan non-batched: %v", err)
	} else if nbBatched {
		t.Fatal("expected batched=false when resident session has batched=false")
	}

	if err := emptySess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, _, err := emptySess.CommitForcedSpan(ctx, 1, []int{2}); err == nil {
		t.Fatal("expected error after Close")
	}

	// End-to-end public CommitForcedSpan SpecObserve notification (both empty and batched spans).
	specDev := &countingInk2Cohort{fakeInk2CohortWithCaps: fakeInk2CohortWithCaps{newFakeInk2Cohort()}}
	specRes, err := newInkling2ResidentSized(m, specDev, 64, 1024)
	if err != nil {
		t.Fatalf("spec newInkling2ResidentSized: %v", err)
	}
	drafter := &recordingSpecDrafter{}
	specSess := &inkling2ResidentSession{s: specRes, specDrafter: drafter}
	defer specSess.Close()

	spPrev, err := specSess.Prefill(ctx, prompt)
	if err != nil {
		t.Fatalf("spec Prefill: %v", err)
	}
	spNext, spBatched, err := specSess.CommitForcedSpan(ctx, spPrev, nil)
	if err != nil || spBatched {
		t.Fatalf("CommitForcedSpan empty sync batched=%v err=%v", spBatched, err)
	}
	forcedSpan := []int{7, 9, 11}
	if _, batchedSpan, err := specSess.CommitForcedSpan(ctx, spNext, forcedSpan); err != nil || !batchedSpan {
		t.Fatalf("CommitForcedSpan batched sync batched=%v err=%v", batchedSpan, err)
	}
	wantEmpty := []int{spPrev}
	wantBatched := append([]int{spNext}, forcedSpan...)
	if len(drafter.observed) != 2 || !slices.Equal(drafter.observed[0], wantEmpty) || !slices.Equal(drafter.observed[1], wantBatched) {
		t.Fatalf("SpecObserve observed=%v, want [%v, %v]", drafter.observed, wantEmpty, wantBatched)
	}
	if len(drafter.keeps) != 2 || drafter.keeps[0] != 1 || drafter.keeps[1] != len(wantBatched) {
		t.Fatalf("SpecObserve keeps=%v, want [1, %d]", drafter.keeps, len(wantBatched))
	}
}
