package ggufload

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

func TestInkling2LoopBatchStreamCohortCtx(t *testing.T) {
	req := func(prompt, maxTok int) Inkling2LoopBatchRequest {
		return Inkling2LoopBatchRequest{Prompt: make([]int, prompt), MaxTokens: maxTok}
	}
	for _, tc := range []struct {
		name     string
		reqs     []Inkling2LoopBatchRequest
		resident int
		want     int
	}{
		{"rounds the longest row up to 64", []Inkling2LoopBatchRequest{req(10, 20), req(100, 1)}, 4096, 128},
		{"exact multiple stays", []Inkling2LoopBatchRequest{req(60, 4), req(1, 1)}, 4096, 64},
		{"short cohort is small", []Inkling2LoopBatchRequest{req(1, 1), req(2, 1)}, 4096, 64},
		{"capped at resident", []Inkling2LoopBatchRequest{req(4000, 97), req(1, 1)}, 4096, 4096},
	} {
		got, err := inkling2BatchCohortCtx(tc.reqs, tc.resident)
		if err != nil || got != tc.want {
			t.Fatalf("%s: ctx=%d err=%v, want %d", tc.name, got, err, tc.want)
		}
	}
	if _, err := inkling2BatchCohortCtx([]Inkling2LoopBatchRequest{req(4000, 98), req(1, 1)}, 4096); err == nil {
		t.Fatal("a row past the resident context must be refused")
	}
}

// The cohort context must satisfy the batch's own position bound and cap rule.
func TestInkling2LoopBatchStreamCohortCtxFitsBatchCaps(t *testing.T) {
	reqs := []Inkling2LoopBatchRequest{{Prompt: make([]int, 37), MaxTokens: 50}, {Prompt: make([]int, 3), MaxTokens: 9}}
	ctx, err := inkling2BatchCohortCtx(reqs, Inkling2LoopResidentContext)
	if err != nil {
		t.Fatal(err)
	}
	if ctx < 1 || ctx > inkling2ResidentCtx {
		t.Fatalf("ctx %d outside newInkling2Batch's 1..%d", ctx, inkling2ResidentCtx)
	}
	// lockstep steps = longest prompt+budget-1; every position must stay below ctx
	if steps := 37 + 50 - 1; steps > ctx {
		t.Fatalf("ctx %d below the %d positions the cohort steps", ctx, steps)
	}
}

func TestInkling2LoopBatchStreamLaneAcceptanceIsRowIsolated(t *testing.T) {
	var first, second []int
	now := time.Now()
	lanes := []inkling2BatchLane{
		newInkling2BatchLane(Inkling2LoopBatchRequest{Context: context.Background(), Prompt: []int{1}, MaxTokens: 4, Stop: func(token int) bool { return token == 9 }, Emit: func(token int) bool { first = append(first, token); return true }}, now),
		newInkling2BatchLane(Inkling2LoopBatchRequest{Context: context.Background(), Prompt: []int{2}, MaxTokens: 1, Emit: func(token int) bool { second = append(second, token); return true }}, now),
	}
	lanes[0].accept(8, now)
	lanes[1].accept(7, now)
	lanes[0].accept(9, now)
	if !reflect.DeepEqual(first, []int{8}) || !reflect.DeepEqual(second, []int{7}) {
		t.Fatalf("row emissions crossed: first=%v second=%v", first, second)
	}
	if !lanes[0].result.Stopped || lanes[0].result.Generated != 1 || !lanes[1].done || lanes[1].result.Generated != 1 {
		t.Fatalf("row states crossed: first=%+v second=%+v", lanes[0].result, lanes[1].result)
	}
}

func TestInkling2LoopBatchStreamCancelledLaneFinishes(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	lane := newInkling2BatchLane(Inkling2LoopBatchRequest{Context: cancelled, Prompt: []int{1}, MaxTokens: 3, Emit: func(int) bool { t.Fatal("emitted on a cancelled row"); return false }}, time.Now())
	lane.accept(5, time.Now())
	if !lane.done || !errors.Is(lane.result.Err, context.Canceled) || lane.result.Generated != 0 {
		t.Fatalf("cancelled lane = %+v", lane.result)
	}
}

func TestInkling2LoopBatchStreamRefusesBadCohort(t *testing.T) {
	if _, err := Inkling2LoopBatchGenerate(nil, nil, make([]Inkling2LoopBatchRequest, 1)); err == nil {
		t.Fatal("a one-row cohort must be refused")
	}
	if _, err := Inkling2LoopBatchGenerate(nil, nil, make([]Inkling2LoopBatchRequest, Inkling2LoopMaxBatch+1)); err == nil {
		t.Fatal("a cohort past the max batch must be refused")
	}
	if Inkling2LoopBatchEligible(nil, nil) {
		t.Fatal("nil model is not batch eligible")
	}
}
