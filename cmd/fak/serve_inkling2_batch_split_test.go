package main

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// splitTestRows queues n rows whose prompt is [i+1] and records what each emits.
func splitTestRows(n int) ([]*inkling2QueuedRequest, [][]int) {
	got := make([][]int, n)
	rows := make([]*inkling2QueuedRequest, n)
	for i := range rows {
		rows[i] = &inkling2QueuedRequest{ctx: context.Background(), prompt: []int{i + 1}, maxNew: 1,
			sink: &inkling2RowSink{emit: func(id int) bool { got[i] = append(got[i], id); return true }},
			done: make(chan inkling2QueuedResult, 1)}
	}
	return rows, got
}

// splitTestEngine runs a batch that refuses device memory for cohorts wider than
// most, and a scalar engine; both emit prompt[0]*10, so every served row emits it once.
func splitTestEngine(t *testing.T, most int) (*inkling2LoopBatchEngine, *[]int, *int) {
	t.Helper()
	var widths []int
	scalar := 0
	e := newInkling2LoopBatchEngineWith(
		func(_ context.Context, prompt []int, _ int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
			scalar++
			emit(prompt[0] * 10)
			return agent.Inkling2LoopGeneration{Generated: 1}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			widths = append(widths, len(reqs))
			if len(reqs) > most {
				return nil, &ggufload.Inkling2BatchMemoryError{Sequences: len(reqs), Need: 1 << 30, Free: 1 << 20}
			}
			out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
			for i, r := range reqs {
				r.Emit(r.Prompt[0] * 10)
				out[i].Generated = 1
			}
			return out, nil
		},
		testInkling2Admission(inkling2RealClock{}), 4096)
	return e, &widths, &scalar
}

func splitTestCheck(t *testing.T, rows []*inkling2QueuedRequest, got [][]int) {
	t.Helper()
	for i, r := range rows {
		select {
		case res := <-r.done:
			if res.err != nil || res.gen.Generated != 1 {
				t.Fatalf("row %d: gen=%+v err=%v", i, res.gen, res.err)
			}
		default:
			t.Fatalf("row %d was never answered", i)
		}
		if !reflect.DeepEqual(got[i], []int{(i + 1) * 10}) {
			t.Fatalf("row %d emitted %v, want [%d]", i, got[i], (i+1)*10)
		}
	}
}

// A cohort refused device memory before any row ran is halved until its halves fit,
// and every row is served once with its own tokens instead of failing.
func TestInkling2BatchEngineHalvesRefusedCohortAndServesEveryRow(t *testing.T) {
	e, widths, scalar := splitTestEngine(t, 4)
	rows, got := splitTestRows(ggufload.Inkling2LoopMaxBatch)
	e.execute(rows)
	splitTestCheck(t, rows, got)
	if want := []int{16, 8, 4, 4, 8, 4, 4}; !reflect.DeepEqual(*widths, want) {
		t.Fatalf("batch widths %v, want %v", *widths, want)
	}
	if *scalar != 0 || e.splitGroups.Load() != 3 {
		t.Fatalf("scalar calls %d, splits %d; want 0 and 3", *scalar, e.splitGroups.Load())
	}
}

// A device that cannot fit even a two-row cohort serves every row on the serialized
// engine.
func TestInkling2BatchEngineRefusedPairsFallBackToScalar(t *testing.T) {
	e, _, scalar := splitTestEngine(t, 1)
	rows, got := splitTestRows(5)
	e.execute(rows)
	splitTestCheck(t, rows, got)
	if *scalar != 5 || e.scalarGroups.Load() != 5 {
		t.Fatalf("scalar calls %d (groups %d), want 5", *scalar, e.scalarGroups.Load())
	}
}

// The memory plan runs before the cohort: its groups run in plan order, a one-row
// group runs on the serialized engine after the cohort arenas are released, and the
// whole cohort is never attempted.
func TestInkling2BatchEngineRunsPlannedSubCohorts(t *testing.T) {
	e, widths, scalar := splitTestEngine(t, ggufload.Inkling2LoopMaxBatch)
	var plans [][]int
	e.split = func(positions []int) ([][]int, error) {
		plans = append(plans, positions)
		short := make([]int, 0, len(positions)-1)
		for i := range positions {
			if i != 6 {
				short = append(short, i)
			}
		}
		return [][]int{short, {6}}, nil
	}
	released := 0
	e.releaseArenas = func() { released++ }
	rows, got := splitTestRows(ggufload.Inkling2LoopMaxBatch)
	rows[6].prompt = append(rows[6].prompt, make([]int, 3999)...)
	e.execute(rows)
	splitTestCheck(t, rows, got)
	if len(plans) != 1 || plans[0][0] != 2 || plans[0][6] != 4001 {
		t.Fatalf("split saw %v, want each row's len(prompt)+maxNew", plans)
	}
	if !reflect.DeepEqual(*widths, []int{ggufload.Inkling2LoopMaxBatch - 1}) || *scalar != 1 || released != 1 {
		t.Fatalf("batch widths %v, scalar %d, releases %d; want [15], 1, 1", *widths, *scalar, released)
	}
}

// A batch failure that is not a memory refusal still fails the cohort's rows.
func TestInkling2BatchEngineNonMemoryFailureStillFailsRows(t *testing.T) {
	e := newInkling2LoopBatchEngineWith(nil,
		func([]ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			return nil, context.DeadlineExceeded
		},
		testInkling2Admission(inkling2RealClock{}), 4096)
	rows, _ := splitTestRows(3)
	e.execute(rows)
	for i, r := range rows {
		if res := <-r.done; res.err != context.DeadlineExceeded {
			t.Fatalf("row %d: err %v, want the batch's error", i, res.err)
		}
	}
	if e.splitGroups.Load() != 0 {
		t.Fatalf("a non-memory failure was split %d times", e.splitGroups.Load())
	}
}
