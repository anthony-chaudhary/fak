// Ported from modular/modular max/python/max/pipelines/kv_cache/checkpoint/{_checkpointer,_triggers,_heuristic/_at_event}.py@88ab8913af6b9298fe914e0f20d87a49e4172d0d (Apache-2.0 WITH LLVM-exception; Copyright (c) 2026 Modular Inc.). Direct port to fak-owned Go.

package model

import (
	"errors"
	"testing"
)

// cpQueryKey keys the fake candidate_at_or_before query's table and call log.
type cpQueryKey struct {
	req RequestID
	pos int
}

// cpFakeQuery is the engine-injected candidate_at_or_before stand-in: it
// answers from an explicit table and logs every ask so the oracle can pin
// which targets the heuristics queried.
type cpFakeQuery struct {
	answers map[cpQueryKey]int // key present => a candidate exists at its value
	calls   []cpQueryKey
}

func newCPFakeQuery() *cpFakeQuery {
	return &cpFakeQuery{answers: map[cpQueryKey]int{}}
}

func (q *cpFakeQuery) run(req RequestID, pos int) (int, bool) {
	k := cpQueryKey{req: req, pos: pos}
	q.calls = append(q.calls, k)
	v, ok := q.answers[k]
	return v, ok
}

func (q *cpFakeQuery) called(k cpQueryKey) bool {
	for _, c := range q.calls {
		if c == k {
			return true
		}
	}
	return false
}

// cpCountingHeuristic pins trigger mapping and once-per-step by counting its
// own runs and emitting fixed checkpoints.
type cpCountingHeuristic struct {
	kinds []EventKind
	runs  int
	out   []Checkpoint
}

func (h *cpCountingHeuristic) Subscribe(triggers *Triggers) {
	triggers.OnEvent(h, h.kinds...)
}

func (h *cpCountingHeuristic) Run(ctx *Context) []Checkpoint {
	h.runs++
	return h.out
}

// TestCheckpointPolicyOracle pins the closed contract of the event-driven
// checkpoint policy layer ported from modular's kv_cache/checkpoint package:
// trigger mapping, once-per-step execution, end-of-prompt dual targets,
// release-boundary selection, and fail-closed factory construction.
func TestCheckpointPolicyOracle(t *testing.T) {
	t.Run("trigger mapping fires exactly registered kinds and drops unhandled", func(t *testing.T) {
		unhandled := EventKind(99)
		a := &cpCountingHeuristic{kinds: []EventKind{EndOfPrompt}, out: []Checkpoint{{RequestID: "ra", Position: 1}}}
		b := &cpCountingHeuristic{kinds: []EventKind{Release}, out: []Checkpoint{{RequestID: "rb", Position: 2}}}
		none := func(RequestID, int) (int, bool) { return 0, false }
		cp, err := NewCheckpointerFactory().WithHeuristics(a, b).WithCandidateAtOrBefore(none).Build()
		if err != nil {
			t.Fatalf("Build: %v", err)
		}

		if !cp.Handles(EndOfPrompt) || !cp.Handles(Release) || cp.Handles(unhandled) {
			t.Fatalf("Handles(eop)=%v Handles(release)=%v Handles(unhandled)=%v, want true true false",
				cp.Handles(EndOfPrompt), cp.Handles(Release), cp.Handles(unhandled))
		}

		// An unhandled kind is dropped without error.
		cp.Record(Event{RequestID: "rx", Kind: unhandled, Position: 7})
		if got := cp.Step(); len(got) != 0 {
			t.Fatalf("Step after unhandled event = %+v, want none", got)
		}
		if a.runs != 0 || b.runs != 0 {
			t.Fatalf("unhandled event ran callbacks: a.runs=%d b.runs=%d, want 0 0", a.runs, b.runs)
		}

		// A registered kind fires exactly its own callback.
		cp.Record(Event{RequestID: "ra", Kind: EndOfPrompt, Position: 1})
		got := cp.Step()
		if a.runs != 1 || b.runs != 0 {
			t.Fatalf("after EndOfPrompt step: a.runs=%d b.runs=%d, want 1 0", a.runs, b.runs)
		}
		if len(got) != 1 || got[0] != (Checkpoint{RequestID: "ra", Position: 1}) {
			t.Fatalf("Step = %+v, want exactly [{ra 1}]", got)
		}

		cp.Record(Event{RequestID: "rb", Kind: Release, Position: 2})
		got = cp.Step()
		if a.runs != 1 || b.runs != 1 {
			t.Fatalf("after Release step: a.runs=%d b.runs=%d, want 1 1", a.runs, b.runs)
		}
		if len(got) != 1 || got[0] != (Checkpoint{RequestID: "rb", Position: 2}) {
			t.Fatalf("Step = %+v, want exactly [{rb 2}]", got)
		}
	})

	t.Run("once per step for a heuristic subscribed to two firing events", func(t *testing.T) {
		both := &cpCountingHeuristic{
			kinds: []EventKind{EndOfPrompt, Release},
			out:   []Checkpoint{{RequestID: "r", Position: 3}},
		}
		none := func(RequestID, int) (int, bool) { return 0, false }
		cp, err := NewCheckpointerFactory().WithHeuristics(both).WithCandidateAtOrBefore(none).Build()
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		cp.Record(Event{RequestID: "r", Kind: EndOfPrompt, Position: 3})
		cp.Record(Event{RequestID: "r", Kind: Release, Position: 3})
		got := cp.Step()
		if both.runs != 1 {
			t.Fatalf("heuristic ran %d times in one step, want exactly 1", both.runs)
		}
		if len(got) != 1 || got[0] != (Checkpoint{RequestID: "r", Position: 3}) {
			t.Fatalf("Step = %+v, want exactly [{r 3}] (deduped)", got)
		}
		// The step consumed its events: a second empty step re-runs nothing.
		if again := cp.Step(); len(again) != 0 || both.runs != 1 {
			t.Fatalf("second Step = %+v runs=%d, want none and still 1 total", again, both.runs)
		}
	})

	t.Run("end_of_prompt selects position and position-1 via the injected query", func(t *testing.T) {
		q := newCPFakeQuery()
		// Distinct answers per target so both targets are observable.
		q.answers[cpQueryKey{req: "r1", pos: 10}] = 7
		q.answers[cpQueryKey{req: "r1", pos: 9}] = 6
		cp, err := NewCheckpointerFactory().
			WithHeuristics(EndOfPromptHeuristic()).
			WithCandidateAtOrBefore(q.run).
			Build()
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		cp.Record(Event{RequestID: "r1", Kind: EndOfPrompt, Position: 10})
		got := cp.Step()
		want := []Checkpoint{{RequestID: "r1", Position: 7}, {RequestID: "r1", Position: 6}}
		if len(got) != len(want) {
			t.Fatalf("Step = %+v, want %+v", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("Step[%d] = %+v, want %+v", i, got[i], want[i])
			}
		}
		if !q.called(cpQueryKey{req: "r1", pos: 10}) || !q.called(cpQueryKey{req: "r1", pos: 9}) {
			t.Fatalf("query calls = %+v, want both targets {r1,10} and {r1,9}", q.calls)
		}
	})

	t.Run("release selects the release boundary", func(t *testing.T) {
		q := newCPFakeQuery()
		q.answers[cpQueryKey{req: "r2", pos: 5}] = 4
		cp, err := NewCheckpointerFactory().
			WithHeuristics(ReleaseHeuristic()).
			WithCandidateAtOrBefore(q.run).
			Build()
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		// r3 releases with no candidate available: no checkpoint, no error.
		cp.Record(Event{RequestID: "r2", Kind: Release, Position: 5})
		cp.Record(Event{RequestID: "r3", Kind: Release, Position: 5})
		got := cp.Step()
		want := []Checkpoint{{RequestID: "r2", Position: 4}}
		if len(got) != len(want) || got[0] != want[0] {
			t.Fatalf("Step = %+v, want %+v", got, want)
		}
		if !q.called(cpQueryKey{req: "r3", pos: 5}) {
			t.Fatalf("query calls = %+v, want the r3 release boundary asked too", q.calls)
		}
	})

	t.Run("factory without the query closure fails closed", func(t *testing.T) {
		cp, err := NewCheckpointerFactory().WithHeuristics(EndOfPromptHeuristic()).Build()
		if !errors.Is(err, ErrMissingCandidateQuery) {
			t.Fatalf("Build without query: err = %v, want ErrMissingCandidateQuery", err)
		}
		if cp != nil {
			t.Fatalf("Build without query returned checkpointer %+v, want nil", cp)
		}

		// Disabled builds and drops every event instead of querying.
		disabled, err := NewCheckpointerFactory().WithHeuristics(EndOfPromptHeuristic()).Disabled().Build()
		if err != nil {
			t.Fatalf("Disabled Build: %v", err)
		}
		disabled.Record(Event{RequestID: "r", Kind: EndOfPrompt, Position: 1})
		if got := disabled.Step(); len(got) != 0 {
			t.Fatalf("disabled Step = %+v, want none", got)
		}
	})
}
