package agent

// write_barrier_test.go — the #1319 acceptance witness: the BEFORE-CONSUMPTION WRITE
// BARRIER. Two layers: (1) the mechanism in isolation against a recording engine — a
// SPECULATIVE effect-free read is served from the prediction WITHOUT engine dispatch
// (the recorder sees nothing, SpecServed increments), a mispredict SQUASHES (BufferSink
// Rollback leaves no committed effect), and the dependent write is BARRED (never reaches
// the engine); and (2) the same barrier driven through RunArm, where a write-shaped
// authoritative call behind a squashed speculation is blocked from dispatch (EngineCalls
// stays below the no-barrier baseline).

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// TestWriteBarrierServesNoDispatchSquashesAndBars drives the specState mechanism directly
// against a recording engine so "the write never reaches the engine" is a literal,
// checkable fact (the recorder's call count), not an inference.
func TestWriteBarrierServesNoDispatchSquashesAndBars(t *testing.T) {
	ctx := context.Background()
	Configure() // registers localtools + the agent policy
	// Register a recorder UNDER localtools AFTER Configure so a dispatch would be visible.
	rec := &recordingEngine{id: "localtools"}
	abi.RegisterEngine("localtools", rec)
	k := kernel.New("localtools")

	spec := abi.NewSpeculator(0)
	spec.Learn(searchPattern(`{"origin":"SFO"}`))
	sp := newSpecState(spec, k)
	m := ArmMetrics{}

	// SUSPEND: speculate the effect-free read. It is served from the prediction — the
	// recorder must see NO dispatch, and SpecServed counts the dispatch-free serve.
	sp.speculate(ctx, 0, "get_user_details", nil, &m)
	if got := len(rec.calls()); got != 0 {
		t.Fatalf("speculative serve DISPATCHED to the engine (%d calls) — it must be served from the prediction without dispatch", got)
	}
	if m.SpecServed != 1 || m.SpecIssued != 1 {
		t.Fatalf("speculative serve counters = issued %d served %d, want 1/1", m.SpecIssued, m.SpecServed)
	}
	if sp.pending == nil || !sp.pending.Suspended() {
		t.Fatal("after speculate the turn must be SUSPENDED, holding the provisional read")
	}
	held := sp.pending // capture before resolve clears it, to inspect the sink's rollback

	// RESUME with a MISMATCH (the model authoritatively books, not searches): squash.
	sp.resolve(ctx, &abi.ToolCall{Tool: "book_flight", Args: inlineRef("{}")}, &m)
	if m.SpecSquashed != 1 {
		t.Fatalf("a mismatching authoritative call must SQUASH; squashed=%d", m.SpecSquashed)
	}
	// BufferSink.Rollback retracted the provisional read — nothing committed, no leak.
	if c := held.Sink().Committed(); len(c) != 0 {
		t.Fatalf("squash left %d committed effects, want 0 (Rollback must retract the provisional read)", len(c))
	}
	if p := held.Sink().PendingEpochs(); p != 0 {
		t.Fatalf("squash left %d pending epochs, want 0", p)
	}

	// BARRIER: the dependent write is barred — and the caller honoring the bar means the
	// engine is never touched. The recorder still shows zero dispatches.
	if !sp.barWrite("book_flight", &m) {
		t.Fatal("write behind a squashed speculation must be BARRED")
	}
	if m.WritesBarred != 1 {
		t.Fatalf("WritesBarred = %d, want 1", m.WritesBarred)
	}
	if got := len(rec.calls()); got != 0 {
		t.Fatalf("the barred write reached the engine (%d calls) — it must never be dispatched", got)
	}
	// A read after the squash is harmless and not barred.
	if sp.barWrite("get_user_details", &m) {
		t.Fatal("a non-write call must NOT be barred")
	}
}

// TestRunArmWriteBarrierBlocksDependentWrite is the production-caller witness: RunArm
// itself, with a speculator, serves a speculative read dispatch-free and BARS the
// write-shaped authoritative call behind the squashed speculation — so it dispatches
// strictly fewer engine calls than the identical run with no speculator (the barred write
// never reached the engine).
func TestRunArmWriteBarrierBlocksDependentWrite(t *testing.T) {
	script := []*Completion{
		toolCallTurn("get_user_details", `{"user_id":"u1"}`), // turn 1: read → speculate next
		toolCallTurn("book_flight", `{"flight_id":"f1"}`),    // turn 2: WRITE → squashes the read-prediction, barred
		{Message: Message{Content: "done"}},                  // turn 3: final answer
	}
	spec := abi.NewSpeculator(0)
	spec.Learn(searchPattern(`{"origin":"SFO"}`)) // predicts a READ after get_user_details

	var log []traceEvent
	m, err := RunArm(context.Background(), &scriptedPlanner{turns: script}, "go", true, 10, &log, WithSpeculator(spec))
	if err != nil {
		t.Fatalf("RunArm (with speculator): %v", err)
	}

	if m.SpecServed == 0 {
		t.Fatal("RunArm did not serve a speculative read (SpecServed=0)")
	}
	if m.SpecSquashed == 0 {
		t.Fatalf("the write turn must squash the read-prediction; squashed=%d", m.SpecSquashed)
	}
	if m.WritesBarred == 0 {
		t.Fatal("the write behind the squashed speculation was not barred (WritesBarred=0)")
	}
	// The barred write never reached the engine: its trace event carries the barrier
	// verdict (BARRED / write-barrier), not an engine-produced result. A dispatched call
	// would carry an adjudication verdict (ALLOW/DENY/...) and By the deciding rung.
	var booked *traceEvent
	for i := range log {
		if log[i].Tool == "book_flight" {
			booked = &log[i]
		}
	}
	if booked == nil {
		t.Fatal("no trace event for the write call book_flight")
	}
	if booked.Verdict != "BARRED" || booked.By != "write-barrier" {
		t.Fatalf("book_flight trace = {verdict=%q by=%q}, want {BARRED, write-barrier} — the write must be barred, not dispatched", booked.Verdict, booked.By)
	}
}
