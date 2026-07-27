package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// The agents pane carries two lineages and used to render one of them as the other.
//
// ParentTrace/Generation are written by exactly one verb — session.Table.Recontinue —
// and internal/session defines them as "the trace this session was re-continued FROM"
// and "how many budget-reset re-continuations preceded this session". That is a hidden
// CONTEXT RESET of the same agent. The pane read a non-empty ParentTrace as "this row is
// a sub-agent" and Generation as its spawn depth, so a single long-running agent that
// had been re-continued three times rendered as "sub g3" and summarized as
// "1 active (1 sub, deepest g3)": a sub-agent fleet that did not exist, at a depth that
// was a reset tally. The honest sub-agent axis was in the same row the whole time —
// SpawnCount, the admitted subagent-spawn count.
//
// These tests build the lineage with the REAL producer rather than hand-written field
// values, so what is pinned is the meaning the session package actually assigns, not a
// fixture's opinion of it. The one hand-carried step is the SessionState -> SessionVars
// field copy in gateway/debug.go, which is a literal per-field copy of the two fields
// under test.

// recontinuedLineage runs an actual budget-reset re-continuation chain of depth resets
// and returns the live child's lineage as the pane would receive it.
func recontinuedLineage(t *testing.T, resets int) (parent string, generation int) {
	t.Helper()
	tbl := session.NewTable()
	trace := "main-trace"
	st := session.State{}
	for i := 0; i < resets; i++ {
		child := trace + "-c"
		st = tbl.Recontinue(trace, child, session.Budget{TurnsLeft: 10, TokensLeft: 1000})
		trace = child
	}
	wire := toGatewaySessionState(st)
	if strings.TrimSpace(wire.ParentTrace) == "" {
		t.Fatalf("a re-continued session must carry its parent trace: %+v", wire)
	}
	if wire.Generation != resets {
		t.Fatalf("Generation = %d after %d re-continuations, want %d — the field counts "+
			"context resets, and this test's premise depends on it", wire.Generation, resets, resets)
	}
	return wire.ParentTrace, wire.Generation
}

// A context reset is not a spawn, and the row that reports it must not say it is.
func TestAContinuedAgentIsNotRenderedAsASubAgent(t *testing.T) {
	parent, gen := recontinuedLineage(t, 3)
	row := guardInfoAgentText(guardInfoSession{
		TraceID: "main-trace-c-c-c", Run: "running", ParentTrace: parent, Generation: gen,
	})
	if !strings.Contains(row, "cont g3") {
		t.Errorf("a thrice-re-continued agent must read as a continuation at g3: %q", row)
	}
	// The specific wrong reading, and the specific wrong number it produced.
	if strings.Contains(row, "sub g") {
		t.Errorf("a continuation must not be rendered as a sub-agent: %q", row)
	}
	// It spawned nothing, so no spawn clause may appear.
	if strings.Contains(row, "spawn") {
		t.Errorf("a row that spawned nothing must claim no spawns: %q", row)
	}
}

// The same claim at the summary level, which is what the compact status line shows and
// therefore what an operator reads most often.
func TestTheAgentsSummarySeparatesContinuationsFromSpawns(t *testing.T) {
	parent, gen := recontinuedLineage(t, 2)
	continued := guardInfoAgentsSummary([]guardInfoSession{
		{TraceID: "main-trace-c-c", Run: "running", ParentTrace: parent, Generation: gen},
	})
	if !strings.Contains(continued, "1 active") || !strings.Contains(continued, "1 continued, deepest g2") {
		t.Errorf("one re-continued agent must summarize as active + continued: %q", continued)
	}
	if strings.Contains(continued, "sub") {
		t.Errorf("nothing spawned a sub-agent here, so the summary must not report one: %q", continued)
	}
	if strings.Contains(continued, "spawned") {
		t.Errorf("zero spawns must render no spawn clause, not a zero: %q", continued)
	}

	// SpawnCount is the axis that a sub-agent actually moves. It is PARENT-side: the
	// count belongs to the trace that issued the spawn, and the summary totals it.
	spawning := guardInfoAgentsSummary([]guardInfoSession{
		{TraceID: "a", Run: "running", SpawnCount: 2},
		{TraceID: "b", Run: "running", SpawnCount: 1},
	})
	if !strings.Contains(spawning, "3 spawned") {
		t.Errorf("the summary must total the admitted spawn counts: %q", spawning)
	}
	if strings.Contains(spawning, "continued") {
		t.Errorf("neither row was re-continued, so no continuation clause may appear: %q", spawning)
	}
}

// The two axes are independent, and a row may carry both: an agent can be a
// continuation of itself AND have spawned children. Rendering must state each from its
// own field rather than letting one imply the other.
func TestTheTwoLineageAxesDoNotImplyEachOther(t *testing.T) {
	parent, gen := recontinuedLineage(t, 1)
	both := guardInfoAgentText(guardInfoSession{
		TraceID: "main-trace-c", Run: "running", ParentTrace: parent, Generation: gen, SpawnCount: 4,
	})
	if !strings.Contains(both, "cont g1") {
		t.Errorf("the continuation axis must survive alongside spawns: %q", both)
	}
	if !strings.Contains(both, "4 spawns") {
		t.Errorf("the spawn axis must survive alongside a continuation: %q", both)
	}
	// A row with spawns but no parent is an original agent that spawned children — the
	// case the old rendering could not express at all, since it had one lineage slot.
	root := guardInfoAgentText(guardInfoSession{TraceID: "main", Run: "running", SpawnCount: 4})
	if !strings.Contains(root, "root") || !strings.Contains(root, "4 spawns") {
		t.Errorf("an original agent that spawned four children must read as root + 4 spawns: %q", root)
	}
}
