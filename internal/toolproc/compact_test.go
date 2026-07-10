package toolproc

import (
	"fmt"
	"testing"
)

// A live (un-exited) spawn must survive compaction even when it is the oldest
// event and the tail window would otherwise exclude it — the #3032 invariant.
func TestCompactJournalKeepsOldestLiveSpawn(t *testing.T) {
	events := []Event{{Kind: EvSpawn, CallID: "live", Tool: "Bash", Session: "s", AtMS: 1}}
	for i := 0; i < 50; i++ {
		id := fmt.Sprintf("done-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: int64(2 + 2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(3 + 2*i), Status: "ok"})
	}
	got := CompactJournal(events, 5)

	foundLive := false
	for _, ev := range got {
		if ev.CallID == "live" && ev.Kind == EvSpawn {
			foundLive = true
		}
	}
	if !foundLive {
		t.Fatalf("live spawn dropped by compaction; got %d events", len(got))
	}
	if len(got) >= len(events) {
		t.Fatalf("compaction did not reduce terminal history: in=%d out=%d", len(events), len(got))
	}
	if id, running := hookResolveID("live", got); !running || id != "live" {
		t.Fatalf("live identity not running after compaction: id=%q running=%v", id, running)
	}
}

// The compacted journal must stay fold-clean even when the tail window begins
// partway through a call — a trailing event whose spawn fell just before the
// cut. Before the boundary fix, CompactJournal kept that trailing event but
// dropped its spawn, and Fold rejected the result with "… for unknown call".
// The fix keys retention on ANY tail CallID, so the straddle must fold for
// every non-spawn kind — exit, pulse, AND kill — not just exit.
func TestCompactJournalStaysFoldCleanAcrossBoundary(t *testing.T) {
	var events []Event
	var at int64 = 1
	for i := 0; i < 40; i++ {
		id := fmt.Sprintf("call-%d", i)
		events = append(events, Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: at})
		at++
		switch i % 3 {
		case 0: // spawn + exit — boundary can land on the exit straddle
			events = append(events, Event{Kind: EvExit, CallID: id, AtMS: at, Status: "ok"})
		case 1: // spawn + pulse + exit — boundary can land on the pulse straddle
			events = append(events, Event{Kind: EvPulse, CallID: id, AtMS: at})
			at++
			events = append(events, Event{Kind: EvExit, CallID: id, AtMS: at, Status: "ok"})
		case 2: // spawn + kill — boundary can land on the kill straddle
			events = append(events, Event{Kind: EvKill, CallID: id, AtMS: at, Reason: "TOOL_DEADLINE_EXCEEDED"})
		}
		at++
	}
	// Sweep tailKeep so the window boundary lands after a spawn whose trailing
	// exit, pulse, OR kill sits in the tail — every straddle kind the inTail fix
	// must keep the spawn for.
	for tailKeep := 1; tailKeep < len(events); tailKeep++ {
		got := CompactJournal(events, tailKeep)
		if _, err := Fold(got, at+10, Config{}); err != nil {
			t.Fatalf("compacted journal not fold-clean at tailKeep=%d: %v", tailKeep, err)
		}
	}
}

// A leaked spawn — un-exited, its owning session already ended — must keep
// folding to an ORPHANED proc after compaction even when the session_end row
// has aged out of the tail window. CompactJournal keeps the live spawn (the
// #3032 invariant); it must ALSO retain the session_end that defines the
// spawn's orphan status, or the leak re-folds as a healthy RUNNING proc and the
// TOOL_ORPHANED signal the subsystem exists to raise is silently erased.
func TestCompactJournalKeepsOrphanSessionEnd(t *testing.T) {
	// The leak and its session boundary, both ancient.
	events := []Event{
		{Kind: EvSpawn, CallID: "leak", Tool: "train_probe", Session: "s1", AtMS: 1},
		{Kind: EvSessionEnd, Session: "s1", AtMS: 2},
	}
	// A long run of later, unrelated terminal history pushes the session_end far
	// outside any small tail window.
	base := int64(10)
	for i := 0; i < 60; i++ {
		id := fmt.Sprintf("done-%d", i)
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s2", AtMS: base + int64(2*i)},
			Event{Kind: EvExit, CallID: id, AtMS: base + int64(2*i) + 1, Status: "ok"})
	}
	got := CompactJournal(events, 8) // session_end (index 1) sits well outside the tail

	foundEnd := false
	for _, ev := range got {
		if ev.Kind == EvSessionEnd && ev.Session == "s1" {
			foundEnd = true
		}
	}
	if !foundEnd {
		t.Fatalf("session_end for the leaked spawn's session dropped by compaction")
	}
	tbl, err := Fold(got, base+10_000, Config{})
	if err != nil {
		t.Fatalf("compacted journal not fold-clean: %v", err)
	}
	var leak *Proc
	for i := range tbl.Procs {
		if tbl.Procs[i].CallID == "leak" {
			leak = &tbl.Procs[i]
		}
	}
	if leak == nil {
		t.Fatalf("leaked spawn missing from folded table")
	}
	if leak.State != StateRunning || !leak.Orphaned {
		t.Fatalf("preserved leak not running+orphaned after compaction: state=%s orphaned=%v", leak.State, leak.Orphaned)
	}
}

// Compaction must not drift a retained proc's terminal State. When a call's
// exit/kill fell before the cut but a LATER straddling event sits in the tail —
// a kill followed by an exit-after-kill, or a done call trailed by a late pulse
// — keeping only the spawn re-folds the wrong state (KILLED→DONE losing
// TOOL_RESULT_AFTER_KILL, or DONE→RUNNING). CompactJournal must keep the
// terminal marker alongside the tail event.
func TestCompactJournalPreservesTerminalStateAcrossBoundary(t *testing.T) {
	filler := func(base int64) []Event {
		var out []Event
		for i := 0; i < 12; i++ {
			id := fmt.Sprintf("f-%d", i)
			out = append(out,
				Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "s", AtMS: base + int64(2*i)},
				Event{Kind: EvExit, CallID: id, AtMS: base + int64(2*i) + 1, Status: "ok"})
		}
		return out
	}
	stateOf := func(t *testing.T, evs []Event) Proc {
		t.Helper()
		tbl, err := Fold(evs, 10_000, Config{})
		if err != nil {
			t.Fatalf("fold: %v", err)
		}
		for i := range tbl.Procs {
			if tbl.Procs[i].CallID == "A" {
				return tbl.Procs[i]
			}
		}
		t.Fatalf("call A missing from fold")
		return Proc{}
	}

	t.Run("kill before cut, exit-after-kill in tail", func(t *testing.T) {
		events := []Event{
			{Kind: EvSpawn, CallID: "A", Tool: "Bash", Session: "s", AtMS: 1, DeadlineMS: 10},
			{Kind: EvKill, CallID: "A", AtMS: 20, Reason: "TOOL_DEADLINE_EXCEEDED"},
		}
		events = append(events, filler(30)...)
		events = append(events, Event{Kind: EvExit, CallID: "A", AtMS: 100, Status: "ok"}) // straddles into the tail
		orig := stateOf(t, events)
		comp := stateOf(t, CompactJournal(events, 1))
		if comp.State != orig.State || comp.KillReason != orig.KillReason {
			t.Fatalf("terminal state drifted: orig={%s kill=%s} comp={%s kill=%s}", orig.State, orig.KillReason, comp.State, comp.KillReason)
		}
	})

	t.Run("exit before cut, late pulse in tail", func(t *testing.T) {
		events := []Event{
			{Kind: EvSpawn, CallID: "A", Tool: "Bash", Session: "s", AtMS: 1},
			{Kind: EvExit, CallID: "A", AtMS: 5, Status: "ok"},
		}
		events = append(events, filler(10)...)
		events = append(events, Event{Kind: EvPulse, CallID: "A", AtMS: 100}) // late pulse straddles into the tail
		orig := stateOf(t, events)
		comp := stateOf(t, CompactJournal(events, 1))
		if comp.State != orig.State {
			t.Fatalf("terminal state drifted: orig=%s comp=%s (DONE resurrected as live)", orig.State, comp.State)
		}
	})
}

// tailKeep >= len(events) returns the input unchanged.
func TestCompactJournalTailKeepAll(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "s", AtMS: 1},
		{Kind: EvExit, CallID: "a", AtMS: 2, Status: "ok"},
	}
	if got := CompactJournal(events, len(events)); len(got) != len(events) {
		t.Fatalf("tailKeep>=len should be unchanged: in=%d out=%d", len(events), len(got))
	}
}
