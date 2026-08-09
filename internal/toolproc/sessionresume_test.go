package toolproc

import (
	"encoding/json"
	"testing"
)

// The defect #3152 names: the harness's SessionEnd hook fires while the session
// id keeps working, so `session_end` records a hook firing rather than the
// session's lifetime. Measured on this workspace's live journal, 224 of 1,517
// session_end rows are followed by a further spawn in the SAME session (shortest
// gap 20 ms), and Fold refuses the whole table at row 100 of 29,525.
//
// This is the unfixed baseline: it must stay refused, because nothing in the
// journal retracts the boundary.
func TestFoldRefusesBarePostEndSpawn(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "S", AtMS: 1000},
		{Kind: EvExit, CallID: "a", AtMS: 1100, Status: "ok"},
		{Kind: EvSessionEnd, Session: "S", AtMS: 1200},
		{Kind: EvSpawn, CallID: "b", Tool: "Bash", Session: "S", AtMS: 1220},
	}
	if _, err := Fold(events, 2000, Config{}); err == nil {
		t.Fatal("a spawn from an ended session with no retraction in the journal must still refuse the fold")
	}
}

// With the producer's own retraction in the journal the table folds — and the
// premature boundary is COUNTED, not swallowed.
func TestFoldAdmitsPostEndSpawnAfterResume(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "S", AtMS: 1000},
		{Kind: EvExit, CallID: "a", AtMS: 1100, Status: "ok"},
		{Kind: EvSessionEnd, Session: "S", AtMS: 1200},
		{Kind: EvSessionResume, Session: "S", AtMS: 1220},
		{Kind: EvSpawn, CallID: "b", Tool: "Bash", Session: "S", AtMS: 1220},
	}
	tab, err := Fold(events, 2000, Config{})
	if err != nil {
		t.Fatalf("retracted boundary must admit the later spawn: %v", err)
	}
	if len(tab.Procs) != 2 {
		t.Fatalf("procs = %d, want 2", len(tab.Procs))
	}
	if tab.Counts.SessionsResumed != 1 {
		t.Fatalf("sessions_resumed = %d, want 1 (the retraction must not be silent)", tab.Counts.SessionsResumed)
	}
	if tab.Counts.Running != 1 {
		t.Fatalf("running = %d, want 1", tab.Counts.Running)
	}
}

// A retracted session is alive, so its still-running procs are not orphans. The
// orphan verdict is what a premature boundary was manufacturing.
func TestResumeClearsOrphanForStillRunningProcs(t *testing.T) {
	base := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "S", AtMS: 1000},
		{Kind: EvSessionEnd, Session: "S", AtMS: 1200},
	}
	orphaned, err := Fold(base, 2000, Config{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	if orphaned.Counts.Orphaned != 1 {
		t.Fatalf("baseline orphaned = %d, want 1 (a live proc past a real session end IS an orphan)", orphaned.Counts.Orphaned)
	}
	resumedTab, err := Fold(append(append([]Event{}, base...),
		Event{Kind: EvSessionResume, Session: "S", AtMS: 1300}), 2000, Config{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	if resumedTab.Counts.Orphaned != 0 {
		t.Fatalf("orphaned = %d, want 0 — a session witnessed alive again cannot have orphaned its own procs", resumedTab.Counts.Orphaned)
	}
}

// A session may legitimately end, resume, and end again. The boundary that
// counts is the last one not yet retracted, and it re-arms at its own time.
func TestSecondSessionEndReArmsBoundary(t *testing.T) {
	events := []Event{
		{Kind: EvSessionEnd, Session: "S", AtMS: 1200},
		{Kind: EvSessionResume, Session: "S", AtMS: 1300},
		{Kind: EvSpawn, CallID: "b", Tool: "Bash", Session: "S", AtMS: 1300},
		{Kind: EvSessionEnd, Session: "S", AtMS: 1400},
	}
	tab, err := Fold(events, 2000, Config{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	if tab.Counts.Orphaned != 1 {
		t.Fatalf("orphaned = %d, want 1 — the re-armed end must orphan the still-running proc", tab.Counts.Orphaned)
	}
	events = append(events, Event{Kind: EvSpawn, CallID: "c", Tool: "Bash", Session: "S", AtMS: 1500})
	if _, err := Fold(events, 2000, Config{}); err == nil {
		t.Fatal("a spawn past the RE-ARMED end must refuse again; a stale retraction must not license it forever")
	}
}

// The producer writes the retraction at the moment its own evidence refutes the
// boundary — ahead of the spawn, because Fold reads in order.
func TestHookEventsEmitsResumeAheadOfPostEndSpawn(t *testing.T) {
	existing := []Event{
		{Kind: EvSpawn, CallID: "a", Tool: "Bash", Session: "S", AtMS: 1000},
		{Kind: EvExit, CallID: "a", AtMS: 1100, Status: "ok"},
		{Kind: EvSessionEnd, Session: "S", AtMS: 1200},
	}
	evs, err := HookEvents("pre", HookPayload{SessionID: "S", ToolName: "Bash", ToolUseID: "b"}, nil, 1220, existing)
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	if len(evs) != 2 {
		t.Fatalf("events = %+v, want [session_resume spawn]", evs)
	}
	if evs[0].Kind != EvSessionResume || evs[0].Session != "S" || evs[0].AtMS != 1220 {
		t.Fatalf("first event = %+v, want a session_resume for S at 1220", evs[0])
	}
	if evs[1].Kind != EvSpawn {
		t.Fatalf("second event = %+v, want the spawn", evs[1])
	}
	if _, err := Fold(append(existing, evs...), 2000, Config{}); err != nil {
		t.Fatalf("the journal the producer just wrote must fold: %v", err)
	}
}

// One retraction per boundary: the next spawn under an already-retracted end
// must not write another, or a chatty session would pad the journal with rows
// that assert nothing new.
func TestHookEventsDoesNotRepeatResume(t *testing.T) {
	existing := []Event{
		{Kind: EvSessionEnd, Session: "S", AtMS: 1200},
		{Kind: EvSessionResume, Session: "S", AtMS: 1220},
		{Kind: EvSpawn, CallID: "b", Tool: "Bash", Session: "S", AtMS: 1220},
	}
	evs, err := HookEvents("pre", HookPayload{SessionID: "S", ToolName: "Bash", ToolUseID: "c"}, nil, 1300, existing)
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	if len(evs) != 1 || evs[0].Kind != EvSpawn {
		t.Fatalf("events = %+v, want the spawn alone", evs)
	}
}

// A healthy session must be byte-identical to before: no retraction row appears
// where no boundary was ever claimed.
func TestHookEventsQuietWithoutSessionEnd(t *testing.T) {
	evs, err := HookEvents("pre", HookPayload{SessionID: "S", ToolName: "Bash", ToolUseID: "b"}, nil, 1220, nil)
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	if len(evs) != 1 || evs[0].Kind != EvSpawn {
		t.Fatalf("events = %+v, want the spawn alone", evs)
	}
}

// The retraction is only ever written by the hook that OBSERVED the session
// alive. The supervisor-brokered spawn path (ArmMonitor) never routes through
// HookEvents, so the leaked-child class #3032 guards keeps its teeth: a spawn
// journaled for a dead session by anything other than that session's own hook
// still refuses the fold.
func TestBrokeredSpawnFromEndedSessionStillRefuses(t *testing.T) {
	spec := MonitorSpec{CallID: "m", Session: "S", HeartbeatEveryMS: 500, Filter: "error", AtMS: 1300}
	spawn, err := ArmMonitor(spec)
	if err != nil {
		t.Fatalf("ArmMonitor: %v", err)
	}
	events := []Event{{Kind: EvSessionEnd, Session: "S", AtMS: 1200}, spawn}
	if _, err := Fold(events, 2000, Config{}); err == nil {
		t.Fatal("a brokered spawn for an ended session must still refuse: no hook witnessed that session alive")
	}
}

// Compaction must retain a retraction on the same terms as the boundary it
// retracts, or the compacted journal re-arms a withdrawn boundary and refuses.
func TestCompactionKeepsRetractionWithItsBoundary(t *testing.T) {
	events := []Event{
		{Kind: EvSessionEnd, Session: "S", AtMS: 1000},
		{Kind: EvSessionResume, Session: "S", AtMS: 1010},
		{Kind: EvSpawn, CallID: "live", Tool: "Bash", Session: "S", AtMS: 1010},
	}
	// Push the lifetime rows well outside the tail window with unrelated history.
	for i := 0; i < 12; i++ {
		id := string(rune('A' + i))
		events = append(events,
			Event{Kind: EvSpawn, CallID: id, Tool: "Bash", Session: "T", AtMS: int64(1100 + i*10)},
			Event{Kind: EvExit, CallID: id, AtMS: int64(1105 + i*10), Status: "ok"})
	}
	kept := CompactJournal(events, 8)
	var sawEnd, sawResume bool
	for _, ev := range kept {
		if ev.Kind == EvSessionEnd && ev.Session == "S" {
			sawEnd = true
		}
		if ev.Kind == EvSessionResume && ev.Session == "S" {
			sawResume = true
		}
	}
	if !sawEnd {
		t.Fatal("compaction dropped the session_end that defines the orphan boundary for a live spawn")
	}
	if !sawResume {
		t.Fatal("compaction kept the boundary but dropped its retraction")
	}
	if _, err := Fold(kept, 3000, Config{}); err != nil {
		t.Fatalf("CompactJournal must return a fold-clean journal: %v", err)
	}
}

// The row is a closed-vocabulary journal event like any other: it parses, it
// round-trips, and it refuses without a session.
func TestSessionResumeRowValidatesAndRoundTrips(t *testing.T) {
	if err := ValidateEvent(Event{Kind: EvSessionResume, Session: "S", AtMS: 1}); err != nil {
		t.Fatalf("valid session_resume refused: %v", err)
	}
	if err := ValidateEvent(Event{Kind: EvSessionResume, AtMS: 1}); err == nil {
		t.Fatal("session_resume without a session must refuse")
	}
	raw, err := json.Marshal(Event{Kind: EvSessionResume, Session: "S", AtMS: 7})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Event
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Kind != EvSessionResume || back.Session != "S" || back.AtMS != 7 {
		t.Fatalf("round trip = %+v", back)
	}
}
