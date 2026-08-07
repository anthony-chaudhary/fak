package toolproc

import (
	"encoding/json"
	"testing"
)

// bgSession builds the payload for a background launch post announcing `job` in
// `session`, plus the pre-spawn row the launch call needs in the journal.
func bgLaunch(session, toolUseID, job string) (Event, HookPayload) {
	spawn := Event{Kind: EvSpawn, CallID: toolUseID, Tool: "Bash", Session: session, AtMS: 1_000}
	p := HookPayload{SessionID: session, ToolName: "Bash", ToolUseID: toolUseID,
		ToolResponse: json.RawMessage(`"Command running in background with ID: ` + job + `"`)}
	return spawn, p
}

// TestBackgroundIDsAreUniqueAcrossSessions is the live-journal failure,
// minimized: two concurrent sessions each announce the harness background id
// "8" — a per-session counter that restarts at 1 — and both bridge into the ONE
// shared workspace journal. The identities must differ, and the journal that
// results must fold, because Fold refuses the whole table on a duplicate spawn.
//
// This is the defect that refused this workspace's live journal on "duplicate
// spawn for call bg:8" (#5880): 11 identities collided there, every one of them
// a bg: id spanning more than one session.
func TestBackgroundIDsAreUniqueAcrossSessions(t *testing.T) {
	spawnA, payloadA := bgLaunch("session-a", "tu-a", "8")
	spawnB, payloadB := bgLaunch("session-b", "tu-b", "8")

	journal := []Event{spawnA, spawnB}
	evsA, err := HookEvents("post", payloadA, hookEnvNone, 2_000, journal)
	if err != nil {
		t.Fatalf("HookEvents(session-a): %v", err)
	}
	journal = append(journal, evsA...)
	evsB, err := HookEvents("post", payloadB, hookEnvNone, 2_100, journal)
	if err != nil {
		t.Fatalf("HookEvents(session-b): %v", err)
	}
	journal = append(journal, evsB...)

	jobA, jobB := evsA[len(evsA)-1], evsB[len(evsB)-1]
	if jobA.Kind != EvSpawn || jobB.Kind != EvSpawn {
		t.Fatalf("want a bridged spawn from each session, got %+v / %+v", jobA, jobB)
	}
	if jobA.CallID == jobB.CallID {
		t.Fatalf("two sessions minted the same identity %q for their own job 8: "+
			"a per-session key cannot be the key in a workspace-shared journal", jobA.CallID)
	}
	if jobA.CallID != "bg:session-a:8" || jobB.CallID != "bg:session-b:8" {
		t.Fatalf("identities = %q / %q, want the owning session qualifying each", jobA.CallID, jobB.CallID)
	}
	tab, err := Fold(journal, 3_000, Config{})
	if err != nil {
		t.Fatalf("two sessions' background jobs must fold together: %v", err)
	}
	if tab.Counts.BackgroundIDsUnqualified != 0 {
		t.Fatalf("freshly written rows counted as unqualified: %d", tab.Counts.BackgroundIDsUnqualified)
	}
}

// TestDuplicateBackgroundSpawnStillRefuses is the other half, and the one that
// matters for keeping the check's teeth: qualifying the identity must not buy
// its uniqueness by making Fold lenient. A journal carrying two spawns of the
// SAME qualified identity — one session genuinely spawning its job 8 twice
// without it ever exiting — must still refuse the whole table, exactly as
// before. The producer must also decline to mint that second spawn.
func TestDuplicateBackgroundSpawnStillRefuses(t *testing.T) {
	spawn, payload := bgLaunch("session-a", "tu-a", "8")
	id := backgroundCallID("session-a", "8")

	events := []Event{
		spawn,
		{Kind: EvSpawn, CallID: id, Tool: "Bash[bg]", Session: "session-a", AtMS: 1_100},
		{Kind: EvSpawn, CallID: id, Tool: "Bash[bg]", Session: "session-a", AtMS: 1_200},
	}
	if _, err := Fold(events, 2_000, Config{}); err == nil {
		t.Fatal("a genuine duplicate spawn of one session's own job must still refuse the fold")
	}

	// And the producer never writes it: an id already journaled as running
	// bridges nothing but the launch call's own exit.
	evs, err := HookEvents("post", payload, hookEnvNone, 2_000, events[:2])
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	if len(evs) != 1 || evs[0].Kind != EvExit {
		t.Fatalf("re-announce of a live job => %+v, want the launch exit alone", evs)
	}
}

// TestBackgroundPollCannotReachAnotherSessionsJob pins the quiet half of the
// defect. The poll bridge resolves the polled id the same way the launch minted
// it, so session-b polling ITS "8" can never pulse session-a's job 8. Before the
// qualifier this correlated across sessions silently — no refusal, just a
// foreign job reading LIVE off someone else's output.
func TestBackgroundPollCannotReachAnotherSessionsJob(t *testing.T) {
	journal := []Event{
		{Kind: EvSpawn, CallID: backgroundCallID("session-a", "8"), Tool: "Bash[bg]", Session: "session-a", AtMS: 1_000},
		{Kind: EvSpawn, CallID: "tu-poll-b", Tool: "BashOutput", Session: "session-b", AtMS: 2_000},
	}
	poll := HookPayload{SessionID: "session-b", ToolName: "BashOutput", ToolUseID: "tu-poll-b",
		ToolInput:    json.RawMessage(`{"bash_id": "8"}`),
		ToolResponse: json.RawMessage(`{"status": "running", "stdout": "chunk"}`)}
	evs, err := HookEvents("post", poll, hookEnvNone, 3_000, journal)
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	for _, ev := range evs {
		if ev.Kind == EvPulse {
			t.Fatalf("session-b's poll pulsed %q — a poll must not reach another session's job", ev.CallID)
		}
	}
	if len(evs) != 1 || evs[0].CallID != "tu-poll-b" {
		t.Fatalf("cross-session poll => %+v, want the poll's own exit alone", evs)
	}

	// Its OWN job of the same name still pulses normally.
	journal = append(journal, Event{Kind: EvSpawn, CallID: backgroundCallID("session-b", "8"),
		Tool: "Bash[bg]", Session: "session-b", AtMS: 2_500})
	evs, err = HookEvents("post", poll, hookEnvNone, 3_000, journal)
	if err != nil {
		t.Fatalf("HookEvents: %v", err)
	}
	if len(evs) != 2 || evs[1].Kind != EvPulse || evs[1].CallID != "bg:session-b:8" {
		t.Fatalf("own-session poll => %+v, want a pulse on bg:session-b:8", evs)
	}
}

// Journals written before the qualifier landed are not retro-fixed (#3152's
// position, restated in bgident.go). They still fold when their bare ids happen
// not to collide, and still refuse when they do — the count reports how much of
// that history is left rather than hiding it.
func TestUnqualifiedBackgroundIDsAreCountedNotTolerated(t *testing.T) {
	legacy := []Event{
		{Kind: EvSpawn, CallID: "bg:7", Tool: "Bash[bg]", Session: "session-a", AtMS: 1_000},
		{Kind: EvSpawn, CallID: backgroundCallID("session-b", "7"), Tool: "Bash[bg]", Session: "session-b", AtMS: 1_100},
	}
	tab, err := Fold(legacy, 2_000, Config{})
	if err != nil {
		t.Fatalf("a non-colliding legacy row must still fold: %v", err)
	}
	if tab.Counts.BackgroundIDsUnqualified != 1 {
		t.Fatalf("unqualified count = %d, want 1 (the bare bg:7 only)", tab.Counts.BackgroundIDsUnqualified)
	}

	// Two sessions' legacy rows still collide, and must still refuse: the cure is
	// on the writer, so nothing here makes the reader tolerate the old shape.
	collide := []Event{
		{Kind: EvSpawn, CallID: "bg:7", Tool: "Bash[bg]", Session: "session-a", AtMS: 1_000},
		{Kind: EvSpawn, CallID: "bg:7", Tool: "Bash[bg]", Session: "session-b", AtMS: 1_100},
	}
	if _, err := Fold(collide, 2_000, Config{}); err == nil {
		t.Fatal("colliding legacy rows must still refuse; the qualifier repairs the writer, not the backlog")
	}
}
