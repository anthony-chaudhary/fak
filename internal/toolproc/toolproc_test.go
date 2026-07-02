package toolproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func findProc(t *testing.T, tab Table, id string) Proc {
	t.Helper()
	for _, p := range tab.Procs {
		if p.CallID == id {
			return p
		}
	}
	t.Fatalf("proc %s not in table", id)
	return Proc{}
}

func hasFinding(p Proc, reason string, advice Advice) bool {
	for _, f := range p.Findings {
		if f.Reason == reason && f.Advice == advice {
			return true
		}
	}
	return false
}

// TestSampleExercisesEveryVerdictClass is the offline proof: the built-in
// journal folds into one row per lifecycle class, each carrying the closed
// verdict token + advice the class demands.
func TestSampleExercisesEveryVerdictClass(t *testing.T) {
	events, now, cfg := Sample()
	tab, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}

	done := findProc(t, tab, "t-done")
	if done.State != StateDone || done.ExitStatus != "ok" || len(done.Findings) != 0 {
		t.Errorf("t-done: want clean DONE/ok, got state=%s status=%s findings=%v", done.State, done.ExitStatus, done.Findings)
	}
	if done.RuntimeMS != 2_000 {
		t.Errorf("t-done runtime: want 2000ms, got %d", done.RuntimeMS)
	}

	live := findProc(t, tab, "t-live")
	if live.State != StateRunning || live.Liveness != LivenessLive || len(live.Findings) != 0 {
		t.Errorf("t-live: want clean RUNNING/LIVE, got state=%s liveness=%s findings=%v", live.State, live.Liveness, live.Findings)
	}
	if live.Pulses != 2 {
		t.Errorf("t-live pulses: want 2, got %d", live.Pulses)
	}

	overdue := findProc(t, tab, "t-overdue")
	if !hasFinding(overdue, ReasonToolDeadlineExceededName, AdviceKill) {
		t.Errorf("t-overdue: want TOOL_DEADLINE_EXCEEDED/kill, got %v", overdue.Findings)
	}
	if overdue.OverdueMS != 65_000 {
		t.Errorf("t-overdue overdue_ms: want 65000, got %d", overdue.OverdueMS)
	}

	stalled := findProc(t, tab, "t-stalled")
	if stalled.Liveness != LivenessStalled || !hasFinding(stalled, ReasonToolHeartbeatStalledName, AdviceProbe) {
		t.Errorf("t-stalled: want STALLED + TOOL_HEARTBEAT_STALLED/probe, got liveness=%s findings=%v", stalled.Liveness, stalled.Findings)
	}

	orphan := findProc(t, tab, "t-orphan")
	if !orphan.Orphaned || !hasFinding(orphan, ReasonToolOrphanedName, AdviceReap) {
		t.Errorf("t-orphan: want orphaned + TOOL_ORPHANED/reap, got orphaned=%t findings=%v", orphan.Orphaned, orphan.Findings)
	}

	late := findProc(t, tab, "t-late")
	if late.State != StateKilled || late.KillReason != ReasonToolDeadlineExceededName {
		t.Errorf("t-late: want KILLED citing TOOL_DEADLINE_EXCEEDED, got state=%s reason=%s", late.State, late.KillReason)
	}
	if !hasFinding(late, ReasonToolResultAfterKillName, AdviceQuarantineResult) {
		t.Errorf("t-late: want TOOL_RESULT_AFTER_KILL/quarantine_result, got %v", late.Findings)
	}

	if !tab.AttentionNeeded {
		t.Error("sample table must need attention")
	}
	want := Counts{Running: 4, Done: 1, Killed: 1, Overdue: 1, Stalled: 1, Orphaned: 1}
	if tab.Counts != want {
		t.Errorf("counts: want %+v, got %+v", want, tab.Counts)
	}
}

// TestFoldDeterministic: same journal + same instant => byte-identical JSON.
func TestFoldDeterministic(t *testing.T) {
	events, now, cfg := Sample()
	a, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Fold(events, now, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Error("fold is not deterministic")
	}
}

// A proc with no declared cadence is QUIET (unknown liveness), never STALLED,
// and with no declared or default deadline it is never OVERDUE — honest, not green.
func TestUndeclaredEnvelopeIsQuietNotStalled(t *testing.T) {
	events := []Event{{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 1_000}}
	tab, err := Fold(events, 10_000_000, Config{})
	if err != nil {
		t.Fatal(err)
	}
	p := findProc(t, tab, "c1")
	if p.Liveness != LivenessQuiet || len(p.Findings) != 0 || p.OverdueMS != 0 {
		t.Errorf("want QUIET with no findings, got liveness=%s findings=%v overdue=%d", p.Liveness, p.Findings, p.OverdueMS)
	}
}

// The config default deadline applies exactly when the spawn declared none.
func TestDefaultDeadlineAppliesWhenUndeclared(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 1_000},
		{Kind: EvSpawn, CallID: "c2", Tool: "y", AtMS: 1_000, DeadlineMS: 60_000},
	}
	tab, err := Fold(events, 31_001, Config{DefaultDeadlineMS: 30_000})
	if err != nil {
		t.Fatal(err)
	}
	if p := findProc(t, tab, "c1"); !hasFinding(p, ReasonToolDeadlineExceededName, AdviceKill) {
		t.Errorf("c1: default deadline should apply, findings=%v", p.Findings)
	}
	if p := findProc(t, tab, "c2"); len(p.Findings) != 0 {
		t.Errorf("c2: declared 60s deadline should override default, findings=%v", p.Findings)
	}
}

// Benign races are tolerated and counted; they never resurrect a terminal proc.
func TestBenignRaces(t *testing.T) {
	events := []Event{
		{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 1_000},
		{Kind: EvExit, CallID: "c1", AtMS: 2_000, Status: "ok"},
		{Kind: EvPulse, CallID: "c1", AtMS: 2_500},                    // chunk in flight when exit landed
		{Kind: EvKill, CallID: "c1", AtMS: 3_000, Reason: "OPERATOR"}, // killer lost the race
	}
	tab, err := Fold(events, 10_000, Config{})
	if err != nil {
		t.Fatal(err)
	}
	p := findProc(t, tab, "c1")
	if p.State != StateDone || p.LatePulses != 1 || p.KillReason != "" {
		t.Errorf("want DONE with 1 late pulse and no kill, got state=%s late=%d kill=%q", p.State, p.LatePulses, p.KillReason)
	}
}

// Impossible transitions refuse the fold (fail closed).
func TestFailClosedTransitions(t *testing.T) {
	cases := []struct {
		name   string
		events []Event
		want   string
	}{
		{"duplicate spawn", []Event{
			{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 1_000},
			{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 2_000},
		}, "duplicate spawn"},
		{"pulse for unknown call", []Event{
			{Kind: EvPulse, CallID: "ghost", AtMS: 1_000},
		}, "unknown call"},
		{"exit for unknown call", []Event{
			{Kind: EvExit, CallID: "ghost", AtMS: 1_000, Status: "ok"},
		}, "unknown call"},
		{"double exit", []Event{
			{Kind: EvSpawn, CallID: "c1", Tool: "x", AtMS: 1_000},
			{Kind: EvExit, CallID: "c1", AtMS: 2_000, Status: "ok"},
			{Kind: EvExit, CallID: "c1", AtMS: 3_000, Status: "ok"},
		}, "double exit"},
		{"spawn from ended session", []Event{
			{Kind: EvSessionEnd, Session: "s1", AtMS: 1_000},
			{Kind: EvSpawn, CallID: "c1", Session: "s1", Tool: "x", AtMS: 2_000},
		}, "which ended"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Fold(tc.events, 10_000, Config{})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("want error containing %q, got %v", tc.want, err)
			}
		})
	}
}

// ParseEvents refuses the closed-vocabulary violations with the line number.
func TestParseFailsClosed(t *testing.T) {
	cases := []struct {
		name, journal, want string
	}{
		{"unknown kind", `{"kind":"resurrect","call_id":"c1","at_unix_ms":1}`, "unknown event kind"},
		{"bad exit status", `{"kind":"exit","call_id":"c1","at_unix_ms":1,"status":"fine"}`, "ok|error"},
		{"kill without reason", `{"kind":"kill","call_id":"c1","at_unix_ms":1}`, "reason token required"},
		{"spawn without tool", `{"kind":"spawn","call_id":"c1","at_unix_ms":1}`, "tool required"},
		{"zero timestamp", `{"kind":"pulse","call_id":"c1"}`, "must be positive"},
		{"malformed json", `{nope`, "line 1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseEvents(strings.NewReader(tc.journal))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("want error containing %q, got %v", tc.want, err)
			}
		})
	}
}

// Comments and blank lines are tolerated; valid rows round-trip.
func TestParseRoundTrip(t *testing.T) {
	journal := strings.Join([]string{
		"# a comment",
		"",
		`{"kind":"spawn","call_id":"c1","tool":"x","at_unix_ms":1000,"deadline_ms":5000}`,
		`{"kind":"pulse","call_id":"c1","at_unix_ms":2000,"via":"poll-9"}`,
		`{"kind":"exit","call_id":"c1","at_unix_ms":3000,"status":"ok"}`,
	}, "\n")
	events, err := ParseEvents(strings.NewReader(journal))
	if err != nil {
		t.Fatalf("ParseEvents: %v", err)
	}
	if len(events) != 3 || events[1].Via != "poll-9" {
		t.Fatalf("want 3 events with via preserved, got %+v", events)
	}
	tab, err := Fold(events, 10_000, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if p := findProc(t, tab, "c1"); p.State != StateDone {
		t.Errorf("want DONE, got %s", p.State)
	}
}

// The reason block stays out of the human-owned core range and the pairs stay
// in lockstep with the constants.
func TestReasonBlockIsOutOfTree(t *testing.T) {
	for _, pr := range ReasonPairs() {
		if pr.Code <= abi.ReasonCoreMax {
			t.Errorf("%s: code %d is inside the human-owned core range (<= %d)", pr.Name, pr.Code, abi.ReasonCoreMax)
		}
		if !strings.HasPrefix(pr.Name, "TOOL_") {
			t.Errorf("%s: verdict tokens are namespaced TOOL_*", pr.Name)
		}
	}
	if len(ReasonPairs()) != 4 {
		t.Errorf("closed vocabulary has 4 tokens, got %d", len(ReasonPairs()))
	}
}

// Guard the fold-input contract.
func TestFoldInputContract(t *testing.T) {
	if _, err := Fold(nil, 0, Config{}); err == nil {
		t.Error("zero now must refuse")
	}
	if _, err := Fold(nil, 1, Config{StallMultiplier: 0.5}); err == nil {
		t.Error("sub-1 stall multiplier must refuse")
	}
	if _, err := Fold(nil, 1, Config{DefaultDeadlineMS: -1}); err == nil {
		t.Error("negative default deadline must refuse")
	}
}
