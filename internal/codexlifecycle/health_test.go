package codexlifecycle

import (
	"strings"
	"testing"
)

// --- synthetic health-row builders (pure-fold inputs) ------------------------

func hStart(id string) HealthRow { return HealthRow{Kind: KindStarted, TurnID: id} }
func hDone(id string) HealthRow  { return HealthRow{Kind: KindComplete, TurnID: id} }
func hAbort(id string) HealthRow { return HealthRow{Kind: KindAborted, TurnID: id} }
func hCall() HealthRow           { return HealthRow{Kind: kindToolCall} }
func hMsg(text string) HealthRow { return HealthRow{Kind: kindAgentMsg, Message: text} }
func hTokens(n int) HealthRow    { return HealthRow{Kind: kindTokens, InputTokens: n} }
func hCompacted() HealthRow      { return HealthRow{Kind: kindCompacted} }
func hModel(m string) HealthRow  { return HealthRow{Kind: kindModel, Model: m} }

func TestClassifyZeroTool(t *testing.T) {
	cases := []struct{ msg, want string }{
		{"All proposed tool calls were refused by the fak kernel: shell_command REQUIRE_WITNESS", ZeroGuardRefused},
		{"I'll use the `super-loop` skill because the goal is to audit workers", ZeroPreambleNoop},
		{"Here is a status update on the work.", ZeroTalkOnly},
		{"", ZeroSilent},
	}
	for _, c := range cases {
		if got := ClassifyZeroTool(c.msg); got != c.want {
			t.Errorf("ClassifyZeroTool(%q) = %q, want %q", c.msg, got, c.want)
		}
	}
}

func TestFoldHealth_FullTurnCountsToolNotZero(t *testing.T) {
	s := FoldHealth([]HealthRow{hModel("gpt-5.6-sol"), hStart("A"), hCall(), hCall(), hMsg("done"), hDone("A")})
	if s.Turns != 1 || s.ToolCalls != 2 || s.ZeroToolTotal() != 0 || s.Model != "gpt-5.6-sol" {
		t.Errorf("stats = %+v, want 1 turn / 2 tools / 0 zero-tool / model gpt-5.6-sol", s)
	}
}

func TestFoldHealth_ZeroToolTurnClassified(t *testing.T) {
	s := FoldHealth([]HealthRow{hStart("A"), hMsg("All proposed tool calls were refused by the fak kernel"), hDone("A")})
	if s.ZeroTool[ZeroGuardRefused] != 1 || s.ZeroToolTotal() != 1 {
		t.Errorf("zero_tool = %v, want {guard_refused: 1}", s.ZeroTool)
	}
}

// THE TICKET (#5063): a second task_started while a turn is open must close the old
// turn as SUPERSEDED — never a success, never zero-tool "completed" — and every
// post-boundary tool call belongs to the NEW turn, not the abandoned one.
func TestFoldHealth_SupersededTurnKeyedByTurnID(t *testing.T) {
	s := FoldHealth([]HealthRow{
		hStart("A"), hCall(), hCall(), // A does real work…
		hStart("B"),          // …then B starts with no terminal for A
		hCall(),              // post-boundary delta: B's, not A's
		hMsg(""), hDone("B"), // B completes having called one tool
	})
	if s.Turns != 2 {
		t.Fatalf("turns = %d, want 2", s.Turns)
	}
	if s.Superseded != 1 {
		t.Errorf("superseded = %d, want 1 (A closed non-success at B's start)", s.Superseded)
	}
	// A did 2 tool calls and B did 1: had B's call leaked into A (the boolean-fold
	// defect) or A's count been reset, B would look zero-tool. It must not.
	if s.ZeroToolTotal() != 0 {
		t.Errorf("zero_tool = %v, want none: B called a tool and A is superseded, not completed", s.ZeroTool)
	}
	if s.ToolCalls != 3 {
		t.Errorf("tool_calls = %d, want 3", s.ToolCalls)
	}
}

// THE TICKET (#5063): turn_aborted closes a turn. The Python fold ignored it
// entirely, leaving the turn open forever.
func TestFoldHealth_TurnAbortedClosesTurn(t *testing.T) {
	s := FoldHealth([]HealthRow{hStart("A"), hMsg("interrupted"), hAbort("A")})
	if s.Aborted != 1 {
		t.Fatalf("aborted = %d, want 1", s.Aborted)
	}
	if s.Unterminated != 0 {
		t.Errorf("unterminated = %d, want 0: the abort IS the terminal", s.Unterminated)
	}
	if s.ZeroToolTotal() != 0 {
		t.Errorf("zero_tool = %v, want none: an aborted turn is not a zero-tool completion", s.ZeroTool)
	}
}

// A legacy unkeyed turn_aborted binds to the open turn (same rule as Fold).
func TestFoldHealth_UnkeyedAbortBindsActiveTurn(t *testing.T) {
	s := FoldHealth([]HealthRow{hStart("A"), hAbort("")})
	if s.Aborted != 1 || s.Unterminated != 0 {
		t.Errorf("aborted/unterminated = %d/%d, want 1/0", s.Aborted, s.Unterminated)
	}
}

// A late OBSERVED task_complete for a turn synthesized closed repairs the row —
// with its FROZEN pre-boundary tool count, so the post-boundary delta still never
// leaks into it.
func TestFoldHealth_LateObservedCompleteRepairsSupersede(t *testing.T) {
	s := FoldHealth([]HealthRow{
		hStart("A"),          // A opens, calls nothing
		hStart("B"), hCall(), // B supersedes A and calls a tool
		hDone("A"), // late observed terminal for A
		hDone("B"),
	})
	if s.Superseded != 0 {
		t.Errorf("superseded = %d, want 0: observed evidence outranks the synthesized close", s.Superseded)
	}
	// A completed with its frozen count of 0 tools — B's call must not have leaked in.
	if s.ZeroToolTotal() != 1 {
		t.Errorf("zero_tool total = %d, want exactly A's zero-tool completion", s.ZeroToolTotal())
	}
}

// The final start with no terminal is unterminated — counted, never completed.
func TestFoldHealth_OpenFinalTurnIsUnterminated(t *testing.T) {
	s := FoldHealth([]HealthRow{hStart("A"), hDone("A"), hStart("B"), hCall()})
	if s.Unterminated != 1 || s.Superseded != 0 {
		t.Errorf("unterminated/superseded = %d/%d, want 1/0", s.Unterminated, s.Superseded)
	}
}

func TestFoldHealth_CompactionRecordsLastInputOccupancy(t *testing.T) {
	s := FoldHealth([]HealthRow{hStart("A"), hTokens(92000), hCall(), hDone("A"), hCompacted()})
	if len(s.Compactions) != 1 || s.Compactions[0] != 92000 {
		t.Errorf("compactions = %v, want [92000]", s.Compactions)
	}
}

// A real compaction emits BOTH a top-level `compacted` and an
// event_msg/context_compacted; only the former may count (no 2x).
func TestParseHealthRollout_NoDoubleCountOfCompaction(t *testing.T) {
	lines := strings.Join([]string{
		`{"timestamp":"t1","type":"event_msg","payload":{"type":"task_started","turn_id":"A"}}`,
		`{"timestamp":"t2","type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":90000}}}}`,
		`{"timestamp":"t3","type":"event_msg","payload":{"type":"task_complete","turn_id":"A"}}`,
		`{"timestamp":"t4","type":"compacted","payload":{}}`,
		`{"timestamp":"t5","type":"event_msg","payload":{"type":"context_compacted"}}`,
	}, "\n")
	_, rows, err := ParseHealthRollout(strings.NewReader(lines + "\n"))
	if err != nil {
		t.Fatalf("ParseHealthRollout: %v", err)
	}
	if s := FoldHealth(rows); len(s.Compactions) != 1 {
		t.Errorf("compactions = %v, want exactly 1 (context_compacted must not double count)", s.Compactions)
	}
}

// End-to-end parse: real-shaped lines, including a tool call and a torn tail line.
func TestParseHealthRollout_Shapes(t *testing.T) {
	lines := strings.Join([]string{
		`{"timestamp":"t0","type":"session_meta","payload":{"id":"r1","model_provider":"fak","cli_version":"0.144.4","cwd":"C:\\work\\fak"}}`,
		`{"timestamp":"t1","type":"turn_context","payload":{"model":"gpt-5.6-sol"}}`,
		`{"timestamp":"t2","type":"event_msg","payload":{"type":"task_started","turn_id":"A"}}`,
		`{"timestamp":"t3","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c1"}}`,
		`{"timestamp":"t4","type":"event_msg","payload":{"type":"agent_message","message":"done"}}`,
		`{"timestamp":"t5","type":"event_msg","payload":{"type":"task_complete","turn_id":"A"}}`,
		`{"torn`,
	}, "\n")
	meta, rows, err := ParseHealthRollout(strings.NewReader(lines + "\n"))
	if err != nil {
		t.Fatalf("ParseHealthRollout: %v", err)
	}
	if meta.Provider != "fak" || meta.RolloutID != "r1" {
		t.Errorf("meta = %+v, want provider fak / id r1", meta)
	}
	s := FoldHealth(rows)
	if s.Turns != 1 || s.ToolCalls != 1 || s.ZeroToolTotal() != 0 || s.Model != "gpt-5.6-sol" {
		t.Errorf("stats = %+v, want 1 turn / 1 tool / 0 zero-tool / model set", s)
	}
}

func TestRollUp_FlagsAndBuckets(t *testing.T) {
	// One healthy session, one guard-refusal loop, one premature compaction.
	healthy := SessionStats{Session: "a", Model: "m", Turns: 10, ToolCalls: 200,
		ZeroTool: map[string]int{}, Compactions: []int{93000}}
	refusal := SessionStats{Session: "b", Model: "gpt-5.6-sol", Turns: 8, ToolCalls: 5,
		ZeroTool: map[string]int{ZeroGuardRefused: 6}}
	stuck := SessionStats{Session: "c", Model: "m", Turns: 6, ToolCalls: 0,
		ZeroTool: map[string]int{ZeroPreambleNoop: 6}, Compactions: []int{7000}}
	rep := RollUp([]SessionStats{healthy, refusal, stuck}, 15)

	if rep.SessionsWithTurns != 3 {
		t.Errorf("sessions_with_turns = %d, want 3", rep.SessionsWithTurns)
	}
	if rep.Compaction.NearBudget96K != 1 || rep.Compaction.PrematureLT40K != 1 {
		t.Errorf("compaction = %+v, want near_budget 1 / premature 1", rep.Compaction)
	}
	if len(rep.GuardRefusalLoops) != 1 || rep.GuardRefusalLoops[0].Session != "b" {
		t.Errorf("guard_refusal_loops = %+v, want exactly session b", rep.GuardRefusalLoops)
	}
	inflated := map[string]bool{}
	for _, r := range rep.TurnInflation {
		inflated[r.Session] = true
	}
	if !inflated["b"] || !inflated["c"] || len(inflated) != 2 {
		t.Errorf("turn_inflation = %+v, want sessions b and c", rep.TurnInflation)
	}
	joined := strings.Join(rep.Flags, " ")
	for _, want := range []string{"GUARD_REFUSAL_LOOPS", "PREMATURE_COMPACTION", "HIGH_ZERO_TOOL_RATE"} {
		if !strings.Contains(joined, want) {
			t.Errorf("flags %v missing %s", rep.Flags, want)
		}
	}
}

func TestRollUp_EmptyCorpusIsClean(t *testing.T) {
	rep := RollUp(nil, 15)
	if rep.Totals.Turns != 0 || len(rep.Flags) != 0 {
		t.Errorf("empty corpus: totals=%+v flags=%v, want zero/none", rep.Totals, rep.Flags)
	}
	if rep.Schema != HealthSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, HealthSchema)
	}
}

// The superseded/aborted counts survive into the corpus totals — the integrity
// numbers the boolean fold could not report at all.
func TestRollUp_CarriesIntegrityTotals(t *testing.T) {
	s := FoldHealth([]HealthRow{hStart("A"), hCall(), hStart("B"), hAbort("B")})
	s.Session = "x"
	rep := RollUp([]SessionStats{s}, 15)
	if rep.Totals.Superseded != 1 || rep.Totals.Aborted != 1 {
		t.Errorf("totals = %+v, want superseded 1 / aborted 1", rep.Totals)
	}
}
