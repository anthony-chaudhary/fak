package agenttest

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// recordingT is a fake T that records whether Errorf was called and the last message, so
// the assertion library can be tested without a real *testing.T.
type recordingT struct {
	failed bool
	last   string
}

func (r *recordingT) Helper() {}
func (r *recordingT) Errorf(format string, args ...any) {
	r.failed = true
	r.last = strings.TrimSpace(fmt.Sprintf(format, args...))
}

// jsonString quotes s as a JSON string literal (for embedding raw args in a mock result).
func jsonString(s string) string { return strconv.Quote(s) }

// --- a small scripted airline-ish workflow reused across tests ---

func bookingScript() *ScriptedPlanner {
	return NewScriptedPlanner(
		CallTurn(Call{Tool: "get_user", Args: `{"id":"mia"}`}),
		CallTurn(Call{Tool: "search_flights", Args: `{"from":"SFO","to":"JFK"}`}),
		CallTurn(Call{Tool: "book_flight", Args: `{"flight":"UA1"}`}),
		AnswerTurn("Booked flight UA1, confirmation ABC123."),
	)
}

func bookingTools() *MockTools {
	return NewMockTools().
		Respond("get_user", `{"id":"mia","tier":"gold"}`).
		Respond("search_flights", `{"flights":[{"id":"UA1","price":199}]}`).
		RespondFunc("book_flight", func(args string) string {
			return `{"confirmation":"ABC123","echo":` + jsonString(args) + `}`
		})
}

// Acceptance 1 + 2 + 3: deterministic workflow over mock responses, asserted by pattern.
func TestRunSession_DeterministicWorkflow(t *testing.T) {
	r, err := RunSession(bookingScript(), bookingTools(), "book the cheapest SFO->JFK", 10)
	if err != nil {
		t.Fatalf("RunSession: %v", err)
	}
	AssertToolSequence(t, r, "get_user", "search_flights", "book_flight")
	AssertToolOrder(t, r, "get_user", "book_flight")
	AssertToolCount(t, r, "book_flight", 1)
	AssertToolNotCalled(t, r, "delete_account")
	AssertCalledWith(t, r, "search_flights", `"to":"JFK"`)
	AssertAllMocked(t, r)
	AssertFinalAnswer(t, r, "ABC123")
	if r.HitTurnCap {
		t.Errorf("workflow hit the turn cap unexpectedly")
	}
	// the RespondFunc echoed the args back into the tool result
	if !strings.Contains(r.Tools[2].Response, "UA1") {
		t.Errorf("book_flight response did not echo args: %q", r.Tools[2].Response)
	}
}

// The matchers must be correct on the negative side too (a green test on every input is
// a vacuous test).
func TestMatchers_NegativePaths(t *testing.T) {
	r, err := RunSession(bookingScript(), bookingTools(), "x", 10)
	if err != nil {
		t.Fatalf("RunSession: %v", err)
	}
	if err := r.MatchToolSequence("get_user", "book_flight"); err == nil {
		t.Errorf("MatchToolSequence should reject a wrong sequence")
	}
	if err := r.MatchToolSubsequence("get_user", "search_flights"); err != nil {
		t.Errorf("MatchToolSubsequence should accept a real subsequence: %v", err)
	}
	if err := r.MatchToolSubsequence("book_flight", "get_user"); err == nil {
		t.Errorf("MatchToolSubsequence should reject an out-of-order subsequence")
	}
	if err := r.MatchToolOrder("book_flight", "get_user"); err == nil {
		t.Errorf("MatchToolOrder should reject reversed order")
	}
	if r.CalledWith("get_user", "nope") {
		t.Errorf("CalledWith should be false for absent args")
	}

	// the Assert* wrappers must drive t.Errorf on a failed match
	ft := &recordingT{}
	AssertToolSequence(ft, r, "only_one")
	if !ft.failed {
		t.Errorf("AssertToolSequence should have failed the fake T")
	}
	ok := &recordingT{}
	AssertToolSequence(ok, r, "get_user", "search_flights", "book_flight")
	if ok.failed {
		t.Errorf("AssertToolSequence false-failed a correct sequence: %s", ok.last)
	}
}

// Acceptance 3: an unmocked tool is a loud, recorded miss — not a silent empty result.
func TestMockTools_UnmockedIsLoud(t *testing.T) {
	planner := NewScriptedPlanner(
		CallTurn(Call{Tool: "ghost", Args: `{}`}),
		AnswerTurn("done"),
	)
	r, err := RunSession(planner, NewMockTools(), "call a ghost", 5)
	if err != nil {
		t.Fatalf("RunSession: %v", err)
	}
	if got := r.UnmockedTools(); len(got) != 1 || got[0] != "ghost" {
		t.Errorf("UnmockedTools = %v, want [ghost]", got)
	}
	if !strings.Contains(r.Tools[0].Response, "no mock registered") {
		t.Errorf("unmocked response should be a loud error, got %q", r.Tools[0].Response)
	}
	ft := &recordingT{}
	AssertAllMocked(ft, r)
	if !ft.failed {
		t.Errorf("AssertAllMocked should fail when a tool was unmocked")
	}
}

// Acceptance 4: reproduce from a recorded transcript. Run once, serialize the transcript,
// replay it over the SAME mock tools, and assert the reproduced tool-call pattern and the
// assistant turns match the original.
func TestReplay_ReproducesFromTranscript(t *testing.T) {
	orig, err := RunSession(bookingScript(), bookingTools(), "book it", 10)
	if err != nil {
		t.Fatalf("original RunSession: %v", err)
	}
	jsonl, err := DumpTranscriptJSONL(orig.Messages)
	if err != nil {
		t.Fatalf("DumpTranscriptJSONL: %v", err)
	}
	msgs, err := LoadTranscriptJSONL(jsonl)
	if err != nil {
		t.Fatalf("LoadTranscriptJSONL: %v", err)
	}
	replay, err := RunSession(NewReplayPlanner(msgs), bookingTools(), "book it", 10)
	if err != nil {
		t.Fatalf("replay RunSession: %v", err)
	}
	// reproduced tool-call pattern matches the original exactly
	AssertToolSequence(t, replay, orig.ToolNames()...)
	if replay.FinalAnswer != orig.FinalAnswer {
		t.Errorf("replay final answer %q != original %q", replay.FinalAnswer, orig.FinalAnswer)
	}
	// the assistant turns of the reproduced transcript equal the original's (verbatim ids)
	origAsst := assistantMessages(orig.Messages)
	replayAsst := assistantMessages(replay.Messages)
	if len(origAsst) != len(replayAsst) {
		t.Fatalf("assistant-turn count: original %d, replay %d", len(origAsst), len(replayAsst))
	}
	for i := range origAsst {
		if len(origAsst[i].ToolCalls) != len(replayAsst[i].ToolCalls) {
			t.Errorf("turn %d tool-call count differs", i)
			continue
		}
		for j := range origAsst[i].ToolCalls {
			a, b := origAsst[i].ToolCalls[j], replayAsst[i].ToolCalls[j]
			if a.ID != b.ID || a.Function.Name != b.Function.Name || a.Function.Arguments != b.Function.Arguments {
				t.Errorf("turn %d call %d differs: %+v vs %+v", i, j, a, b)
			}
		}
	}
}

// Acceptance 1 (reproducibility guarantee) + fixtures: a JSON-authored fixture verifies
// its expectations and reproduces bit-for-bit across runs.
func TestFixture_RoundTripAndReproducible(t *testing.T) {
	f := Fixture{
		Name: "booking",
		Task: "book the cheapest SFO->JFK",
		Mocks: map[string]string{
			"get_user":       `{"id":"mia"}`,
			"search_flights": `{"flights":[{"id":"UA1"}]}`,
			"book_flight":    `{"confirmation":"ABC123"}`,
		},
		Script: []FixtureTurn{
			{Calls: []FixtureCall{{Tool: "get_user", Args: `{"id":"mia"}`}}},
			{Calls: []FixtureCall{{Tool: "search_flights", Args: `{"to":"JFK"}`}}},
			{Calls: []FixtureCall{{Tool: "book_flight", Args: `{"flight":"UA1"}`}}},
			{Answer: "Booked UA1 (ABC123)."},
		},
		ExpectTools:  []string{"get_user", "search_flights", "book_flight"},
		ExpectAnswer: "ABC123",
	}

	// JSON round-trip preserves the fixture
	raw, err := f.JSON()
	if err != nil {
		t.Fatalf("Fixture.JSON: %v", err)
	}
	loaded, err := LoadFixture(raw)
	if err != nil {
		t.Fatalf("LoadFixture: %v", err)
	}

	// the loaded fixture verifies its declared expectations
	ft := &recordingT{}
	loaded.Verify(ft)
	if ft.failed {
		t.Errorf("Verify failed on a correct fixture: %s", ft.last)
	}

	// and it is reproducible
	rt := &recordingT{}
	AssertReproducible(rt, loaded)
	if rt.failed {
		t.Errorf("AssertReproducible failed: %s", rt.last)
	}

	// a wrong expectation must be caught (the assertion is not vacuous)
	bad := loaded
	bad.ExpectTools = []string{"get_user"}
	bt := &recordingT{}
	bad.Verify(bt)
	if !bt.failed {
		t.Errorf("Verify should fail when ExpectTools is wrong")
	}

	// unknown JSON keys are rejected
	if _, err := LoadFixture([]byte(`{"name":"x","typo_field":1}`)); err == nil {
		t.Errorf("LoadFixture should reject unknown fields")
	}
}

// The harness drives the REAL deterministic agent.MockPlanner end-to-end (not just our
// own ScriptedPlanner), proving the framework tests genuine agent.Planner implementations.
// The mock tool vocabulary is sourced from agent.ToolCatalog() so the test never guesses
// the planner's tool names, and every advertised tool gets a non-error response so the
// stateful planner (which retries on a tool result containing "error") advances to its
// final answer instead of spinning to the turn cap.
func TestRunSession_DrivesRealMockPlanner(t *testing.T) {
	mocks := func() *MockTools {
		m := NewMockTools()
		for _, td := range agent.ToolCatalog() {
			m.Respond(td.Function.Name, `{"ok":true}`)
		}
		return m
	}

	r, err := RunSession(agent.NewMockPlanner("unit"), mocks(), agent.DefaultTask, 20)
	if err != nil {
		t.Fatalf("RunSession with real MockPlanner: %v", err)
	}
	if len(r.Tools) == 0 {
		t.Fatalf("real planner made no tool calls")
	}
	if r.HitTurnCap {
		t.Errorf("real planner hit the turn cap; tool calls were %v", r.ToolNames())
	}
	// the real planner naturally starts by looking up the user
	AssertToolCalled(t, r, "get_user_details")
	// every tool the real planner called was one the catalog advertised (no unmocked miss)
	AssertAllMocked(t, r)
	// determinism: a second identical drive yields the same tool-name pattern
	r2, err := RunSession(agent.NewMockPlanner("unit"), mocks(), agent.DefaultTask, 20)
	if err != nil {
		t.Fatalf("second RunSession: %v", err)
	}
	if err := r2.MatchToolSequence(r.ToolNames()...); err != nil {
		t.Errorf("real MockPlanner not deterministic across runs: %v", err)
	}
}

func TestScriptedPlanner_ResetReproduces(t *testing.T) {
	p := bookingScript()
	a, err := RunSession(p, bookingTools(), "t", 10)
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	p.Reset()
	b, err := RunSession(p, bookingTools(), "t", 10)
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if err := b.MatchToolSequence(a.ToolNames()...); err != nil {
		t.Errorf("Reset did not reproduce: %v", err)
	}
	if a.Tools[0].ID != b.Tools[0].ID {
		t.Errorf("Reset should restore deterministic ids: %q vs %q", a.Tools[0].ID, b.Tools[0].ID)
	}
}

// --- small local helpers (kept here so the package's non-test API stays minimal) ---

func assistantMessages(msgs []agent.Message) []agent.Message {
	var out []agent.Message
	for _, m := range msgs {
		if m.Role == agent.RoleAssistant {
			out = append(out, m)
		}
	}
	return out
}
