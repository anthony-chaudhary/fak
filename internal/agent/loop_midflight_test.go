package agent

// loop_midflight_test.go — the #5158 acceptance witnesses: structured mid-flight
// session verbs (interrupt / drop-pending-call / set-budget) land at the loop's next
// CLEAN turn boundary — never mid-tool, never mid-adjudication — with a tamper-
// evident journal record of the verb, its arrival, and the boundary it landed at.
// The byte-for-byte invariant (no verb issued => the historical loop) is proven
// alongside, the same guard the read half (#5148) honors.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// midflightCallPlanner emits one turn with two named tool calls (call_a / call_b),
// then a final answer, recording the messages each Complete call was handed so a
// test can prove which tool results reached the model.
type midflightCallPlanner struct {
	calls int
	turns [][]Message
}

func (p *midflightCallPlanner) Model() string { return "midflight-calls" }

func (p *midflightCallPlanner) Complete(_ context.Context, msgs []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.calls++
	snapshot := make([]Message, len(msgs))
	copy(snapshot, msgs)
	p.turns = append(p.turns, snapshot)
	if p.calls == 1 {
		return &Completion{
			Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
				{ID: "call_a", Type: "function", Function: Func{Name: "search", Arguments: `{"query":"first"}`}},
				{ID: "call_b", Type: "function", Function: Func{Name: "search", Arguments: `{"query":"second"}`}},
			}},
			FinishReason: "tool_calls",
			Usage:        Usage{CompletionTokens: 1},
		}, nil
	}
	return &Completion{
		Message:      Message{Role: RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage:        Usage{CompletionTokens: 1},
	}, nil
}

// TestInterruptLandsAtTurnBoundary is the #5158 acceptance witness: an interrupt
// issued while turn N is dispatching tool calls does NOT cut a tool or an
// adjudication short — the loop completes turn N's admitted results, then stops at
// the boundary carrying the closed witness token, and the journal records the verb,
// its arrival, and the boundary it landed at.
func TestInterruptLandsAtTurnBoundary(t *testing.T) {
	verbs := NewMidflightVerbs()
	p := &midTurnFlipPlanner{flip: func() {
		if r := verbs.Interrupt(); r != nil {
			t.Errorf("mid-flight interrupt refused on a live run: %v", r)
		}
	}}
	var log []traceEvent
	m, err := RunArm(context.Background(), p, "task", false, 3, &log, WithMidflightVerbs(verbs))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	// Turn N completed: BOTH announced tool calls dispatched and were admitted.
	if m.ToolCalls != 2 || len(log) != 2 {
		t.Fatalf("interrupt cut turn N short: %d tool calls / %d trace events, want 2/2", m.ToolCalls, len(log))
	}
	if m.Turns != 1 {
		t.Fatalf("interrupted arm ran %d turns, want exactly the in-flight turn", m.Turns)
	}
	// The stop carries the closed witness token, at the boundary — not TERMINATED
	// (mid-turn), not DRAINING (drive-state), not a turn-cap inference.
	if m.StoppedBySession != session.ReasonInterrupted {
		t.Fatalf("StoppedBySession = %q, want %q", m.StoppedBySession, session.ReasonInterrupted)
	}
	if m.HitTurnCap {
		t.Fatal("interrupted arm reported HitTurnCap — the stop must be the verb's, not the cap's")
	}
	// The journal witnesses the verb, its arrival, and the boundary it landed at.
	j := verbs.Journal()
	var queued, applied *MidflightRecord
	for i := range j {
		r := &j[i]
		if r.Verb != MidflightInterrupt {
			continue
		}
		switch r.Status {
		case MidflightQueued:
			queued = r
		case MidflightApplied:
			applied = r
		}
	}
	if queued == nil || applied == nil {
		t.Fatalf("journal missing the interrupt lifecycle (queued=%v applied=%v): %+v", queued != nil, applied != nil, j)
	}
	if applied.BoundaryTurn != 2 {
		t.Fatalf("interrupt landed at boundary %d, want 2 (the boundary after the in-flight turn)", applied.BoundaryTurn)
	}
	if queued.ArrivedAtUnixNano == 0 || applied.ArrivedAtUnixNano != queued.ArrivedAtUnixNano {
		t.Fatalf("journal arrival drifted: queued %d vs applied %d", queued.ArrivedAtUnixNano, applied.ArrivedAtUnixNano)
	}
	if !VerifyMidflightJournal(j) {
		t.Fatal("journal hash chain does not verify")
	}
}

// TestDropPendingCallSkipsExactlyNamedCall proves the net-new per-call skip consult:
// the named call_id is skipped BEFORE dispatch with a typed status=skipped receipt,
// and nothing else — the sibling call on the same turn dispatches normally.
func TestDropPendingCallSkipsExactlyNamedCall(t *testing.T) {
	verbs := NewMidflightVerbs()
	if r := verbs.DropPendingCall("call_a"); r != nil {
		t.Fatalf("drop-pending-call refused on a live run: %v", r)
	}
	p := &midflightCallPlanner{}
	var log []traceEvent
	m, err := RunArm(context.Background(), p, "task", false, 3, &log, WithMidflightVerbs(verbs))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.FinalAnswer == "" {
		t.Fatal("run did not complete — a dropped call must not derail the loop")
	}
	// Exactly one DROPPED verdict, on the named call's tool slot; the sibling ran.
	var dropped, ran int
	for _, ev := range log {
		switch ev.Verdict {
		case "DROPPED":
			dropped++
		case "naive-exec":
			ran++
		}
	}
	if dropped != 1 || ran != 1 {
		t.Fatalf("verdicts dropped=%d ran=%d, want exactly 1 drop (call_a) and 1 normal dispatch (call_b): %+v", dropped, ran, log)
	}
	// The model saw a typed skipped receipt on call_a and a REAL result on call_b.
	if len(p.turns) < 2 {
		t.Fatalf("planner saw %d turns, want 2", len(p.turns))
	}
	var skippedReceipt, realResult bool
	for _, msg := range p.turns[1] {
		if msg.Role != RoleTool {
			continue
		}
		switch msg.ToolCallID {
		case "call_a":
			skippedReceipt = strings.Contains(msg.Content, `"status":"skipped"`) && strings.Contains(msg.Content, "CALL_DROPPED_BY_OPERATOR")
		case "call_b":
			realResult = !strings.Contains(msg.Content, `"status":"skipped"`)
		}
	}
	if !skippedReceipt {
		t.Fatal("call_a's tool result is not the typed skipped receipt")
	}
	if !realResult {
		t.Fatal("call_b was skipped too — drop-pending-call must skip exactly the named call")
	}
	// Journaled: the drop, its arrival, and the boundary (turn) it landed at.
	var landed bool
	for _, r := range verbs.Journal() {
		if r.Verb == MidflightDropPendingCall && r.Status == MidflightApplied && r.CallID == "call_a" && r.BoundaryTurn == 1 {
			landed = true
		}
	}
	if !landed {
		t.Fatalf("journal carries no APPLIED drop record for call_a: %+v", verbs.Journal())
	}
	if !VerifyMidflightJournal(verbs.Journal()) {
		t.Fatal("journal hash chain does not verify")
	}
}

// TestSetBudgetAppliedAtTurnBoundary proves the mid-flight budget setter: a
// set-budget issued while turn N is in flight is written through to the live drive
// state at the NEXT boundary — turn N completes untouched, and the same boundary's
// gate adjudicates the fresh allotment with its closed exhaustion reason.
func TestSetBudgetAppliedAtTurnBoundary(t *testing.T) {
	const trace = "midflight-set-budget"
	tbl := session.NewTable()
	verbs := NewMidflightVerbs()
	p := &midTurnFlipPlanner{flip: func() {
		if r := verbs.SetBudget(session.Budget{TurnsLeft: 0, TokensLeft: session.Unbounded, ContextTokensLeft: session.Unbounded}); r != nil {
			t.Errorf("mid-flight set-budget refused on a live run: %v", r)
		}
	}}
	m, err := RunArm(context.Background(), p, "task", false, 5, nil, WithSessionTable(tbl, trace), WithMidflightVerbs(verbs))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	// Turn N completed untouched (both announced calls dispatched)...
	if m.Turns != 1 || m.ToolCalls != 2 {
		t.Fatalf("set-budget disturbed the in-flight turn: turns=%d tool_calls=%d, want 1/2", m.Turns, m.ToolCalls)
	}
	// ...then the boundary's gate read the written-through allotment and stopped the
	// arm with the EXISTING closed exhaustion reason (the OpBudget witness).
	if m.StoppedBySession != session.ReasonBudgetTurns {
		t.Fatalf("StoppedBySession = %q, want %q (the staged budget never reached the gate)", m.StoppedBySession, session.ReasonBudgetTurns)
	}
	var landed bool
	for _, r := range verbs.Journal() {
		if r.Verb == MidflightSetBudget && r.Status == MidflightApplied && r.BoundaryTurn == 2 {
			landed = true
		}
	}
	if !landed {
		t.Fatalf("journal carries no APPLIED set-budget record at boundary 2: %+v", verbs.Journal())
	}
}

// TestSetBudgetWithoutTableRefusesInJournal: with no session table wired there is no
// budget sink — the staged write drains as REFUSED (never silently applied), and the
// run itself is untouched.
func TestSetBudgetWithoutTableRefusesInJournal(t *testing.T) {
	verbs := NewMidflightVerbs()
	if r := verbs.SetBudget(session.Budget{TurnsLeft: 1}); r != nil {
		t.Fatalf("set-budget refused at enqueue: %v", r)
	}
	p := &recordingPlanner{}
	m, err := RunArm(context.Background(), p, "task", false, 1, nil, WithMidflightVerbs(verbs))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.StoppedBySession != "" || m.FinalAnswer == "" {
		t.Fatalf("sinkless set-budget disturbed the run: stopped=%q final=%q", m.StoppedBySession, m.FinalAnswer)
	}
	var refused bool
	for _, r := range verbs.Journal() {
		if r.Verb == MidflightSetBudget && r.Status == MidflightRefused && strings.Contains(r.Detail, "no budget sink") {
			refused = true
		}
	}
	if !refused {
		t.Fatalf("journal carries no REFUSED no-sink record: %+v", verbs.Journal())
	}
}

// TestMidflightNoVerbHistoricalLoop is the byte-for-byte invariant: with a mailbox
// wired but NO verb issued — and with a nil mailbox — the loop behaves exactly like
// the historical loop (final answer, no stop, empty journal).
func TestMidflightNoVerbHistoricalLoop(t *testing.T) {
	for _, verbs := range []*MidflightVerbs{NewMidflightVerbs(), nil} {
		p := &midflightCallPlanner{}
		m, err := RunArm(context.Background(), p, "task", false, 3, nil, WithMidflightVerbs(verbs))
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if m.StoppedBySession != "" || m.FinalAnswer != "done" || m.Turns != 2 || m.ToolCalls != 2 {
			t.Fatalf("no-verb run drifted from the historical loop: %+v", m)
		}
		if verbs != nil && len(verbs.Journal()) != 0 {
			t.Fatalf("no-verb journal is not empty: %+v", verbs.Journal())
		}
	}
}

// TestMidflightVerbsSealedAfterRun: once the arm returns, every verb refuses with the
// closed terminal-session token — a finished run cannot take a mid-flight verb.
func TestMidflightVerbsSealedAfterRun(t *testing.T) {
	verbs := NewMidflightVerbs()
	if _, err := RunArm(context.Background(), &recordingPlanner{}, "task", false, 1, nil, WithMidflightVerbs(verbs)); err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if r := verbs.Interrupt(); r == nil || r.Reason != session.ReasonControlSessionTerminal {
		t.Fatalf("sealed interrupt refusal = %v, want %s", r, session.ReasonControlSessionTerminal)
	}
	if r := verbs.DropPendingCall("call_x"); r == nil || r.Reason != session.ReasonControlSessionTerminal {
		t.Fatalf("sealed drop refusal = %v, want %s", r, session.ReasonControlSessionTerminal)
	}
	if r := verbs.SetBudget(session.Budget{}); r == nil || r.Reason != session.ReasonControlSessionTerminal {
		t.Fatalf("sealed set-budget refusal = %v, want %s", r, session.ReasonControlSessionTerminal)
	}
}

// TestMidflightJournalTamperEvident: the journal is hash-chained — an edited field,
// a dropped record, or a reorder breaks verification.
func TestMidflightJournalTamperEvident(t *testing.T) {
	verbs := NewMidflightVerbs()
	if r := verbs.DropPendingCall("call_a"); r != nil {
		t.Fatalf("drop: %v", r)
	}
	if r := verbs.Interrupt(); r != nil {
		t.Fatalf("interrupt: %v", r)
	}
	if r := verbs.SetBudget(session.Budget{TurnsLeft: 3}); r != nil {
		t.Fatalf("set-budget: %v", r)
	}
	j := verbs.Journal()
	if len(j) < 3 {
		t.Fatalf("journal has %d records, want >= 3", len(j))
	}
	if !VerifyMidflightJournal(j) {
		t.Fatal("intact journal does not verify")
	}
	edited := append([]MidflightRecord(nil), j...)
	edited[1].CallID = "forged"
	if VerifyMidflightJournal(edited) {
		t.Fatal("edited journal still verifies — the chain is not tamper-evident")
	}
	dropped := append(append([]MidflightRecord(nil), j[:1]...), j[2:]...)
	if VerifyMidflightJournal(dropped) {
		t.Fatal("journal with a dropped record still verifies")
	}
}
