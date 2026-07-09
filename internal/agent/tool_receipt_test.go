package agent

// tool_receipt_test.go — #2414 acceptance witnesses for the OWNED loop's typed
// tool-result receipts: a denied call becomes a structured error bound to the
// originating call ID (reason+disposition+fix) that the NEXT planner input carries,
// and a legitimately not-sent call surfaces status=skipped instead of feigning success.

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// scriptRecordingPlanner returns a fixed per-turn script (like scriptedPlanner) AND
// records the exact messages it was handed each turn — so a test can assert what the
// OWNED loop placed in the NEXT planner input (the tool_result receipt on the call ID).
type scriptRecordingPlanner struct {
	turns []*Completion
	seen  [][]Message
	n     int
}

func (p *scriptRecordingPlanner) Model() string { return "script-recording" }

func (p *scriptRecordingPlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.seen = append(p.seen, append([]Message(nil), messages...))
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}

// callTurn is toolCallTurn with an explicit call ID so the receipt binding is checkable.
func callTurn(id, tool, args string) *Completion {
	return &Completion{Message: Message{ToolCalls: []ToolCall{{ID: id, Function: Func{Name: tool, Arguments: args}}}}}
}

// findReceipt returns the parsed ToolReceipt on the tool message bound to callID within
// msgs, or fails the test if no such tool_result block is present.
func findReceipt(t *testing.T, msgs []Message, callID string) ToolReceipt {
	t.Helper()
	for _, msg := range msgs {
		if msg.Role != RoleTool || msg.ToolCallID != callID {
			continue
		}
		var rc ToolReceipt
		if err := json.Unmarshal([]byte(msg.Content), &rc); err != nil {
			t.Fatalf("tool_result on call %q is not a typed receipt: %v; content=%q", callID, err, msg.Content)
		}
		return rc
	}
	t.Fatalf("no tool_result block bound to originating call id %q in the next planner input; msgs=%+v", callID, msgs)
	return ToolReceipt{}
}

func TestDeniedCallReturnsStructuredError(t *testing.T) {
	// Turn 1 proposes the policy-DENIED delete_account; turn 2 is the model's adaptation.
	p := &scriptRecordingPlanner{turns: []*Completion{
		callTurn("call_del", "delete_account", `{"user_id":"mia_li_3668"}`),
		{Message: Message{Content: "I cannot delete the account."}},
	}}

	m, err := RunArm(context.Background(), p, "delete my account", true, 6, nil)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.Denies == 0 {
		t.Fatalf("the owned loop did not deny delete_account (Denies=0)")
	}

	// The NEXT planner input (turn 2's messages) must carry the structured error on the
	// ORIGINATING call id — not a prose "[fak]" note, and not a vanished call.
	if len(p.seen) < 2 {
		t.Fatalf("planner ran %d turns, want >= 2 (the deny must drive an adaptation turn)", len(p.seen))
	}
	rc := findReceipt(t, p.seen[1], "call_del")
	if rc.Status != ToolResultError {
		t.Fatalf("denied call receipt status = %q, want %q", rc.Status, ToolResultError)
	}
	if rc.Reason != "POLICY_BLOCK" {
		t.Fatalf("denied call receipt reason = %q, want POLICY_BLOCK", rc.Reason)
	}
	if rc.Disposition == "" {
		t.Fatalf("denied call receipt carried no disposition; want a closed loopback class")
	}
	// The prose adjudicationNote voice must be absent on the owned path: the receipt is
	// structured JSON, never a "[fak] ..." splice.
	for _, msg := range p.seen[1] {
		if msg.Role == RoleTool && msg.ToolCallID == "call_del" {
			if got := msg.Content; got == "" || got[0] != '{' {
				t.Fatalf("owned-loop deny is not a structured tool_result (got prose?): %q", got)
			}
		}
	}
}

// TestDeniedReceiptSurfacesFix is the focused witness that the receipt carries the
// closed-vocabulary fix/remedy when the deciding verdict supplies one (the arg-predicate
// rung's Meta["fix"] / reversibility rung's Meta["dry_run_hint"]) — the SAME source the
// gateway wire's renderVerdict reads. A bare policy block carries none, so this proves
// the field flows through rather than asserting a non-empty fix on delete_account.
func TestDeniedReceiptSurfacesFix(t *testing.T) {
	result := &abi.Result{Meta: map[string]string{"reason": "MISROUTE", "disposition": "RETRYABLE"}}
	v := abi.Verdict{Kind: abi.VerdictDeny, Meta: map[string]string{"fix": "call search_direct_flight instead"}}
	rc := denyToolReceipt(result, v)
	if rc.Status != ToolResultError || rc.Reason != "MISROUTE" || rc.Disposition != "RETRYABLE" {
		t.Fatalf("receipt = %+v, want error/MISROUTE/RETRYABLE", rc)
	}
	if rc.Fix != "call search_direct_flight instead" {
		t.Fatalf("receipt fix = %q, want the verdict's Meta[fix]", rc.Fix)
	}
	// The dry_run_hint fallback (reversibility rung) also lands in Fix.
	rc2 := denyToolReceipt(&abi.Result{Meta: map[string]string{"reason": "SELF_MODIFY", "disposition": "ESCALATE"}},
		abi.Verdict{Meta: map[string]string{"dry_run_hint": "preview with --dry-run first"}})
	if rc2.Fix != "preview with --dry-run first" {
		t.Fatalf("receipt fix = %q, want the dry_run_hint fallback", rc2.Fix)
	}
}

func TestNoOpReportsSkipped(t *testing.T) {
	// The write barrier (#1319) is the owned loop's canonical NOT-SENT no-op: a write
	// behind a squashed speculative read is never dispatched. It must surface as a typed
	// status=skipped receipt, not a feigned success.
	p := &scriptRecordingPlanner{turns: []*Completion{
		callTurn("call_read", "get_user_details", `{"user_id":"u1"}`), // read → speculate a follow-on read
		callTurn("call_book", "book_flight", `{"flight_id":"f1"}`),    // write → squashes the read-prediction, barred
		{Message: Message{Content: "done"}},                           // final answer
	}}
	spec := abi.NewSpeculator(0)
	spec.Learn(searchPattern(`{"origin":"SFO"}`)) // predicts a READ after get_user_details

	m, err := RunArm(context.Background(), p, "book me a flight", true, 10, nil, WithSpeculator(spec))
	if err != nil {
		t.Fatalf("RunArm (with speculator): %v", err)
	}
	if m.WritesBarred == 0 {
		t.Fatalf("the write behind the squashed speculation was not barred (WritesBarred=0)")
	}
	if len(p.seen) < 3 {
		t.Fatalf("planner ran %d turns, want >= 3", len(p.seen))
	}
	rc := findReceipt(t, p.seen[2], "call_book")
	if rc.Status != ToolResultSkipped {
		t.Fatalf("barred (not-sent) call receipt status = %q, want %q", rc.Status, ToolResultSkipped)
	}
	if rc.Reason != "WRITE_BARRED" {
		t.Fatalf("skipped receipt reason = %q, want WRITE_BARRED", rc.Reason)
	}
}
