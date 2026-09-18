package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// floor_2088_test.go — adversarial tests for #2088, CONTRACT B: a forced
// tool_choice whose advertised schema could not be formed into an adjudicable
// call must surface as a TYPED refusal whose error `code` is the closed reason
// NAME (MALFORMED / OVERSIZE), instead of the historical opaque 502. ReasonNone
// keeps the legacy opaque behavior fail-closed.
//
// These drive the REAL gateway rendering entrypoint
// (*Server).validateChatCompletionConformance — the same function the served
// chat path calls — so the assertion is on the production code path, not a
// re-implementation.

// floor2088Server returns a Server enough to run validateChatCompletionConformance:
// a non-nil logf (the gate logs the drop) and the fields the conformance gate reads.
func floor2088Server() *Server {
	return &Server{logf: func(string, ...any) {}}
}

// decodeErrEnvelope pulls the OpenAI error envelope out of a written body.
func decodeErrEnvelope(t *testing.T, rec *httptest.ResponseRecorder) (code string, hasCode bool, message string) {
	t.Helper()
	var body struct {
		Error struct {
			Message string `json:"message"`
			Code    any    `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode error envelope: %v (%s)", err, rec.Body.String())
	}
	if body.Error.Code != nil {
		if s, ok := body.Error.Code.(string); ok {
			code = s
			hasCode = true
		}
	}
	return code, hasCode, body.Error.Message
}

// TestFloor2088MalformedDropRendersTypedCode is the #2088 acceptance: a forced
// tool call dropped with ReasonMalformed must render error code "MALFORMED" and
// the message must still refuse to skip adjudication.
func TestFloor2088MalformedDropRendersTypedCode(t *testing.T) {
	s := floor2088Server()
	comp := &agent.Completion{
		ToolCallsDropped:       true,
		ToolCallsDroppedReason: abi.ReasonMalformed,
		Message:                agent.Message{Role: agent.RoleAssistant},
	}
	asst := comp.Message
	asst.Role = agent.RoleAssistant

	rec := httptest.NewRecorder()
	ok := s.validateChatCompletionConformance(rec, nil, comp, asst, false, false, false, nil)
	if ok {
		t.Fatal("conformance gate returned true for a dropped tool call; want false (fail-closed)")
	}
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	code, hasCode, msg := decodeErrEnvelope(t, rec)
	if !hasCode || code != "MALFORMED" {
		t.Fatalf("error code = (%q, hasCode=%v), want typed `MALFORMED`", code, hasCode)
	}
	if !strings.Contains(msg, "refusing to skip adjudication") {
		t.Fatalf("refusal message %q must still state it refuses to skip adjudication", msg)
	}
}

// TestFloor2088OversizeDropRendersTypedCode is the length-stop sibling: a drop
// with ReasonOversize must render code "OVERSIZE".
func TestFloor2088OversizeDropRendersTypedCode(t *testing.T) {
	s := floor2088Server()
	comp := &agent.Completion{
		ToolCallsDropped:       true,
		ToolCallsDroppedReason: abi.ReasonOversize,
		Message:                agent.Message{Role: agent.RoleAssistant},
	}
	rec := httptest.NewRecorder()
	if s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
		t.Fatal("conformance gate returned true for a dropped oversize call; want false")
	}
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	code, hasCode, _ := decodeErrEnvelope(t, rec)
	if !hasCode || code != "OVERSIZE" {
		t.Fatalf("error code = (%q, hasCode=%v), want typed `OVERSIZE`", code, hasCode)
	}
}

// TestFloor2088ReasonNoneKeepsOpaquePath is the fail-closed generic path: with
// ReasonNone (unspecified unparseable upstream format) the historical opaque
// message is emitted and NO typed code appears (code stays null).
func TestFloor2088ReasonNoneKeepsOpaquePath(t *testing.T) {
	s := floor2088Server()
	comp := &agent.Completion{
		ToolCallsDropped:       true,
		ToolCallsDroppedReason: abi.ReasonNone,
		Message:                agent.Message{Role: agent.RoleAssistant},
	}
	rec := httptest.NewRecorder()
	if s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
		t.Fatal("conformance gate returned true for a dropped call; want false")
	}
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", rec.Code)
	}
	code, hasCode, msg := decodeErrEnvelope(t, rec)
	if hasCode || code != "" {
		t.Fatalf("ReasonNone must NOT carry a typed code; got code=%q hasCode=%v", code, hasCode)
	}
	if !strings.Contains(msg, "tool-call format not recognized") {
		t.Fatalf("ReasonNone message = %q, want the historical opaque format", msg)
	}
}

// TestFloor2088DroppedButCallsPresentNoRefusalEdge: ToolCallsDropped=true but
// len(ToolCalls)>0 must NOT trigger the refusal — the gate is keyed on the
// AND of the drop flag and an EMPTY parsed call set. A regression here would
// refuse a turn that actually produced adjudicable calls.
func TestFloor2088DroppedButCallsPresentNoRefusalEdge(t *testing.T) {
	s := floor2088Server()
	comp := &agent.Completion{
		ToolCallsDropped:       true,
		ToolCallsDroppedReason: abi.ReasonMalformed,
		Message: agent.Message{
			Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{{
				ID:       "call_1",
				Type:     "function",
				Function: agent.Func{Name: "Read", Arguments: `{"file_path":"x"}`},
			}},
		},
	}
	rec := httptest.NewRecorder()
	// receipt/decode flags false so no later gate fires; the call should pass.
	ok := s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil)
	if !ok {
		t.Fatalf("a dropped flag WITH parsed calls must not trigger the refusal; got false (body=%s)", rec.Body.String())
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("no refusal should be written when calls are present; body=%s", rec.Body.String())
	}
}

// TestFloor2088NotDroppedNoRefusalEdge: ToolCallsDropped=false must pass through
// even with zero calls (a benign empty/text turn is not a conformance refusal).
func TestFloor2088NotDroppedNoRefusalEdge(t *testing.T) {
	s := floor2088Server()
	comp := &agent.Completion{
		ToolCallsDropped: false,
		Message:          agent.Message{Role: agent.RoleAssistant, Content: "hello"},
	}
	rec := httptest.NewRecorder()
	if !s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
		t.Fatalf("ToolCallsDropped=false must pass; got false (body=%s)", rec.Body.String())
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("no refusal should be written; body=%s", rec.Body.String())
	}
}
