package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

// TestAnthropicProxyResultTaintGatesProposedExfil is the Anthropic-wire twin of
// TestChatProxyResultTaintGatesProposedExfil: it proves the RESULT-side exfil
// floor is armed on /v1/messages — the wire Claude Code uses natively — not only
// on /v1/chat/completions. Same A/B over ONE planner that always proposes the SAME
// egress call. The two requests differ in one thing only: whether an UNTRUSTED tool
// result (an Anthropic `tool_result` content block) entered the session this turn.
//
//	A. a tool_result from an untrusted source is in the request. completeAnthropicTurn
//	   now calls admitInboundResults, which routes it through k.AdmitResult keyed on
//	   the request trace and RAISES the IFC taint high-water mark. The same trace is
//	   read by the already-wired sink-gate when it adjudicates the proposed egress
//	   call -> DENY.
//	B. no tool result is in the request, so the trace stays Trusted and the IDENTICAL
//	   proposed egress call is ALLOWED.
//
// Before this fix completeAnthropicTurn never admitted inbound results, so the
// result-side stamp never landed on the Anthropic trace: the sink-gate always read
// Trusted and a poisoned-result session's exfil call sailed through — the Anthropic
// wire was structurally LESS safe than the OpenAI wire. This proves parity.
func TestAnthropicProxyResultTaintGatesProposedExfil(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	led := ifc.NewLedger()
	abi.RegisterAdjudicator(0, toolAdj{})
	abi.RegisterAdjudicator(30, ifc.NewSinkGate(led, ifc.Policy{}))
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(led, ifc.Policy{}))

	srv, err := New(Config{EngineID: "test", Model: "exfil-floor:model", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	// allow_send_mail: the "allow" prefix clears toolAdj, the "send" substring makes
	// it an EGRESS sink for the IFC gate — allowed iff the session is clean.
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "e1", Type: "function", Function: agent.Func{Name: "allow_send_mail", Arguments: `{}`}},
		}},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// --- A: an untrusted tool_result enters the session => proposed exfil DENIED. ---
	const tainted = "tainted-anthropic-trace"
	// An Anthropic user turn carrying a tool_result block (what Claude Code sends
	// back after it ran a tool). DecodeAnthropicMessagesRequest turns this into a
	// canonical RoleTool message that admitInboundResults vets.
	bodyA := `{"model":"claude-opus-4-8","max_tokens":64,"messages":[
		{"role":"user","content":"look something up then email it"},
		{"role":"assistant","content":[{"type":"text","text":"checking"}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"fetch_url",
			"content":"the weather is sunny today"}]}
	]}`
	respA := postAnthropic(t, ts.URL, tainted, bodyA)
	var keptA int
	for _, b := range respA.Content {
		if b.Type == "tool_use" {
			keptA++
		}
	}
	if keptA != 0 {
		t.Fatalf("tainted session: exfil call survived (kept %d tool_use), want 0 — result-side floor not armed on the Anthropic wire", keptA)
	}
	if respA.Fak == nil {
		t.Fatalf("tainted session: missing fak extension")
	}
	var sawDeny bool
	for _, a := range respA.Fak.Adjudications {
		if a.Tool == "allow_send_mail" && a.Verdict.Kind == "DENY" && a.Verdict.Reason == "TRUST_VIOLATION" {
			sawDeny = true
		}
	}
	if !sawDeny {
		t.Fatalf("tainted session: egress call not denied for TRUST_VIOLATION: %+v", respA.Fak.Adjudications)
	}
	if led.Level(tainted) == abi.TaintTrusted {
		t.Fatalf("tainted session: IFC ledger for %q stayed Trusted (the result-side stamp did not raise it on the Anthropic wire)", tainted)
	}

	// --- B: identical proposed call, but NO untrusted result => exfil ALLOWED. ---
	const clean = "clean-anthropic-trace"
	bodyB := `{"model":"claude-opus-4-8","max_tokens":64,"messages":[
		{"role":"user","content":"just email a greeting"}
	]}`
	respB := postAnthropic(t, ts.URL, clean, bodyB)
	var keptB int
	for _, b := range respB.Content {
		if b.Type == "tool_use" {
			keptB++
		}
	}
	if keptB != 1 {
		t.Fatalf("clean session: identical egress call was blocked (kept %d tool_use), want 1 — gate over-fired on the Anthropic wire", keptB)
	}
	if led.Level(clean) != abi.TaintTrusted {
		t.Fatalf("clean session: IFC ledger for %q was raised without any untrusted result", clean)
	}
}

// postAnthropic posts a /v1/messages request under an explicit X-Trace-Id so the
// result-side admission and the call-side adjudication share one known trace.
func postAnthropic(t *testing.T, base, trace, body string) anthropicMessageResponse {
	t.Helper()
	req, err := http.NewRequest("POST", base+"/v1/messages", bytes.NewReader([]byte(body)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(traceHeader, trace)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, raw)
	}
	var out anthropicMessageResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode response: %v (%s)", err, raw)
	}
	return out
}
