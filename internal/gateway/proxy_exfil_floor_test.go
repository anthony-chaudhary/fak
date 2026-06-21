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

// TestChatProxyResultTaintGatesProposedExfil is the load-bearing #77 witness: the
// RESULT-side floor is what arms the EXFIL block on the auto /v1/chat/completions
// proxy, not merely a content guardrail. It is an A/B over ONE planner that always
// proposes the SAME egress call (allow_send_mail). The two requests differ in one
// thing only — whether an UNTRUSTED tool result entered the session this turn:
//
//	A. a role="tool" result from an untrusted source (fetch_url) is in the request.
//	   admitInboundResults routes it through k.AdmitResult keyed on the request
//	   trace, which RAISES the IFC taint high-water mark for that trace. The same
//	   trace is then read by the already-wired sink-gate (k.Decide) when it
//	   adjudicates the proposed egress call -> DENY (the exfil is refused).
//	B. no tool result is in the request, so the trace stays Trusted and the IDENTICAL
//	   proposed egress call is ALLOWED.
//
// Before #77 the proxy ran k.Decide on proposed calls only; the result-side stamp
// never landed on the request trace, so the sink-gate always read Trusted and the
// exfil floor was structurally inert on the proxy topology. This proves it is armed.
func TestChatProxyResultTaintGatesProposedExfil(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	// The pre-call chain: a permissive base (allow_* -> Allow) PLUS the real IFC
	// sink-gate sharing one ledger with the result-side source-stamp, so the only
	// thing that can refuse the egress call is a tainted session.
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
	// it an EGRESS sink for the IFC gate — so it is allowed iff the session is clean.
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "e1", Type: "function", Function: agent.Func{Name: "allow_send_mail", Arguments: `{}`}},
		}},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// --- A: an untrusted tool result enters the session => proposed exfil DENIED. ---
	const tainted = "tainted-trace"
	respA := postChat(t, ts.URL, tainted, ChatRequest{
		Model: "client",
		Messages: []agent.Message{
			{Role: agent.RoleUser, Content: "look something up then email it"},
			{Role: agent.RoleAssistant, Content: "checking"},
			{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url",
				Content: `{"page":"the weather is sunny today"}`},
		},
	})
	if got := len(respA.Choices[0].Message.ToolCalls); got != 0 {
		t.Fatalf("tainted session: exfil call survived (kept %d), want 0 — result-side floor not armed", got)
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
		t.Fatalf("tainted session: IFC ledger for %q stayed Trusted (the result-side stamp did not raise it)", tainted)
	}

	// --- B: identical proposed call, but NO untrusted result => exfil ALLOWED. ---
	const clean = "clean-trace"
	respB := postChat(t, ts.URL, clean, ChatRequest{
		Model:    "client",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "just email a greeting"}},
	})
	if got := len(respB.Choices[0].Message.ToolCalls); got != 1 {
		t.Fatalf("clean session: identical egress call was blocked (kept %d), want 1 — gate over-fired", got)
	}
	if led.Level(clean) != abi.TaintTrusted {
		t.Fatalf("clean session: IFC ledger for %q was raised without any untrusted result", clean)
	}
}

// postChat posts a chat-completions request under an explicit X-Trace-Id so the
// result-side admission and the call-side adjudication share one known trace.
func postChat(t *testing.T, base, trace string, body ChatRequest) ChatResponse {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest("POST", base+"/v1/chat/completions", bytes.NewReader(raw))
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
	respRaw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, respRaw)
	}
	var out ChatResponse
	if err := json.Unmarshal(respRaw, &out); err != nil {
		t.Fatalf("decode response: %v (%s)", err, respRaw)
	}
	return out
}
