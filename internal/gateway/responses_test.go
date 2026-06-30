package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

// responses_test.go proves the OpenAI Responses wire (POST /v1/responses) runs the
// SAME kernel adjudication every other inbound wire does — a benign function call is
// admitted as a function_call output item, a policy-denied call is absent, a
// grammar-repaired call carries the rewritten arguments, and an inbound
// function_call_output result actually transits the result-side floor. These are the
// #925 acceptance witnesses: the wire is not a benign passthrough.

// postResponses posts a /v1/responses request and decodes the buffered response. It
// returns the HTTP status and (on 200) the decoded body, mirroring postJSON but for
// the Responses shape. body is posted as a raw map so the real string|array `input`
// union decoder runs end to end.
func postResponses(t *testing.T, base string, body any) (int, responsesResponse) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	httpResp, err := http.Post(base+"/v1/responses", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		return httpResp.StatusCode, responsesResponse{}
	}
	var out responsesResponse
	if err := json.Unmarshal(respRaw, &out); err != nil {
		t.Fatalf("decode 200 body: %v: %s", err, respRaw)
	}
	return httpResp.StatusCode, out
}

// functionCallItems filters a Responses output to its function_call items, keyed by
// name, for assertion convenience.
func functionCallItems(out []responsesOutputItem) map[string]responsesOutputItem {
	m := map[string]responsesOutputItem{}
	for _, it := range out {
		if it.Type == "function_call" {
			m[it.Name] = it
		}
	}
	return m
}

// messageText concatenates the output_text of every message item in a Responses
// output.
func messageText(out []responsesOutputItem) string {
	var b strings.Builder
	for _, it := range out {
		if it.Type != "message" {
			continue
		}
		for _, p := range it.Content {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// TestResponsesProxyAllowsBenignAndDropsDenied is the core #925 witness: the kernel
// verdict pass runs on the Responses wire. The stub proposes three calls — allow_a
// (ALLOW), deny_b (DENY/POLICY_BLOCK), transform_c (TRANSFORM, args repaired). The
// response must carry function_call items for allow_a (verbatim args) and
// transform_c (repaired args), NONE for deny_b, the full 3-call adjudication in the
// fak extension, and status "completed".
func TestResponsesProxyAllowsBenignAndDropsDenied(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "c_a", Type: "function", Function: agent.Func{Name: "allow_a", Arguments: `{"x":1}`}},
			{ID: "c_b", Type: "function", Function: agent.Func{Name: "deny_b", Arguments: `{}`}},
			{ID: "c_c", Type: "function", Function: agent.Func{Name: "transform_c", Arguments: `{"secret":"y"}`}},
		}},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "m",
		"input": "do the three things",
		"tools": []map[string]any{
			{"type": "function", "name": "allow_a"},
			{"type": "function", "name": "deny_b"},
			{"type": "function", "name": "transform_c"},
		},
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if resp.Object != "response" || resp.Status != "completed" {
		t.Fatalf("object/status = %q/%q, want response/completed", resp.Object, resp.Status)
	}
	calls := functionCallItems(resp.Output)
	if _, ok := calls["deny_b"]; ok {
		t.Errorf("denied call deny_b leaked into output: %+v", resp.Output)
	}
	a, ok := calls["allow_a"]
	if !ok {
		t.Fatalf("allowed call allow_a missing from output: %+v", resp.Output)
	}
	if a.Arguments != `{"x":1}` || a.CallID != "c_a" {
		t.Errorf("allow_a item = %+v, want args {\"x\":1} call_id c_a", a)
	}
	c, ok := calls["transform_c"]
	if !ok {
		t.Fatalf("transformed call transform_c missing from output: %+v", resp.Output)
	}
	if c.Arguments != `{"redacted":true}` {
		t.Errorf("transform_c arguments = %q, want the repaired {\"redacted\":true}", c.Arguments)
	}
	if resp.Fak == nil || len(resp.Fak.Adjudications) != 3 {
		t.Fatalf("fak.adjudications = %+v, want 3 (the verdict pass must have run on this wire)", resp.Fak)
	}
}

// TestResponsesAllDeniedSynthesizesText proves that when every proposed call is
// refused, the assistant message carries the deny summary as output_text (so even a
// fak-unaware Responses client gets something actionable) and there are no
// function_call items.
func TestResponsesAllDeniedSynthesizesText(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "c_b", Type: "function", Function: agent.Func{Name: "deny_b", Arguments: `{}`}},
		}},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "m",
		"input": "do the denied thing",
		"tools": []map[string]any{{"type": "function", "name": "deny_b"}},
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(functionCallItems(resp.Output)) != 0 {
		t.Errorf("all-denied turn still emitted function_call items: %+v", resp.Output)
	}
	if !strings.Contains(messageText(resp.Output), "refused by the fak kernel") {
		t.Errorf("output_text = %q, want the deny summary", messageText(resp.Output))
	}
}

// TestResponsesInboundResultGatesProposedExfil proves the RESULT-side floor is armed
// on the Responses wire: a function_call_output from an untrusted source
// (fetch_url) raises the trace taint via admitInboundResults, which then DENIES an
// otherwise-allowed proposed egress call. It is the Responses analogue of
// TestChatProxyResultTaintGatesProposedExfil — an A/B over one planner that always
// proposes allow_send_mail; the arms differ only in whether an untrusted
// function_call_output entered this turn.
func TestResponsesInboundResultGatesProposedExfil(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	led := ifc.NewLedger()
	abi.RegisterAdjudicator(0, toolAdj{})
	abi.RegisterAdjudicator(30, ifc.NewSinkGate(led, ifc.Policy{}))
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(led, ifc.Policy{}))

	srv, err := New(Config{EngineID: "test", Model: "responses-exfil:model", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "e1", Type: "function", Function: agent.Func{Name: "allow_send_mail", Arguments: `{}`}},
		}},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// A: an untrusted function_call_output enters the turn => proposed exfil DENIED.
	const tainted = "responses-tainted"
	respA := postResponsesTrace(t, ts.URL, tainted, map[string]any{
		"model": "client",
		"input": []map[string]any{
			{"type": "message", "role": "user", "content": "look something up then email it"},
			{"type": "function_call", "call_id": "call_1", "name": "fetch_url", "arguments": "{}"},
			{"type": "function_call_output", "call_id": "call_1", "output": `{"page":"the weather is sunny today"}`},
		},
	})
	if got := len(functionCallItems(respA.Output)); got != 0 {
		t.Fatalf("tainted turn: exfil call survived (kept %d), want 0 — result-side floor not armed on /v1/responses", got)
	}
	if led.Level(tainted) == abi.TaintTrusted {
		t.Fatalf("tainted turn: IFC ledger for %q stayed Trusted — admitInboundResults did not run on the decoded Responses input", tainted)
	}

	// B: identical proposed call, no untrusted result => exfil ALLOWED.
	const clean = "responses-clean"
	respB := postResponsesTrace(t, ts.URL, clean, map[string]any{
		"model": "client",
		"input": "just email a greeting",
	})
	if got := len(functionCallItems(respB.Output)); got != 1 {
		t.Fatalf("clean turn: identical egress call was blocked (kept %d), want 1 — gate over-fired", got)
	}
}

// postResponsesTrace posts a /v1/responses request under an explicit X-Trace-Id so
// the result-side admission and the call-side adjudication share one known trace.
func postResponsesTrace(t *testing.T, base, trace string, body any) responsesResponse {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest("POST", base+"/v1/responses", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(traceHeader, trace)
	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/responses (trace %s) = %d, want 200: %s", trace, httpResp.StatusCode, respRaw)
	}
	var out responsesResponse
	if err := json.Unmarshal(respRaw, &out); err != nil {
		t.Fatalf("decode body: %v: %s", err, respRaw)
	}
	return out
}

// TestResponsesMalformedBodyIsBadRequest proves a non-JSON body is rejected with a
// 400 before the planner is reached.
func TestResponsesMalformedBodyIsBadRequest(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "planner must not be reached"},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(`{not json`))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", httpResp.StatusCode)
	}
}

// TestResponsesEmptyInputIsBadRequest proves a missing/empty input is rejected with
// a 400 ("input: field required") rather than forwarded as a degenerate request.
func TestResponsesEmptyInputIsBadRequest(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "planner must not be reached"},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for _, body := range []string{`{"model":"m"}`, `{"model":"m","input":[]}`, `{"model":"m","input":""}`} {
		httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		respRaw, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		if httpResp.StatusCode != http.StatusBadRequest {
			t.Fatalf("body %s: status = %d, want 400: %s", body, httpResp.StatusCode, respRaw)
		}
		if !strings.Contains(string(respRaw), "input: field required") {
			t.Errorf("body %s: response = %s, want \"input: field required\"", body, respRaw)
		}
	}
}

// TestResponsesStreamEmitsSSE proves the Responses wire synthesizes a well-formed
// SSE sequence when stream:true: response.created → response.output_item.added
// → response.output_item.done → response.completed.
func TestResponsesStreamEmitsSSE(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "hello back"},
		FinishReason: "stop",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json",
		strings.NewReader(`{"model":"m","input":"hi","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, body)
	}

	ct := httpResp.Header.Get("Content-Type")
	if ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	// Parse SSE events and verify the sequence
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		t.Fatal(err)
	}
	events := parseTypedSSE(t, string(body))

	// Should have at least: response.created, response.output_item.added,
	// response.output_item.done, response.completed
	if len(events) < 4 {
		t.Fatalf("got %d SSE events, want at least 4 (created, added, item done, completed): %v", len(events), events)
	}

	// response.created should be first
	if events[0].Event != "response.created" {
		t.Errorf("first event = %q, want response.created", events[0].Event)
	}

	// response.completed should be last; real Codex treats streams that close before
	// this event as incomplete and retries.
	last := events[len(events)-1]
	if last.Event != "response.completed" {
		t.Errorf("last event = %q, want response.completed", events[len(events)-1].Event)
	}
	if !strings.Contains(last.Data, `"type":"response.completed"`) || !strings.Contains(last.Data, `"response":`) {
		t.Errorf("response.completed data = %s, want OpenAI-style event envelope with type and response", last.Data)
	}

	// Verify we have output_item events
	var foundAdded, foundDone bool
	for _, ev := range events {
		if ev.Event == "response.output_item.added" {
			foundAdded = true
			if !strings.Contains(ev.Data, `"type":"response.output_item.added"`) || !strings.Contains(ev.Data, `"output_index":0`) {
				t.Errorf("response.output_item.added data = %s, want type and output_index", ev.Data)
			}
		}
		if ev.Event == "response.output_item.done" {
			foundDone = true
			if !strings.Contains(ev.Data, `"type":"response.output_item.done"`) || !strings.Contains(ev.Data, `"output_index":0`) {
				t.Errorf("response.output_item.done data = %s, want type and output_index", ev.Data)
			}
		}
	}
	if !foundAdded {
		t.Error("no response.output_item.added event found")
	}
	if !foundDone {
		t.Error("no response.output_item.done event found")
	}
}

// typedSSEEvent is a parsed SSE event with a typed event name (like Anthropic/Responses)
type typedSSEEvent struct {
	Event string
	Data  string
}

// parseTypedSSE parses an SSE body with typed events (event: ..., data: ...) into
// a slice of typed events.
func parseTypedSSE(t *testing.T, body string) []typedSSEEvent {
	t.Helper()
	var out []typedSSEEvent
	var ev, data string
	flush := func() {
		if data != "" {
			out = append(out, typedSSEEvent{Event: ev, Data: data})
		}
		ev, data = "", ""
	}
	for _, line := range strings.Split(body, "\n") {
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, "event:"):
			ev = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data += strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
	}
	flush()
	return out
}

// TestResponsesInputStringForm proves the bare-string input form decodes to a single
// user message and yields a normal completed response with no fak extension when no
// tools were proposed.
func TestResponsesInputStringForm(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "hello back"},
		FinishReason: "stop",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{"model": "m", "input": "hello"})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if resp.Status != "completed" {
		t.Errorf("status = %q, want completed", resp.Status)
	}
	if resp.OutputText != "hello back" || messageText(resp.Output) != "hello back" {
		t.Errorf("output_text = %q / message = %q, want \"hello back\"", resp.OutputText, messageText(resp.Output))
	}
	if resp.Fak != nil {
		t.Errorf("fak extension present on a no-tool turn: %+v", resp.Fak)
	}
}
