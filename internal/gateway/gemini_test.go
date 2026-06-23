package gateway

import (
	"bufio"
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

// TestGeminiGenerateContentAdjudicates is the end-to-end proof of the
// Gemini-native front door: a Gemini-CLI/google-genai-shaped client POSTs
// generateContent, fak decodes it, forwards to a Gemini upstream, runs every
// proposed functionCall through the kernel, and renders the survivors back on the
// Gemini wire. deny_b is dropped, transform_c is repaired, allow_a survives, and
// the fak extension carries all three adjudications — exact parity with the
// OpenAI/Anthropic proxy paths.
func TestGeminiGenerateContentAdjudicates(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	inbound := []byte(`{"contents":[{"role":"user","parts":[{"text":"call tools"}]}],` +
		`"tools":[{"functionDeclarations":[{"name":"allow_a","parameters":{"type":"object"}},` +
		`{"name":"deny_b","parameters":{"type":"object"}},{"name":"transform_c","parameters":{"type":"object"}}]}]}`)

	upstreamHits := 0
	var gotKey, gotPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		gotPath = r.URL.Path
		gotKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[` +
			`{"text":"checking"},` +
			`{"functionCall":{"name":"allow_a","args":{"x":1},"id":"g1"}},` +
			`{"functionCall":{"name":"deny_b","args":{},"id":"g2"}},` +
			`{"functionCall":{"name":"transform_c","args":{"secret":"y"},"id":"g3"}}]},` +
			`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"totalTokenCount":10}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "gemini-test", BaseURL: upstream.URL, Provider: "gemini", APIKey: "sekret", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1beta/models/gemini-test:generateContent", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", "sekret")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if upstreamHits != 1 {
		t.Fatalf("upstream hits = %d, want 1", upstreamHits)
	}
	if gotPath != "/models/gemini-test:generateContent" {
		t.Errorf("upstream path = %q (model pass-through)", gotPath)
	}
	if gotKey != "sekret" {
		t.Errorf("upstream x-goog-api-key = %q, want sekret", gotKey)
	}

	var body geminiGenerateContentResponse
	raw, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode: %v (%s)", err, raw)
	}
	if len(body.Candidates) != 1 {
		t.Fatalf("candidates = %d", len(body.Candidates))
	}
	// The rendered candidate carries the model role and the adjudicated parts:
	// text + allow_a (survived) + transform_c (repaired). deny_b must be absent.
	// When the kernel dropped or repaired a call, an in-band [fak] text note is
	// PREPENDED (parity with the Anthropic wire) so a client that reads only parts
	// still sees the decision — so assert by content, not position.
	parts := body.Candidates[0].Content.Parts
	var texts, names []string
	var repairedArgs string
	for _, p := range parts {
		if p.Text != "" {
			texts = append(texts, p.Text)
		}
		if p.FunctionCall != nil {
			names = append(names, p.FunctionCall.Name)
			if p.FunctionCall.Name == "deny_b" {
				t.Error("denied functionCall must NOT reach the Gemini client")
			}
			if p.FunctionCall.Name == "transform_c" {
				repairedArgs = string(p.FunctionCall.Args)
			}
		}
	}
	allText := strings.Join(texts, "\n")
	if !strings.Contains(allText, "checking") {
		t.Errorf("model text part missing: %v", texts)
	}
	if !strings.Contains(allText, "refused") {
		t.Errorf("in-band adjudication note missing (should be prepended): %v", texts)
	}
	if len(names) != 2 || names[0] != "allow_a" || names[1] != "transform_c" {
		t.Fatalf("surviving functionCalls = %v, want [allow_a transform_c]", names)
	}
	// transform_c's args were repaired by the kernel to {"redacted":true}.
	if !strings.Contains(repairedArgs, "redacted") {
		t.Errorf("transform_c args not repaired: %s", repairedArgs)
	}
	if body.Fak == nil || len(body.Fak.Adjudications) != 3 {
		t.Fatalf("fak adjudications = %+v, want 3", body.Fak)
	}
	if body.UsageMetadata.PromptTokenCount != 7 || body.UsageMetadata.CandidatesTokenCount != 3 {
		t.Errorf("usageMetadata not forwarded: %+v", body.UsageMetadata)
	}
	if body.Candidates[0].FinishReason != "STOP" {
		t.Errorf("finishReason = %q, want STOP (Gemini signals tool use via parts)", body.Candidates[0].FinishReason)
	}
}

// TestGeminiStreamGenerateContentSynthesizesSSE proves the streaming route emits a
// well-formed Gemini SSE data frame carrying the finished, already-adjudicated
// candidate — so a Gemini CLI client (which defaults to streamGenerateContent)
// parses the turn identically to a buffered response, with denied calls still
// stripped before the client sees them.
func TestGeminiStreamGenerateContentSynthesizesSSE(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	inbound := []byte(`{"contents":[{"role":"user","parts":[{"text":"call tools"}]}],` +
		`"tools":[{"functionDeclarations":[{"name":"allow_a","parameters":{"type":"object"}},{"name":"deny_b","parameters":{"type":"object"}}]}]}`)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[` +
			`{"text":"checking"},` +
			`{"functionCall":{"name":"allow_a","args":{"x":1},"id":"g1"}},` +
			`{"functionCall":{"name":"deny_b","args":{},"id":"g2"}}]},` +
			`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "gemini-test", BaseURL: upstream.URL, Provider: "gemini", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1beta/models/gemini-test:streamGenerateContent", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}
	// Parse the SSE body: one or more `data: {json}` frames.
	var frame geminiGenerateContentResponse
	sawData := false
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if d, ok := strings.CutPrefix(line, "data: "); ok && d != "" {
			sawData = true
			if err := json.Unmarshal([]byte(d), &frame); err != nil {
				t.Fatalf("decode data frame: %v (%s)", err, d)
			}
		}
	}
	if !sawData {
		t.Fatal("no data frame in stream")
	}
	var names []string
	for _, p := range frame.Candidates[0].Content.Parts {
		if p.FunctionCall != nil {
			names = append(names, p.FunctionCall.Name)
		}
	}
	if len(names) != 1 || names[0] != "allow_a" {
		t.Fatalf("stream functionCalls = %v, want [allow_a] (deny_b stripped)", names)
	}
	if frame.UsageMetadata.TotalTokenCount != 6 {
		t.Errorf("stream usageMetadata total = %d, want 6", frame.UsageMetadata.TotalTokenCount)
	}
}

// TestGeminiAuthViaGoogApiKey proves an authenticated (non-loopback) gateway admits
// a Gemini-native client that authenticates with the x-goog-api-key header — the
// arm added so Gemini CLI / google-genai (which never send Authorization: Bearer)
// do not 401 against a RequireKey gateway.
func TestGeminiAuthViaGoogApiKey(t *testing.T) {
	srv := newTestServer(t)
	srv.requireKey = "goog-secret"
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	inbound := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	cases := []struct {
		name   string
		hdr    string
		hdrVal string
		wantOK bool
	}{
		{"x-goog-api-key", "X-Goog-Api-Key", "goog-secret", true},
		{"x-api-key still works", "X-Api-Key", "goog-secret", true},
		{"wrong key rejected", "X-Goog-Api-Key", "nope", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req, _ := http.NewRequest("POST", ts.URL+"/v1beta/models/m:generateContent", bytes.NewReader(inbound))
			req.Header.Set("Content-Type", "application/json")
			if c.hdrVal != "" {
				req.Header.Set(c.hdr, c.hdrVal)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if c.wantOK && resp.StatusCode != 200 {
				t.Errorf("status = %d, want 200", resp.StatusCode)
			}
			if !c.wantOK && resp.StatusCode != 401 {
				t.Errorf("status = %d, want 401", resp.StatusCode)
			}
		})
	}
}

// TestGeminiRouteRejectsUnknownMethod proves the /v1beta/ subtree hands only the
// documented :generateContent / :streamGenerateContent methods to the handler and
// 404s anything else (so a stray /v1beta/models path cannot masquerade as a served
// model turn).
func TestGeminiRouteRejectsUnknownMethod(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	cases := []struct {
		path   string
		status int
	}{
		{"/v1beta/models/m:bogus", 404},           // unknown method
		{"/v1beta/notmodels", 404},                // not a models/ route
		{"/v2beta/models/m:generateContent", 404}, // wrong version prefix
	}
	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			req, _ := http.NewRequest("POST", ts.URL+c.path, bytes.NewReader([]byte(`{"contents":[{"role":"user","parts":[{"text":"x"}]}]}`)))
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != c.status {
				t.Errorf("status = %d, want %d", resp.StatusCode, c.status)
			}
		})
	}
}

// TestGeminiResultFloorArmsExfilGate is the Gemini-wire twin of
// TestChatProxyResultTaintGatesProposedExfil: an inbound functionResponse (a tool
// result the Gemini client executed) flows through admitInboundResults, which routes
// it through k.AdmitResult keyed on the trace, RAISING the IFC taint high-water
// mark. The same trace is then read by the sink-gate when it adjudicates the
// proposed egress call -> DENY (TRUST_VIOLATION). This proves the Gemini wire has
// the SAME result-side floor the OpenAI and Anthropic wires have — not merely a
// call-side guardrail.
func TestGeminiResultFloorArmsExfilGate(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	led := ifc.NewLedger()
	abi.RegisterAdjudicator(0, toolAdj{})
	abi.RegisterAdjudicator(30, ifc.NewSinkGate(led, ifc.Policy{}))
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(led, ifc.Policy{}))

	srv, err := New(Config{EngineID: "test", Model: "gemini-floor-test", VDSO: true})
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

	// A: an untrusted functionResponse (fetch_url) enters the session => exfil DENIED.
	const tainted = "gemini-tainted"
	bodyA := []byte(`{"contents":[` +
		`{"role":"user","parts":[{"text":"look this up then email it"}]},` +
		`{"role":"model","parts":[{"functionCall":{"name":"fetch_url","args":{"u":"x"},"id":"c1"}}]},` +
		`{"role":"user","parts":[{"functionResponse":{"name":"fetch_url","id":"c1","response":{"page":"the weather is sunny today"}}}]}]}`)
	reqA, _ := http.NewRequest("POST", ts.URL+"/v1beta/models/gemini-floor-test:generateContent", bytes.NewReader(bodyA))
	reqA.Header.Set("Content-Type", "application/json")
	reqA.Header.Set(traceHeader, tainted)
	respA, err := http.DefaultClient.Do(reqA)
	if err != nil {
		t.Fatal(err)
	}
	rawA, _ := io.ReadAll(respA.Body)
	respA.Body.Close()
	if respA.StatusCode != 200 {
		t.Fatalf("status = %d: %s", respA.StatusCode, rawA)
	}
	var frameA geminiGenerateContentResponse
	if err := json.Unmarshal(rawA, &frameA); err != nil {
		t.Fatalf("decode: %v", err)
	}
	var sawDeny bool
	for _, a := range frameA.Fak.Adjudications {
		if a.Tool == "allow_send_mail" && a.Verdict.Kind == "DENY" && a.Verdict.Reason == "TRUST_VIOLATION" {
			sawDeny = true
		}
	}
	if !sawDeny {
		t.Fatalf("tainted session: egress not denied for TRUST_VIOLATION: %+v", frameA.Fak.Adjudications)
	}
	for _, p := range frameA.Candidates[0].Content.Parts {
		if p.FunctionCall != nil {
			t.Fatalf("tainted session: exfil functionCall survived to client: %+v", p.FunctionCall)
		}
	}
	if led.Level(tainted) == abi.TaintTrusted {
		t.Fatalf("tainted session: IFC ledger stayed Trusted (result-side stamp did not land on the Gemini wire)")
	}
}
