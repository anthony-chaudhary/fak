package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"
)

func TestToolResultBudgetBoundsEachResult(t *testing.T) {
	multibyte := strings.Repeat("é", 6000) // 2 bytes per rune: a cut can land mid-rune
	cases := []struct {
		name    string
		tool    string
		content string
		limit   int
		wantLen int // 0 = must not change
		kind    string
	}{
		{name: "under bound is untouched", tool: "Read", content: "short", limit: 100, kind: ""},
		{name: "over bound is truncated", tool: "Bash", content: strings.Repeat("x", 20000), limit: 8192, wantLen: 8192, kind: "bytes"},
		{name: "multibyte cut stays valid", tool: "Read", content: multibyte, limit: 8192, wantLen: -1, kind: "bytes"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, kind := boundLocalToolResult(tc.tool, tc.content, tc.limit)
			if kind != tc.kind {
				t.Fatalf("kind=%q want %q", kind, tc.kind)
			}
			if tc.wantLen == 0 {
				if got != tc.content {
					t.Fatalf("under-bound content changed")
				}
				return
			}
			if len(got) > tc.limit {
				t.Fatalf("bounded len=%d exceeds limit %d", len(got), tc.limit)
			}
			if !utf8.ValidString(got) {
				t.Fatalf("truncation split a rune")
			}
			if !strings.Contains(got, "bytes omitted") || !strings.Contains(got, "Narrow the request") {
				t.Fatalf("truncation notice missing: %q", got[max(0, len(got)-200):])
			}
		})
	}
}

func TestToolResultBudgetGrepCapsMatches(t *testing.T) {
	type grepRow struct {
		File string `json:"file"`
		Line int    `json:"line"`
		Text string `json:"text"`
	}
	rows := make([]grepRow, 0, 120)
	for i := range 120 {
		rows = append(rows, grepRow{File: fmt.Sprintf("f%03d.go", i), Line: i + 1, Text: "match"})
	}
	raw, err := json.Marshal(map[string]any{"matches": rows, "truncated": false})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name        string
		tool        string
		content     string
		wantRows    int
		wantDropped bool
	}{
		{name: "grep caps rows", tool: "Grep", content: string(raw), wantRows: localGrepMatchCap, wantDropped: true},
		{name: "non-grep tool keeps matches field", tool: "Bash", content: string(raw), wantRows: 120},
		{name: "grep under cap unchanged", tool: "Grep", content: `{"matches":[{"file":"a.go","line":1,"text":"x"}]}`, wantRows: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := boundLocalToolResult(tc.tool, tc.content, 1<<20)
			var doc struct {
				Matches        []grepRow `json:"matches"`
				MatchesDropped int       `json:"matches_dropped"`
			}
			if err := json.Unmarshal([]byte(got), &doc); err != nil {
				t.Fatalf("result not JSON: %v", err)
			}
			if len(doc.Matches) != tc.wantRows {
				t.Fatalf("rows=%d want %d", len(doc.Matches), tc.wantRows)
			}
			if tc.wantDropped != (doc.MatchesDropped > 0) {
				t.Fatalf("matches_dropped=%d, wantDropped=%v", doc.MatchesDropped, tc.wantDropped)
			}
			if tc.wantDropped && doc.MatchesDropped != 120-localGrepMatchCap {
				t.Fatalf("matches_dropped=%d want %d", doc.MatchesDropped, 120-localGrepMatchCap)
			}
		})
	}
}

func TestContextFitStubsOldestUntilPromptFits(t *testing.T) {
	bigResult := func(id string) Message {
		return Message{Role: RoleTool, ToolCallID: id, Name: "Grep", Content: strings.Repeat("r", 6000)}
	}
	transcript := func() []Message {
		var msgs []Message
		for i := range 6 {
			id := fmt.Sprintf("c%d", i)
			msgs = append(msgs,
				Message{Role: RoleUser, Content: "look"},
				Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: id, Type: "function", Function: Func{Name: "Grep", Arguments: `{"pattern":"x"}`}}}},
				bigResult(id),
			)
		}
		return msgs
	}
	// Six 6000-byte results are about 10300 estimated tokens. A 5000-token window
	// (4744 usable) must stub the four oldest and leave the newest two intact, which
	// fit in about 3400 tokens.
	out, rep := fitLocalTranscript(transcript(), localContextFit{perResultBytes: localToolResultBudgetBytes, contextTokens: 5000, reserveTokens: 256, keepRecent: 2})
	if rep.Stubbed == 0 {
		t.Fatalf("expected stubs, report=%+v", rep)
	}
	var stubbedIdx []int
	for i, m := range out {
		if m.Role == RoleTool && isLocalStub(m.Content) {
			stubbedIdx = append(stubbedIdx, i)
		}
	}
	if len(stubbedIdx) != rep.Stubbed {
		t.Fatalf("stub count %d != report %d", len(stubbedIdx), rep.Stubbed)
	}
	if stubbedIdx[0] != 2 {
		t.Fatalf("oldest result must be stubbed first, got index %d", stubbedIdx[0])
	}
	for i := len(out) - 1; i >= 0 && i >= len(out)-2; i-- {
		if out[i].Role == RoleTool && isLocalStub(out[i].Content) {
			t.Fatalf("newest tool result %d was stubbed", i)
		}
	}
	// Stubbing stops as soon as the estimate fits, so the prompt is within budget
	// (the newest results are never stubbed, so a fit is always reachable here).
	if rep.EstTokens > 5000-256 {
		t.Fatalf("prompt still over budget after fit: est=%d", rep.EstTokens)
	}
	// Message count and roles are preserved: the fit never drops a turn.
	if len(out) != len(transcript()) {
		t.Fatalf("fit changed message count: %d", len(out))
	}
}

func TestContextFitUnknownWindowAppliesOnlyPerResultBound(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Function: Func{Name: "Grep", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "a", Name: "Grep", Content: strings.Repeat("g", 20000)},
	}
	out, rep := fitLocalTranscript(msgs, localContextFit{perResultBytes: localToolResultBudgetBytes})
	if rep.Stubbed != 0 {
		t.Fatalf("unknown window must not stub, got %d", rep.Stubbed)
	}
	if len(out[1].Content) > localToolResultBudgetBytes {
		t.Fatalf("per-result bound not applied: %d", len(out[1].Content))
	}
}

func TestContextFitCompactedRetryTightensBound(t *testing.T) {
	ctx := withLocalCompact(agentHardenCtx(t))
	p := NewHTTPPlanner("http://127.0.0.1:1", "m", "")
	p.MaxTokens = 512
	msgs := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Function: Func{Name: "Bash", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "a", Name: "Bash", Content: strings.Repeat("b", 20000)},
	}
	out := p.fitLocalRequest(ctx, msgs, nil)
	if len(out[1].Content) > localCompactedToolResultBytes {
		t.Fatalf("compacted bound not applied: %d", len(out[1].Content))
	}
}

func TestContextFitReservesToolDefinitionTokens(t *testing.T) {
	t.Setenv(EnvAgentContextTokens, "5000")
	msgs := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Function: Func{Name: "Grep", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "a", Name: "Grep", Content: strings.Repeat("r", 4000)},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "b", Function: Func{Name: "Grep", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "b", Name: "Grep", Content: strings.Repeat("r", 4000)},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c", Function: Func{Name: "Grep", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "c", Name: "Grep", Content: strings.Repeat("r", 4000)},
	}
	stubbed := func(out []Message) int {
		n := 0
		for _, m := range out {
			if m.Role == RoleTool && isLocalStub(m.Content) {
				n++
			}
		}
		return n
	}
	p := &HTTPPlanner{}
	ctx := agentHardenCtx(t)
	if got := stubbed(p.fitLocalRequest(ctx, msgs, nil)); got != 0 {
		t.Fatalf("transcript fits without tools, got %d stubs", got)
	}
	// About 6000 bytes of schema is roughly 1700 tokens, enough to push the
	// three-result transcript over the 5000-token window.
	big := ToolDef{Type: "function", Function: ToolDefFunction{Name: "Grep", Description: strings.Repeat("d", 6000)}}
	out := p.fitLocalRequest(ctx, msgs, []ToolDef{big})
	if got := stubbed(out); got == 0 {
		t.Fatal("tool definitions over budget must stub the oldest result")
	}
	if !isLocalStub(out[1].Content) {
		t.Fatalf("oldest result must be the stub, got %q", out[1].Content[:min(40, len(out[1].Content))])
	}
}

// maxRunOf returns the longest unbroken run of byte c in s.
func maxRunOf(s string, c byte) int {
	best, run := 0, 0
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			run++
			best = max(best, run)
		} else {
			run = 0
		}
	}
	return best
}

func TestLocalRequestBodyCarriesBoundedResult(t *testing.T) {
	agentHardenFastRetry(t)
	t.Setenv(EnvAgentContextTokens, "")
	var mu sync.Mutex
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(raw))
		n := len(bodies)
		mu.Unlock()
		if n == 1 {
			agentHardenStatus(w, http.StatusBadGateway, `{"error":{"message":"upstream model error"}}`)
			return
		}
		agentHardenStatus(w, http.StatusOK, agentHardenOK)
	}))
	t.Cleanup(srv.Close)
	p := agentHardenPlanner(srv, 3)
	msgs := []Message{
		{Role: RoleUser, Content: "find it"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Type: "function", Function: Func{Name: "Grep", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "a", Name: "Grep", Content: strings.Repeat("m", 60000)},
	}
	if _, err := p.Complete(agentHardenCtx(t), msgs, nil); err != nil {
		t.Fatalf("compacted retry should succeed, got %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("posts=%d want 2", len(bodies))
	}
	if got := maxRunOf(bodies[0], 'm'); got == 0 || got > localToolResultBudgetBytes {
		t.Fatalf("first post result run=%d want (0, %d]", got, localToolResultBudgetBytes)
	}
	if got := maxRunOf(bodies[1], 'm'); got == 0 || got > localCompactedToolResultBytes {
		t.Fatalf("compacted post result run=%d want (0, %d]", got, localCompactedToolResultBytes)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestHostedRequestBodyIsNotFit(t *testing.T) {
	t.Setenv(EnvAgentContextTokens, "")
	var body string
	p := NewHTTPPlanner("https://api.example.com/v1", "m", "")
	p.Client = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		raw, _ := io.ReadAll(r.Body)
		body = string(raw)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(agentHardenOK)),
			Request:    r,
		}, nil
	})}
	msgs := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Type: "function", Function: Func{Name: "Grep", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "a", Name: "Grep", Content: strings.Repeat("m", 60000)},
	}
	if _, err := p.Complete(agentHardenCtx(t), msgs, nil); err != nil {
		t.Fatalf("hosted complete: %v", err)
	}
	// Hosted requests keep the pre-send quarantine path; the local window fit must
	// not rewrite them, so no local truncation or elision marker may appear.
	if strings.Contains(body, "truncated for the local context window") || strings.Contains(body, "elided to fit") {
		t.Fatalf("local fit applied to a hosted request: %.300s", body)
	}
}

func TestLocalEndpointGatesFitAndRetry(t *testing.T) {
	overflow := &ContextOverflowError{Status: http.StatusBadGateway, Cause: errors.New("x")}
	cases := []struct {
		name      string
		baseURL   string
		ctxCompat bool
		err       error
		wantRetry bool
	}{
		{name: "local overflow retries once", baseURL: "http://127.0.0.1:18102/v1", err: overflow, wantRetry: true},
		{name: "local compacted attempt never retries", baseURL: "http://127.0.0.1:18102/v1", ctxCompat: true, err: overflow},
		{name: "local non-overflow error does not retry", baseURL: "http://127.0.0.1:18102/v1", err: errors.New("boom")},
		{name: "hosted overflow never retries", baseURL: "https://api.openai.com/v1", err: overflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := &HTTPPlanner{BaseURL: tc.baseURL}
			ctx := agentHardenCtx(t)
			if tc.ctxCompat {
				ctx = withLocalCompact(ctx)
			}
			if got := p.localContextRetry(ctx, tc.err); got != tc.wantRetry {
				t.Fatalf("localContextRetry=%v want %v", got, tc.wantRetry)
			}
		})
	}
}

func TestLocalEndpointRefusalRecognizesServeBody(t *testing.T) {
	body := []byte(`{"error":{"code":"upstream_error","message":"upstream model error"}}`)
	local := &HTTPPlanner{BaseURL: "http://aegis2.local:18102/v1"}
	hosted := &HTTPPlanner{BaseURL: "https://api.openai.com/v1"}
	if !localContextRefusal(local, http.StatusBadGateway, body) {
		t.Fatal("local 502 upstream model error must read as a context refusal")
	}
	if localContextRefusal(hosted, http.StatusBadGateway, body) {
		t.Fatal("hosted 502 must keep the generic classification")
	}
	if localContextRefusal(local, http.StatusInternalServerError, body) {
		t.Fatal("only 502 is the context refusal shape")
	}
}

func TestLocalEndpointCompactRetryOnContextRefusal(t *testing.T) {
	agentHardenFastRetry(t)
	srv, posts := agentHardenServer(t, func(n int32, w http.ResponseWriter) {
		if n == 1 {
			agentHardenStatus(w, http.StatusBadGateway, `{"error":{"message":"upstream model error"}}`)
			return
		}
		agentHardenStatus(w, http.StatusOK, agentHardenOK)
	})
	p := agentHardenPlanner(srv, 3)
	msgs := []Message{
		{Role: RoleUser, Content: "find it"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Type: "function", Function: Func{Name: "Grep", Arguments: "{}"}}}},
		{Role: RoleTool, ToolCallID: "a", Name: "Grep", Content: strings.Repeat("m", 60000)},
	}
	comp, err := p.Complete(agentHardenCtx(t), msgs, nil)
	if err != nil {
		t.Fatalf("compacted retry should succeed, got %v", err)
	}
	if comp == nil {
		t.Fatal("nil completion")
	}
	if got := posts.Load(); got != 2 {
		t.Fatalf("posts=%d want 2 (one refusal, one compacted retry)", got)
	}
}

func TestIdenticalRetryLocalContextRefusalIsNotCounted(t *testing.T) {
	agentHardenFastRetry(t)
	srv, posts := agentHardenServer(t, func(n int32, w http.ResponseWriter) {
		agentHardenStatus(w, http.StatusBadGateway, `{"error":{"message":"upstream model error"}}`)
	})
	p := agentHardenPlanner(srv, 1)
	_, err := p.Complete(agentHardenCtx(t), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if !errors.Is(err, ErrContextOverflow) {
		t.Fatalf("err=%v want ErrContextOverflow", err)
	}
	if errors.Is(err, ErrIdenticalRetry) {
		t.Fatalf("context refusal was counted as an identical retry: %v", err)
	}
	if got := posts.Load(); got != 2 {
		t.Fatalf("posts=%d want 2: one refusal and one compacted retry, then stop", got)
	}
}
