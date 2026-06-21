package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// threeCallPlanner proposes one allow, one deny, one transform — the same shape the
// chat-completions proxy test uses, so the Anthropic wire is asserted to run the
// IDENTICAL kernel filter.
func threeCallPlanner() stubPlanner {
	return stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "toolu_a", Type: "function", Function: agent.Func{Name: "allow_a", Arguments: `{"x":1}`}},
			{ID: "toolu_b", Type: "function", Function: agent.Func{Name: "deny_b", Arguments: `{}`}},
			{ID: "toolu_c", Type: "function", Function: agent.Func{Name: "transform_c", Arguments: `{"secret":"y"}`}},
		}},
		FinishReason: "tool_calls",
		Usage:        agent.Usage{PromptTokens: 12, CompletionTokens: 7, TotalTokens: 19},
	}}
}

func TestAnthropicMessagesFiltersAndRepairs(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = threeCallPlanner()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := json.RawMessage(`{"model":"claude-opus-4-8","messages":[{"role":"user","content":"go"}],
		"tools":[{"name":"allow_a","input_schema":{"type":"object"}}]}`)
	var resp anthropicMessageResponse
	if code := postJSON(t, ts.URL+"/v1/messages", body, &resp); code != 200 {
		t.Fatalf("status = %d", code)
	}
	if resp.Type != "message" || resp.Role != "assistant" {
		t.Errorf("envelope wrong: %+v", resp)
	}
	if resp.Model != "claude-opus-4-8" {
		t.Errorf("requested model must be echoed, got %q", resp.Model)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", resp.StopReason)
	}
	// allow_a verbatim + transform_c repaired; deny_b must be gone.
	var ids []string
	var repaired string
	for _, b := range resp.Content {
		if b.Type != "tool_use" {
			continue
		}
		ids = append(ids, b.ID)
		if b.Name == "transform_c" {
			repaired = string(b.Input)
		}
		if b.Name == "deny_b" {
			t.Error("denied tool call must NOT reach the caller")
		}
	}
	if len(ids) != 2 {
		t.Fatalf("kept %d tool_use blocks, want 2: %+v", len(ids), resp.Content)
	}
	if ids[0] != "toolu_a" {
		t.Errorf("tool_use id must survive the round trip, got %v", ids)
	}
	if repaired != `{"redacted":true}` {
		t.Errorf("transform not applied to input: %q", repaired)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 7 {
		t.Errorf("usage not forwarded: %+v", resp.Usage)
	}

	// The kernel dropped deny_b and repaired transform_c. On the Anthropic wire —
	// the one Claude Code reads — that must be LEGIBLE, not silent: a leading
	// in-band text note names the refusal + repair so the agent adapts instead of
	// re-proposing the dropped call or proceeding unaware its args were rewritten.
	if len(resp.Content) == 0 || resp.Content[0].Type != "text" {
		t.Fatalf("first block must be the [fak] decision note, got %+v", resp.Content)
	}
	note := resp.Content[0].Text
	if !strings.Contains(note, "[fak]") || !strings.Contains(note, "deny_b") {
		t.Errorf("in-band note must name the refused call deny_b, got %q", note)
	}
	if !strings.Contains(note, "transform_c") {
		t.Errorf("in-band note must name the repaired call transform_c, got %q", note)
	}
	// The structured verdicts also ride back as the `fak` extension, the Anthropic
	// twin of the OpenAI wire's resp.Fak — one ToolAdjudication per proposed call.
	if resp.Fak == nil || len(resp.Fak.Adjudications) != 3 {
		t.Fatalf("fak extension must carry all 3 adjudications, got %+v", resp.Fak)
	}
	var sawDeny, sawTransform bool
	for _, a := range resp.Fak.Adjudications {
		if a.Tool == "deny_b" && !a.Admitted {
			sawDeny = true
		}
		if a.Tool == "transform_c" && a.Admitted && a.Verdict.Kind == "TRANSFORM" {
			sawTransform = true
		}
	}
	if !sawDeny || !sawTransform {
		t.Errorf("fak extension must record the deny + transform verdicts: %+v", resp.Fak.Adjudications)
	}
}

func TestAnthropicMessagesAllDeniedNote(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "x", Function: agent.Func{Name: "deny_a"}}}},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp anthropicMessageResponse
	postJSON(t, ts.URL+"/v1/messages", json.RawMessage(`{"messages":[{"role":"user","content":"x"}]}`), &resp)
	if resp.StopReason != "end_turn" {
		t.Errorf("all-denied must end_turn, got %q", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" || !strings.Contains(resp.Content[0].Text, "refused") {
		t.Errorf("expected an in-band deny note, got %+v", resp.Content)
	}
}

// A clean all-allow turn must NOT inject an in-band [fak] note (that channel is
// reserved for drops/repairs the agent must react to), while the structured `fak`
// extension still records the allow for fak-aware tooling — the same asymmetry the
// OpenAI wire has. This is what keeps the legibility signal HIGH-SIGNAL.
func TestAnthropicMessagesCleanAllowNoNote(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
			{ID: "toolu_a", Type: "function", Function: agent.Func{Name: "allow_a", Arguments: `{"x":1}`}},
		}},
		FinishReason: "tool_calls",
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp anthropicMessageResponse
	postJSON(t, ts.URL+"/v1/messages", json.RawMessage(`{"messages":[{"role":"user","content":"go"}]}`), &resp)
	for _, b := range resp.Content {
		if b.Type == "text" && strings.Contains(b.Text, "[fak]") {
			t.Errorf("clean all-allow turn must not inject a [fak] note: %q", b.Text)
		}
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("surviving allow must keep stop_reason tool_use, got %q", resp.StopReason)
	}
	if resp.Fak == nil || len(resp.Fak.Adjudications) != 1 || !resp.Fak.Adjudications[0].Admitted {
		t.Errorf("fak extension must still record the allow verdict: %+v", resp.Fak)
	}
}

func TestAnthropicMessagesSSE(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = threeCallPlanner()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/messages", "application/json",
		bytes.NewReader([]byte(`{"model":"claude-opus-4-8","stream":true,"messages":[{"role":"user","content":"go"}]}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}
	raw, _ := io.ReadAll(r.Body)
	body := string(raw)
	// The full, ordered event sequence must be present.
	for _, want := range []string{
		"event: message_start",
		"event: content_block_start",
		"event: content_block_delta",
		`"type":"input_json_delta"`,
		`redacted`, // the repaired args streamed (JSON-escaped) as the tool_use input
		`[fak]`,    // the decision note must survive STREAMING (Claude Code's path)
		`deny_b`,   // ...naming the dropped call so the agent does not re-propose it
		"event: content_block_stop",
		"event: message_delta",
		`"stop_reason":"tool_use"`,
		"event: message_stop",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("SSE stream missing %q\n---\n%s", want, body)
		}
	}
	// Ordering: message_start precedes message_stop.
	if strings.Index(body, "message_start") > strings.Index(body, "message_stop") {
		t.Error("message_start must precede message_stop")
	}
}

type delayedPlanner struct {
	delay time.Duration
	comp  *agent.Completion
}

func (p delayedPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	select {
	case <-time.After(p.delay):
		return p.comp, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (delayedPlanner) Model() string { return "delayed" }

func TestAnthropicMessagesSSEPingsDuringSlowPlanner(t *testing.T) {
	old := anthropicStreamPingInterval
	anthropicStreamPingInterval = 5 * time.Millisecond
	defer func() { anthropicStreamPingInterval = old }()

	srv := newTestServer(t)
	srv.planner = delayedPlanner{
		delay: 25 * time.Millisecond,
		comp: &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "pong"},
			FinishReason: "stop",
			Usage:        agent.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11},
		},
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/messages", "application/json",
		bytes.NewReader([]byte(`{"model":"m","stream":true,"messages":[{"role":"user","content":"go"}]}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	raw, _ := io.ReadAll(r.Body)
	body := string(raw)
	if !strings.Contains(body, "event: ping") {
		t.Fatalf("slow stream did not emit ping:\n%s", body)
	}
	if strings.Index(body, "event: ping") > strings.Index(body, "event: message_stop") {
		t.Fatalf("ping must arrive before message_stop:\n%s", body)
	}
}

func TestAnthropicCountTokens(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp struct {
		InputTokens int `json:"input_tokens"`
	}
	code := postJSON(t, ts.URL+"/v1/messages/count_tokens",
		json.RawMessage(`{"model":"m","system":"you are helpful","messages":[{"role":"user","content":"count these tokens please"}]}`), &resp)
	if code != 200 {
		t.Fatalf("status = %d", code)
	}
	if resp.InputTokens <= 0 {
		t.Errorf("input_tokens = %d, want > 0", resp.InputTokens)
	}
}
