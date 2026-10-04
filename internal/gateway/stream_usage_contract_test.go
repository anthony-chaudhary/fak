package gateway

// stream_usage_contract_test.go — the OpenAI chat `stream_options.include_usage`
// wire contract on the native `/v1/chat/completions` SSE surface.
//
// The gateway has TWO native chat stream emitters: the LIVE one (streamChatLive,
// taken by any agent.StreamingPlanner) and the BUFFERED fallback
// (writeChatCompletionStream via chatStreamWriter, taken by a Complete-only
// planner). The OpenAI chat reference requires both to answer identically:
//
//   - omitted or false: no `usage` field on any chunk and no usage-only chunk.
//   - true: every non-usage chunk carries `usage:null`, and the finish-bearing
//     choice is followed by EXACTLY ONE usage-only chunk (`choices: []`) and then
//     `[DONE]`.
//
// A test that only exercises one emitter (or that decodes into the typed struct,
// which cannot distinguish an omitted field from an explicit null) cannot witness
// the contract. This test drives both emitters over real HTTP and inspects the raw
// JSON key presence of every frame.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// usageStreamPlanner advertises the live streaming capability (agent.StreamingPlanner)
// and streams its content through the sink, so the request takes streamChatLive.
type usageStreamPlanner struct{ content string }

func (p *usageStreamPlanner) Model() string { return "test-model" }

func (p *usageStreamPlanner) StreamingSupported() bool { return true }

func (p *usageStreamPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.completion(), nil
}

func (p *usageStreamPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	for _, seg := range segmentContent(p.content) {
		if err := sink(seg); err != nil {
			return nil, err
		}
	}
	return p.completion(), nil
}

func (p *usageStreamPlanner) completion() *agent.Completion {
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: p.content},
		FinishReason: "stop",
		Model:        "test-model",
		Usage:        agent.Usage{PromptTokens: 6, CompletionTokens: 4, TotalTokens: 10},
	}
}

// usageBufferedPlanner implements the bare agent.Planner surface only — NOT
// agent.StreamingPlanner — so the gateway is forced onto the buffered fallback.
type usageBufferedPlanner struct{ content string }

func (p *usageBufferedPlanner) Model() string { return "test-model" }

func (p *usageBufferedPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return (&usageStreamPlanner{content: p.content}).completion(), nil
}

var (
	_ agent.Planner          = (*usageStreamPlanner)(nil)
	_ agent.StreamingPlanner = (*usageStreamPlanner)(nil)
	_ agent.Planner          = (*usageBufferedPlanner)(nil)
	_                        = (*usageBufferedPlanner)(nil)
)

// TestChatStreamIncludeUsageContract is the ticket witness. It drives both native
// emitters and all three option states at the HTTP SSE boundary.
func TestChatStreamIncludeUsageContract(t *testing.T) {
	const content = "usage contract witness"

	planners := []struct {
		name    string
		planner agent.Planner
	}{
		{"live", &usageStreamPlanner{content: content}},
		{"buffered", &usageBufferedPlanner{content: content}},
	}

	// The three request forms. `set` distinguishes an omitted stream_options object
	// from an explicit `include_usage:false`/`true`.
	options := []struct {
		name          string
		set           bool
		includeUsage  bool
		wantUsageNull bool
	}{
		{"omitted", false, false, false},
		{"false", true, false, false},
		{"true", true, true, true},
	}

	for _, pl := range planners {
		for _, opt := range options {
			t.Run(pl.name+"/"+opt.name, func(t *testing.T) {
				raw, stat := driveChatStream(t, pl.planner, opt.set, opt.includeUsage)
				if stat != http.StatusOK {
					t.Fatalf("status = %d, want 200; body: %s", stat, raw)
				}
				assertChatStreamUsageContract(t, raw, content, opt.wantUsageNull)
			})
		}
	}
}

// driveChatStream issues one stream:true native chat request and returns the raw SSE
// body and HTTP status.
func driveChatStream(t *testing.T, planner agent.Planner, set, includeUsage bool) (string, int) {
	t.Helper()
	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	reqBody := map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": "go"}},
		"stream":   true,
	}
	if set {
		reqBody["stream_options"] = map[string]any{"include_usage": includeUsage}
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if got := resp.Header.Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", got)
	}
	return string(raw), resp.StatusCode
}

// assertChatStreamUsageContract parses every SSE frame and asserts the exact usage
// tri-state and ordering required by the OpenAI chat reference.
func assertChatStreamUsageContract(t *testing.T, raw, wantContent string, optedIn bool) {
	t.Helper()

	var (
		frames     []map[string]json.RawMessage
		typed      []ChatStreamResponse
		sawDone    bool
		rebuilt    strings.Builder
		finishAt   = -1
		usageOnly  = -1
		usageNulls int
	)
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			t.Fatalf("non-SSE line on the wire: %q", line)
		}
		if data == "[DONE]" {
			sawDone = true
			break
		}
		var frame map[string]json.RawMessage
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			t.Fatalf("decode SSE data %q: %v", data, err)
		}
		frames = append(frames, frame)

		var chunk ChatStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("decode SSE chunk %q: %v", data, err)
		}
		idx := len(frames) - 1
		typed = append(typed, chunk)

		usageKey, hasUsage := frame["usage"]
		if optedIn {
			if !hasUsage {
				t.Fatalf("opted-in chunk %d omitted the usage key; want it present (%s)", idx, data)
			}
			if string(usageKey) == "null" {
				usageNulls++
			}
		} else if hasUsage {
			t.Fatalf("non-opted chunk %d carried a usage key; want it omitted (%s)", idx, data)
		}

		if len(chunk.Choices) == 0 {
			// The usage-only terminal frame: empty choices, non-null usage object.
			if usageOnly != -1 {
				t.Fatalf("more than one empty-choices frame (first at %d, again at %d)", usageOnly, idx)
			}
			usageOnly = idx
			if string(usageKey) == "null" || !hasUsage {
				t.Fatalf("usage-only frame %d must carry the usage object, got %s", idx, data)
			}
			if chunk.Usage == nil || chunk.Usage.TotalTokens != 10 {
				t.Fatalf("usage-only frame %d usage = %+v, want total_tokens 10", idx, chunk.Usage)
			}
			continue
		}

		rebuilt.WriteString(chunk.Choices[0].Delta.Content)
		if chunk.Choices[0].FinishReason != nil {
			if finishAt != -1 {
				t.Fatalf("more than one finish-bearing chunk (first at %d, again at %d)", finishAt, idx)
			}
			finishAt = idx
			if *chunk.Choices[0].FinishReason != "stop" {
				t.Fatalf("finish_reason = %q, want stop", *chunk.Choices[0].FinishReason)
			}
		}
	}

	if !sawDone {
		t.Fatalf("stream never terminated with [DONE]: %s", raw)
	}
	if finishAt == -1 {
		t.Fatalf("no finish-bearing chunk in the stream: %s", raw)
	}
	if got := rebuilt.String(); got != wantContent {
		t.Fatalf("reassembled content = %q, want %q", got, wantContent)
	}

	if !optedIn {
		if usageOnly != -1 {
			t.Fatalf("non-opted stream emitted a usage-only frame at %d: %s", usageOnly, raw)
		}
		if usageNulls != 0 {
			t.Fatalf("non-opted stream nulled usage %d time(s): %s", usageNulls, raw)
		}
		return
	}

	if usageOnly == -1 {
		t.Fatalf("opted-in stream emitted no usage-only frame: %s", raw)
	}
	if usageOnly != finishAt+1 {
		t.Fatalf("usage-only frame at %d must immediately follow the finish frame at %d: %s", usageOnly, finishAt, raw)
	}
	// Every frame other than the usage-only one must carry `usage:null`.
	if want := len(frames) - 1; usageNulls != want {
		t.Fatalf("opted-in stream nulled usage on %d chunk(s), want every non-usage chunk (%d): %s", usageNulls, want, raw)
	}
}
