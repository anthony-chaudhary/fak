package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// reasoningStreamPlanner streams its content through the sink and returns a
// completion that also carries reasoning, without using any live reasoning seam.
type reasoningStreamPlanner struct{ content, reasoning string }

func (p *reasoningStreamPlanner) Model() string { return "test-model" }

func (p *reasoningStreamPlanner) StreamingSupported() bool { return true }

func (p *reasoningStreamPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return p.completion(), nil
}

func (p *reasoningStreamPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if err := sink(p.content); err != nil {
		return nil, err
	}
	return p.completion(), nil
}

func (p *reasoningStreamPlanner) completion() *agent.Completion {
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: p.content, ReasoningContent: p.reasoning},
		FinishReason: "stop",
		Model:        "test-model",
	}
}

// chatStreamFieldTally folds choices[0].delta across every SSE frame.
type chatStreamFieldTally struct {
	reasoning, content             strings.Builder
	reasoningFrames, contentFrames int
	firstReasoning, firstContent   int
}

func tallyChatStreamFields(t *testing.T, raw string) *chatStreamFieldTally {
	t.Helper()
	tally := &chatStreamFieldTally{firstReasoning: -1, firstContent: -1}
	frame := 0
	for _, line := range strings.Split(raw, "\n") {
		data, ok := strings.CutPrefix(strings.TrimSpace(line), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("decode SSE data %q: %v", data, err)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		frame++
		delta := chunk.Choices[0].Delta
		if delta.ReasoningContent != "" {
			tally.reasoningFrames++
			tally.reasoning.WriteString(delta.ReasoningContent)
			if tally.firstReasoning < 0 {
				tally.firstReasoning = frame
			}
		}
		if delta.Content != "" {
			tally.contentFrames++
			tally.content.WriteString(delta.Content)
			if tally.firstContent < 0 {
				tally.firstContent = frame
			}
		}
	}
	return tally
}

// fak-test:runtime fast est=100ms lane=default
func TestStreamChatLiveEmitsReasoningContentDelta(t *testing.T) {
	t.Run("upstream_live", func(t *testing.T) {
		abi.ResetForTest()
		abi.RegisterRegionBackend(inlineBackend{})
		abi.RegisterEngine("test", echoEngine{})
		abi.RegisterAdjudicator(0, toolAdj{})
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"two plus\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" two\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"content\":\"4\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
				"data: [DONE]\n\n")
		}))
		t.Cleanup(upstream.Close)
		srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL + "/compat", Provider: "openai-compatible"})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(srv.Close)
		ts := httptest.NewServer(srv.Handler())
		t.Cleanup(ts.Close)
		body, _ := json.Marshal(map[string]any{
			"model":    "x:model",
			"messages": []map[string]string{{"role": "user", "content": "two plus two"}},
			"stream":   true,
		})
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.StatusCode, raw)
		}
		tally := tallyChatStreamFields(t, string(raw))
		if got := tally.reasoning.String(); got != "two plus two" {
			t.Fatalf("reasoning_content = %q, want %q: %s", got, "two plus two", raw)
		}
		if got := tally.content.String(); got != "4" {
			t.Fatalf("content = %q, want 4: %s", got, raw)
		}
		if tally.reasoningFrames != 2 || tally.firstReasoning > tally.firstContent {
			t.Fatalf("reasoning frames = %d first=%d content first=%d, want 2 live frames before content: %s", tally.reasoningFrames, tally.firstReasoning, tally.firstContent, raw)
		}
	})
	t.Run("upstream_live_before_done", func(t *testing.T) {
		// The upstream holds its stream open after two reasoning deltas until the
		// client has read one. A gateway that holds reasoning until the completion
		// (33ccf41871) only delivers it after the hold times out.
		abi.ResetForTest()
		abi.RegisterRegionBackend(inlineBackend{})
		abi.RegisterEngine("test", echoEngine{})
		abi.RegisterAdjudicator(0, toolAdj{})
		seen := make(chan struct{})
		release := sync.OnceFunc(func() { close(seen) })
		t.Cleanup(release)
		var resumed atomic.Bool
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"two plus\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" two\"},\"finish_reason\":null}]}\n\n")
			w.(http.Flusher).Flush()
			select {
			case <-seen:
			case <-time.After(3 * time.Second):
			}
			resumed.Store(true)
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"4\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
				"data: [DONE]\n\n")
		}))
		t.Cleanup(upstream.Close)
		srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL + "/compat", Provider: "openai-compatible"})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(srv.Close)
		ts := httptest.NewServer(srv.Handler())
		t.Cleanup(ts.Close)
		body, _ := json.Marshal(map[string]any{
			"model":    "x:model",
			"messages": []map[string]string{{"role": "user", "content": "two plus two"}},
			"stream":   true,
		})
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var raw strings.Builder
		liveBeforeDone := false
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			line := sc.Text()
			raw.WriteString(line + "\n")
			if !liveBeforeDone && strings.Contains(line, "reasoning_content") {
				liveBeforeDone = !resumed.Load()
				release()
			}
		}
		if !liveBeforeDone {
			t.Fatalf("reasoning_content reached the client only after the upstream resumed: %s", raw.String())
		}
		tally := tallyChatStreamFields(t, raw.String())
		if got := tally.reasoning.String(); got != "two plus two" || tally.content.String() != "4" {
			t.Fatalf("reasoning=%q content=%q, want each exactly once: %s", got, tally.content.String(), raw.String())
		}
	})
	t.Run("post_hoc", func(t *testing.T) {
		raw, status := driveChatStream(t, &reasoningStreamPlanner{content: "4", reasoning: "two plus two"}, false, false)
		if status != http.StatusOK {
			t.Fatalf("status = %d: %s", status, raw)
		}
		tally := tallyChatStreamFields(t, raw)
		// A planner without a live reasoning split gets the finalized reasoning held
		// until adjudication, segmented like content (33ccf41871), exactly once.
		if got := tally.reasoning.String(); got != "two plus two" || tally.reasoningFrames == 0 {
			t.Fatalf("reasoning_content = %q over %d frames, want the post-hoc reasoning exactly once: %s", got, tally.reasoningFrames, raw)
		}
		if got := tally.content.String(); got != "4" {
			t.Fatalf("content = %q, want 4: %s", got, raw)
		}
	})
	t.Run("no_reasoning", func(t *testing.T) {
		raw, status := driveChatStream(t, &reasoningStreamPlanner{content: "4"}, false, false)
		if status != http.StatusOK || strings.Contains(raw, "reasoning_content") {
			t.Fatalf("status = %d; a turn without reasoning must not carry the field: %s", status, raw)
		}
	})
}

// fak-test:runtime fast est=5ms lane=default
func TestWriteChatCompletionStreamCarriesReasoningContent(t *testing.T) {
	rec := httptest.NewRecorder()
	writeChatCompletionStream(newChatStreamWriter(rec, "test-model", false), ChatResponse{
		Choices: []ChatChoice{{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "4", ReasoningContent: "two plus two"},
			FinishReason: "stop",
		}},
	})
	raw := rec.Body.String()
	tally := tallyChatStreamFields(t, raw)
	if got := tally.reasoning.String(); got != "two plus two" {
		t.Fatalf("reasoning_content = %q, want %q: %s", got, "two plus two", raw)
	}
	if got := tally.content.String(); got != "4" {
		t.Fatalf("content = %q, want 4: %s", got, raw)
	}
	if tally.firstReasoning < 0 || tally.firstReasoning > tally.firstContent {
		t.Fatalf("reasoning first=%d content first=%d, want reasoning before content: %s", tally.firstReasoning, tally.firstContent, raw)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCoalescingSenderNeverMergesReasoningWithContent(t *testing.T) {
	var got []DeltaChunk
	release := make(chan struct{})
	sender := NewCoalescingSender(func(chunk DeltaChunk) error {
		<-release
		got = append(got, chunk)
		return nil
	})
	// The consumer is blocked, so every later send queues and is eligible to merge.
	for _, step := range []struct {
		text      string
		reasoning bool
	}{{"r1", true}, {"r2", true}, {"c1", false}, {"c2", false}, {"r3", true}} {
		var err error
		if step.reasoning {
			err = sender.reasoning(step.text)
		} else {
			err = sender.Send(step.text, 0)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	close(release)
	if err := sender.Close(); err != nil {
		t.Fatal(err)
	}
	var reasoning, content strings.Builder
	order := ""
	for _, chunk := range got {
		if chunk.reasoning {
			reasoning.WriteString(chunk.Text)
			order += "R"
		} else {
			content.WriteString(chunk.Text)
			order += "C"
		}
	}
	if reasoning.String() != "r1r2r3" || content.String() != "c1c2" {
		t.Fatalf("reasoning=%q content=%q, want r1r2r3 and c1c2 on their own fields", reasoning.String(), content.String())
	}
	if strings.Trim(order, "R") == "" || !strings.HasPrefix(order, "R") || !strings.HasSuffix(order, "R") || strings.Contains(order, "CRC") {
		t.Fatalf("delivery order = %s, want reasoning, then content, then reasoning", order)
	}
}
