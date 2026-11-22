package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// turnkeyReasoningBufferedPlanner is Complete-only, so a stream:true request takes
// the buffered SSE twin in handleChatCompletions.
type turnkeyReasoningBufferedPlanner struct{}

func (turnkeyReasoningBufferedPlanner) Model() string { return "local" }

func (turnkeyReasoningBufferedPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "4", ReasoningContent: "two plus two"},
		FinishReason: "stop",
	}, nil
}

type turnkeyStreamDeltaTally struct {
	reasoning, content           strings.Builder
	reasoningFrames              int
	firstReasoning, firstContent int
}

func tallyTurnkeyStreamDeltas(t *testing.T, raw string) *turnkeyStreamDeltaTally {
	t.Helper()
	tally := &turnkeyStreamDeltaTally{firstReasoning: -1, firstContent: -1}
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
			tally.content.WriteString(delta.Content)
			if tally.firstContent < 0 {
				tally.firstContent = frame
			}
		}
	}
	return tally
}

// fak-test:runtime fast est=100ms lane=default
func TestTurnkeyChatStreamCarriesReasoningContent(t *testing.T) {
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

	for _, tc := range []struct {
		name          string
		planner       agent.Planner
		wantLiveFrame bool
	}{
		{name: "live_upstream", planner: agent.NewHTTPPlanner(upstream.URL, "local", ""), wantLiveFrame: true},
		{name: "buffered", planner: turnkeyReasoningBufferedPlanner{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ts := newTurnkeyStreamTestServer(tc.planner)
			defer ts.Close()
			resp := postTurnkeyStream(t, context.Background(), ts.URL)
			defer resp.Body.Close()
			raw, _ := io.ReadAll(resp.Body)
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d: %s", resp.StatusCode, raw)
			}
			tally := tallyTurnkeyStreamDeltas(t, string(raw))
			if got := tally.reasoning.String(); got != "two plus two" {
				t.Fatalf("reasoning_content = %q, want %q: %s", got, "two plus two", raw)
			}
			if got := tally.content.String(); got != "4" {
				t.Fatalf("content = %q, want 4: %s", got, raw)
			}
			if tally.firstReasoning > tally.firstContent {
				t.Fatalf("reasoning first=%d content first=%d, want reasoning before content: %s", tally.firstReasoning, tally.firstContent, raw)
			}
			if tc.wantLiveFrame && tally.reasoningFrames != 2 {
				t.Fatalf("reasoning frames = %d, want the 2 upstream deltas streamed live: %s", tally.reasoningFrames, raw)
			}
		})
	}
}
