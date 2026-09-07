package gateway

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Exercise the HTTP completion path: legitimate discussion must reach the
// planner while genuine, repeated kernel banners still terminate the loop.
func TestAnthropicMessagesKernelEchoBoundary(t *testing.T) {
	for _, tc := range []struct {
		name, text, fresh string
		turns             int
		steer             bool
	}{
		{name: "inline", text: "Explanation %d mentions [fak] refused in prose.", turns: 4},
		{name: "quoted", text: "> [fak] refused %d tool call(s): example", turns: 4},
		{name: "fenced", text: "Example:\n```text\n[fak] refused %d tool call(s): example\n```", turns: 4},
		{name: "below_threshold", text: "[fak] refused %d tool call(s): Write", turns: 3},
		{name: "refusal", text: "[fak] refused %d tool call(s): Write", turns: 4, steer: true},
		{name: "current_refusal", text: "[fak] Allowed next step for %d refused tool call(s): choose a permitted tool.", turns: 4, steer: true},
		{name: "held_result", text: "[fak] %d tool results were held out of context (OVERSIZE)", turns: 4, steer: true},
		{name: "fresh_question", text: "[fak] refused %d tool call(s): Write", turns: 4, fresh: "Explain what [fak] means in this output."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			srv.planner = stubPlanner{comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "planner reached"}, FinishReason: "stop"}}
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			messages := []map[string]string{{"role": "user", "content": "again"}}
			for i := 0; i < tc.turns; i++ {
				messages = append(messages, map[string]string{"role": "assistant", "content": fmt.Sprintf(tc.text, i+1)}, map[string]string{"role": "user", "content": "again"})
			}
			if tc.fresh != "" {
				messages[len(messages)-1]["content"] = tc.fresh
			}
			body, err := json.Marshal(map[string]any{"messages": messages})
			if err != nil {
				t.Fatal(err)
			}
			var resp anthropicMessageResponse
			postJSON(t, ts.URL+"/v1/messages", json.RawMessage(body), &resp)
			var text string
			for _, block := range resp.Content {
				text += block.Text
			}
			if got := strings.Contains(text, "repeating myself"); got != tc.steer {
				t.Fatalf("steer=%v, want %v: %q", got, tc.steer, text)
			}
			if !tc.steer && !strings.Contains(text, "planner reached") {
				t.Fatalf("planner did not answer: %q", text)
			}
			if tc.steer && resp.StopReason != "end_turn" {
				t.Fatalf("stop=%q", resp.StopReason)
			}
		})
	}
}
