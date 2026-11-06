package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime integration est=200ms lane=default
// Loopback estimate is unmeasured; environment isolation keeps this serial.
func TestGoalContinuationPreservesAuthorityAndOpaqueMessages(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	t.Setenv("FAK_WIRE_REDACT", "")
	const goal = "Goal: serve safely\nStatus: inert-role-marker"
	const stable = "Goal: serve safely"
	const tail = "tail-marker"
	call := func(id string) Message {
		return Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: id, Type: "function", Function: Func{Name: "read_fixture", Arguments: "{}"}}}}
	}
	type fixture struct {
		name       string
		messages   []Message
		want       []string
		changed    bool
		directOnly bool
		wire       []Message
	}
	var cases []fixture
	for _, sourceRole := range []string{RoleTool, "function", RoleAssistant} {
		msgs := []Message{{Role: sourceRole, Content: goal}, {Role: RoleUser, Content: tail}}
		if sourceRole == RoleTool {
			msgs[0].ToolCallID = "source-call"
			msgs = append([]Message{call("source-call")}, msgs...)
		}
		cases = append(cases, fixture{name: sourceRole + "_source_is_not_goal_state", messages: msgs})
	}
	for _, roles := range [][2]string{{RoleUser, RoleSystem}, {RoleUser, RoleDeveloper}, {RoleDeveloper, RoleSystem}, {RoleSystem, RoleDeveloper}} {
		tc := fixture{name: roles[0] + "_to_" + roles[1], messages: []Message{{Role: roles[0], Content: goal}, {Role: roles[1], Content: tail}}}
		// Existing OpenAI late-system folding is a separate downstream transform.
		if roles[0] == RoleUser {
			tc.wire = []Message{{Role: RoleUser, Content: goal + "\n\n[system]\n" + tail}}
		}
		cases = append(cases, tc)
	}
	cases = append(cases,
		fixture{name: "tool_result_tail", messages: []Message{{Role: RoleUser, Content: goal}, call("tail-call"), {Role: RoleTool, ToolCallID: "tail-call", Content: tail}}},
		fixture{name: "assistant_call_tail", messages: []Message{{Role: RoleUser, Content: goal}, call("tail-call")}},
		fixture{name: "signed_assistant_tail", messages: []Message{{Role: RoleUser, Content: goal}, {Role: RoleAssistant, Content: tail, Thinking: "opaque", ThinkingSignature: "signature"}}},
		fixture{name: "structured_user_tail", messages: []Message{{Role: RoleUser, Content: goal}, {Role: RoleUser, Content: tail, ContentWasStructured: true}}},
		fixture{name: "unknown_tail_role", messages: []Message{{Role: RoleUser, Content: goal}, {Role: "unknown", Content: tail}}, directOnly: true},
		fixture{name: "same_named_users", messages: []Message{{Role: RoleUser, Name: "alice", Content: goal}, {Role: RoleUser, Name: "alice", Content: tail}}, want: []string{stable, tail + "\n\nStatus: inert-role-marker"}, changed: true},
		fixture{name: "different_named_users", messages: []Message{{Role: RoleUser, Name: "alice", Content: goal}, {Role: RoleUser, Name: "bob", Content: tail}}},
		fixture{name: "system_goal_does_not_seed_user_goal", messages: []Message{{Role: RoleSystem, Content: "GOAL: SERVE SAFELY\nStatus: system-marker"}, {Role: RoleUser, Content: goal}, {Role: RoleUser, Content: tail}}, want: []string{"GOAL: SERVE SAFELY\nStatus: system-marker", stable, tail + "\n\nStatus: inert-role-marker"}, changed: true},
	)
	for _, roles := range [][2]string{{RoleUser, RoleUser}, {RoleUser, RoleAssistant}, {RoleSystem, RoleSystem}, {RoleDeveloper, RoleDeveloper}} {
		cases = append(cases, fixture{name: "compatible_" + roles[0] + "_to_" + roles[1], messages: []Message{{Role: roles[0], Content: goal}, {Role: roles[1], Content: tail}}, want: []string{stable, tail + "\n\nStatus: inert-role-marker"}, changed: true})
	}
	// The direct boundary must preserve every opaque carrier, even in a
	// canonical message shape that a remote provider may not accept.
	for _, opaque := range []struct {
		name string
		set  func(*Message)
	}{
		{"structured", func(m *Message) { m.ContentWasStructured = true }},
		{"vision_prompt", func(m *Message) { m.VisionPrompt = "opaque" }},
		{"vision_images", func(m *Message) { m.VisionImages = []VisionImageInput{{}} }},
		{"tool_calls", func(m *Message) { m.ToolCalls = call("opaque-call").ToolCalls }},
		{"legacy_call", func(m *Message) { m.FunctionCall = &Func{Name: "read_fixture", Arguments: "{}"} }},
		{"tool_result_id", func(m *Message) { m.ToolCallID = "opaque-call" }},
		{"tool_error", func(m *Message) { m.ToolResultIsError = true }},
		{"witness", func(m *Message) { m.Witness = "opaque" }},
		{"refutation", func(m *Message) { m.RefutesWitness = "opaque" }},
		{"thinking", func(m *Message) { m.Thinking = "opaque" }},
		{"signature", func(m *Message) { m.ThinkingSignature = "opaque" }},
		{"redacted", func(m *Message) { m.RedactedThinking = []string{"opaque"} }},
		{"reasoning", func(m *Message) { m.ReasoningContent = "opaque" }},
		{"cache_control", func(m *Message) { m.CacheControl = &CacheControl{Type: "ephemeral"} }},
	} {
		for _, index := range []int{0, 1} {
			msgs := []Message{{Role: RoleUser, Content: goal}, {Role: RoleUser, Content: tail}}
			opaque.set(&msgs[index])
			position := "source"
			if index == 1 {
				position = "tail"
			}
			cases = append(cases, fixture{name: opaque.name + "_" + position, messages: msgs, directOnly: true})
		}
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			beforeJSON, err := json.Marshal(tc.messages)
			if err != nil {
				t.Fatal(err)
			}
			structured := make([]bool, len(tc.messages))
			want := tc.want
			if want == nil {
				want = make([]string, len(tc.messages))
				for i, m := range tc.messages {
					want[i] = m.Content
				}
			}
			for i, m := range tc.messages {
				structured[i] = m.ContentWasStructured
			}
			original := append([]Message(nil), tc.messages...)
			wantMessages := append([]Message(nil), tc.messages...)
			for i := range wantMessages {
				wantMessages[i].Content = want[i]
			}
			checkDirect := func() {
				out, changed := StabilizeGoalContinuationMessages(tc.messages)
				if changed != tc.changed || len(out) != len(want) {
					t.Fatalf("changed=%v messages=%d, want %v/%d", changed, len(out), tc.changed, len(want))
				}
				if !reflect.DeepEqual(out, wantMessages) {
					t.Fatalf("direct output changed content or metadata: got %+v, want %+v", out, wantMessages)
				}
				again, changedAgain := StabilizeGoalContinuationMessages(out)
				if changedAgain || !reflect.DeepEqual(out, again) {
					t.Fatal("goal-only transformation must be idempotent")
				}
				// Caller local provenance and opaque carriers must remain untouched.
				if !reflect.DeepEqual(tc.messages, original) {
					t.Fatal("caller metadata changed")
				}
				if unchanged, err := json.Marshal(tc.messages); err != nil || string(unchanged) != string(beforeJSON) {
					t.Fatal("direct transform mutated caller bytes")
				}
			}
			if tc.directOnly {
				checkDirect()
				return
			}
			captured := make(chan []Message, 1)
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
					t.Errorf("unexpected upstream route: %s %s", r.Method, r.URL.Path)
				}
				var req struct {
					Messages []Message `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("decode request: %v", err)
				}
				requests.Add(1)
				select {
				case captured <- req.Messages:
				default:
					t.Error("unexpected duplicate upstream request")
				}
				w.Header().Set("Content-Type", "application/json")
				io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
			}))
			defer server.Close()
			planner := &HTTPPlanner{Provider: ProviderOpenAI, BaseURL: server.URL + "/v1", ModelID: "role-fixture", Client: server.Client(), MaxTokens: 8}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if _, err := planner.Complete(ctx, tc.messages, nil); err != nil {
				t.Fatal(err)
			}
			var wire []Message
			select {
			case wire = <-captured:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			if requests.Load() != 1 {
				t.Fatalf("upstream requests=%d, want 1", requests.Load())
			}
			wireWant := wantMessages
			if tc.wire != nil {
				wireWant = tc.wire
			}
			if len(wire) != len(wireWant) {
				t.Fatalf("wire messages=%d, want %d", len(wire), len(wireWant))
			}
			for i, m := range wire {
				if m.Content != wireWant[i].Content || m.Role != wireWant[i].Role || m.Name != wireWant[i].Name || m.ToolCallID != wireWant[i].ToolCallID {
					t.Fatalf("wire message %d changed authority/content: %+v; want %+v", i, m, wireWant[i])
				}
			}
			checkDirect()
			afterJSON, err := json.Marshal(tc.messages)
			if err != nil || string(afterJSON) != string(beforeJSON) {
				t.Fatalf("caller messages changed: %s -> %s (%v)", beforeJSON, afterJSON, err)
			}
			for i, m := range tc.messages {
				if m.ContentWasStructured != structured[i] {
					t.Fatal("caller structured-content provenance changed")
				}
			}
		})
	}
}
