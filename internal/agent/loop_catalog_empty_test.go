package agent

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=100ms lane=default
// fak-test:justify why=regression when=changed:internal/agent/loop*.go
func TestNativeNoToolsCatalogOnHTTPWire(t *testing.T) {
	DisarmCodeTools()
	t.Cleanup(DisarmCodeTools)
	for _, tc := range []struct {
		name   string
		opts   []RunOption
		closed bool
		custom string
		armed  bool
	}{
		{name: "omitted-default"},
		{name: "omitted-with-addons", opts: []RunOption{WithTodoTools(), WithTaskTools()}},
		{name: "explicit-nil", opts: []RunOption{WithToolCatalog(nil)}, closed: true},
		{name: "explicit-empty-with-addons", opts: []RunOption{WithToolCatalog([]ToolDef{}), WithTodoTools(), WithTaskTools()}, closed: true},
		{name: "explicit-nil-while-code-armed", opts: []RunOption{WithToolCatalog(nil)}, closed: true, armed: true},
		{name: "explicit-nonempty", opts: []RunOption{WithToolCatalog(adapterTestTools())}},
		{name: "custom-prompt", opts: []RunOption{WithToolCatalog(nil), WithSystemPrompt("CUSTOM_NO_TOOLS_SENTINEL")}, closed: true, custom: "CUSTOM_NO_TOOLS_SENTINEL"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.armed {
				if _, err := ArmCodeTools(t.TempDir()); err != nil {
					t.Fatal(err)
				}
				defer DisarmCodeTools()
			}
			requests := make(chan map[string]json.RawMessage, 2)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					http.Error(w, "bad request", 400)
					return
				}
				requests <- body
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1}}`))
			}))
			defer srv.Close()
			planner, err := NewProviderHTTPPlanner("openai", srv.URL, "fixture-model", "loopback")
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			m, err := RunArm(ctx, planner, "Return done.", true, 1, nil, tc.opts...)
			if err != nil || m.FinalAnswer != "done" {
				t.Fatalf("answer=%q error=%v", m.FinalAnswer, err)
			}
			var request map[string]json.RawMessage
			select {
			case request = <-requests:
			default:
				t.Fatal("HTTP planner did not send request")
			}
			var tools []ToolDef
			if raw, ok := request["tools"]; ok {
				if err := json.Unmarshal(raw, &tools); err != nil {
					t.Fatal(err)
				}
				if tc.closed {
					t.Error("explicit empty catalog must omit HTTP tools field")
				}
			}
			if tc.name == "explicit-nonempty" {
				if len(tools) != 1 || tools[0].Function.Name != "lookup" {
					t.Error("nonempty request catalog changed")
				}
			} else if !tc.closed {
				if len(tools) < len(ToolCatalog()) || len(tools) == 0 {
					t.Error("omitted catalog lost default tools")
				}
				if tc.name == "omitted-with-addons" {
					found := map[string]bool{}
					for _, tool := range tools {
						found[tool.Function.Name] = true
					}
					if !found[ToolTodoWrite] || !found[ToolTaskSpawn] {
						t.Error("omitted catalog lost requested addon tools")
					}
				}
			}
			var messages []Message
			if err := json.Unmarshal(request["messages"], &messages); err != nil {
				t.Fatal(err)
			}
			wantPrompt := SystemPrompt
			if tc.closed {
				wantPrompt = NoToolsAgentSystemPrompt
			}
			if tc.custom != "" {
				wantPrompt = tc.custom
			}
			if len(messages) == 0 || messages[0].Role != RoleSystem || !strings.Contains(messages[0].Content, wantPrompt) {
				t.Error("standing prompt does not match declared catalog/custom override")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
// fak-test:justify why=guard when=changed:internal/agent/loop*.go twin=TestNativeNoToolsCatalogOnHTTPWire
func TestNativeNoToolsRejectsHostileStructuredAndSpeculativeCalls(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(map[bool]string{false: "structured", true: "streaming-speculative"}[streaming], func(t *testing.T) {
			turn := toolCallTurn("get_user_details", `{"user_id":"u1"}`)
			turn.Message.Content = `<tool_call>{"name":"get_user_details","arguments":{"user_id":"u1"}}</tool_call>`
			spec := abi.NewSpeculator(0)
			spec.Learn(searchPattern(`{"origin":"SFO"}`))
			opts := []RunOption{WithToolCatalog(nil), WithSpeculator(spec), WithStreamingSpeculation(true)}
			fsmCalls := 0
			opts = append(opts, WithStreamingFSMHook(func(*StreamingToolFSM) { fsmCalls++ }))
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var m ArmMetrics
			var err error
			if streaming {
				m, err = RunArmStream(ctx, &scriptedStreamingPlanner{turns: []*Completion{turn}}, "Return a final answer without tools.", true, 1, nil, nil, opts...)
			} else {
				m, err = RunArm(ctx, &scriptedPlanner{turns: []*Completion{turn}}, "Return a final answer without tools.", true, 1, nil, opts...)
			}
			if !errors.Is(err, ErrToolCatalogClosed) {
				t.Errorf("closed catalog error=%v", err)
			}
			if m.EngineCalls != 0 || m.SpecIssued != 0 || m.SpecCommitted != 0 || fsmCalls != 0 {
				t.Errorf("closed catalog dispatched engine/speculation: engine=%d issued=%d committed=%d fsm=%d", m.EngineCalls, m.SpecIssued, m.SpecCommitted, fsmCalls)
			}
		})
	}
}
