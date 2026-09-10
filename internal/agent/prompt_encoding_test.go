package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeEncodePromptParity(t *testing.T) {
	cfg := model.Config{
		ModelType:             "qwen3_5",
		Architectures:         []string{"Qwen3_5ForCausalLM"},
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		MaxPositionEmbeddings: 32768,
	}
	p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, loadProbeTok(t), "qwen3.8-fixture", false, nil, false, InKernelPlannerConfig{ContextTokens: 8192})
	messages := []Message{
		{Role: RoleSystem, Content: "Work precisely."},
		{Role: RoleUser, Content: "Read and repair retry.go"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Type: "function", Function: Func{Name: "Read", Arguments: `{"file_path":"retry.go"}`}}}},
		{Role: RoleTool, Name: "Read", ToolCallID: "call-1", Content: `{"content":"broken"}`},
	}
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read a file", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`)}},
		{Type: "function", Function: ToolDefFunction{Name: "Write", Description: "write a file", Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}`)}},
	}
	responseFormat := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}}`)
	toolChoice := json.RawMessage(`{"type":"function","function":{"name":"Read"}}`)
	opts := []SampleOpt{WithMaxTokens(73), WithResponseFormat(responseFormat), WithToolChoice(toolChoice), WithReasoningEffort(EffortTierNone), WithThinkingBudget(0)}

	wantIDs, wantDigest := promptEncodingOracle(t, p, messages, tools, opts...)
	got, err := p.EncodePrompt(context.Background(), messages, tools, opts...)
	if err != nil {
		t.Fatalf("EncodePrompt: %v", err)
	}
	if !reflect.DeepEqual(got.TokenIDs, wantIDs) || got.PromptTokens != len(wantIDs) || got.RenderedSHA256 != wantDigest {
		t.Fatalf("render/tokenize mismatch: ids=%v want=%v tokens=%d digest=%q want=%q", got.TokenIDs, wantIDs, got.PromptTokens, got.RenderedSHA256, wantDigest)
	}
	if got.ModelID != p.Model() || got.ContextWindowTokens != 8192 || got.ReservedOutputTokens != 73 {
		t.Fatalf("resolved metadata = model %q context %d reserve %d", got.ModelID, got.ContextWindowTokens, got.ReservedOutputTokens)
	}
	if got.RendererID == "" || got.TokenizerID == "" || got.TokenizerID == got.ModelID {
		t.Fatalf("closed operational identities required: model=%q renderer=%q tokenizer=%q", got.ModelID, got.RendererID, got.TokenizerID)
	}
	originalFirst := got.TokenIDs[0]
	got.TokenIDs[0]++
	again, err := p.EncodePrompt(context.Background(), messages, tools, opts...)
	if err != nil || again.TokenIDs[0] != originalFirst || again.TokenizerID != got.TokenizerID {
		t.Fatalf("result must copy IDs and tokenizer identity must be request-independent: again=%+v err=%v", again, err)
	}

	mutations := []struct {
		name  string
		msgs  []Message
		tools []ToolDef
		opts  []SampleOpt
	}{
		{name: "message", msgs: append(append([]Message(nil), messages...), Message{Role: RoleUser, Content: "extra"}), tools: tools, opts: opts},
		{name: "tools", msgs: messages, tools: []ToolDef{
			{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read a file", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}},
			tools[1],
		}, opts: opts},
		{name: "response_format", msgs: messages, tools: tools, opts: []SampleOpt{WithMaxTokens(73), WithResponseFormat(json.RawMessage(`{"type":"json_object"}`)), WithToolChoice(toolChoice), WithReasoningEffort(EffortTierNone)}},
		{name: "tool_choice", msgs: messages, tools: tools, opts: []SampleOpt{WithMaxTokens(73), WithResponseFormat(responseFormat), WithToolChoice(json.RawMessage(`{"type":"function","function":{"name":"Write"}}`)), WithReasoningEffort(EffortTierNone)}},
		{name: "thinking", msgs: messages, tools: tools, opts: []SampleOpt{WithMaxTokens(73), WithResponseFormat(responseFormat), WithToolChoice(toolChoice), WithReasoningEffort(EffortTierHigh), WithThinkingBudget(32)}},
	}
	for _, tt := range mutations {
		t.Run(tt.name, func(t *testing.T) {
			enc, err := p.EncodePrompt(context.Background(), tt.msgs, tt.tools, tt.opts...)
			if err != nil {
				t.Fatal(err)
			}
			if reflect.DeepEqual(enc.TokenIDs, wantIDs) || enc.RenderedSHA256 == wantDigest {
				t.Fatalf("render-affecting mutation retained baseline encoding/digest")
			}
		})
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	bad := []*InKernelPlanner{nil, {}, {m: &model.Model{Cfg: cfg}}, {m: &model.Model{Cfg: cfg}, tok: loadProbeTok(t)}}
	contexts := []context.Context{context.Background(), context.Background(), context.Background(), canceled}
	for i := range bad {
		if _, err := bad[i].EncodePrompt(contexts[i], messages, tools, opts...); err == nil {
			t.Fatalf("invalid planner case %d returned an approximate encoding", i)
		}
	}
}

func promptEncodingOracle(t *testing.T, p *InKernelPlanner, messages []Message, tools []ToolDef, opts ...SampleOpt) ([]int, string) {
	t.Helper()
	sp := applySampleOpts(opts...)
	messages, tools, _ = p.ApplyPromptShrink(context.Background(), messages, tools, opts...)
	rendered := renderInKernelChatMLRequest(messages, tools, p.m.Cfg, sp.ResponseFormat, sp.ToolChoice, sp)
	ids, err := p.tok.Encode(rendered)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte(rendered))
	return ids, hex.EncodeToString(sum[:])
}
