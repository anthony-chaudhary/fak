package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelNativePromptReceiptMatchesPreparedEncoding(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	m := model.NewSynthetic(cfg)
	backend := &cudaUploadSnapshotBackend{Backend: compute.Default()}
	p := NewInKernelPlanner(m, loadProbeTok(t), "native-prompt-receipt", false, backend, false)
	p.SetPromptShrinkLevers(80, true, true)

	messages := []Message{
		{Role: RoleSystem, Content: "Answer with the selected tool."},
		{Role: RoleUser, Content: "Inspect the old result."},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "read-1", Type: "function", Function: Func{Name: "Read", Arguments: `{"file_path":"old.txt"}`}}}},
		{Role: RoleTool, Name: "Read", ToolCallID: "read-1", Content: strings.Repeat("stale output ", 80)},
		{Role: RoleAssistant, Content: "The old result was incomplete."},
		{Role: RoleUser, Content: "Return the current result."},
	}
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "read a file"}},
		{Type: "function", Function: ToolDefFunction{Name: "cold_database", Description: "query old records"}},
	}
	responseFormat := json.RawMessage(`{"type":"json_object"}`)
	toolChoice := json.RawMessage(`{"type":"function","function":{"name":"Read"}}`)
	opts := []SampleOpt{
		WithMaxTokens(2),
		WithResponseFormat(responseFormat),
		WithToolChoice(toolChoice),
		WithReasoningEffort(EffortTierNone),
		WithThinkingBudget(0),
		WithCompactHistoryBudget(20),
		WithDeferColdTools(true),
	}

	want, err := p.EncodePrompt(context.Background(), messages, tools, opts...)
	if err != nil {
		t.Fatalf("EncodePrompt: %v", err)
	}
	rawRendered := renderInKernelChatMLRequest(messages, tools, cfg, responseFormat, toolChoice, applySampleOpts(opts...))
	rawIDs, err := p.tok.Encode(rawRendered)
	if err != nil {
		t.Fatalf("encode unshrunk prompt: %v", err)
	}
	if reflect.DeepEqual(rawIDs, want.TokenIDs) {
		t.Fatal("fixture did not distinguish the prepared prompt from raw pre-shrink messages/tools")
	}

	comp, err := p.Complete(context.Background(), messages, tools, append(opts, WithNativeInferenceReceipt(true))...)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	r := comp.NativeInference
	if r == nil {
		t.Fatal("native prompt receipt missing")
	}
	if !reflect.DeepEqual(r.PromptTokenIDs, want.TokenIDs) || len(r.PromptTokenIDs) != comp.Usage.PromptTokens {
		t.Fatalf("prompt IDs/count = %v/%d, want %v/%d", r.PromptTokenIDs, comp.Usage.PromptTokens, want.TokenIDs, want.PromptTokens)
	}
	if r.TokenizerID != want.TokenizerID || r.RendererID != want.RendererID || r.RenderedSHA256 != want.RenderedSHA256 {
		t.Fatalf("prompt identity = tokenizer %q renderer %q digest %q, want %q/%q/%q", r.TokenizerID, r.RendererID, r.RenderedSHA256, want.TokenizerID, want.RendererID, want.RenderedSHA256)
	}
	wire, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal native receipt: %v", err)
	}
	var projected struct {
		PromptTokenIDs []int  `json:"prompt_token_ids"`
		TokenizerID    string `json:"tokenizer_id"`
		RendererID     string `json:"renderer_id"`
		RenderedSHA256 string `json:"rendered_sha256"`
	}
	if err := json.Unmarshal(wire, &projected); err != nil {
		t.Fatalf("unmarshal native receipt projection: %v", err)
	}
	if !reflect.DeepEqual(projected.PromptTokenIDs, want.TokenIDs) || projected.TokenizerID != want.TokenizerID || projected.RendererID != want.RendererID || projected.RenderedSHA256 != want.RenderedSHA256 {
		t.Fatalf("JSON prompt projection = %+v, want exact prepared encoding", projected)
	}

	originalFirst := r.PromptTokenIDs[0]
	r.PromptTokenIDs[0]++
	again, err := p.Complete(context.Background(), messages, tools, append(opts, WithNativeInferenceReceipt(true))...)
	if err != nil {
		t.Fatalf("second Complete: %v", err)
	}
	if again.NativeInference.PromptTokenIDs[0] != originalFirst {
		t.Fatalf("receipt prompt IDs alias prior request: got first ID %d, want %d", again.NativeInference.PromptTokenIDs[0], originalFirst)
	}
}

func TestInKernelNativePromptReceiptDefaultOff(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "native-prompt-default-off", false, nil, false)
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "plain request"}}, nil, WithMaxTokens(1))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if comp.NativeInference != nil {
		t.Fatalf("ordinary completion exposed native receipt: %+v", comp.NativeInference)
	}
}
