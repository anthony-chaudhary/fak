package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type turnkeyToolRecordingPlanner struct {
	messages         [][]agent.Message
	tools            [][]agent.ToolDef
	receiptRequested []bool
}

type turnkeyBlockingPlanner struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (p *turnkeyBlockingPlanner) Model() string { return "local" }

func (p *turnkeyBlockingPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.once.Do(func() { close(p.started) })
	<-p.release
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}}, nil
}

func (p *turnkeyToolRecordingPlanner) Model() string      { return "local" }
func (p *turnkeyToolRecordingPlanner) ContextWindow() int { return 8192 }

func (p *turnkeyToolRecordingPlanner) Complete(_ context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.messages = append(p.messages, append([]agent.Message(nil), messages...))
	p.tools = append(p.tools, append([]agent.ToolDef(nil), tools...))
	var params agent.SampleParams
	for _, opt := range opts {
		opt(&params)
	}
	p.receiptRequested = append(p.receiptRequested, params.NativeInferenceReceipt)
	return &agent.Completion{
		Message:         agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "call_1", Type: "function", Function: agent.Func{Name: "read_file", Arguments: `{"path":"README.md"}`}}}},
		FinishReason:    "tool_calls",
		Usage:           agent.Usage{PromptTokens: 8, CompletionTokens: 3, TotalTokens: 11, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 5}},
		NativeInference: &model.NativeInferenceReceipt{Model: "receipt-sentinel"},
	}, nil
}

func TestTurnkeyChatCompletionsPreservesOpenAIToolLoop(t *testing.T) {
	planner := &turnkeyToolRecordingPlanner{}
	turnkey := &turnkeyServer{planner: planner, plan: macfit.TurnkeyProfile{ContextBudgetTokens: 8192, Tier: macfit.ModelTier{ModelID: "local"}}}
	testServer := httptest.NewServer(http.HandlerFunc(turnkey.handleChatCompletions))
	defer testServer.Close()

	tool := agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: "read_file", Description: "read a file", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}}
	first := map[string]any{
		"model": "local", "stream": true, "tool_choice": "auto", "tools": []agent.ToolDef{tool},
		"messages": []map[string]any{{"role": "user", "content": []map[string]any{{"type": "text", "text": "read the README"}}}},
	}
	streamBody := postTurnkeyToolRequest(t, testServer.URL, first)
	for _, want := range [][]byte{[]byte(`"tool_calls"`), []byte(`"call_1"`), []byte(`"read_file"`), []byte(`"finish_reason":"tool_calls"`), []byte(`"cached_tokens":5`), []byte("data: [DONE]")} {
		if !bytes.Contains(streamBody, want) {
			t.Fatalf("stream omitted %s: %s", want, streamBody)
		}
	}
	if len(planner.tools) != 1 || len(planner.tools[0]) != 1 || planner.tools[0][0].Function.Name != "read_file" || planner.messages[0][0].Content != "read the README" {
		t.Fatalf("first planner input lost content/tools: messages=%#v tools=%#v", planner.messages, planner.tools)
	}

	zero := 0.0
	second := gateway.ChatRequest{Model: "local", Temperature: &zero, Fak: &gateway.FakRequestExt{NativeInferenceReceipt: true}, Messages: []agent.Message{
		{Role: agent.RoleUser, Content: "read the README"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "call_1", Type: "function", Function: agent.Func{Name: "read_file", Arguments: `{"path":"README.md"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "read_file", Content: "contents"},
	}, Tools: []agent.ToolDef{tool}}
	bufferedBody := postTurnkeyToolRequest(t, testServer.URL, second)
	var got gateway.ChatResponse
	if err := json.Unmarshal(bufferedBody, &got); err != nil {
		t.Fatalf("decode buffered response: %v (%s)", err, bufferedBody)
	}
	if len(got.Choices) != 1 || got.Choices[0].FinishReason != "tool_calls" || len(got.Choices[0].Message.ToolCalls) != 1 || got.Choices[0].Message.ToolCalls[0].Function.Name != "read_file" {
		t.Fatalf("buffered response lost tool call: %#v", got)
	}
	if !planner.receiptRequested[1] || got.Fak == nil || got.Fak.NativeInferenceReceipt == nil || got.Fak.NativeInferenceReceipt.Model != "receipt-sentinel" {
		t.Fatalf("native receipt opt-in was not forwarded: requested=%v fak=%#v", planner.receiptRequested, got.Fak)
	}
	rejected := gateway.ChatRequest{Model: "local", Stream: true, Fak: &gateway.FakRequestExt{NativeInferenceReceipt: true}, Messages: []agent.Message{{Role: agent.RoleUser, Content: "receipt"}}}
	raw, err := json.Marshal(rejected)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(testServer.URL, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("streamed receipt status = %d, want 400", resp.StatusCode)
	}
	msgs := planner.messages[1]
	if len(msgs) != 3 || len(msgs[1].ToolCalls) != 1 || msgs[1].ToolCalls[0].ID != "call_1" || msgs[2].ToolCallID != "call_1" || msgs[2].Name != "read_file" || msgs[2].Content != "contents" {
		t.Fatalf("continuation lost assistant call or tool result: %#v", msgs)
	}
}

func TestTurnkeyModelsAdvertisesEnforcedCapacity(t *testing.T) {
	srv := &turnkeyServer{planner: &turnkeyToolRecordingPlanner{}, plan: macfit.TurnkeyProfile{ContextBudgetTokens: 8192, Tier: macfit.ModelTier{ModelID: "local"}}}
	rec := httptest.NewRecorder()
	srv.handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var got struct {
		Data []struct {
			ContextWindow   uint64 `json:"context_window"`
			MaxOutputTokens int    `json:"max_output_tokens"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 1 || got.Data[0].ContextWindow != 8192 || got.Data[0].MaxOutputTokens != 1024 {
		t.Fatalf("model capacity = %#v", got.Data)
	}
}

func TestTurnkeyHealthReportsNativeStartupMemory(t *testing.T) {
	srv := &turnkeyServer{native: &turnkeyNativeResources{Startup: turnkeyNativeStartup{HostAvailableBefore: 900, HostAvailableAfter: 700, LoadMode: "gguf-resident-q4k"}}}
	rec := httptest.NewRecorder()
	srv.handleHealthz(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	var got struct {
		NativeStartup turnkeyNativeStartup `json:"native_startup"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.NativeStartup.HostAvailableBefore != 900 || got.NativeStartup.HostAvailableAfter != 700 || got.NativeStartup.LoadMode != "gguf-resident-q4k" {
		t.Fatalf("native startup health = %+v", got.NativeStartup)
	}
}

func TestTurnkeyShutdownKeepsNativeResourcesUntilActiveRequestDrains(t *testing.T) {
	planner := &turnkeyBlockingPlanner{started: make(chan struct{}), release: make(chan struct{})}
	var nativeCloses atomic.Int32
	srv := &turnkeyServer{planner: planner, native: &turnkeyNativeResources{closeModel: func() error { nativeCloses.Add(1); return nil }}}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv.httpServer = &http.Server{Handler: mux}
	go func() { _ = srv.httpServer.Serve(listener) }()

	requestDone := make(chan error, 1)
	go func() {
		resp, err := http.Post("http://"+listener.Addr().String()+"/v1/chat/completions", "application/json", bytes.NewBufferString(`{"messages":[{"role":"user","content":"wait"}]}`))
		if err == nil {
			_ = resp.Body.Close()
		}
		requestDone <- err
	}()
	select {
	case <-planner.started:
	case <-time.After(5 * time.Second):
		t.Fatal("planner did not start")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := srv.Shutdown(ctx); err == nil {
		t.Fatal("canceled shutdown unexpectedly succeeded")
	}
	if got := nativeCloses.Load(); got != 0 {
		t.Fatalf("native resources closed with active request: %d", got)
	}
	close(planner.release)
	if err := <-requestDone; err != nil {
		t.Fatalf("request completion: %v", err)
	}
	ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		t.Fatalf("drained shutdown: %v", err)
	}
	if got := nativeCloses.Load(); got != 1 {
		t.Fatalf("native closes = %d, want 1", got)
	}
}

func postTurnkeyToolRequest(t *testing.T, url string, body any) []byte {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, body=%s", resp.StatusCode, got)
	}
	return got
}
