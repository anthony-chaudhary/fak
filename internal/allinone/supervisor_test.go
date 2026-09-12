package allinone

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type chatCompletionEngine struct {
	mu    sync.Mutex
	calls []*abi.ToolCall
}

func (e *chatCompletionEngine) Complete(_ context.Context, call *abi.ToolCall) (*abi.Result, error) {
	e.mu.Lock()
	copyCall := *call
	copyCall.Args.Inline = bytes.Clone(call.Args.Inline)
	e.calls = append(e.calls, &copyCall)
	e.mu.Unlock()

	const answer = "engine says hello"
	return &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(answer), Len: int64(len(answer))},
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"input_tokens":  "7",
			"output_tokens": "3",
		},
	}, nil
}

func (*chatCompletionEngine) Caps() []abi.Capability { return nil }

func (e *chatCompletionEngine) Calls() []*abi.ToolCall {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]*abi.ToolCall(nil), e.calls...)
}

func TestAllInOneChatCompletionsEndpoint(t *testing.T) {
	engine := &chatCompletionEngine{}
	sup, err := NewSupervisor(Config{
		LockPath:     writeContractLock(t, `[]`, `[]`),
		Addr:         "127.0.0.1:0",
		EngineDriver: engine,
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() {
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		if err := sup.Shutdown(shutdownCtx); err != nil {
			t.Errorf("Shutdown: %v", err)
		}
	})

	client := &http.Client{Timeout: 5 * time.Second}
	endpoint := "http://" + sup.Addr() + "/v1/chat/completions"
	request := func(t *testing.T, stream bool) *http.Response {
		t.Helper()
		body, err := json.Marshal(map[string]any{
			"model":  "custom-model",
			"stream": stream,
			"messages": []map[string]string{
				{"role": "system", "content": "answer briefly"},
				{"role": "user", "content": "say hello"},
			},
		})
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("POST /v1/chat/completions: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, raw)
		}
		return resp
	}

	t.Run("non-streaming JSON", func(t *testing.T) {
		resp := request(t, false)
		defer resp.Body.Close()
		if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
			t.Fatalf("Content-Type = %q, want application/json", got)
		}
		var completion struct {
			Object  string `json:"object"`
			Choices []struct {
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
			t.Fatalf("decode completion: %v", err)
		}
		if completion.Object != "chat.completion" || len(completion.Choices) != 1 {
			t.Fatalf("completion envelope = %+v, want one chat.completion choice", completion)
		}
		choice := completion.Choices[0]
		if choice.Message.Role != "assistant" || choice.Message.Content != "engine says hello" || choice.FinishReason != "stop" {
			t.Fatalf("choice = %+v, want assistant engine result with stop", choice)
		}
		if completion.Usage.PromptTokens != 7 || completion.Usage.CompletionTokens != 3 || completion.Usage.TotalTokens != 10 {
			t.Fatalf("usage = %+v, want 7 prompt + 3 completion tokens", completion.Usage)
		}
	})

	t.Run("streaming SSE", func(t *testing.T) {
		resp := request(t, true)
		defer resp.Body.Close()
		if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
			t.Fatalf("Content-Type = %q, want text/event-stream", got)
		}

		var content strings.Builder
		seenStop := false
		seenDone := false
		scan := bufio.NewScanner(resp.Body)
		for scan.Scan() {
			line := scan.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				seenDone = true
				continue
			}
			var chunk struct {
				Object  string `json:"object"`
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				t.Fatalf("decode SSE data %q: %v", data, err)
			}
			if chunk.Object != "chat.completion.chunk" || len(chunk.Choices) != 1 {
				t.Fatalf("SSE chunk = %+v, want one chat.completion.chunk choice", chunk)
			}
			content.WriteString(chunk.Choices[0].Delta.Content)
			if reason := chunk.Choices[0].FinishReason; reason != nil && *reason == "stop" {
				seenStop = true
			}
		}
		if err := scan.Err(); err != nil {
			t.Fatalf("scan SSE: %v", err)
		}
		if got := content.String(); got != "engine says hello" {
			t.Fatalf("streamed content = %q, want engine result", got)
		}
		if !seenStop || !seenDone {
			t.Fatalf("stream terminators: finish_reason stop=%v [DONE]=%v", seenStop, seenDone)
		}
	})

	calls := engine.Calls()
	if len(calls) != 2 {
		t.Fatalf("engine call count = %d, want 2", len(calls))
	}
	for i, call := range calls {
		if call.Tool != "chat.completions" || call.Engine != "custom" {
			t.Errorf("call[%d] route = tool %q engine %q, want chat.completions/custom", i, call.Tool, call.Engine)
		}
		var forwarded struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(call.Args.Inline, &forwarded); err != nil {
			t.Fatalf("call[%d] Args is not original request JSON: %v", i, err)
		}
		if len(forwarded.Messages) != 2 || forwarded.Messages[1].Role != "user" || forwarded.Messages[1].Content != "say hello" {
			t.Errorf("call[%d] messages = %+v, want preserved system and user messages", i, forwarded.Messages)
		}
	}
}
