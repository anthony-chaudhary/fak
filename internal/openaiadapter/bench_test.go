package openaiadapter

import (
	"bytes"
	"context"
	"net/http/httptest"
	"testing"
)

func BenchmarkOpenAIAdapter(b *testing.B) {
	s := &Server{
		Token: "bench-token",
		Aliases: map[string]string{
			"gpt-4o-mini": "task/job-apply/v1",
		},
		Execute: func(_ context.Context, model string, r Request) (Result, error) {
			return Result{
				Content: `{"company":"Acme","role":"Engineer","status":"draft"}`,
				ToolCalls: []ToolCall{
					{
						ID:   "call_bench_0",
						Type: "function",
						Function: FunctionCall{
							Name:      "save_application",
							Arguments: `{"status":"draft"}`,
						},
					},
				},
				Usage: Usage{
					PromptTokens:     16,
					CompletionTokens: 12,
					TotalTokens:      28,
				},
			}, nil
		},
	}
	handler := s.Handler()
	rawBody := []byte(`{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Benchmark prompt for OpenAI adapter wire conversion"}],"tools":[{"type":"function","function":{"name":"save_application"}}]}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(rawBody))
		req.Header.Set("Authorization", "Bearer bench-token")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != 200 {
			b.Fatalf("unexpected HTTP status %d: %s", w.Code, w.Body.String())
		}
	}
}

func TestBenchmarkOpenAIAdapterSanity(t *testing.T) {
	s := &Server{
		Token: "test-token",
		Aliases: map[string]string{
			"m": "target",
		},
		Execute: func(_ context.Context, _ string, _ Request) (Result, error) {
			return Result{Content: "ok", Usage: Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}}, nil
		},
	}
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader([]byte(`{"model":"m","messages":[{"role":"user","content":"ping"}]}`)))
	req.Header.Set("Authorization", "Bearer test-token")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}
