package openaiadapter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func server() *Server {
	return &Server{Token: "app-secret", Aliases: map[string]string{"job-apply": "task/job-apply/v1"}, Execute: func(_ context.Context, model string, r Request) (Result, error) {
		if model != "task/job-apply/v1" {
			panic(model)
		}
		return Result{Content: map[string]any{"company": "Acme", "role": "Engineer"}, ToolCalls: []ToolCall{{ID: "call_1", Type: "function", Function: FunctionCall{Name: "save_application", Arguments: `{"status":"draft"}`}}}, Usage: Usage{PromptTokens: 12, CompletionTokens: 8, TotalTokens: 20}}, nil
	}}
}
func TestGoldenJobApplyWire(t *testing.T) {
	s := server()
	body := `{"model":"job-apply","messages":[{"role":"user","content":"Apply"}],"tools":[{"type":"function","function":{"name":"save_application"}}],"response_format":{"type":"json_schema","json_schema":{"name":"application"}}}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer app-secret")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	got := w.Body.String()
	want, err := os.ReadFile("testdata/job-apply.golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g, v any
	if json.Unmarshal([]byte(got), &g) != nil || json.Unmarshal(want, &v) != nil {
		t.Fatal("invalid JSON")
	}
	gb, _ := json.Marshal(g)
	vb, _ := json.Marshal(v)
	if string(gb) != string(vb) {
		t.Fatalf("wire mismatch\ngot %s\nwant %s", gb, vb)
	}
}
func TestStreamingAuthCancellationAndCompatibility(t *testing.T) {
	s := server()
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"job-apply","messages":[],"stream":true}`))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Fatal(w.Code)
	}
	r = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"job-apply","messages":[],"stream":true}`))
	r.Header.Set("Authorization", "Bearer app-secret")
	w = httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	if !strings.Contains(w.Body.String(), "data: [DONE]") || !strings.Contains(w.Body.String(), `"usage"`) {
		t.Fatal(w.Body.String())
	}
	s.Execute = func(ctx context.Context, _ string, _ Request) (Result, error) { return Result{}, context.Canceled }
	r = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"job-apply","messages":[]}`))
	r.Header.Set("Authorization", "Bearer app-secret")
	w = httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	if w.Code != 499 || !strings.Contains(w.Body.String(), "request_cancelled") {
		t.Fatal(w.Body.String())
	}
	if Compatibility()["images"] != "unsupported" || Compatibility()["tool_calls"] != "supported" {
		t.Fatal(Compatibility())
	}
	if l, e := s.Listen("tcp", "0.0.0.0:0"); e == nil {
		l.Close()
		t.Fatal("general-purpose LAN listener admitted")
	}
}
