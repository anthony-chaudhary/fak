package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fak-test:justify why=integration when=changed:internal/gateway/**,internal/agent/chat.go
// fak-test:runtime fast est=1s
func TestChatProxyPreservesEmptyToolCallPresence(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		present     bool
	}{
		{"absent", "", false},
		{"null", `,"tool_calls":null`, false},
		{"empty", `,"tool_calls":[]`, true},
		{"refused", `,"tool_calls":[{"id":"call-denied","type":"function","function":{"name":"deny_write","arguments":"{}"}}]`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"model":"test-model","choices":[{"message":{"role":"assistant","content":"hello"%s},"finish_reason":"stop"}]}`, tc.field)
			}))
			defer upstream.Close()
			srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible"})
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"deny_write","parameters":{"type":"object"}}}]}`)))
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d", w.Code)
			}
			var response struct {
				Choices []struct {
					Message map[string]json.RawMessage `json:"message"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if len(response.Choices) != 1 {
				t.Fatalf("choices = %d", len(response.Choices))
			}
			calls, present := response.Choices[0].Message["tool_calls"]
			if present != tc.present || (present && string(calls) != "[]") {
				t.Fatalf("tool_calls presence = %t, value = %s; want presence %t", present, calls, tc.present)
			}
		})
	}
}
