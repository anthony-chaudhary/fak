package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestLiveMCPFilterProofAB(t *testing.T) {
	var descriptorCounts []int
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var request struct {
			Messages []map[string]any `json:"messages"`
			Tools    []map[string]any `json:"tools"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			t.Fatal(err)
		}
		descriptorCounts = append(descriptorCounts, len(request.Tools))
		want := requiredToolForPrompt(request.Messages)
		name := want
		args := `{}`
		if !toolPresent(request.Tools, want) {
			name = "fak_tools_search"
			args = `{"query":"` + strings.TrimPrefix(want, "fak_") + `"}`
		}
		response := map[string]any{"choices": []any{map[string]any{"message": map[string]any{
			"role": "assistant", "content": "", "tool_calls": []any{map[string]any{
				"id": "call-1", "type": "function", "function": map[string]any{"name": name, "arguments": args},
			}},
		}}}}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer provider.Close()

	srv, err := gateway.New(gateway.Config{Model: "proof"})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	proof, err := runLiveMCPFilterProof(context.Background(), srv, provider.URL, "test-key", "test-model")
	if err != nil {
		t.Fatal(err)
	}
	if proof.Verdict != "PASS" || proof.Active.TaskSuccessRate != 1 || proof.Control.TaskSuccessRate != 1 || proof.Active.SearchRecall != 1 || proof.Active.SavedDescriptorBytes <= 0 {
		t.Fatalf("proof=%+v", proof)
	}
	if len(descriptorCounts) != 9 { // active search+route x3, then control direct x3
		t.Fatalf("provider calls=%d counts=%v", len(descriptorCounts), descriptorCounts)
	}
}

func requiredToolForPrompt(messages []map[string]any) string {
	for _, message := range messages {
		text, _ := message["content"].(string)
		switch {
		case strings.Contains(text, "memory drivers"):
			return "fak_memory_drivers"
		case strings.Contains(text, "context budget"):
			return "fak_context_change"
		case strings.Contains(text, "available features"):
			return "fak_feature_query"
		}
	}
	return ""
}

func toolPresent(tools []map[string]any, want string) bool {
	for _, tool := range tools {
		fn, _ := tool["function"].(map[string]any)
		if fn["name"] == want {
			return true
		}
	}
	return false
}

func TestClassifyLiveProofErrorDoesNotLeakProviderBody(t *testing.T) {
	got := classifyLiveProofError(io.ErrUnexpectedEOF)
	if got != "provider_request_failed" {
		t.Fatal(got)
	}
	got = classifyLiveProofError(&providerTestError{text: "secret tool payload rejected"})
	if got != "provider_rejected_tool_call" || strings.Contains(got, "secret") {
		t.Fatal(got)
	}
}

type providerTestError struct{ text string }

func (e *providerTestError) Error() string { return e.text }

func TestCompactJSON(t *testing.T) {
	if got := compactJSON([]byte("{ \"a\": 1 }")); !bytes.Equal(got, []byte(`{"a":1}`)) {
		t.Fatal(string(got))
	}
}
