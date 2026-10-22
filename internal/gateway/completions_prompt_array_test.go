package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// An OpenAI prompt array of two or more strings is a batch; merging it into one
// prompt silently returns one choice for N prompts, so the route refuses it
// before admission and before any SSE preamble.
// fak-test:runtime fast est=50ms
func TestCompletionsPromptArrayBatchRefused(t *testing.T) {
	srv := newTestServer(t)
	seen := "unset"
	srv.planner = capturePlanner{
		comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"},
		last: &seen,
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	cases := []struct {
		name   string
		body   string
		stream bool
	}{
		{"two", `{"model":"m","prompt":["a","b"]}`, false},
		{"two-stream", `{"model":"m","prompt":["a","b"],"stream":true}`, true},
		{"three", `{"model":"m","prompt":["a","b","c"]}`, false},
	}
	for _, tc := range cases {
		r, err := http.Post(ts.URL+"/v1/completions", "application/json", strings.NewReader(tc.body))
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(r.Body)
		r.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if r.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400", tc.name, r.StatusCode)
		}
		if got := chatErrorCode(t, body); got != "prompt_batch_unsupported" {
			t.Errorf("%s: error.code = %q, want prompt_batch_unsupported", tc.name, got)
		}
		if tc.stream {
			if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
				t.Errorf("%s: Content-Type = %q, want application/json (refusal before the SSE preamble)", tc.name, ct)
			}
		}
	}
	if seen != "unset" {
		t.Errorf("planner saw %q, want it never invoked", seen)
	}
}

// fak-test:runtime fast est=50ms
func TestCompletionsPromptArraySingleElementServes(t *testing.T) {
	srv := newTestServer(t)
	var seen string
	srv.planner = capturePlanner{
		comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"},
		last: &seen,
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var resp CompletionResponse
	code := postJSON(t, ts.URL+"/v1/completions", CompletionRequest{
		Model:  "test-model",
		Prompt: json.RawMessage(`["only one"]`),
	}, &resp)
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Text != "ok" {
		t.Fatalf("choices = %+v, want exactly one choice with text=ok", resp.Choices)
	}
	if seen != "only one" {
		t.Errorf("planner saw %q, want %q", seen, "only one")
	}
}
