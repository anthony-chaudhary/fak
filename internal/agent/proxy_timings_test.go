package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// llamaServerBody is a buffered llama.cpp server response captured from the
// appliance's llama-server (build dff6004), trimmed to the fields the adapter reads.
const llamaServerBody = `{"choices":[{"finish_reason":"length","index":0,"message":{"role":"assistant","content":"Hi"}}],` +
	`"model":"Qwen3.8-27B-UD-Q2_K_XL","object":"chat.completion",` +
	`"usage":{"completion_tokens":4,"prompt_tokens":13,"total_tokens":17,"prompt_tokens_details":{"cached_tokens":0}},` +
	`"timings":{"cache_n":0,"prompt_n":13,"prompt_ms":250.368,"prompt_per_token_ms":19.259,"prompt_per_second":51.92,` +
	`"predicted_n":4,"predicted_ms":165.636,"predicted_per_token_ms":55.211,"predicted_per_second":18.11}}`

func TestProxiedLlamaServerTimingsParsed(t *testing.T) {
	comp, err := openAIAdapter{}.ParseResponse([]byte(llamaServerBody))
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	want := Timings{CacheN: 0, PromptN: 13, PromptMS: 250.368, PromptPerTokenMS: 19.259, PromptPerSecond: 51.92,
		PredictedN: 4, PredictedMS: 165.636, PredictedPerTokenMS: 55.211, PredictedPerSecond: 18.11}
	if comp.Timings == nil || *comp.Timings != want {
		t.Fatalf("Timings = %+v, want %+v", comp.Timings, want)
	}

	plain, err := openAIAdapter{}.ParseResponse([]byte(`{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	if err != nil {
		t.Fatalf("ParseResponse without timings: %v", err)
	}
	if plain.Timings != nil {
		t.Fatalf("a provider without timings must leave Timings nil, got %+v", plain.Timings)
	}
}

func TestProxiedLlamaServerStreamTimingsParsed(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}],"object":"chat.completion.chunk"}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[{"finish_reason":"length","index":0,"delta":{}}],"object":"chat.completion.chunk"}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[],"object":"chat.completion.chunk","usage":{"completion_tokens":3,"prompt_tokens":13,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":9}},`+
			`"timings":{"cache_n":9,"prompt_n":4,"prompt_ms":123.315,"predicted_n":3,"predicted_ms":111.079}}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer up.Close()

	comp, err := NewHTTPPlanner(up.URL+"/v1", "qwen", "").CompleteStream(context.Background(), nil, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	want := Timings{CacheN: 9, PromptN: 4, PromptMS: 123.315, PredictedN: 3, PredictedMS: 111.079}
	if comp.Timings == nil || *comp.Timings != want {
		t.Fatalf("stream Timings = %+v, want %+v", comp.Timings, want)
	}
}

// fak-test:runtime fast est=100ms
func TestProxiedMalformedTimingsPreserveCompletion(t *testing.T) {
	raw := strings.Replace(llamaServerBody, `"predicted_ms":165.636`, `"predicted_ms":"invalid"`, 1)
	check := func(t *testing.T, comp *Completion, err error, wantTimings *Timings) {
		t.Helper()
		if err != nil {
			t.Fatalf("completion rejected for optional timings: %v", err)
		}
		if comp == nil {
			t.Fatal("completion is nil")
		}
		if comp.Message.Role != RoleAssistant || comp.Message.Content != "Hi" ||
			comp.FinishReason != "length" || comp.Model != "Qwen3.8-27B-UD-Q2_K_XL" {
			t.Errorf("completion fields lost: %+v", comp)
		}
		if comp.Usage.PromptTokens != 13 || comp.Usage.CompletionTokens != 4 ||
			comp.Usage.TotalTokens != 17 || comp.Usage.PromptTokensDetails == nil ||
			comp.Usage.PromptTokensDetails.CachedTokens != 0 {
			t.Errorf("usage lost: %+v", comp.Usage)
		}
		if wantTimings == nil {
			if comp.Timings != nil {
				t.Errorf("malformed timings must be unmeasured, got %+v", comp.Timings)
			}
		} else if comp.Timings == nil || *comp.Timings != *wantTimings {
			t.Errorf("prior valid timings lost: got %+v, want %+v", comp.Timings, wantTimings)
		}
	}

	t.Run("buffered", func(t *testing.T) {
		comp, err := openAIAdapter{}.ParseResponse([]byte(raw))
		check(t, comp, err, nil)
	})
	t.Run("stream", func(t *testing.T) {
		// Put content, usage and finish on the malformed-timings chunk: none may
		// disappear when only its optional metadata fails to decode.
		chunk := strings.Replace(raw, `"message":`, `"delta":`, 1)
		chunk = strings.Replace(chunk, `"chat.completion"`, `"chat.completion.chunk"`, 1)
		up, _ := sseServer(t, "data: "+chunk+"\n\ndata: [DONE]\n\n")
		var got strings.Builder
		comp, err := NewHTTPPlanner(up.URL+"/v1", "qwen", "").CompleteStream(context.Background(), func(s string) error {
			got.WriteString(s)
			return nil
		}, []Message{{Role: RoleUser, Content: "hi"}}, nil)
		check(t, comp, err, nil)
		if got.String() != "Hi" {
			t.Errorf("sink content = %q, want Hi", got.String())
		}
	})

	t.Run("valid_then_malformed_stream", func(t *testing.T) {
		valid := `{"choices":[],"timings":{"prompt_ms":250.368,"predicted_ms":165.636}}`
		chunk := strings.Replace(raw, `"message":`, `"delta":`, 1)
		chunk = strings.Replace(chunk, `"chat.completion"`, `"chat.completion.chunk"`, 1)
		up, _ := sseServer(t, "data: "+valid+"\n\ndata: "+chunk+"\n\ndata: [DONE]\n\n")
		var got strings.Builder
		comp, err := NewHTTPPlanner(up.URL+"/v1", "qwen", "").CompleteStream(context.Background(), func(s string) error {
			got.WriteString(s)
			return nil
		}, []Message{{Role: RoleUser, Content: "hi"}}, nil)
		check(t, comp, err, &Timings{PromptMS: 250.368, PredictedMS: 165.636})
		if got.String() != "Hi" {
			t.Errorf("sink content = %q, want Hi", got.String())
		}
	})
}
