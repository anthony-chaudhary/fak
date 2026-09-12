package agentbench

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestDiscoverNativeReference(t *testing.T) {
	wantRequest := referenceRequest{
		Messages:     []chatMessage{{Role: "system", Content: "stable"}, {Role: "user", Content: "repair"}},
		Tools:        []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"Read","parameters":{"type":"object"}}}`)},
		OutputTokens: 128,
	}
	var tokenizeRequests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model", "context_length": 32768, "fak_capabilities": map[string]any{"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"}}}}})
		case "/v1/fak/tokenize":
			var got referenceRequest
			if err := json.NewDecoder(r.Body).Decode(&got); err != nil || !reflect.DeepEqual(got, wantRequest) {
				http.Error(w, "request mismatch", http.StatusBadRequest)
				return
			}
			tokenizeRequests++
			_ = json.NewEncoder(w).Encode(referenceEncoding{ModelID: "fixture-model", RendererID: "renderer-v1", TokenizerID: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", TokenIDs: []int{7, 11, 13}, PromptTokens: 3, ContextWindowTokens: 32768, ReservedOutputTokens: 128, RenderedSHA256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	reference, err := discoverNativeReference(context.Background(), server.Client(), server.URL, "fixture-model")
	if err != nil {
		t.Fatal(err)
	}
	if reference.ModelID != "fixture-model" || reference.ContextWindowTokens != 32768 || reference.TokenizeURL != server.URL+"/v1/fak/tokenize" {
		t.Fatalf("discovery = %+v", reference)
	}
	first, err := reference.Encode(context.Background(), wantRequest)
	if err != nil {
		t.Fatal(err)
	}
	second, err := reference.Encode(context.Background(), wantRequest)
	if err != nil {
		t.Fatal(err)
	}
	if tokenizeRequests != 2 || first.PromptTokens != len(first.TokenIDs) || first.ContextWindowTokens != 32768 || first.ReservedOutputTokens != 128 || first.ModelID != reference.ModelID || first.RendererID == "" || first.TokenizerID == "" || first.RenderedSHA256 == "" {
		t.Fatalf("encoding receipt = %+v requests=%d", first, tokenizeRequests)
	}
	if first.RendererID != second.RendererID || first.TokenizerID != second.TokenizerID {
		t.Fatalf("request-independent identities changed: %+v %+v", first, second)
	}
	first.TokenIDs[0] = 999
	if second.TokenIDs[0] != 7 {
		t.Fatal("Encode returned aliased token IDs")
	}

	t.Run("fails closed", func(t *testing.T) {
		cases := []struct {
			name, models string
		}{
			{"missing model", `{"data":[{"id":"other","context_length":32768,"fak_capabilities":{"prompt_tokenization":{"endpoint":"/v1/fak/tokenize"}}}]}`},
			{"small context", `{"data":[{"id":"fixture-model","context_length":16384,"fak_capabilities":{"prompt_tokenization":{"endpoint":"/v1/fak/tokenize"}}}]}`},
			{"missing capability", `{"data":[{"id":"fixture-model","context_length":32768}]}`},
			{"cross origin", `{"data":[{"id":"fixture-model","context_length":32768,"fak_capabilities":{"prompt_tokenization":{"endpoint":"https://evil.example/tokenize"}}}]}`},
			{"malformed", `{`},
			{"oversize", `{"data":[]}` + strings.Repeat(" ", 1<<20)},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(tc.models)) }))
				defer s.Close()
				if _, err := discoverNativeReference(context.Background(), s.Client(), s.URL, "fixture-model"); err == nil {
					t.Fatal("invalid discovery metadata accepted")
				}
			})
		}
	})

	t.Run("encoding validation and cancellation", func(t *testing.T) {
		for _, response := range []string{
			`{"model_id":"other","renderer_id":"r","tokenizer_id":"t","token_ids":[1],"prompt_tokens":1,"context_window_tokens":32768,"reserved_output_tokens":128,"rendered_sha256":"d"}`,
			`{"model_id":"fixture-model","renderer_id":"","tokenizer_id":"","token_ids":[1],"prompt_tokens":2,"context_window_tokens":32768,"reserved_output_tokens":128,"rendered_sha256":""}`,
			`{"model_id":"fixture-model","renderer_id":"r","tokenizer_id":"t","token_ids":[1],"prompt_tokens":1,"context_window_tokens":32768,"reserved_output_tokens":40000,"rendered_sha256":"d"}`,
		} {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if strings.HasSuffix(r.URL.Path, "/models") {
					_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model", "context_length": 32768, "fak_capabilities": map[string]any{"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"}}}}})
					return
				}
				_, _ = w.Write([]byte(response))
			}))
			ref, err := discoverNativeReference(context.Background(), s.Client(), s.URL, "fixture-model")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ref.Encode(context.Background(), wantRequest); err == nil {
				t.Fatalf("invalid encoding accepted: %s", response)
			}
			s.Close()
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := discoverNativeReference(ctx, server.Client(), server.URL, "fixture-model"); err == nil {
			t.Fatal("canceled discovery succeeded")
		}
	})
}
