package qwen38quantrun

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestRunAllFixturesAndRepetitions(t *testing.T) {
	c := qwen38quant.DefaultCorpus()
	calls := 0
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "exact"}}})
			return
		}
		calls++
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Error("missing auth")
		}
		var in map[string]any
		json.NewDecoder(r.Body).Decode(&in)
		messages := in["messages"].([]any)
		prompt := messages[0].(map[string]any)["content"].(string)
		_ = prompt
		msg := map[string]any{"content": "ok"}
		switch {
		case calls <= 3:
			msg["content"] = "ok"
		case calls <= 6:
			msg["content"] = `{"ok":true}`
		case calls <= 9:
			msg["content"] = ""
			msg["tool_calls"] = []any{map[string]any{"function": map[string]any{"name": "x", "arguments": `null`}}}
		default:
			msg["content"] = "ok"
		}
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": msg}}, "usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 2}})
	}))
	defer s.Close()
	got, err := (Runner{}).Run(context.Background(), Config{Endpoint: s.URL, APIKey: "secret", Model: "exact"}, c)
	if err != nil {
		t.Fatal(err)
	}
	if calls != len(c.Fixtures)*c.MinimumRepetitions || len(got) != calls {
		t.Fatalf("calls=%d results=%d", calls, len(got))
	}
	for _, r := range got {
		if r.Quality != "PASS" {
			t.Fatalf("%s: %s", r.FixtureID, r.Failure)
		}
	}
}
func TestFailureRetained(t *testing.T) {
	c := qwen38quant.DefaultCorpus()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "exact"}}})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"model": "exact", "choices": []any{map[string]any{"message": map[string]any{"content": "wrong"}}}})
	}))
	defer s.Close()
	got, err := (Runner{}).Run(context.Background(), Config{Endpoint: s.URL, APIKey: "x", Model: "exact"}, c)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Quality != "FAIL" || got[0].Failure == "" {
		t.Fatal("failure not retained")
	}
}
func TestRefusesTooFewRepetitions(t *testing.T) {
	_, err := (Runner{}).Run(context.Background(), Config{Endpoint: "x", APIKey: "x", Model: "x", Repetitions: 2}, qwen38quant.DefaultCorpus())
	if err == nil {
		t.Fatal("accepted under-repeated run")
	}
}
func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestModelIdentityMismatchDeniedBeforeMeasurement(t *testing.T) {
	chatCalls := 0
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "substitute"}}})
			return
		}
		chatCalls++
	}))
	defer s.Close()
	_, err := (Runner{}).Run(context.Background(), Config{Endpoint: s.URL, APIKey: "secret", Model: "exact"}, qwen38quant.DefaultCorpus())
	if err == nil || !contains(err.Error(), "exact model") || chatCalls != 0 {
		t.Fatalf("err=%v chatCalls=%d", err, chatCalls)
	}
}

func TestRunOneCarriesExplicitEffectContracts(t *testing.T) {
	tests := []struct {
		name    string
		fixture qwen38quant.Fixture
		assert  func(*testing.T, map[string]any)
	}{
		{
			name: "JSON schema",
			fixture: qwen38quant.Fixture{
				ID: "strict-json", Workload: "json_schema", Prompt: "return JSON", MaxOutputTokens: 32,
				ExpectedJSON: map[string]any{"ok": true, "count": float64(1)},
			},
			assert: func(t *testing.T, body map[string]any) {
				t.Helper()
				format, ok := body["response_format"].(map[string]any)
				if !ok || format["type"] != "json_schema" {
					t.Fatalf("response_format=%#v", body["response_format"])
				}
				wrapped := format["json_schema"].(map[string]any)
				if wrapped["name"] != "strict-json" || wrapped["strict"] != true {
					t.Fatalf("json_schema=%#v", wrapped)
				}
				schema := wrapped["schema"].(map[string]any)
				if schema["additionalProperties"] != false || !reflect.DeepEqual(schema["required"], []any{"count", "ok"}) {
					t.Fatalf("schema=%#v", schema)
				}
			},
		},
		{
			name: "required tool",
			fixture: qwen38quant.Fixture{
				ID: "required-tool", Workload: "correlated_tools", Prompt: "call it", MaxOutputTokens: 32,
				Tools: []map[string]any{{"type": "function", "function": map[string]any{"name": "lookup"}}},
			},
			assert: func(t *testing.T, body map[string]any) {
				t.Helper()
				if body["tool_choice"] != "required" || len(body["tools"].([]any)) != 1 {
					t.Fatalf("tool contract=%#v", body)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var captured map[string]any
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
					t.Fatal(err)
				}
				json.NewEncoder(w).Encode(map[string]any{"model": "exact", "choices": []any{map[string]any{"message": map[string]any{"content": "ok"}}}})
			}))
			defer s.Close()
			if _, err := runOne(context.Background(), s.Client(), Config{Endpoint: s.URL, APIKey: "secret", Model: "exact"}, tc.fixture); err != nil {
				t.Fatal(err)
			}
			tc.assert(t, captured)
		})
	}
}
func TestRunOneDecodesOpenAIUsageObject(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"model":   "exact",
			"choices": []any{map[string]any{"message": map[string]any{"content": "Q38"}}},
			"usage": map[string]any{
				"prompt_tokens": 41, "completion_tokens": 2, "total_tokens": 43,
				"prompt_tokens_details": map[string]any{"cached_tokens": 17},
			},
		})
	}))
	defer s.Close()

	got, err := runOne(context.Background(), s.Client(), Config{Endpoint: s.URL, APIKey: "secret", Model: "exact"}, qwen38quant.Fixture{ID: "usage", Workload: "text", Prompt: "reply Q38", MaxOutputTokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Usage, map[string]int{"prompt_tokens": 41, "completion_tokens": 2, "total_tokens": 43}) {
		t.Fatalf("usage = %#v", got.Usage)
	}
	if got.UsageDetails.CachedTokens != 17 {
		t.Fatalf("cached tokens = %d", got.UsageDetails.CachedTokens)
	}
}
