package qwen38quantrun

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestRunAllFixturesAndRepetitions(t *testing.T) {
	c := qwen38quant.DefaultCorpus()
	calls := 0
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "wrong"}}}})
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
