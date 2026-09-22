package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestRouterProviderSpeaksResponsesWire drives the REAL planner against a real
// httptest server: "router" must resolve to the Responses provider and emit the
// /responses path, bearer auth, and an input-carrying JSON body, then parse a
// Responses-shaped completion.
func TestRouterProviderSpeaksResponsesWire(t *testing.T) {
	if pv, ok := ParseProvider("router"); !ok || pv != ProviderOpenAIResponses {
		t.Fatalf("ParseProvider(router) = (%q, %v), want (%q, true)", pv, ok, ProviderOpenAIResponses)
	}

	var gotPath, gotAuth string
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("router", ts.URL, "deepseek-v4-flash", "sekret")
	if err != nil {
		t.Fatalf("NewProviderHTTPPlanner: %v", err)
	}
	comp, err := planner.Complete(context.Background(), adapterTestMessages(""), nil)
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if gotPath != "/responses" {
		t.Fatalf("request path = %q, want /responses", gotPath)
	}
	if gotAuth != "Bearer sekret" {
		t.Fatalf("Authorization = %q, want Bearer sekret", gotAuth)
	}
	if gotBody["model"] != "deepseek-v4-flash" {
		t.Fatalf("body model = %v, want deepseek-v4-flash", gotBody["model"])
	}
	if _, ok := gotBody["input"]; !ok {
		t.Fatalf("body missing input: %v", gotBody)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("completion content = %q, want ok", comp.Message.Content)
	}
}
