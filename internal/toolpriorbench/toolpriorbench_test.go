package toolpriorbench

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRunCapturesRawCallsMetricsAndRejectsSemanticMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Tools []Tool `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		name := req.Tools[0].Function.Name
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"message": map[string]any{"tool_calls": []any{map[string]any{"function": map[string]any{"name": name, "arguments": map[string]any{"query": "registration_digest"}}}}}, "prompt_eval_count": 10, "eval_count": 3})
	}))
	defer srv.Close()
	l, err := Run(context.Background(), Config{Endpoint: srv.URL, Model: "test-model", Now: func() time.Time { return time.Date(2026, 8, 14, 0, 0, 0, 0, time.UTC) }})
	if err != nil {
		t.Fatal(err)
	}
	if l.Date != "2026-08-14" || l.SnapshotDigest == "" {
		t.Fatalf("metadata %#v", l)
	}
	if len(l.Trials) == 0 || len(l.Trials[0].Request) == 0 || len(l.Trials[0].Response) == 0 {
		t.Fatal("raw calls not captured")
	}
	found := false
	for _, c := range l.Compatibility {
		if c.Variant == "shell" && c.Verdict == "rejected_semantic_mismatch" {
			found = true
		}
	}
	if !found {
		t.Fatal("semantic mismatch not rejected")
	}
}
func TestSnapshotDigestStable(t *testing.T) {
	a, e := snapshotDigest(variants, schema)
	if e != nil {
		t.Fatal(e)
	}
	b, e := snapshotDigest(variants, schema)
	if e != nil || a != b {
		t.Fatalf("digest %q %q %v", a, b, e)
	}
}
