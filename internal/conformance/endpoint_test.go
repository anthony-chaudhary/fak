package conformance

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func endpointFixture(content string, usage bool) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"ok": true, "engine": "fixture", "model": "same-model"})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "same-model", "owned_by": "fak"}}})
	})
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Params struct {
				Arguments struct {
					Tool    string `json:"tool"`
					TraceID string `json:"trace_id"`
				} `json:"arguments"`
			} `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		kind, reason, disposition := "ALLOW", "", ""
		if req.Params.Arguments.Tool == "refund_payment" {
			kind, reason, disposition = "DENY", "DEFAULT_DENY", "TERMINAL"
		}
		text, _ := json.Marshal(map[string]any{"verdict": map[string]any{"kind": kind, "reason": reason, "by": "monitor", "disposition": disposition}, "trace_id": req.Params.Arguments.TraceID})
		json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "result": map[string]any{"content": []map[string]any{{"type": "text", "text": string(text)}}, "isError": false}})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		v := map[string]any{"id": "deployment-specific-id", "object": "chat.completion", "choices": []map[string]any{{"finish_reason": "stop", "message": map[string]any{"content": content}}}}
		if usage {
			v["usage"] = map[string]any{"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
		}
		json.NewEncoder(w).Encode(v)
	})
	return httptest.NewServer(mux)
}

func TestProbeEndpointPairPassesAndKeepsUnsupportedFieldsNotYet(t *testing.T) {
	a, b := endpointFixture("FAK_ENDPOINT_CONFORMANCE_PASS", true), endpointFixture("FAK_ENDPOINT_CONFORMANCE_PASS", true)
	defer a.Close()
	defer b.Close()
	p := ProbeEndpointPair(context.Background(), a.Client(), a.URL, b.URL)
	if p.Verdict != "PASS" {
		t.Fatalf("packet = %+v", p)
	}
	if p.Endpoints[0].Admission.Status != "PASS" || p.Endpoints[0].Quota.Status != "NOT_YET" {
		t.Fatalf("coverage = %+v", p.Endpoints[0])
	}
	if p.Endpoints[0].Usage.Status != "PASS" || p.Endpoints[0].Receipt.Status != "PASS" {
		t.Fatalf("observed coverage missing: %+v", p.Endpoints[0])
	}
}

func TestProbeEndpointPairFailsSemanticDrift(t *testing.T) {
	a, b := endpointFixture("same", true), endpointFixture("different", true)
	defer a.Close()
	defer b.Close()
	p := ProbeEndpointPair(context.Background(), a.Client(), a.URL, b.URL)
	if p.Verdict != "FAIL" || p.Reason == "" {
		t.Fatalf("packet = %+v", p)
	}
}

func TestProbeEndpointPairFailsAdmissionSemanticDrift(t *testing.T) {
	a := endpointFixture("same", true)
	defer a.Close()
	// A second fixture with the same transport task but no portable adjudication path
	// must not be treated as parity merely because both chat completions succeed.
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"ok": true, "engine": "fixture", "model": "same-model"})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "same-model", "owned_by": "fak"}}})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"id": "id", "object": "chat.completion", "choices": []map[string]any{{"finish_reason": "stop", "message": map[string]any{"content": "same"}}}, "usage": map[string]any{"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}})
	})
	missing := httptest.NewServer(mux)
	defer missing.Close()
	p := ProbeEndpointPair(context.Background(), a.Client(), a.URL, missing.URL)
	if p.Verdict != "FAIL" || !strings.Contains(p.Reason, "admission status drift") {
		t.Fatalf("packet = %+v", p)
	}
}

func TestProbeEndpointPairDoesNotTreatMissingUsageAsZero(t *testing.T) {
	a, b := endpointFixture("same", false), endpointFixture("same", false)
	defer a.Close()
	defer b.Close()
	p := ProbeEndpointPair(context.Background(), a.Client(), a.URL, b.URL)
	if p.Verdict != "PASS" || p.Endpoints[0].Usage.Status != "NOT_YET" || p.Endpoints[0].Usage.Observed != nil {
		t.Fatalf("packet = %+v", p)
	}
}
