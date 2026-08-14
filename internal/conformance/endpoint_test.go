package conformance

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
	if p.Endpoints[0].Admission.Status != "NOT_YET" || p.Endpoints[0].Quota.Status != "NOT_YET" {
		t.Fatalf("unsupported coverage inferred: %+v", p.Endpoints[0])
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

func TestProbeEndpointPairDoesNotTreatMissingUsageAsZero(t *testing.T) {
	a, b := endpointFixture("same", false), endpointFixture("same", false)
	defer a.Close()
	defer b.Close()
	p := ProbeEndpointPair(context.Background(), a.Client(), a.URL, b.URL)
	if p.Verdict != "PASS" || p.Endpoints[0].Usage.Status != "NOT_YET" || p.Endpoints[0].Usage.Observed != nil {
		t.Fatalf("packet = %+v", p)
	}
}
