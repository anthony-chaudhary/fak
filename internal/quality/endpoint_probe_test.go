package quality

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeEndpointClassifiesCapabilitiesIndependently(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "exact-model"}}})
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if r.URL.Path == "/v1/completions" {
			json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "B", "logprobs": nil}}})
			return
		}
		if _, reasoning := body["reasoning_effort"]; reasoning {
			http.Error(w, "unsupported parameter", http.StatusBadRequest)
			return
		}
		w.Header().Set("X-Fak-Engine", "fak-native")
		w.Header().Set("X-Fak-Fallbacks", "0")
		json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": "A"}}}})
	}))
	defer server.Close()

	report := ProbeEndpoint(context.Background(), server.Client(), server.URL, "exact-model")
	if report.Models.Status != probeOK || report.Generation.Status != probeOK {
		t.Fatalf("supported probes: %+v", report)
	}
	if report.PromptLogprobs.Status != probeNo {
		t.Fatalf("prompt logprobs = %q", report.PromptLogprobs.Status)
	}
	if report.Reasoning.Status != probeNo {
		t.Fatalf("reasoning = %q", report.Reasoning.Status)
	}
	if !report.Native || report.Engine != "fak-native" || report.Fallbacks != 0 {
		t.Fatalf("provenance: %+v", report)
	}
	if report.AccuracyEvaluated {
		t.Fatal("probe must not report accuracy")
	}
}

func TestProbeEndpointMissingModelIsInfrastructureError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []any{}})
	}))
	defer server.Close()
	report := ProbeEndpoint(context.Background(), server.Client(), server.URL, "missing")
	for name, probe := range map[string]ProbeResult{"models": report.Models, "generation": report.Generation, "logprobs": report.PromptLogprobs, "reasoning": report.Reasoning} {
		if probe.Status != probeInfra {
			t.Fatalf("%s = %q", name, probe.Status)
		}
	}
}

func TestProbeEndpointUnreachableIsInfrastructureError(t *testing.T) {
	report := ProbeEndpoint(context.Background(), &http.Client{}, "http://127.0.0.1:1", "model")
	if report.Models.Status != probeInfra || report.Generation.Status != probeInfra {
		t.Fatalf("report: %+v", report)
	}
}

func TestProbeEndpointFallbackPreventsNativeClaim(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Fak-Engine", "fak-native")
		w.Header().Set("X-Fak-Fallbacks", "1")
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "m"}}})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"logprobs": map[string]any{"tokens": []string{"A"}}}}})
	}))
	defer server.Close()
	report := ProbeEndpoint(context.Background(), server.Client(), server.URL, "m")
	if report.Native {
		t.Fatalf("fallback receipt claimed native: %+v", report)
	}
}
