package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRunQualityProbeEmitsCapabilityReceipt(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "m"}}})
			return
		}
		if r.URL.Path == "/v1/completions" {
			json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"logprobs": nil}}})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": "A"}}}})
	}))
	defer server.Close()
	var stdout, stderr bytes.Buffer
	if code := runQualityProbe(&stdout, &stderr, []string{"--endpoint", server.URL, "--model", "m"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var receipt map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt["accuracy_evaluated"] != false {
		t.Fatalf("accuracy field: %#v", receipt)
	}
	if receipt["generation"].(map[string]any)["status"] != "supported" || receipt["prompt_logprobs"].(map[string]any)["status"] != "unsupported" {
		t.Fatalf("receipt: %#v", receipt)
	}
}

func TestRunQualityProbeRequiresExactInputs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runQualityProbe(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("code=%d", code)
	}
}
