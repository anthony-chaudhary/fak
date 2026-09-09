package amdgpu

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestStrixHILReceipt_ValidateAndDigest(t *testing.T) {
	target := StrixTarget{
		Host:         "strix-test",
		Mode:         "ssh",
		Reachable:    true,
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		GPUName:      "AMD Radeon 8060S Graphics",
	}

	receipt := StrixHILReceipt{
		Schema:         StrixHILReceiptSchema,
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
		Verdict:        "PASS",
		Verified:       true,
		Target:         target,
		SubkernelCount: 19,
		AblationCount:  2,
		InferenceWitness: &StrixHILInferenceWitness{
			Endpoint:         "http://127.0.0.1:8080/v1/chat/completions",
			Model:            "Qwen3.8-27B-Q4_K_M",
			Prompt:           "ping",
			Response:         "pong",
			PromptTokens:     10,
			CompletionTokens: 2,
			TotalTokens:      12,
			LatencyMS:        45.2,
			TokensPerSec:     44.2,
			Verified:         true,
		},
	}

	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatalf("compute digest: %v", err)
	}
	if !strings.HasPrefix(digest, "sha256:") {
		t.Fatalf("invalid digest prefix: %s", digest)
	}
	receipt.Digest = digest

	if err := receipt.Validate(); err != nil {
		t.Fatalf("validate receipt: %v", err)
	}

	// Mismatched digest fails validation
	receipt.Digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
	if err := receipt.Validate(); err == nil {
		t.Fatal("expected failure on mismatched digest, got nil")
	}

	// Unreachable target fails validation
	receipt.Target.Reachable = false
	receipt.Digest, _ = receipt.ComputeDigest()
	if err := receipt.Validate(); err == nil {
		t.Fatal("expected failure on unreachable target, got nil")
	}
}

func TestExecuteStrixInferenceCheck_MockServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		auth := r.Header.Get("Authorization")
		if !strings.Contains(auth, "test-key") {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		resp := map[string]any{
			"id":    "chatcmpl-test",
			"model": "Qwen3.8-27B-Q4_K_M",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": "4",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     10,
				"completion_tokens": 1,
				"total_tokens":      11,
			},
		}
		data, _ := json.Marshal(resp)
		w.Header().Set("Content-Type", "application/json")
		w.Write(data)
	}))
	defer server.Close()

	addr := strings.TrimPrefix(server.URL, "http://")
	target := &StrixTarget{
		Host:      addr,
		Mode:      "ssh",
		Reachable: true,
	}

	witness := ExecuteStrixInferenceCheck(context.Background(), target, "Qwen3.8-27B-Q4_K_M", "2+2", "test-key")
	if !witness.Verified {
		t.Fatalf("expected verified inference witness, got error: %s", witness.Error)
	}
	if witness.Response != "4" {
		t.Errorf("response = %q, want '4'", witness.Response)
	}
	if witness.TotalTokens != 11 {
		t.Errorf("total tokens = %d, want 11", witness.TotalTokens)
	}
	if witness.TokensPerSec <= 0 {
		t.Errorf("tokens per sec = %f, want > 0", witness.TokensPerSec)
	}
}

func TestStrixHIL_SubkernelCoverage(t *testing.T) {
	// Verify that DefaultCreditableSubkernelSelectors covers at least 18 subkernels
	if len(DefaultCreditableSubkernelSelectors) < 18 {
		t.Fatalf("expected at least 18 creditable subkernels, got %d", len(DefaultCreditableSubkernelSelectors))
	}

	for _, selector := range DefaultCreditableSubkernelSelectors {
		contract, ok := LookupSubkernelParityContract(selector)
		if !ok {
			t.Errorf("missing contract for creditable selector %q", selector)
			continue
		}
		if !contract.DeviceObserved {
			t.Errorf("creditable selector %q must have DeviceObserved = true", selector)
		}
		if contract.Engine != StrixVulkanEngine {
			t.Errorf("creditable selector %q must use engine %q, got %q", selector, StrixVulkanEngine, contract.Engine)
		}
	}
}
