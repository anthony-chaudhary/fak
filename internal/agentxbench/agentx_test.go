package agentxbench_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

func TestQuantile(t *testing.T) {
	values := []float64{10.0, 20.0, 30.0, 40.0, 50.0}

	if got := agentxbench.Quantile(values, 0.0); got != 10.0 {
		t.Fatalf("p=0 got %v, want 10.0", got)
	}
	if got := agentxbench.Quantile(values, 0.5); got != 30.0 {
		t.Fatalf("p=0.5 got %v, want 30.0", got)
	}
	if got := agentxbench.Quantile(values, 1.0); got != 50.0 {
		t.Fatalf("p=1.0 got %v, want 50.0", got)
	}
	if got := agentxbench.Quantile(nil, 0.5); got != 0.0 {
		t.Fatalf("empty slice got %v, want 0.0", got)
	}
}

func TestComputeInteractivity(t *testing.T) {
	t0 := time.Now().UnixNano()
	timestamps := []int64{
		t0,
		t0 + 10*1e6, // 10ms
		t0 + 20*1e6, // 10ms
		t0 + 40*1e6, // 20ms
		t0 + 70*1e6, // 30ms
	}
	ttftMS := 50.0
	execMS := 120.0

	metrics := agentxbench.ComputeInteractivity(timestamps, ttftMS, execMS)

	if metrics.TTFTMS != 50.0 {
		t.Fatalf("expected TTFT 50.0, got %v", metrics.TTFTMS)
	}
	if metrics.ITLMedianMS < 9.0 || metrics.ITLMedianMS > 16.0 {
		t.Fatalf("unexpected ITLMedianMS: %v", metrics.ITLMedianMS)
	}
	if metrics.ITLMaxMS != 30.0 {
		t.Fatalf("expected ITLMaxMS 30.0, got %v", metrics.ITLMaxMS)
	}
	if metrics.NormalizedInteractivity <= 0.0 {
		t.Fatalf("expected positive NormalizedInteractivity, got %v", metrics.NormalizedInteractivity)
	}
}

func TestValidateReceiptInvariants(t *testing.T) {
	validReceipt := &agentxbench.AgentXReceipt{
		Schema:        agentxbench.SchemaIdentifier,
		BenchmarkID:   "bench-1",
		Model:         "Qwen3.8-27B-Q4_K_M",
		Engine:        "fak-inkernel-cuda",
		Hardware:      "NVIDIA A100-SXM4-40GB",
		AgentCount:    1,
		TurnsPerAgent: 1,
		Aggregated: agentxbench.AggregatedMetrics{
			TotalRequests: 1,
		},
		Requests: []agentxbench.RequestRecord{
			{
				RequestID:        "agent-1-t0",
				AgentID:          "agent-1",
				TurnIndex:        0,
				PromptTokens:     1024,
				CompletionTokens: 16,
				CachedTokens:     0,
				ClientPhases: agentxbench.ClientPhases{
					QueueWaitMS:      5.0,
					AgentExecutionMS: 200.0,
					EvaluationMS:     10.0,
					TotalLifecycleMS: 215.0,
				},
				Interactivity: agentxbench.InteractivityMetrics{
					TTFTMS:                  100.0,
					ITLMedianMS:             10.0,
					NormalizedInteractivity: 50.0,
				},
				TokenTimestampsUnixNano: []int64{100, 200, 300},
				Success:                 true,
			},
		},
	}

	errs := agentxbench.ValidateReceipt(validReceipt)
	if len(errs) != 0 {
		t.Fatalf("expected valid receipt to pass, got errs: %v", errs)
	}

	// 1. Nil receipt
	if errs := agentxbench.ValidateReceipt(nil); len(errs) == 0 {
		t.Fatalf("expected error on nil receipt")
	}

	// 2. Schema mismatch
	badSchema := *validReceipt
	badSchema.Schema = "bad.schema"
	if errs := agentxbench.ValidateReceipt(&badSchema); len(errs) == 0 {
		t.Fatalf("expected error on bad schema")
	}

	// 3. Non-monotonic timestamps
	badTimestamps := *validReceipt
	reqs := make([]agentxbench.RequestRecord, len(validReceipt.Requests))
	copy(reqs, validReceipt.Requests)
	reqs[0].TokenTimestampsUnixNano = []int64{300, 100, 200}
	badTimestamps.Requests = reqs
	if errs := agentxbench.ValidateReceipt(&badTimestamps); len(errs) == 0 {
		t.Fatalf("expected error on non-monotonic timestamps")
	}

	// 4. Cached tokens exceeding prompt tokens
	badTokens := *validReceipt
	reqs = make([]agentxbench.RequestRecord, len(validReceipt.Requests))
	copy(reqs, validReceipt.Requests)
	reqs[0].CachedTokens = 2048
	reqs[0].PromptTokens = 1024
	badTokens.Requests = reqs
	if errs := agentxbench.ValidateReceipt(&badTokens); len(errs) == 0 {
		t.Fatalf("expected error on cached tokens > prompt tokens")
	}
}

func TestRunnerMockExecution(t *testing.T) {
	runner := agentxbench.NewRunner(nil)
	cfg := agentxbench.Config{
		Model:             "Qwen3.8-27B-Q4_K_M",
		Engine:            "fak-inkernel-cuda",
		Hardware:          "GCP A100-SXM4-40GB",
		AgentCount:        3,
		TurnsPerAgent:     3,
		MaxTokens:         16,
		DeterministicSeed: 42,
		MockExecution:     true,
	}

	receipt, err := runner.Run(context.Background(), cfg)
	if err != nil {
		t.Fatalf("runner failed: %v", err)
	}

	if receipt.ValidationStatus != "VERIFIED_PASS" {
		t.Fatalf("expected VERIFIED_PASS, got %s: %v", receipt.ValidationStatus, receipt.ValidationErrors)
	}

	if receipt.Aggregated.TotalRequests != 9 {
		t.Fatalf("expected 9 requests, got %d", receipt.Aggregated.TotalRequests)
	}
	if receipt.Aggregated.SuccessfulRequests != 9 {
		t.Fatalf("expected 9 successful requests, got %d", receipt.Aggregated.SuccessfulRequests)
	}
	if receipt.Aggregated.PrefixSpeedupRatio <= 1.5 {
		t.Fatalf("expected PrefixSpeedupRatio > 1.5, got %v", receipt.Aggregated.PrefixSpeedupRatio)
	}
	if receipt.Aggregated.ClusterTokenThroughputPerSec <= 0.0 {
		t.Fatalf("expected positive cluster token throughput, got %v", receipt.Aggregated.ClusterTokenThroughputPerSec)
	}
}

func TestRunnerLiveHTTPStreaming(t *testing.T) {
	// Create mock OpenAI-compatible SSE streaming server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}

		tokens := []string{"Ready", ".", " Task", " completed", " successfully", "."}
		for _, tok := range tokens {
			chunk := fmt.Sprintf(`data: {"choices":[{"delta":{"content":%q}}]}`+"\n\n", tok)
			w.Write([]byte(chunk))
			flusher.Flush()
			time.Sleep(5 * time.Millisecond)
		}
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer server.Close()

	runner := agentxbench.NewRunner(server.Client())
	cfg := agentxbench.Config{
		EndpointURL:       server.URL,
		Model:             "Qwen3.8-27B-Q4_K_M",
		Engine:            "fak-inkernel-cuda",
		Hardware:          "GCP A100-SXM4-40GB",
		AgentCount:        2,
		TurnsPerAgent:     2,
		MaxTokens:         16,
		DeterministicSeed: 123,
		MockExecution:     false,
	}

	receipt, err := runner.Run(context.Background(), cfg)
	if err != nil {
		t.Fatalf("live runner failed: %v", err)
	}

	if receipt.ValidationStatus != "VERIFIED_PASS" {
		t.Fatalf("expected VERIFIED_PASS, got %s: %v", receipt.ValidationStatus, receipt.ValidationErrors)
	}

	if len(receipt.Requests) != 4 {
		t.Fatalf("expected 4 requests, got %d", len(receipt.Requests))
	}

	for _, req := range receipt.Requests {
		if !req.Success {
			t.Fatalf("request %s failed: %s", req.RequestID, req.Error)
		}
		if req.CompletionTokens != 6 {
			t.Fatalf("expected 6 completion tokens, got %d", req.CompletionTokens)
		}
		if req.Interactivity.TTFTMS <= 0 {
			t.Fatalf("expected positive TTFT, got %v", req.Interactivity.TTFTMS)
		}
	}
}
