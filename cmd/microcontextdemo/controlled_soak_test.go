package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestControlledSoakExercisesFaultsAndReconciles(t *testing.T) {
	var calls atomic.Int64
	var attemptsMu sync.Mutex
	attempts := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		id := req.Messages[len(req.Messages)-1].Content
		attemptsMu.Lock()
		attempts[id]++
		attempt := attempts[id]
		attemptsMu.Unlock()
		if id == "ctx-2" && attempt == 1 {
			http.Error(w, "controlled provider overload", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()
	r, err := run(context.Background(), config{Contexts: 10, Workers: 2, Endpoint: srv.URL, Model: "fixture", Provider: "test", Hardware: "test", ControlledSoak: true, RequestTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 12 || r.Completed != 10 || r.Failed != 0 || r.UsageResponses != 10 {
		t.Fatalf("accounting: calls=%d report=%+v", calls.Load(), r)
	}
	if r.RetryInjected != 1 || r.RetryRecovered != 1 || r.CancellationInjected != 1 || r.CancellationRecovered != 1 {
		t.Fatalf("fault evidence: %+v", r)
	}
	if r.ProviderFailures != 1 || r.ProviderRecovered != 1 || r.MaxAttempts != 3 {
		t.Fatalf("provider recovery evidence: %+v", r)
	}
	if r.CanaryContexts != 1 || r.CanaryPassed != 1 || r.QueuePeakContexts != 10 || r.HibernatedContexts != 8 || r.RestoredContexts != 8 || r.BaseRollbackCount != 1 {
		t.Fatalf("soak evidence: %+v", r)
	}
}

func TestVerifyTenThousandRequiresControlledSoakEvidence(t *testing.T) {
	r := report{Schema: "fak-microcontext-spine/1", Verdict: "PASS", LogicalShards: 10000, PhysicalWorkers: 8, PeakInFlight: 8, Completed: 10000, SharedBaseInstalls: 1, TurnCount: 10000, Mode: "openai-compatible", Provider: "p", Model: "m", Hardware: "h", Scope: "real", BaseFingerprint: "b", PromptTokens: 1, CompletionTokens: 1, UsageResponses: 10000, TTFTP50MS: 1, TTFTP95MS: 1, ElapsedMS: 1, PromptTokensPerSec: 1, DecodeTokensPerSec: 1, ResourceSamples: 2, ClientPeakRSSBytes: 1, ServerPeakRSSBytes: 1, ServerPeakHeapBytes: 1, EndpointPeakRequests: 1, KVCapacityEvidence: "bounded", QueueEvidence: "bounded", ResultCheck: "nonempty", VerifiedResultsPerSec: 1}
	if err := verifyReport(r); err == nil {
		t.Fatal("accepted 10k artifact without soak evidence")
	}
	r.SoakContract = "controlled-10k-v1"
	r.CanaryContexts = 1
	r.CanaryPassed = 1
	r.BaseRollbackCount = 1
	r.RetryInjected = 1
	r.RetryRecovered = 1
	r.CancellationInjected = 1
	r.CancellationRecovered = 1
	r.MaxAttempts = 3
	r.QueuePeakContexts = 10000
	r.HibernatedContexts = 9992
	r.RestoredContexts = 9992
	if err := verifyReport(r); err != nil {
		b, _ := json.Marshal(r)
		t.Fatalf("rejected controlled soak: %v\n%s", err, b)
	}
}
