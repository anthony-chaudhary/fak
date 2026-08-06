package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
)

func TestPrefixABRunsHonestThreeArmComparison(t *testing.T) {
	var mu sync.Mutex
	systems := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/batches" {
			http.NotFound(w, r)
			return
		}
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		mu.Lock()
		systems = append(systems, request.Messages[0].Content)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()
	r, err := runAB(context.Background(), config{Contexts: 4, Workers: 2, Endpoint: server.URL, Model: "test", Provider: "test", Hardware: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Arms) != 3 || r.ClaimVerdict != "not-yet" || r.NativeBatch.Supported {
		t.Fatalf("report=%+v", r)
	}
	if r.Arms[2].PhysicalWorkers != 1 {
		t.Fatalf("sequential arm=%+v", r.Arms[2])
	}
	mu.Lock()
	defer mu.Unlock()
	unique := map[string]bool{}
	shared := 0
	for _, system := range systems {
		if !strings.Contains(system, "00000000") {
			unique[system] = true
		} else {
			shared++
		}
	}
	if len(unique) != 4 || shared != 8 {
		t.Fatalf("unique=%d shared=%d total=%d", len(unique), shared, len(systems))
	}
}

func TestVerifyPrefixABArtifact(t *testing.T) {
	r := abReport{Schema: "fak-microcontext-prefix-ab/1", Verdict: "PASS", ShardsPerArm: 2, BaseFingerprint: "base", ReuseEvidence: "none", ClaimVerdict: "not-yet", Scope: "test"}
	for _, name := range []string{"unique", "shared", "sequential"} {
		r.Arms = append(r.Arms, abArm{Name: name, Completed: 2, PromptTokens: 2, TTFTP50MS: 1})
	}
	data, _ := json.Marshal(r)
	path := t.TempDir() + "/ab.json"
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := verifyABArtifact(path); err != nil {
		t.Fatal(err)
	}
}
