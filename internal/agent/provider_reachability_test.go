package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHTTPPlannerProbeReachabilityDoesNotCreateModelTurn(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Method != http.MethodOptions || r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("probe request = %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusMethodNotAllowed)
	}))
	defer server.Close()
	planner := NewHTTPPlanner(server.URL+"/v1", "test-model", "secret")
	status, err := planner.ProbeReachability(context.Background())
	if err != nil || status != http.StatusMethodNotAllowed || requests != 1 {
		t.Fatalf("probe status=%d err=%v requests=%d", status, err, requests)
	}
}

func TestHTTPPlannerProbeReachabilityRejectsUpstreamFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "upstream down", http.StatusBadGateway)
	}))
	defer server.Close()
	planner := NewHTTPPlanner(server.URL+"/v1", "test-model", "")
	status, err := planner.ProbeReachability(context.Background())
	if status != http.StatusBadGateway || err == nil {
		t.Fatalf("probe status=%d err=%v, want typed failure", status, err)
	}
}
