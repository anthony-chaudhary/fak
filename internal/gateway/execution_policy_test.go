package gateway

import (
	"net/http/httptest"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestHealthzExecutionPolicyBlock(t *testing.T) {
	srv, err := New(Config{Model: "mock", EngineID: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var health map[string]any
	getJSON(t, ts.URL+"/healthz", &health)
	if _, ok := health["execution_policy"]; ok {
		t.Fatalf("default /healthz grew execution_policy: %v", health["execution_policy"])
	}

	srv.SetExecutionPolicy(&ExecutionPolicyReport{Policy: "bringup", Model: "m", Dedicated: true, Bypassed: []string{"cpu-expert-offload"}})
	health = nil
	getJSON(t, ts.URL+"/healthz", &health)
	ep, ok := health["execution_policy"].(map[string]any)
	if !ok || ep["policy"] != "bringup" || ep["model"] != "m" || ep["dedicated"] != true {
		t.Fatalf("execution_policy=%v", health["execution_policy"])
	}
	if b, _ := ep["bypassed"].([]any); len(b) != 1 || b[0] != "cpu-expert-offload" {
		t.Fatalf("bypassed=%v", ep["bypassed"])
	}
}
