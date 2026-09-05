package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRunMacObs_JSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runMacObs(&stdout, &stderr, []string{"--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d. stderr: %s", rc, stderr.String())
	}

	var m map[string]interface{}
	if err := json.Unmarshal(stdout.Bytes(), &m); err != nil {
		t.Fatalf("failed to parse JSON output: %v, raw: %s", err, stdout.String())
	}

	if schema, ok := m["schema"].(string); !ok || schema != "fak.macobs.v1" {
		t.Errorf("expected schema 'fak.macobs.v1', got %v", m["schema"])
	}

	if _, ok := m["analysis"]; !ok {
		t.Errorf("expected analysis section in JSON output")
	}
}

func TestRunMacObs_CheckHeadroom(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runMacObs(&stdout, &stderr, []string{"--check-headroom", "--agents", "2"})
	// Admission check returns 0 on gate passed, 1 on gate failure. Either way stdout has admission line.
	out := stdout.String()
	if !strings.HasPrefix(out, "macobs: [") {
		t.Errorf("expected output to start with 'macobs: [', got %q", out)
	}
	if !strings.Contains(out, "recommended_agents=") {
		t.Errorf("expected output to contain 'recommended_agents=', got %q", out)
	}
	if !strings.Contains(out, "gate_passed=") {
		t.Errorf("expected output to contain 'gate_passed=', got %q", out)
	}
	if !strings.Contains(out, "bottleneck=") {
		t.Errorf("expected output to contain 'bottleneck=', got %q", out)
	}
	if strings.Contains(out, "gate_passed=true") && rc != 0 {
		t.Errorf("gate passed but got non-zero rc: %d", rc)
	}
}

func TestRunMacObs_Human(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runMacObs(&stdout, &stderr, []string{"--agents", "4", "--prefix-tokens", "4096", "--tail-tokens", "2048"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d. stderr: %s", rc, stderr.String())
	}

	out := stdout.String()
	for _, want := range []string{
		"fak macobs — Apple Silicon & MLX Agent Observability",
		"Hardware:",
		"MLX Serving:",
		"Agent Concurrency Headroom:",
		"Verdict & Action:",
		"Verdict: [",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("expected output to contain %q, got:\n%s", want, out)
		}
	}
}

func TestRunMacObs_InvalidFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want int
	}{
		{"unknown flag", []string{"--unknown-xyz"}, 2},
		{"extra positional", []string{"extra-arg"}, 2},
		{"json and check-headroom mutually exclusive", []string{"--json", "--check-headroom"}, 2},
		{"zero agents", []string{"--agents", "0"}, 2},
		{"negative agents", []string{"--agents", "-1"}, 2},
		{"zero prefix tokens", []string{"--prefix-tokens", "0"}, 2},
		{"zero tail tokens", []string{"--tail-tokens", "0"}, 2},
		{"zero interval", []string{"--interval", "0s"}, 2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			rc := runMacObs(&stdout, &stderr, tc.args)
			if rc != tc.want {
				t.Errorf("runMacObs(%v) = %d, want %d", tc.args, rc, tc.want)
			}
		})
	}
}

func TestRunMacObs_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runMacObs(&stdout, &stderr, []string{"--help"})
	if rc != 0 {
		t.Errorf("expected rc 0 for --help, got %d", rc)
	}
}

func TestRunMacObs_CustomEndpoint(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`
vllm:num_requests_running 3
vllm:num_requests_waiting 1
vllm:kv_cache_usage_perc 65.5
`))
	}))
	defer ts.Close()

	var stdout, stderr bytes.Buffer
	rc := runMacObs(&stdout, &stderr, []string{"--mlx-endpoint", ts.URL, "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d. stderr: %s", rc, stderr.String())
	}

	var m struct {
		MLXServing struct {
			Available      bool    `json:"available"`
			ActiveRequests int     `json:"active_requests"`
			QueuedRequests int     `json:"queued_requests"`
			KVCacheUsage   float64 `json:"kv_cache_usage_pct"`
		} `json:"mlx_serving"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &m); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if !m.MLXServing.Available {
		t.Errorf("expected MLX serving to be available")
	}
	if m.MLXServing.ActiveRequests != 3 {
		t.Errorf("expected 3 active requests, got %d", m.MLXServing.ActiveRequests)
	}
	if m.MLXServing.QueuedRequests != 1 {
		t.Errorf("expected 1 queued request, got %d", m.MLXServing.QueuedRequests)
	}
	if m.MLXServing.KVCacheUsage != 65.5 {
		t.Errorf("expected 65.5 kv cache usage, got %f", m.MLXServing.KVCacheUsage)
	}
}

func TestRunMacObs_MultiAgentScenario(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runMacObs(&stdout, &stderr, []string{
		"--agents", "8",
		"--prefix-tokens", "4096",
		"--tail-tokens", "2048",
		"--json",
	})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d. stderr: %s", rc, stderr.String())
	}

	var m struct {
		Schema   string `json:"schema"`
		Headroom struct {
			Available            bool    `json:"available"`
			SharedPrefixTokens   uint64  `json:"shared_prefix_tokens"`
			PrivateTailTokens    uint64  `json:"private_tail_tokens"`
			MaxSharedAgents      int     `json:"max_shared_agents"`
			MaxIsolatedAgents    int     `json:"max_isolated_agents"`
			ConcurrencyAdvantage float64 `json:"concurrency_advantage"`
		} `json:"headroom"`
		Analysis struct {
			Verdict           string `json:"verdict"`
			RecommendedAgents int    `json:"recommended_agents"`
			PrimaryBottleneck string `json:"primary_bottleneck"`
		} `json:"analysis"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &m); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if m.Schema != "fak.macobs.v1" {
		t.Errorf("expected schema 'fak.macobs.v1', got %s", m.Schema)
	}
	if m.Headroom.SharedPrefixTokens != 4096 {
		t.Errorf("expected 4096 shared prefix tokens, got %d", m.Headroom.SharedPrefixTokens)
	}
	if m.Headroom.PrivateTailTokens != 2048 {
		t.Errorf("expected 2048 private tail tokens, got %d", m.Headroom.PrivateTailTokens)
	}
	if m.Analysis.RecommendedAgents <= 0 {
		t.Errorf("expected positive recommended agents, got %d", m.Analysis.RecommendedAgents)
	}
	if m.Analysis.Verdict == "" {
		t.Errorf("expected non-empty verdict")
	}
}

func TestRunMacObs_CheckHeadroom_ExceededAdmissionGate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	// Requesting an absurdly high agent count (999999) must fail the admission gate with rc 1
	rc := runMacObs(&stdout, &stderr, []string{
		"--check-headroom",
		"--agents", "999999",
	})
	if rc != 1 {
		t.Fatalf("expected rc 1 for exceeded headroom, got %d", rc)
	}
	out := stdout.String()
	if !strings.Contains(out, "gate_passed=false") {
		t.Errorf("expected output to contain 'gate_passed=false', got: %s", out)
	}
	if !strings.HasPrefix(out, "macobs: [") {
		t.Errorf("expected output to start with 'macobs: [', got: %s", out)
	}
	if !strings.Contains(out, "recommended_agents=") {
		t.Errorf("expected output to contain 'recommended_agents=', got: %s", out)
	}
}
