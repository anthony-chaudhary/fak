package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelops"
)

func TestRunModelCanaryGateEmitsRollbackAndExitThree(t *testing.T) {
	in := topThreeModelopsCLIInput()
	in.Observations[0].SuccessRate = .5
	path := filepath.Join(t.TempDir(), "input.json")
	data, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runModelCanaryGate(&stdout, &stderr, []string{"--input", path}); code != 3 {
		t.Fatalf("exit = %d stderr=%s output=%s, want 3", code, stderr.String(), stdout.String())
	}
	var got modelops.Decision
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Action != modelops.Rollback || got.Selected != "claude-sonnet-4-6" {
		t.Fatalf("decision = %+v, want rollback to Sonnet", got)
	}
}

func TestRunModelCanaryGateRejectsUnknownJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input.json")
	if err := os.WriteFile(path, []byte(`{"candidate":"x","unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runModelCanaryGate(&stdout, &stderr, []string{"--input", path}); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
}

func topThreeModelopsCLIInput() modelops.Input {
	policy := func(model string, tier int, fallback ...string) modelops.Policy {
		return modelops.Policy{Model: model, CapabilityTier: tier, Fallbacks: fallback,
			Alert: modelops.AlertContract{Owner: "model-ops", Route: "ops", AckSLAMinutes: 15, Runbook: "runbooks/model-canary"}, WindowMinutes: 60, MinSamples: 20,
			MinSuccessRate: .95, MaxProviderErrorRate: .02, MaxInvalidToolRate: .01,
			MaxP95LatencyMS: 5000, MaxThrottleRate: .03, MaxFallbackRate: .05}
	}
	healthy := func(model string) modelops.Observation {
		return modelops.Observation{Model: model, Samples: 30, SuccessRate: .98, ProviderErrorRate: .01,
			P95LatencyMS: 4000, ThrottleRate: .01, FallbackRate: .02}
	}
	return modelops.Input{Candidate: "claude-opus-4-8", RequiredTier: 1,
		Policies:     []modelops.Policy{policy("claude-opus-4-8", 0, "claude-sonnet-4-6"), policy("claude-sonnet-4-6", 1, "claude-haiku-4-5-20251001"), policy("claude-haiku-4-5-20251001", 2)},
		Observations: []modelops.Observation{healthy("claude-opus-4-8"), healthy("claude-sonnet-4-6"), healthy("claude-haiku-4-5-20251001")}}
}

func TestRunModelCanaryGateRejectsTrailingJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input.json")
	if err := os.WriteFile(path, []byte(`{"candidate":"x"} {"candidate":"y"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runModelCanaryGate(&stdout, &stderr, []string{"--input", path}); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
}

func TestRunModelCanaryGateHelpMatchesModelUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runModelCanaryGate(&stdout, &stderr, []string{"--help"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stderr.Bytes(), []byte("usage: "+modelCanaryGateSynopsis)) {
		t.Fatalf("help missing synopsis %q: %s", modelCanaryGateSynopsis, stderr.String())
	}
	if !bytes.Contains([]byte(modelUsage), []byte(modelCanaryGateSynopsis)) {
		t.Fatalf("parent usage missing synopsis %q: %s", modelCanaryGateSynopsis, modelUsage)
	}
}
