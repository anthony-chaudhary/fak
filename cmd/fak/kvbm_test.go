package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunKVBMReplayArtifactCheck(t *testing.T) {
	artifact := filepath.FromSlash("../../internal/compute/testdata/kvbm_agent_replay_issue2666.json")
	var stdout, stderr bytes.Buffer
	code := runKVBM(&stdout, &stderr, []string{"replay", "--artifact", artifact, "--check"})
	if code != 0 {
		t.Fatalf("runKVBM replay exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"kvbm replay: issue2666-agent-session-hot-prefix-pin-restore",
		"cost-aware: hits=150/450",
		"lru:        hits=100/450",
		"oracle:    hits=150/450 exact=true",
		"oracle_bounds=true",
		"pins_safe=true",
		"restore_bytes_stable=true",
		"verdict: PASS",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("kvbm replay output missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestRunKVBMReplayJSON(t *testing.T) {
	artifact := filepath.FromSlash("../../internal/compute/testdata/kvbm_agent_replay_issue2666.json")
	var stdout, stderr bytes.Buffer
	code := runKVBM(&stdout, &stderr, []string{"replay", "--artifact", artifact, "--json"})
	if code != 0 {
		t.Fatalf("runKVBM replay --json exit=%d stderr=%s", code, stderr.String())
	}
	var payload struct {
		OK     bool `json:"ok"`
		Checks struct {
			CostAwareAtLeastLRU  bool `json:"cost_aware_at_least_lru"`
			OracleBoundsPolicies bool `json:"oracle_bounds_policies"`
			PinPressureExercised bool `json:"pin_pressure_exercised"`
			PinsSafe             bool `json:"pins_safe"`
			RestoreExercised     bool `json:"restore_exercised"`
			RestoreBytesStable   bool `json:"restore_bytes_stable"`
		} `json:"checks"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("kvbm replay JSON did not parse: %v\n%s", err, stdout.String())
	}
	if !payload.OK || !payload.Checks.CostAwareAtLeastLRU || !payload.Checks.OracleBoundsPolicies ||
		!payload.Checks.PinPressureExercised || !payload.Checks.PinsSafe ||
		!payload.Checks.RestoreExercised || !payload.Checks.RestoreBytesStable {
		t.Fatalf("kvbm replay JSON did not carry passing checks: %+v", payload)
	}
}

func TestRunKVBMReplayCheckFailsOnWeakArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weak.json")
	if err := os.WriteFile(path, []byte(`{
		"schema":"fak.kvbm.replay/v1",
		"name":"weak",
		"budget_bytes":100,
		"events":[
			{"span_id":"a","tokens":50,"payload":"a"},
			{"span_id":"b","tokens":50,"payload":"b"}
		]
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runKVBM(&stdout, &stderr, []string{"replay", "--artifact", path, "--check"})
	if code != 1 {
		t.Fatalf("weak artifact check exit=%d, want 1; stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "verdict: FAIL") {
		t.Fatalf("weak artifact output missing failure verdict:\n%s", stdout.String())
	}
}
