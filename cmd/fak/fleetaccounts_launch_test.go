package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func TestFleetLaunchLedgerCapturesDecisionWithoutSecrets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "launches.jsonl")
	d := fleetaccounts.LaunchDecision{OK: true, Account: "codex-tier1", Product: "codex", ConfiguredModel: "gpt-5-codex", InvokedModel: "gpt-5-codex", EndpointClass: "subscription", TaskTier: 1, Env: map[string]string{"CODEX_HOME": "/secret/path"}}
	if err := appendFleetLaunchLedger(path, d); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, want := range []string{`"resolved_account":"codex-tier1"`, `"product":"codex"`, `"configured_model":"gpt-5-codex"`, `"invoked_model":"gpt-5-codex"`, `"endpoint_class":"subscription"`, `"task_tier":1`} {
		if !strings.Contains(text, want) {
			t.Errorf("ledger %s missing %s", text, want)
		}
	}
	if strings.Contains(text, "/secret/path") || strings.Contains(text, "CODEX_HOME") {
		t.Fatalf("ledger leaked launch environment: %s", text)
	}
	var row map[string]any
	if err := json.Unmarshal(raw, &row); err != nil {
		t.Fatal(err)
	}
}
