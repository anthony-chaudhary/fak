package fleetaccounts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLaneModelExactAndCaseInsensitive(t *testing.T) {
	pol := DefaultPolicy()
	pol.LaneModels["docs"] = "claude-sonnet-5"
	pol.LaneModels["Gateway"] = " claude-opus-4-8 "

	if got := pol.LaneModel("docs"); got != "claude-sonnet-5" {
		t.Fatalf("exact lane pin: got %q", got)
	}
	// Case-insensitive fallback + trimming.
	if got := pol.LaneModel("gateway"); got != "claude-opus-4-8" {
		t.Fatalf("case-insensitive lane pin: got %q", got)
	}
	// A blank or unpinned lane resolves to no override.
	if got := pol.LaneModel("  "); got != "" {
		t.Fatalf("blank lane should have no pin: got %q", got)
	}
	if got := pol.LaneModel("relay"); got != "" {
		t.Fatalf("unpinned lane should have no pin: got %q", got)
	}
}

func TestProfileModelFallsBackToDefault(t *testing.T) {
	pol := DefaultPolicy()
	// A worker claude row with no explicit profile override resolves the claude-opus default.
	row := Account{Account: ".claude-gem8", Tag: "gem8", Product: "claude", Kind: KindWorker}
	if got := pol.ProfileModel(row); got != "opus" {
		t.Fatalf("default claude profile model: got %q, want opus", got)
	}
	// An operator override on the tag wins.
	pol.AccountProfiles["gem8"] = ProfileOverride{ModelTier: 1, Model: "claude-fable-5"}
	if got := pol.ProfileModel(row); got != "claude-fable-5" {
		t.Fatalf("overridden profile model: got %q", got)
	}
}

func TestLoadPolicyParsesLaneModels(t *testing.T) {
	dir := t.TempDir()
	policyPath := filepath.Join(dir, "accounts_policy.json")
	blob := `{"lane_models": {"docs": "claude-sonnet-5", "gateway": "claude-opus-4-8"}}`
	if err := os.WriteFile(policyPath, []byte(blob), 0o644); err != nil {
		t.Fatal(err)
	}
	pol := LoadPolicy(Paths{PolicyPath: policyPath})
	if got := pol.LaneModel("docs"); got != "claude-sonnet-5" {
		t.Fatalf("parsed docs lane model: got %q", got)
	}
	if got := pol.LaneModel("gateway"); got != "claude-opus-4-8" {
		t.Fatalf("parsed gateway lane model: got %q", got)
	}
	// The defaults still backstop the other keys (exclude preserved).
	if len(pol.Exclude) == 0 {
		t.Fatalf("default exclude list should backstop a partial policy")
	}
}
