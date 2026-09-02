package issue9714witness

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// definitionReceipt is the read-only observation captured by the definition
// deliverable's own surface (`fak model incumbent preflight --json`): the
// reviewed `com.fak.qwen36-model` service-definition tooling must reproduce this
// issue's HOLD state deterministically, binding the same preserved identities
// the drill receipts recorded. It is an observation receipt only: no lifecycle
// operation was admitted, so there is no signal, bootout, or restoration field
// to tamper with — its honesty is exactly that it classifies the unchanged
// external state.
type definitionReceipt struct {
	Schema        string `json:"schema"`
	Issue         int    `json:"issue"`
	ObservedAt    string `json:"observed_at"`
	ReadOnly      bool   `json:"read_only"`
	LaunchdTarget string `json:"launchd_target"`
	ListenerPort  int    `json:"listener_port"`
	ExpectedOwner struct {
		ServiceLabel      string `json:"service_label"`
		LaunchdJobPresent bool   `json:"launchd_job_present"`
	} `json:"expected_owner"`
	Incumbent struct {
		ListenerPresent         bool   `json:"listener_present"`
		ListenerPID             int    `json:"listener_pid"`
		CommandSHA256           string `json:"command_sha256"`
		CommandMatchesPreserved bool   `json:"command_matches_preserved_identity"`
		OwnerLabelSHA256        string `json:"owner_label_sha256"`
		OwnerResolved           bool   `json:"owner_resolved"`
		HealthStatus            int    `json:"health_status"`
		ModelsStatus            int    `json:"models_status"`
		ModelAlias              string `json:"model_alias"`
		AliasMatches            bool   `json:"alias_matches"`
	} `json:"incumbent"`
	Verdict      string `json:"verdict"`
	Reason       string `json:"reason"`
	WitnessClass string `json:"witness_class"`
}

func TestDefinitionPreflightReceiptReproducesHoldState(t *testing.T) {
	b, err := os.ReadFile("definition-preflight-receipt.json")
	if err != nil {
		t.Fatal(err)
	}
	var r definitionReceipt
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if r.Schema != "fak.model-incumbent-preflight/1" || r.Issue != 9714 || !r.ReadOnly {
		t.Fatalf("receipt identity drift: %+v", r)
	}
	if !strings.Contains(r.LaunchdTarget, "com.fak.qwen36-model") || strings.Contains(r.LaunchdTarget, "gui/5") {
		t.Fatalf("launchd target must name the expected label with the uid scrubbed: %q", r.LaunchdTarget)
	}
	if r.Verdict != "EXPECTED_JOB_ABSENT" || r.Reason != "alternate_launchd_supervisor_owns_incumbent" {
		t.Fatalf("verdict drift: %s/%s", r.Verdict, r.Reason)
	}
	if r.ExpectedOwner.ServiceLabel != "com.fak.qwen36-model" || r.ExpectedOwner.LaunchdJobPresent {
		t.Fatalf("expected owner drift: %+v", r.ExpectedOwner)
	}
	inc := r.Incumbent
	if !inc.ListenerPresent || inc.ListenerPID <= 0 {
		t.Fatalf("listener observation drift: %+v", inc)
	}
	// The read-only observation must bind the same preserved identities the
	// drill receipts recorded: the incumbent command and the alternate owner.
	if inc.CommandSHA256 != "sha256:"+incumbentCommandSHA || !inc.CommandMatchesPreserved {
		t.Fatalf("command identity drift: %+v", inc)
	}
	if inc.OwnerLabelSHA256 != "sha256:"+alternateOwnerSHA || !inc.OwnerResolved {
		t.Fatalf("owner identity drift: %+v", inc)
	}
	if inc.HealthStatus != 200 || inc.ModelsStatus != 200 || inc.ModelAlias != "qwen3.6-27b" || !inc.AliasMatches {
		t.Fatalf("endpoint observation drift: %+v", inc)
	}
	if r.WitnessClass != "read_only_observation_no_mutation" {
		t.Fatalf("witness class drift: %q", r.WitnessClass)
	}
}

func TestDefinitionPreflightReceiptRefusesTampering(t *testing.T) {
	b, err := os.ReadFile("definition-preflight-receipt.json")
	if err != nil {
		t.Fatal(err)
	}
	var r definitionReceipt
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	mutants := []func(*definitionReceipt){
		func(v *definitionReceipt) { v.Verdict = "OWNED_EXPECTED_JOB" },
		func(v *definitionReceipt) { v.ExpectedOwner.LaunchdJobPresent = true },
		func(v *definitionReceipt) { v.ReadOnly = false },
		func(v *definitionReceipt) { v.Incumbent.CommandSHA256 = "sha256:0" },
		func(v *definitionReceipt) { v.Incumbent.OwnerLabelSHA256 = "sha256:0" },
		func(v *definitionReceipt) { v.Issue = 9999 },
	}
	for i, mutate := range mutants {
		v := r
		mutate(&v)
		if err := validateDefinitionReceipt(v); err == nil {
			t.Fatalf("tamper mutant %d passed", i)
		}
	}
}

func validateDefinitionReceipt(r definitionReceipt) error {
	if r.Schema != "fak.model-incumbent-preflight/1" || r.Issue != 9714 || !r.ReadOnly {
		return errDefinition("identity")
	}
	if r.Verdict != "EXPECTED_JOB_ABSENT" || r.Reason != "alternate_launchd_supervisor_owns_incumbent" {
		return errDefinition("verdict")
	}
	if r.Incumbent.CommandSHA256 != "sha256:"+incumbentCommandSHA ||
		r.Incumbent.OwnerLabelSHA256 != "sha256:"+alternateOwnerSHA || !r.Incumbent.OwnerResolved {
		return errDefinition("identities")
	}
	if r.ExpectedOwner.LaunchdJobPresent || !r.Incumbent.AliasMatches {
		return errDefinition("ownership")
	}
	return nil
}

type errDefinition string

func (e errDefinition) Error() string { return "definition receipt drift: " + string(e) }
