package agenticbench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestALEPreflightRefusesNativeResumeAliases(t *testing.T) {
	root := filepath.Join("testdata", "ale-resume", "shared")
	base := aleFixtureIdentity()
	if !stockALEAlreadyDone(t, root, base) {
		t.Fatal("fixture must reproduce stock ALE treating the prior direct run as done")
	}

	tests := []struct {
		name   string
		mutate func(*ALEExperimentIdentity)
	}{
		{"different arm", func(id *ALEExperimentIdentity) { id.Arm = "fak" }},
		{"different effort", func(id *ALEExperimentIdentity) { id.Effort = "high" }},
		{"different snapshot", func(id *ALEExperimentIdentity) { id.Snapshot = "2026-08-15" }},
		{"different source sha", func(id *ALEExperimentIdentity) { id.SourceSHA = strings.Repeat("b", 40) }},
		{"different repetition", func(id *ALEExperimentIdentity) { id.Repetition = 2 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			planned := base
			tc.mutate(&planned)
			receipt, err := CheckALEResumeAlias(ALELaunchSpec{
				Identity:  planned,
				Output:    ALEOutputRoot{Path: root},
				Crossover: tc.name == "different arm",
			})
			if !IsALEAliasReason(err, ReasonALEResumeIdentityMismatch) {
				t.Fatalf("error = %v, want %s", err, ReasonALEResumeIdentityMismatch)
			}
			if receipt.Decision != ALEResumeRefused || receipt.SourceRunID != "ale-direct-complete" {
				t.Fatalf("receipt = %+v, want refused decision bound to source run", receipt)
			}
		})
	}
}

func TestALEPreflightAdmitsExactResumeAndRetainsSource(t *testing.T) {
	identity := aleFixtureIdentity()
	receipt, err := CheckALEResumeAlias(ALELaunchSpec{
		Identity: identity,
		Output:   ALEOutputRoot{Path: filepath.Join("testdata", "ale-resume", "shared")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Decision != ALEResumed || receipt.SourceRunID != "ale-direct-complete" {
		t.Fatalf("receipt = %+v, want exact resume from ale-direct-complete", receipt)
	}
	wantIdentity, err := ALEManifestIdentity(identity)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.ManifestIdentity != wantIdentity {
		t.Fatalf("manifest identity = %q, want %q", receipt.ManifestIdentity, wantIdentity)
	}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"decision":"resumed"`) || !strings.Contains(string(b), `"source_run_id":"ale-direct-complete"`) {
		t.Fatalf("comparison receipt lost resume provenance: %s", b)
	}
}

func TestALEPreflightAdmitsFreshIdentityScopedCrossoverRoot(t *testing.T) {
	identity := aleFixtureIdentity()
	identity.Arm = "fak"
	identity.Repetition = 2
	manifestIdentity, err := ALEManifestIdentity(identity)
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(t.TempDir(), "fak-repetition-2")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	marker, err := json.Marshal(ALEIdentityRecord{Schema: ALEExperimentIdentitySchema, Identity: identity})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ALEOutputIdentityFile), marker, 0o644); err != nil {
		t.Fatal(err)
	}
	receipt, err := CheckALEResumeAlias(ALELaunchSpec{
		Identity: identity,
		Output: ALEOutputRoot{
			Path:             root,
			ManifestIdentity: manifestIdentity,
		},
		Crossover: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Decision != ALEResumeFresh || receipt.SourceRunID != "" {
		t.Fatalf("receipt = %+v, want fresh decision without a source run", receipt)
	}
}

func TestALEPreflightRequiresDisableResumeForSharedCrossoverRoot(t *testing.T) {
	identity := aleFixtureIdentity()
	identity.Arm = "fak"
	receipt, err := CheckALEResumeAlias(ALELaunchSpec{
		Identity:  identity,
		Output:    ALEOutputRoot{Path: t.TempDir()},
		Crossover: true,
	})
	if !IsALEAliasReason(err, ReasonALECrossoverResumeUnscoped) || receipt.Decision != ALEResumeRefused {
		t.Fatalf("receipt/error = %+v / %v, want unscoped crossover refusal", receipt, err)
	}

	receipt, err = CheckALEResumeAlias(ALELaunchSpec{
		Identity:      identity,
		Output:        ALEOutputRoot{Path: filepath.Join(t.TempDir(), "shared")},
		Crossover:     true,
		DisableResume: true,
	})
	if err != nil || receipt.Decision != ALEResumeFresh {
		t.Fatalf("disable-resume receipt/error = %+v / %v, want fresh", receipt, err)
	}
}

func TestALEManifestIdentityCoversEveryMaterialTerm(t *testing.T) {
	base := aleFixtureIdentity()
	want, err := ALEManifestIdentity(base)
	if err != nil {
		t.Fatal(err)
	}
	mutations := []struct {
		name   string
		mutate func(*ALEExperimentIdentity)
	}{
		{"task", func(id *ALEExperimentIdentity) { id.TaskPath = "/tasks/finance/other" }},
		{"source repo", func(id *ALEExperimentIdentity) { id.SourceRepo += "-mirror" }},
		{"source sha", func(id *ALEExperimentIdentity) { id.SourceSHA = strings.Repeat("b", 40) }},
		{"harness", func(id *ALEExperimentIdentity) { id.Harness = "ale-codex" }},
		{"agent", func(id *ALEExperimentIdentity) { id.AgentID = "ale-codex" }},
		{"model", func(id *ALEExperimentIdentity) { id.Model = "gpt-5.7" }},
		{"effort", func(id *ALEExperimentIdentity) { id.Effort = "high" }},
		{"endpoint", func(id *ALEExperimentIdentity) { id.Endpoint = "http://127.0.0.1:8080/v1" }},
		{"arm", func(id *ALEExperimentIdentity) { id.Arm = "fak" }},
		{"snapshot", func(id *ALEExperimentIdentity) { id.Snapshot = "2026-08-15" }},
		{"wall budget", func(id *ALEExperimentIdentity) { id.BudgetSeconds++ }},
		{"token budget", func(id *ALEExperimentIdentity) { id.MaxTokens++ }},
		{"retry policy", func(id *ALEExperimentIdentity) { id.RetryPolicy = "disabled" }},
		{"prompt suffix", func(id *ALEExperimentIdentity) { id.PromptSuffixSHA256 = "sha256:" + strings.Repeat("b", 64) }},
		{"repetition", func(id *ALEExperimentIdentity) { id.Repetition++ }},
		{"variant", func(id *ALEExperimentIdentity) { id.Variant++ }},
	}
	for _, mutation := range mutations {
		t.Run(mutation.name, func(t *testing.T) {
			changed := base
			mutation.mutate(&changed)
			got, err := ALEManifestIdentity(changed)
			if err != nil {
				t.Fatal(err)
			}
			if got == want {
				t.Fatalf("%s mutation did not move manifest identity %s", mutation.name, got)
			}
		})
	}
}

func aleFixtureIdentity() ALEExperimentIdentity {
	return ALEExperimentIdentity{
		TaskPath:           "/tasks/finance/forecast",
		SourceRepo:         "https://github.com/rdi-berkeley/agents-last-exam",
		SourceSHA:          "75a3f866535946b67f9a57e4f158eb30ad50be8a",
		Harness:            "ale-claw",
		AgentID:            "ale-claw",
		Model:              "gpt-5.6",
		Effort:             "medium",
		Endpoint:           "https://api.openai.com/v1",
		Arm:                "direct",
		Snapshot:           "2026-08-01",
		BudgetSeconds:      3600,
		MaxTokens:          32768,
		RetryPolicy:        "terminal-only-max-2",
		PromptSuffixSHA256: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Repetition:         1,
		Variant:            0,
	}
}

// stockALEAlreadyDone mirrors the pinned upstream key: agent, model, task, and
// variant only. It intentionally does not read the fak identity sidecar.
func stockALEAlreadyDone(t *testing.T, root string, identity ALEExperimentIdentity) bool {
	t.Helper()
	variantDir := filepath.Join(root, slugALEAgent(identity.AgentID), slugALEModel(identity.Model), slugALETask(identity.TaskPath), "v"+strconv.Itoa(identity.Variant))
	entries, err := os.ReadDir(variantDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		b, err := os.ReadFile(filepath.Join(variantDir, entry.Name(), "run.json"))
		if err != nil {
			continue
		}
		var run struct {
			Status string `json:"status"`
		}
		if json.Unmarshal(b, &run) == nil && (run.Status == "completed" || run.Status == "timeout") {
			return true
		}
	}
	return false
}
