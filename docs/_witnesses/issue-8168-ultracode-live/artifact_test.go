package issue8168live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

type campaignArtifact struct {
	Schema              string   `json:"schema"`
	Live                bool     `json:"live"`
	Verdict             string   `json:"verdict"`
	ValueClaimSupported bool     `json:"value_claim_supported"`
	Reasons             []string `json:"reasons"`
	Activation          struct {
		Expected          int     `json:"treatment_children_expected"`
		Verified          int     `json:"treatment_children_verified_active"`
		Coverage          float64 `json:"coverage"`
		Required          float64 `json:"required_coverage"`
		CausalAttribution bool    `json:"causal_attribution_allowed"`
	} `json:"activation"`
	Equivalence struct {
		TaskEqual             bool `json:"task_equal"`
		ModelEqual            bool `json:"model_equal"`
		EnvironmentEqual      bool `json:"environment_equal"`
		BudgetsEqualDeclared  bool `json:"budgets_equal_declared"`
		BudgetsEqualObserved  bool `json:"budgets_equal_observed"`
		OutcomeEqualAndPassed bool `json:"acceptance_outcome_equal_and_passing"`
		WitnessJoined         bool `json:"independent_witness_joined"`
		ComparisonValid       bool `json:"comparison_valid"`
	} `json:"equivalence"`
	Axes struct {
		CacheTokens struct {
			BilledAvailable bool   `json:"billed_tokens_available"`
			SpendAvailable  bool   `json:"spend_available"`
			Verdict         string `json:"verdict"`
		} `json:"cache_and_tokens"`
	} `json:"axes"`
}

func TestLiveCampaignAbstainsWithoutActivationOrAuthoritativeAccounting(t *testing.T) {
	var campaign campaignArtifact
	readJSON(t, "campaign.json", &campaign)
	if campaign.Schema != "fak.ultracode.live-paired-campaign/1" || !campaign.Live {
		t.Fatalf("not a live campaign artifact: %+v", campaign)
	}
	if campaign.Verdict != "ABSTAIN" || campaign.ValueClaimSupported || campaign.Equivalence.ComparisonValid {
		t.Fatalf("unsupported value claim: %+v", campaign)
	}
	if campaign.Activation.Expected != 3 || campaign.Activation.Verified != 0 || campaign.Activation.Coverage != 0 || campaign.Activation.Required != 1 || campaign.Activation.CausalAttribution {
		t.Fatalf("activation gate not fail-closed: %+v", campaign.Activation)
	}
	if !campaign.Equivalence.TaskEqual || !campaign.Equivalence.ModelEqual || !campaign.Equivalence.EnvironmentEqual || !campaign.Equivalence.BudgetsEqualDeclared || !campaign.Equivalence.OutcomeEqualAndPassed {
		t.Fatalf("declared controls or accepted outcome differ: %+v", campaign.Equivalence)
	}
	if campaign.Equivalence.BudgetsEqualObserved || campaign.Equivalence.WitnessJoined || campaign.Axes.CacheTokens.BilledAvailable || campaign.Axes.CacheTokens.SpendAvailable || campaign.Axes.CacheTokens.Verdict != "ABSTAIN" {
		t.Fatalf("unverified comparison was admitted: equivalence=%+v accounting=%+v", campaign.Equivalence, campaign.Axes.CacheTokens)
	}
	joined := strings.Join(campaign.Reasons, "\n")
	for _, want := range []string{"activation_unverified", "outcome_unverified", "budget_unverified", "spend_unavailable"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("reasons missing %q: %v", want, campaign.Reasons)
		}
	}
}

func TestPairReplaysToCapturedAbstain(t *testing.T) {
	var pair ultracodebench.Pair
	readJSON(t, "pair.json", &pair)
	if pair.Single.Identity != pair.Fleet.Identity {
		t.Fatal("pair controls differ")
	}
	if pair.Single.AcceptedEffects != 3 || pair.Fleet.AcceptedEffects != 3 || pair.Single.Contradictions != 0 || pair.Fleet.Contradictions != 0 {
		t.Fatalf("quality counts changed: single=%+v fleet=%+v", pair.Single, pair.Fleet)
	}
	if pair.Single.InputTokens+pair.Single.CacheReadTokens != 59067 || pair.Fleet.InputTokens+pair.Fleet.CacheReadTokens != 236386 {
		t.Fatalf("input/cache accounting changed: single=%+v fleet=%+v", pair.Single, pair.Fleet)
	}
	report, err := ultracodebench.Evaluate(pair)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "ABSTAIN" {
		t.Fatalf("verdict=%s reasons=%v", report.Verdict, report.Reasons)
	}
	var captured struct {
		Verdict string               `json:"verdict"`
		Reasons []string             `json:"reasons"`
		Gains   ultracodebench.Gains `json:"gains"`
	}
	readJSON(t, "paired-report.witness.json", &captured)
	if captured.Verdict != "ABSTAIN" || len(captured.Reasons) != 1 || captured.Reasons[0] != "both modes require independent witness digests" {
		t.Fatalf("captured pre-activation report lost its abstention: %+v", captured)
	}
	if report.Gains != captured.Gains {
		t.Fatalf("captured metrics drifted:\n got  %+v\n want %+v", report.Gains, captured.Gains)
	}
}

func TestTaskAndCommittedArtifactsAreRedacted(t *testing.T) {
	task, err := os.ReadFile("task.txt")
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte(strings.TrimSuffix(string(task), "\n")))
	if got := "sha256:" + hex.EncodeToString(digest[:]); got != "sha256:df4e6199d8ae6809e759153916fc36bb1d3354c134fd41fee52f4bfd62177861" {
		t.Fatalf("task digest=%s", got)
	}
	for _, path := range []string{"README.md", "campaign.json", "single-receipt.json", "fleet-receipt.json", "pair.json", "paired-report.witness.json", "task.txt"} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		lower := strings.ToLower(string(raw))
		for _, forbidden := range []string{`c:\\users\\`, "session_id", "run_id", "account_uuid", "access_token", "api_key"} {
			if strings.Contains(lower, forbidden) {
				t.Fatalf("%s contains forbidden private marker %q", path, forbidden)
			}
		}
	}
}

func readJSON(t *testing.T, path string, dst any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}
