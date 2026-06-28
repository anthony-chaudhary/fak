package toolsandbox

import (
	"encoding/json"
	"strings"
	"testing"
)

func sampleTau2PolicyDeltaInput() Tau2PolicyDeltaContractInput {
	return Tau2PolicyDeltaContractInput{
		GeneratedAt:    "2026-06-27T00:00:00Z",
		Suite:          sampleContractSuite(),
		SuitePath:      "testdata/toolsandbox/policy_state_smoke.json",
		Domain:         "retail",
		Domains:        []string{"retail", "airline", "telecom", "banking_knowledge"},
		Model:          "gpt-4.1",
		UserModel:      "gpt-4.1",
		Trials:         4,
		RawCommand:     "tau2 run raw",
		FakCommand:     "tau2 run fak",
		RawOutputDir:   "experiments/raw",
		FakOutputDir:   "experiments/fak",
		FakGateway:     "http://localhost:8080/v1",
		SubmissionFile: "web/leaderboard/public/submissions/gpt-4.1_fak_20260627/submission.json",
	}
}

func TestBuildTau2PolicyDeltaContractKeepsResultGated(t *testing.T) {
	c := BuildTau2PolicyDeltaContract(sampleTau2PolicyDeltaInput())
	if c.Schema != Tau2PolicyDeltaContractSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if c.Status != "READY_FOR_EXTERNAL_HARNESS" {
		t.Fatalf("status = %q", c.Status)
	}
	if c.EvidenceClass != "EXTERNAL_RUN_CONTRACT" {
		t.Fatalf("evidence class = %q", c.EvidenceClass)
	}
	if c.ResultClaimAllowed {
		t.Fatal("policy-delta contract must not allow a result claim until grader artifacts land")
	}
	if c.Issue != 1069 || c.ParentEpic != 1063 {
		t.Fatalf("issue/epic binding = %d/%d", c.Issue, c.ParentEpic)
	}
	if len(c.RequiredBeforeClaim) == 0 || !strings.Contains(strings.Join(c.RequiredBeforeClaim, " "), "tau2 submit validate") {
		t.Fatalf("requirements do not name the submission gate: %+v", c.RequiredBeforeClaim)
	}
	if !c.ScoreEvidenceLink.Required || len(c.ScoreEvidenceLink.JoinKeys) == 0 {
		t.Fatalf("score evidence link = %+v", c.ScoreEvidenceLink)
	}
}

func TestBuildTau2PolicyDeltaContractParityGatesAllYes(t *testing.T) {
	c := BuildTau2PolicyDeltaContract(sampleTau2PolicyDeltaInput())
	want := map[string]bool{
		"same_task_ids_required":               true,
		"same_model_required":                  true,
		"same_user_simulator_required":         true,
		"same_budget_required":                 true,
		"trials_ge_4_for_pass_k":               true,
		"banking_knowledge_domain_for_overall": true,
		"solve_rate_neutrality_required":       true,
	}
	got := map[string]bool{}
	for _, g := range c.Gates {
		got[g.Name] = g.OK
	}
	for name := range want {
		ok, present := got[name]
		if !present {
			t.Fatalf("parity gate %q missing from %+v", name, c.Gates)
		}
		if !ok {
			t.Fatalf("parity gate %q must be yes; got no", name)
		}
	}
}

func TestBuildTau2PolicyDeltaContractTrialsBelowFourIsIncomplete(t *testing.T) {
	in := sampleTau2PolicyDeltaInput()
	in.Trials = 2
	c := BuildTau2PolicyDeltaContract(in)
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status with trials<4 = %q", c.Status)
	}
	var found bool
	for _, g := range c.Gates {
		if g.Name == "trials_ge_4_for_pass_k" {
			found = true
			if g.OK {
				t.Fatalf("trials_ge_4 gate must fail at trials=2: %+v", g)
			}
		}
	}
	if !found {
		t.Fatalf("trials_ge_4 gate not recorded: %+v", c.Gates)
	}
}

func TestBuildTau2PolicyDeltaContractMissingBankingKnowledgeIsIncomplete(t *testing.T) {
	in := sampleTau2PolicyDeltaInput()
	in.Domains = []string{"retail", "airline", "telecom"}
	c := BuildTau2PolicyDeltaContract(in)
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status without banking_knowledge = %q", c.Status)
	}
}

func TestBuildTau2PolicyDeltaContractIncompleteWithoutCommands(t *testing.T) {
	c := BuildTau2PolicyDeltaContract(Tau2PolicyDeltaContractInput{
		Suite:   sampleContractSuite(),
		Domains: []string{"banking_knowledge"},
	})
	if c.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status = %q", c.Status)
	}
	var missingRaw bool
	for _, g := range c.Gates {
		if g.Name == "raw_arm_command" && !g.OK {
			missingRaw = true
		}
	}
	if !missingRaw {
		t.Fatalf("missing raw command gate not recorded: %+v", c.Gates)
	}
}

// The conflation fence: every task-resolve number is OBSERVED (the model's,
// relayed) and every policy/cost/evidence number is WITNESSED (fak's); no policy
// metric folds task success.
func TestBuildTau2PolicyDeltaContractLabelsResolveObservedDeltaWitnessed(t *testing.T) {
	c := BuildTau2PolicyDeltaContract(sampleTau2PolicyDeltaInput())
	if len(c.DeltaMetrics) == 0 {
		t.Fatal("no delta metrics")
	}
	var sawResolve, sawPolicy bool
	for _, m := range c.DeltaMetrics {
		switch m.Kind {
		case MetricKindTaskResolve:
			sawResolve = true
			if m.Provenance != ProvenanceObserved {
				t.Fatalf("task-resolve metric %q must be OBSERVED, got %q", m.Name, m.Provenance)
			}
		case MetricKindPolicyCompliance, MetricKindCost, MetricKindEvidence:
			if m.Kind == MetricKindPolicyCompliance {
				sawPolicy = true
			}
			if m.Provenance != ProvenanceWitnessed {
				t.Fatalf("fak-authored metric %q must be WITNESSED, got %q", m.Name, m.Provenance)
			}
			if m.FoldsTaskSuccess {
				t.Fatalf("policy/cost/evidence metric %q must not fold task success", m.Name)
			}
		default:
			t.Fatalf("metric %q has unknown kind %q", m.Name, m.Kind)
		}
	}
	if !sawResolve || !sawPolicy {
		t.Fatalf("contract must report both task-resolve and policy-compliance metrics: resolve=%t policy=%t", sawResolve, sawPolicy)
	}
}

func TestBuildTau2PolicyDeltaContractHonestyFenceHoldsAndQuotesNoNaiveMultiple(t *testing.T) {
	c := BuildTau2PolicyDeltaContract(sampleTau2PolicyDeltaInput())
	f := c.HonestyFence
	if !f.TaskSuccessNeverFoldedIntoPolicy || !f.RawResolveMustBeStatisticallyFlat ||
		!f.ComplianceGainByDepressingSuccessIsNotAWin || !f.NoVsNaiveCacheMultiple ||
		!f.CacheFigureMarginalOverTunedWarmKV || !f.HeadlineNumberIsModelsNotFak {
		t.Fatalf("honesty fence invariants must all hold: %+v", f)
	}
	if c.Submission.SubmissionType != "custom" {
		t.Fatalf("submission_type = %q", c.Submission.SubmissionType)
	}
	// The contract output must obey its own fence: it may NAME the no-vs-naive
	// rule, but it must never QUOTE a vs-baseline cache multiple.
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	blob := strings.ToLower(string(b)) + strings.ToLower(RenderTau2PolicyDeltaContractMarkdown(c))
	for _, banned := range []string{"17.9", "23.4", "44.9", "45x"} {
		if strings.Contains(blob, banned) {
			t.Fatalf("contract output quotes forbidden vs-baseline figure %q", banned)
		}
	}
}

func TestRenderTau2PolicyDeltaContractMarkdown(t *testing.T) {
	c := BuildTau2PolicyDeltaContract(sampleTau2PolicyDeltaInput())
	md := RenderTau2PolicyDeltaContractMarkdown(c)
	for _, want := range []string{
		"tau2-bench Policy-Delta Contract",
		"Delta Metrics (provenance-labeled)",
		"Honesty Fence",
		"Required Before Any Result Claim",
		"OBSERVED",
		"WITNESSED",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}
