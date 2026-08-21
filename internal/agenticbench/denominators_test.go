package agenticbench

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

type denominatorFixture struct {
	Funnel struct {
		Plan []CohortUnit `json:"plan"`
		Arm  ArmRun       `json:"arm"`
	} `json:"funnel"`
	Comparison struct {
		Plan []CohortUnit `json:"plan"`
		Arms []ArmRun     `json:"arms"`
	} `json:"comparison"`
}

func TestDenominatorReceiptReconcilesEveryStageAndMissingReason(t *testing.T) {
	fixture := loadDenominatorFixture(t)
	cohort, err := FreezeCohort(fixture.Funnel.Plan)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(cohort.Digest, "sha256:") {
		t.Fatalf("cohort digest = %q, want content address", cohort.Digest)
	}
	reordered := append([]CohortUnit(nil), fixture.Funnel.Plan...)
	for left, right := 0, len(reordered)-1; left < right; left, right = left+1, right-1 {
		reordered[left], reordered[right] = reordered[right], reordered[left]
	}
	same, err := FreezeCohort(reordered)
	if err != nil {
		t.Fatal(err)
	}
	if same.Digest != cohort.Digest {
		t.Fatalf("same planned population produced different digests: %s != %s", same.Digest, cohort.Digest)
	}
	mutated := append([]CohortUnit(nil), fixture.Funnel.Plan...)
	mutated[0].Stratum = "easy"
	different, err := FreezeCohort(mutated)
	if err != nil {
		t.Fatal(err)
	}
	if different.Digest == cohort.Digest {
		t.Fatal("cohort digest did not bind planned-unit content")
	}

	receipt, err := ReconcileArm(cohort, fixture.Funnel.Arm)
	if err != nil {
		t.Fatal(err)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("reconciled receipt does not validate: %v", err)
	}
	wantCounts := FunnelCounts{Planned: 7, Admitted: 6, Attempted: 5, AgentTerminal: 5, EvaluatorTerminal: 3, Scored: 2, Missing: 5}
	if receipt.Counts != wantCounts {
		t.Fatalf("counts = %+v, want %+v", receipt.Counts, wantCounts)
	}
	for _, reason := range []MissingReason{
		MissingAdmissionRefused,
		MissingLaunchFailed,
		MissingAgentTimeout,
		MissingEvaluatorFailed,
		MissingArtifactIncomplete,
	} {
		if got := receipt.MissingByReason[reason]; got != 1 {
			t.Errorf("missing reason %q count = %d, want 1", reason, got)
		}
	}
	if receipt.Score.DenominatorCount != 2 || receipt.Counts.Missing != 5 {
		t.Fatalf("zero/missing conflated: score=%+v counts=%+v", receipt.Score, receipt.Counts)
	}
	if receipt.Score.Value == nil || math.Abs(*receipt.Score.Value-0.4) > 1e-12 {
		t.Fatalf("score aggregate = %+v, want zero retained in 0.8/2 mean", receipt.Score)
	}
	if receipt.Score.Numerator != "sum_official_scores" || receipt.Score.Denominator != "scored_rows" {
		t.Fatalf("aggregate does not name numerator/denominator: %+v", receipt.Score)
	}

	_, err = ReconcileArm(cohort, ArmRun{Arm: "unreported", Results: fixture.Funnel.Arm.Results[:len(fixture.Funnel.Arm.Results)-1]})
	if err == nil || !strings.Contains(err.Error(), "missing planned result") {
		t.Fatalf("unreported missingness err = %v", err)
	}
}

func TestDenominatorGateRefusesNaiveEqualityAndLimitsDeclaredSubset(t *testing.T) {
	fixture := loadDenominatorFixture(t)
	cohort, err := FreezeCohort(fixture.Comparison.Plan)
	if err != nil {
		t.Fatal(err)
	}
	report, err := BuildDenominatorReport(cohort, fixture.Comparison.Arms, []ComparisonRequest{
		{LeftArm: "complete", RightArm: "sparse", Rule: AnalysisRule{Scope: ScopeFullCohort}},
		{LeftArm: "complete", RightArm: "sparse", Rule: AnalysisRule{Scope: ScopeScoredSubset, MaxMissingRateDelta: 0.49}},
		{LeftArm: "complete", RightArm: "sparse", Rule: AnalysisRule{Scope: ScopeScoredSubset, MaxMissingRateDelta: 0.5}},
	})
	if err != nil {
		t.Fatal(err)
	}
	receipts := report.Arms
	if receipts[0].Score.Value == nil || receipts[1].Score.Value == nil || *receipts[0].Score.Value != *receipts[1].Score.Value {
		t.Fatalf("fixture must expose naive completed-score equality: %+v", receipts)
	}

	planned := report.Comparisons[0]
	if planned.State != ClaimRefused || !strings.Contains(planned.Detail, "2/4 missing") {
		t.Fatalf("planned-cohort comparison = %+v, want asymmetric missingness refusal", planned)
	}
	imbalanced := report.Comparisons[1]
	if imbalanced.State != ClaimRefused || !strings.Contains(imbalanced.Detail, "exceeds declared") {
		t.Fatalf("imbalanced scored-subset comparison = %+v, want declared-rule refusal", imbalanced)
	}
	subset := report.Comparisons[2]
	if subset.State != ClaimLimitedToScoredSubset || subset.Scope != ScopeScoredSubset {
		t.Fatalf("declared completed-subset comparison = %+v, want explicit limitation", subset)
	}

	differentPlan := append([]CohortUnit(nil), fixture.Comparison.Plan...)
	differentPlan[0].Stratum = "changed"
	differentCohort, err := FreezeCohort(differentPlan)
	if err != nil {
		t.Fatal(err)
	}
	differentReceipt, err := ReconcileArm(differentCohort, fixture.Comparison.Arms[0])
	if err != nil {
		t.Fatal(err)
	}
	mismatch := CompareArms(receipts[0], differentReceipt, AnalysisRule{Scope: ScopeFullCohort})
	if mismatch.State != ClaimRefused || !strings.Contains(mismatch.Detail, "planned cohorts differ") {
		t.Fatalf("cohort mismatch comparison = %+v, want refusal", mismatch)
	}

	markdown := FormatDenominatorMarkdown(report)
	for _, want := range []string{
		"Planned | Admitted | Attempted | Agent terminal | Evaluator terminal | Scored | Missing",
		"sum_official_scores",
		"launch_failed",
		"REFUSED",
		"LIMITED_TO_SCORED_SUBSET",
	} {
		if !strings.Contains(markdown, want) {
			t.Fatalf("denominator markdown missing %q:\n%s", want, markdown)
		}
	}
}

func loadDenominatorFixture(t *testing.T) denominatorFixture {
	t.Helper()
	body, err := os.ReadFile("testdata/denominator_funnel.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture denominatorFixture
	if err := json.Unmarshal(body, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}
