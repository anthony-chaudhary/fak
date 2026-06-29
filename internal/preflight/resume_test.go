package preflight

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

var resumeTestPricing = resume.Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}

func TestResumeRungColdCutUsesResumePlan(t *testing.T) {
	v := PlanResume(ResumeInput{Plan: resume.Input{
		ResidentTokens: 250000,
		IdleSeconds:    7200,
		Pricing:        resumeTestPricing,
	}})

	if v.Schema != ResumePreflightSchema || v.Phase != "projected_at_preflight" {
		t.Fatalf("schema/phase = %q/%q, want %q/projected_at_preflight", v.Schema, v.Phase, ResumePreflightSchema)
	}
	if v.Path != ResumePathCut || v.Recommended != resume.StrategyCut || v.Reason != resume.ReasonColdPrefillShed {
		t.Fatalf("selection = path %q strategy %q reason %q, want CUT/cut/%s",
			v.Path, v.Recommended, v.Reason, resume.ReasonColdPrefillShed)
	}
	if v.Posture != resume.PostureCold {
		t.Fatalf("posture = %q, want cold", v.Posture)
	}
	if v.ProjectedPrefillTokens != resume.DefaultShedBudgetTokens {
		t.Fatalf("projected prefill = %d, want shed budget %d", v.ProjectedPrefillTokens, resume.DefaultShedBudgetTokens)
	}
	cut := candidate(t, v, ResumePathCut)
	if cut.ProjectedPromptCostUSD != v.ProjectedPromptCostUSD {
		t.Fatalf("selected prompt cost %.9f must equal CUT candidate %.9f", v.ProjectedPromptCostUSD, cut.ProjectedPromptCostUSD)
	}
	if len(v.Candidates) != 4 {
		t.Fatalf("candidate count = %d, want four paths", len(v.Candidates))
	}
	for _, p := range []ResumePath{ResumePathWarmSplice, ResumePathWarm, ResumePathCut, ResumePathReset} {
		candidate(t, v, p)
	}
}

func TestResumeRungWarmSpliceIsCheapestAdvisoryPath(t *testing.T) {
	v := PlanResume(ResumeInput{
		Plan: resume.Input{
			ResidentTokens: 200000,
			IdleSeconds:    60,
			HorizonTurns:   3,
			Pricing:        resumeTestPricing,
		},
		WarmSpliceAvailable: true,
	})

	if v.Path != ResumePathWarmSplice {
		t.Fatalf("path = %q, want WARM-SPLICE", v.Path)
	}
	if v.Recommended != resume.StrategyResumeFull {
		t.Fatalf("plan recommendation = %q, want resume_full retained from resume.Plan", v.Recommended)
	}
	if v.ProjectedPrefillTokens != 0 || v.ProjectedPromptCostUSD != 0 {
		t.Fatalf("warm splice must project no re-prefill, got tokens=%d cost=%.9f",
			v.ProjectedPrefillTokens, v.ProjectedPromptCostUSD)
	}
	if !candidate(t, v, ResumePathWarmSplice).Available {
		t.Fatal("warm-splice candidate must be marked available")
	}
	if !v.Advisory || v.BudgetExceeded {
		t.Fatalf("warm splice should be advisory/non-refusal, advisory=%v budgetExceeded=%v", v.Advisory, v.BudgetExceeded)
	}
}

func TestResumeRungWarmKeepsFullPrefix(t *testing.T) {
	v := PlanResume(ResumeInput{Plan: resume.Input{
		ResidentTokens: 200000,
		IdleSeconds:    60,
		HorizonTurns:   3,
		Pricing:        resumeTestPricing,
	}})

	if v.Path != ResumePathWarm || v.Posture != resume.PostureWarm || v.Recommended != resume.StrategyResumeFull {
		t.Fatalf("selection = path %q posture %q strategy %q, want WARM/warm/resume_full",
			v.Path, v.Posture, v.Recommended)
	}
	want := float64(v.ResidentTokens) * resumeTestPricing.InputPerMTokUSD / 1_000_000 * resume.CacheReadMultiplier
	if v.ProjectedPromptCostUSD != want {
		t.Fatalf("warm prompt cost = %.9f, want cache-read projection %.9f", v.ProjectedPromptCostUSD, want)
	}
	if !candidate(t, v, ResumePathWarm).Available {
		t.Fatal("warm candidate must be available for warm posture")
	}
}

func TestResumeRungSmallColdSessionPreservesResumeFullRecommendation(t *testing.T) {
	v := PlanResume(ResumeInput{Plan: resume.Input{
		ResidentTokens: 1000,
		IdleSeconds:    7200,
		Pricing:        resumeTestPricing,
	}})

	if v.Recommended != resume.StrategyResumeFull || v.Reason != resume.ReasonSmallSession {
		t.Fatalf("recommendation = %q/%q, want resume_full/%s",
			v.Recommended, v.Reason, resume.ReasonSmallSession)
	}
	if v.Path != ResumePathWarm || v.ProjectedPrefillTokens != v.ResidentTokens {
		t.Fatalf("small resume_full path = %q tokens=%d, want WARM/full %d",
			v.Path, v.ProjectedPrefillTokens, v.ResidentTokens)
	}
}

func TestResumeRungBudgetExceededLabelsButDoesNotBlock(t *testing.T) {
	v := PlanResume(ResumeInput{
		Plan: resume.Input{
			ResidentTokens: 250000,
			IdleSeconds:    7200,
			Pricing:        resumeTestPricing,
		},
		BudgetTokens: 1000,
	})

	if !v.Advisory {
		t.Fatal("resume preflight rung must be advisory/fail-open")
	}
	if !v.BudgetExceeded || v.BudgetReason != ResumeBudgetColdPrefillExceeded {
		t.Fatalf("budget label = exceeded %v reason %q, want true/%s",
			v.BudgetExceeded, v.BudgetReason, ResumeBudgetColdPrefillExceeded)
	}
	if v.Path != ResumePathCut {
		t.Fatalf("budget label must not change the selected path, got %q", v.Path)
	}
}

func TestResumeRungDeterministicRecord(t *testing.T) {
	in := ResumeInput{Plan: resume.Input{
		ResidentTokens: 123456,
		IdleSeconds:    901,
		Pricing:        resumeTestPricing,
	}, BudgetTokens: 4096}

	a, err := json.Marshal(PlanResume(in))
	if err != nil {
		t.Fatalf("marshal first verdict: %v", err)
	}
	b, err := json.Marshal(PlanResume(in))
	if err != nil {
		t.Fatalf("marshal second verdict: %v", err)
	}
	if string(a) != string(b) {
		t.Fatalf("resume preflight is not deterministic:\n%s\n%s", a, b)
	}
}

func TestResumeTokensFromBytesUsesBytesOverFourProxy(t *testing.T) {
	for _, tc := range []struct {
		bytes int
		want  int
	}{
		{-1, 0},
		{0, 0},
		{1, 1},
		{4, 1},
		{5, 2},
		{4096, 1024},
	} {
		if got := ResumeTokensFromBytes(tc.bytes); got != tc.want {
			t.Fatalf("ResumeTokensFromBytes(%d) = %d, want %d", tc.bytes, got, tc.want)
		}
	}
}

func candidate(t *testing.T, v ResumeVerdict, path ResumePath) ResumePathCost {
	t.Helper()
	for _, c := range v.Candidates {
		if c.Path == path {
			return c
		}
	}
	t.Fatalf("missing candidate path %q in %+v", path, v.Candidates)
	return ResumePathCost{}
}
