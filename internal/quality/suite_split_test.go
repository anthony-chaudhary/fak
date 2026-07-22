package quality

import (
	"strings"
	"testing"
)

// validCase builds a canonical, routable quality case for the split tests. Every
// field ValidateCanonical requires is present, so the case places into a suite
// unless a test deliberately breaks one.
func validCase(id, tier, family string, cost CostSpec) QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      id,
		Version: 1,
		Prompt:  "route this case",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 2},
		Reference: Trace{
			Runner: "reference",
			Tokens: []string{"ok"},
			Text:   "ok",
		},
		Oracles: []string{"greedy-token-diff"},
		Metadata: CaseMetadata{
			Model:     Revision{Name: "m", Revision: "sha256:m"},
			Tokenizer: Revision{Name: "t", Revision: "sha256:t"},
			Engine:    EngineSpec{Name: "fak", Backend: "cpu", Flags: map[string]string{"dtype": "f32"}},
			Code:      Revision{Name: "github.com/anthony-chaudhary/fak", Revision: "git:deadbeef"},
			Oracle:    OracleEvidence{Kind: "exact-greedy-trace", Revision: "sha256:o"},
			Tolerance: ToleranceSpec{Metric: "exact-token", Revision: "policy:v1"},
			Baseline:  BaselineSpec{ID: "b", Revision: "sha256:b"},
			Tier:      TierSpec{Name: tier},
			Cost:      cost,
			Owner:     "quality-team",
			Family:    family,
		},
	}
}

func cost(runtime, timeout int64, cpu int, mem int64, accel int) CostSpec {
	return CostSpec{RuntimeSeconds: runtime, TimeoutSeconds: timeout, CPU: cpu, MemoryMiB: mem, Accelerators: accel}
}

// TestSplitSuitesRoutesByTierAndOrdersByCost is the happy path: a mixed corpus
// splits into the three tiers, each suite is ordered cheapest-evidence-first, and
// per-tier cost is summed so an operator sees what each suite costs.
func TestSplitSuitesRoutesByTierAndOrdersByCost(t *testing.T) {
	cases := []QualityCase{
		validCase("pr-slow", "pr", "deterministic", cost(10, 90, 2, 128, 0)),
		validCase("pr-fast", "pr", "deterministic", cost(1, 20, 1, 32, 0)),
		validCase("night-stats", "nightly", "statistics", cost(600, 1800, 8, 4096, 0)),
		validCase("rel-gpu", "release", "gpu_parity", cost(1200, 7200, 16, 65536, 8)),
		validCase("rel-review", "release", "review", cost(300, 3600, 4, 2048, 0)),
	}
	plan := SplitSuites(cases, nil)

	if len(plan.Rejected) != 0 {
		t.Fatalf("well-formed corpus rejected cases: %+v", plan.Rejected)
	}
	byTier := map[Tier]Suite{}
	for _, s := range plan.Suites {
		byTier[s.Tier] = s
	}
	pr := byTier[TierPR]
	if len(pr.Cases) != 2 || pr.Cases[0].CaseID != "pr-fast" || pr.Cases[1].CaseID != "pr-slow" {
		t.Fatalf("PR suite not ordered cheapest-first: %+v", pr.Cases)
	}
	if pr.TotalRuntimeSec != 11 || pr.MaxMemoryMiB != 128 || pr.MaxAccelerators != 0 {
		t.Fatalf("PR suite cost envelope wrong: %+v", pr)
	}
	if rel := byTier[TierRelease]; rel.MaxAccelerators != 8 || len(rel.Cases) != 2 {
		t.Fatalf("release suite should carry the GPU case and 8 accelerators: %+v", rel)
	}
	// The suites are returned in fixed PR, nightly, release order.
	if plan.Suites[0].Tier != TierPR || plan.Suites[1].Tier != TierNightly || plan.Suites[2].Tier != TierRelease {
		t.Fatalf("suite order not deterministic: %v", []Tier{plan.Suites[0].Tier, plan.Suites[1].Tier, plan.Suites[2].Tier})
	}
}

// TestSplitSuitesRejectsExpensiveEvidenceInCheapTier is the #4574 witness: a
// representative defect — an expensive GPU-parity case mis-declared as a per-PR
// check — is REJECTED by the cost split (never placed in the PR suite), and after
// the fix (routing it to release) the identical corpus splits clean. Fail-then-pass
// in one hermetic, independently replayed run.
func TestSplitSuitesRejectsExpensiveEvidenceInCheapTier(t *testing.T) {
	// A GPU-parity case needs an accelerator — the defining reason it cannot ride
	// the CPU-only PR lane even when its wall is short. The planted defect labels it
	// tier=pr, as if it were a cheap CPU push gate.
	defect := validCase("gpu-parity-1", "pr", "gpu_parity", cost(50, 90, 16, 65536, 8))
	baseline := []QualityCase{
		validCase("det-1", "pr", "deterministic", cost(1, 20, 1, 32, 0)),
		defect,
	}

	planted := SplitSuites(baseline, nil)
	prCases := suiteFor(planted, TierPR).Cases
	for _, sc := range prCases {
		if sc.CaseID == "gpu-parity-1" {
			t.Fatal("planted defect: expensive GPU case was admitted to the PR suite")
		}
	}
	rej := rejectFor(planted, "gpu-parity-1")
	if rej == nil {
		t.Fatal("planted defect was neither placed nor rejected — silently dropped")
	}
	if !strings.Contains(rej.Reason, "accelerator") {
		t.Fatalf("rejection did not localize the first broken budget: %q", rej.Reason)
	}
	// The clean deterministic PR case still routes — one bad case does not sink the split.
	if len(prCases) != 1 || prCases[0].CaseID != "det-1" {
		t.Fatalf("clean PR case did not survive the defect: %+v", prCases)
	}

	// The fix: route the same GPU case to the release tier it can afford.
	fixed := make([]QualityCase, len(baseline))
	copy(fixed, baseline)
	fixed[1].Metadata.Tier.Name = "release"

	repaired := SplitSuites(fixed, nil)
	if len(repaired.Rejected) != 0 {
		t.Fatalf("after fix the corpus still rejected cases: %+v", repaired.Rejected)
	}
	rel := suiteFor(repaired, TierRelease)
	if len(rel.Cases) != 1 || rel.Cases[0].CaseID != "gpu-parity-1" || rel.MaxAccelerators != 8 {
		t.Fatalf("fixed GPU case did not land in the release suite: %+v", rel)
	}
}

// TestSplitSuitesRejectsIncompleteRoutingHeader proves a case missing a required
// routing field (owner) is refused, not routed — missing routing evidence is never
// a pass. The rejection names the field so the gap is actionable.
func TestSplitSuitesRejectsIncompleteRoutingHeader(t *testing.T) {
	noOwner := validCase("orphan", "pr", "deterministic", cost(1, 20, 1, 32, 0))
	noOwner.Metadata.Owner = ""
	plan := SplitSuites([]QualityCase{noOwner}, nil)

	if len(suiteFor(plan, TierPR).Cases) != 0 {
		t.Fatal("ownerless case was routed into a suite")
	}
	rej := rejectFor(plan, "orphan")
	if rej == nil || !strings.Contains(rej.Reason, "owner") {
		t.Fatalf("ownerless case not rejected with an owner reason: %+v", rej)
	}
}

func suiteFor(p SuitePlan, tier Tier) Suite {
	for _, s := range p.Suites {
		if s.Tier == tier {
			return s
		}
	}
	return Suite{}
}

func rejectFor(p SuitePlan, id string) *SuiteReject {
	for i := range p.Rejected {
		if p.Rejected[i].CaseID == id {
			return &p.Rejected[i]
		}
	}
	return nil
}
