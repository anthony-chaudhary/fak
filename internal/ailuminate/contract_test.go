package ailuminate

import (
	"strings"
	"testing"
)

// TestBuildNoGoWhenNoContentFilter is the load-bearing case: with no completion-
// path content filter (today's reality), the contract must recommend NO-GO and
// never allow a result claim.
func TestBuildNoGoWhenNoContentFilter(t *testing.T) {
	c := Build(ContractInput{
		GeneratedAt:                 "2026-06-27T00:00:00Z",
		FakCommit:                   "deadbeef",
		CompletionPathContentFilter: false,
	})
	if c.Schema != ContractSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if c.ResultClaimAllowed {
		t.Fatal("AILuminate grade is never a fak result; ResultClaimAllowed must be false")
	}
	if c.Status != StatusNoGoPrereqUnmet {
		t.Fatalf("status = %q, want %q", c.Status, StatusNoGoPrereqUnmet)
	}
	if c.Recommendation != RecNoGoScopedOut {
		t.Fatalf("recommendation = %q, want %q", c.Recommendation, RecNoGoScopedOut)
	}
	if !strings.HasPrefix(c.Prerequisite.Verdict, "NO") {
		t.Fatalf("prerequisite verdict = %q, want NO-prefixed", c.Prerequisite.Verdict)
	}
	// The prerequisite gate must be the one that is failing.
	var sawPrereqGate bool
	for _, g := range c.Gates {
		if g.Name == "completion_path_content_filter" {
			sawPrereqGate = true
			if g.OK {
				t.Fatal("completion_path_content_filter gate must be NOT OK with no filter")
			}
		}
	}
	if !sawPrereqGate {
		t.Fatalf("missing completion_path_content_filter gate: %+v", c.Gates)
	}
}

// TestBuildGoWhenContentFilterInPath confirms the conditional branch: if a
// content filter is wired into the completion path, the contract flips to a
// fenced operator-gated GO — but still never allows a fak result claim.
func TestBuildGoWhenContentFilterInPath(t *testing.T) {
	c := Build(ContractInput{
		GeneratedAt:                 "2026-06-27T00:00:00Z",
		CompletionPathContentFilter: true,
	})
	if c.Status != StatusReadyOperatorRun {
		t.Fatalf("status = %q, want %q", c.Status, StatusReadyOperatorRun)
	}
	if c.Recommendation != RecGoOperatorGated {
		t.Fatalf("recommendation = %q, want %q", c.Recommendation, RecGoOperatorGated)
	}
	if c.ResultClaimAllowed {
		t.Fatal("even on GO, the AILuminate grade is OBSERVED — ResultClaimAllowed must stay false")
	}
}

// TestTwelveCategoriesMapped asserts the full AILuminate v1.1 taxonomy is mapped
// and labeled with a known movability value.
func TestTwelveCategoriesMapped(t *testing.T) {
	c := Build(ContractInput{GeneratedAt: "t"})
	if len(c.Adjacency) != 12 {
		t.Fatalf("adjacency categories = %d, want 12", len(c.Adjacency))
	}
	groups := map[string]int{}
	for _, h := range c.Adjacency {
		if h.Movability != RidesOnModel && h.Movability != ToolMediatedNotGradedHere {
			t.Fatalf("category %q has unknown movability %q", h.Category, h.Movability)
		}
		if strings.TrimSpace(h.Rationale) == "" {
			t.Fatalf("category %q missing rationale", h.Category)
		}
		groups[h.Group]++
	}
	// v1.1 taxonomy: 5 physical, 5 non-physical, 2 contextual.
	if groups["physical"] != 5 || groups["non-physical"] != 5 || groups["contextual"] != 2 {
		t.Fatalf("group distribution = %+v, want physical:5 non-physical:5 contextual:2", groups)
	}
	for _, g := range c.Gates {
		if g.Name == "twelve_categories_mapped" && !g.OK {
			t.Fatal("twelve_categories_mapped gate not OK")
		}
	}
}

// TestLineageAndProvenancePresent enforces #9 lineage + #72 provenance discipline:
// every lineage field present (placeholder allowed pre-run) and the grade labeled
// OBSERVED while the gateway delta is labeled WITNESSED.
func TestLineageAndProvenancePresent(t *testing.T) {
	c := Build(ContractInput{GeneratedAt: "t"})
	if c.Lineage.AILuminateVersion == "" || c.Lineage.RunDateTime == "" || c.Lineage.FakCommit == "" ||
		c.Lineage.FrontedModelID == "" || c.Lineage.ModelProvider == "" || c.Lineage.ModelDate == "" ||
		c.Lineage.HarnessCommit == "" || c.Lineage.EvaluatorNote == "" {
		t.Fatalf("lineage has an empty field: %+v", c.Lineage)
	}
	var sawObservedGrade, sawWitnessedDelta bool
	for _, p := range c.Provenance {
		switch p.Label {
		case "OBSERVED", "WITNESSED":
		default:
			t.Fatalf("provenance label must be OBSERVED or WITNESSED, got %q for %q", p.Label, p.Number)
		}
		if p.Label == "OBSERVED" && strings.Contains(p.Number, "grade") {
			sawObservedGrade = true
		}
		if p.Label == "WITNESSED" && strings.Contains(p.Number, "delta") {
			sawWitnessedDelta = true
		}
	}
	if !sawObservedGrade {
		t.Fatal("the five-tier grade must be labeled OBSERVED")
	}
	if !sawWitnessedDelta {
		t.Fatal("the gateway-attributable delta must be labeled WITNESSED")
	}
}

// TestBaselineVsSystemArms asserts a bare-model comparator exists alongside the
// model+fak arm (a system arm with no baseline is un-witnessable).
func TestBaselineVsSystemArms(t *testing.T) {
	c := Build(ContractInput{GeneratedAt: "t"})
	if len(c.Arms) != 2 {
		t.Fatalf("arms = %d, want 2 (bare-model + model+fak)", len(c.Arms))
	}
	var sawBare, sawSystem bool
	for _, a := range c.Arms {
		if a.Name == "bare-model-sut" {
			sawBare = true
		}
		if a.Name == "model-plus-fak-gateway-sut" {
			sawSystem = true
		}
		if a.GradeIs != "OBSERVED" {
			t.Fatalf("arm %q grade must be OBSERVED (MLCommons-graded), got %q", a.Name, a.GradeIs)
		}
	}
	if !sawBare || !sawSystem {
		t.Fatalf("missing an arm: bare=%t system=%t", sawBare, sawSystem)
	}
}

// TestRenderMarkdown checks the evidence doc renders the verdict, fence, and the
// conflation-trap line.
func TestRenderMarkdown(t *testing.T) {
	c := Build(ContractInput{GeneratedAt: "2026-06-27T00:00:00Z", CompletionPathContentFilter: false})
	md := RenderMarkdown(c)
	for _, want := range []string{
		"AILuminate Model+Guardrail SUT",
		"NO_GO_SCOPED_OUT",
		"Adjacency verdict",
		"Child Sexual Exploitation",
		"Result claim allowed: `false`",
		"never a fak score",
		"Honesty fence",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}
