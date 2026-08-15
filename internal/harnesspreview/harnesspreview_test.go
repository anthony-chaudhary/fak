package harnesspreview

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessclassify"
	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestCompareUnchangedIsSilent(t *testing.T) {
	lock := fixtureLock("sha256:same", policy("company", []string{"search"}, []string{"shell"}, true))
	got := Compare(Input{Current: &lock, Candidate: &lock, CurrentDomain: "coding", CandidateDomain: "coding"})
	if got.RequiresDecision || got.Verdict != VerdictQuiet || RenderCLI(got) != "" || RenderTUI(got) != "" {
		t.Fatalf("unchanged launch was not quiet: %#v cli=%q tui=%q", got, RenderCLI(got), RenderTUI(got))
	}
}

func TestCompareDistinctReasonsNameLayerCapabilityConsequenceAndChoice(t *testing.T) {
	current := fixtureLock("sha256:old", policy("company", []string{"search"}, []string{"shell"}, true))
	candidate := fixtureLock("sha256:new",
		policy("task-legal", []string{"search", "shell"}, nil, false),
		harnesscompose.EffectiveAsset{Kind: "tool", ID: "payments", Source: "task-legal"},
	)
	classification := harnessclassify.Result{Confidence: .4, NeedsDecision: true, DecisionRequest: &harnessclassify.DecisionRequest{Reason: "legal and integrated signals tie", Scope: "project:matter-7"}}
	got := Compare(Input{Current: &current, Candidate: &candidate, CurrentDomain: "coding", CandidateDomain: "legal", Classification: &classification, Conflict: "component route requires incompatible contract v2"})
	for _, reason := range []string{"conflict", "novel-domain", "privilege-widening", "low-confidence"} {
		if !hasReason(got, reason) {
			t.Errorf("missing reason %q: %#v", reason, got.Changes)
		}
	}
	if !got.RequiresDecision || len(got.Recovery) != 3 {
		t.Fatalf("decision contract incomplete: %#v", got)
	}
	for _, c := range got.Changes {
		if c.Layer == "" || c.Capability == "" || c.Consequence == "" || c.ReversibleChoice == "" {
			t.Errorf("incomplete change: %#v", c)
		}
	}
}

func TestComparePolicyDenyRemovalIsPrivilegeWidening(t *testing.T) {
	old := fixtureLock("old", policy("company", []string{"search"}, []string{"shell"}, true))
	next := fixtureLock("new", policy("repo", []string{"search"}, nil, true))
	got := Compare(Input{Current: &old, Candidate: &next})
	if !hasCapability(got, "policy:tools:deny:shell") {
		t.Fatalf("deny removal not surfaced: %#v", got.Changes)
	}
}

func TestCapturedRiskRenderIsBoundedAndANSIFree(t *testing.T) {
	old := fixtureLock("old")
	next := fixtureLock("new", harnesscompose.EffectiveAsset{Kind: "tool", ID: "deploy", Source: "task:release"})
	got := Compare(Input{Current: &old, Candidate: &next})
	want := "HARNESS PREVIEW | decision required\n- privilege-widening | task:release | tool:deploy\n  makes the deploy tool callable; choice: keep the current lock\nchoices: approve-once | remember | keep-current\n"
	if text := RenderTUI(got); text != want {
		t.Fatalf("captured render mismatch\nwant=%q\n got=%q", want, text)
	}
	if strings.Contains(RenderTUI(got), "\x1b") || strings.Count(RenderTUI(got), "HARNESS PREVIEW") != 1 {
		t.Fatalf("pane corruption: %q", RenderTUI(got))
	}
}

func TestCapturedCLIRenderNamesDecisionWithoutANSI(t *testing.T) {
	old := fixtureLock("old")
	next := fixtureLock("new", harnesscompose.EffectiveAsset{Kind: "tool", ID: "deploy", Source: "task:release"})
	got := Compare(Input{Current: &old, Candidate: &next})
	want := "contextual harness decision required\n- privilege-widening | task:release | tool:deploy\n  makes the deploy tool callable; choice: keep the current lock\nchoices: approve-once | remember | keep-current\n"
	if text := RenderCLI(got); text != want || strings.Contains(text, "\x1b") {
		t.Fatalf("captured CLI render mismatch\nwant=%q\n got=%q", want, text)
	}
}

func TestFirstRiskyLockRequiresDecision(t *testing.T) {
	next := fixtureLock("new", harnesscompose.EffectiveAsset{Kind: "secret", ID: "prod", Ref: "vault:prod", Source: "task:deploy"})
	got := Compare(Input{Candidate: &next})
	if !hasCapability(got, "secret:prod") {
		t.Fatalf("first risky lock launched quietly: %#v", got)
	}
}
func fixtureLock(id string, assets ...harnesscompose.EffectiveAsset) harnessresolve.Lock {
	return harnessresolve.Lock{Schema: harnessresolve.LockSchema, ID: id, Assets: assets}
}
func policy(source string, grants, denies []string, locked bool) harnesscompose.EffectiveAsset {
	return harnesscompose.EffectiveAsset{Kind: "policy", ID: "tools", Source: source, Grants: grants, Denies: denies, Locked: locked}
}
func hasReason(p Preview, reason string) bool {
	for _, c := range p.Changes {
		if c.Reason == reason {
			return true
		}
	}
	return false
}
func hasCapability(p Preview, capability string) bool {
	for _, c := range p.Changes {
		if c.Capability == capability {
			return true
		}
	}
	return false
}
