package issuepolicy

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/flowmetrics"
)

// TestRequiredSectionsHeadingsPresentInCanonicalDraft asserts every heading the
// contract claims is required actually appears in the canonical dispatchable
// draft, so the contract never advertises a heading the canonical body omits.
func TestRequiredSectionsHeadingsPresentInCanonicalDraft(t *testing.T) {
	base := CanonicalDispatchableDraft()
	// The filing-time checks (scope_class, definition_of_done) gate only a draft
	// with no issue number; probe them against a not-yet-filed variant.
	unfiled := base
	unfiled.Number = 0
	body := base.Body
	sections := markdownSections(body)
	for _, s := range RequiredSections().Sections {
		if !s.Required {
			continue
		}
		if s.Source == "discoverability" {
			// scope_class and definition_of_done are predicate checks over the
			// body (a task-list and a done-condition phrase), not heading-block
			// lookups; assert those predicates hold separately below.
			continue
		}
		found := false
		for _, h := range s.Headings {
			if strings.TrimSpace(sections[normalizeHeading(h)]) != "" {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("required section %q: none of %v appear in CanonicalDispatchableDraft body", s.Field, s.Headings)
		}
	}
	// scope_class and definition_of_done are filing-time predicate checks that
	// fire only for an unfiled draft (Number==0). The canonical body omits both,
	// so the unfiled variant is TriageOnly: that is the witness that the gate
	// demands them, and that the contract lists them as required.
	unfiledReview := ReviewIssueDraft(unfiled, Options{})
	if unfiledReview.Dispatchability == Dispatchable {
		t.Errorf("unfiled canonical draft dispatched without a scope class / done-condition phrase; scope_class and definition_of_done are not gating")
	}
	if !containsString(unfiledReview.MissingSections, "scope_class") {
		t.Errorf("unfiled draft not held for scope_class: %v", unfiledReview.MissingSections)
	}
	if !containsString(unfiledReview.MissingSections, "definition_of_done") {
		t.Errorf("unfiled draft not held for definition_of_done: %v", unfiledReview.MissingSections)
	}
	if flowmetrics.ClassifyScope(unfiled.Body) != flowmetrics.ScopeUndeclared {
		t.Errorf("canonical draft unexpectedly declares a scope class; the scope_class contract entry is untested")
	}
	if flowmetrics.HasDoD(unfiled.Body) {
		t.Errorf("canonical draft unexpectedly names a done-condition phrase; the definition_of_done contract entry is untested")
	}
	_ = body
}

// TestRequiredSectionsGateOnDispatch asserts each required section names at
// least one heading whose removal from the canonical draft flips dispatchability
// away from Dispatchable, cross-checked against the behavior-derived
// ReachabilityContract GatingSections.
func TestRequiredSectionsGateOnDispatch(t *testing.T) {
	base := CanonicalDispatchableDraft()
	if got := ReviewIssueDraft(base, Options{}).Dispatchability; got != Dispatchable {
		t.Fatalf("canonical draft is not dispatchable (%s); contract probe is meaningless", got)
	}

	gating := map[string]bool{}
	for _, h := range ReachabilityContract().GatingSections {
		gating[h] = true
	}

	sections := markdownSections(base.Body)
	for _, s := range RequiredSections().Sections {
		if !s.Required || s.Source == "discoverability" {
			continue
		}
		gated := false
		for _, h := range s.Headings {
			norm := normalizeHeading(h)
			if strings.TrimSpace(sections[norm]) == "" {
				continue
			}
			if gating[norm] || sectionRemovalFlipsDispatch(base, norm) {
				gated = true
				break
			}
		}
		if !gated {
			t.Errorf("required section %q does not gate dispatch; remove it from the contract or mark Required:false", s.Field)
		}
	}
}

// sectionRemovalFlipsDispatch removes one normalized heading block from the
// canonical draft and reports whether dispatchability left Dispatchable.
func sectionRemovalFlipsDispatch(base IssueDraft, heading string) bool {
	variant := base
	variant.Body = removeHeadingBlock(base.Body, heading)
	return ReviewIssueDraft(variant, Options{}).Dispatchability != Dispatchable
}

// TestRequiredSectionsDoesNotOverClaimNonGatingSections asserts that the
// brief-only sections (Parent context, Why this is next, Working spine,
// Acceptance gate, Closure binding) are NOT advertised as required, because the
// reviewer does not list them in missing_required and removing them from the
// canonical draft does not strand it.
func TestRequiredSectionsDoesNotOverClaimNonGatingSections(t *testing.T) {
	base := CanonicalDispatchableDraft()
	for _, heading := range []string{"Parent context", "Why this is next", "Working spine", "Acceptance gate", "Closure binding"} {
		if sectionRemovalFlipsDispatch(base, normalizeHeading(heading)) {
			t.Errorf("%q now gates dispatch; the contract must list it as required", heading)
		}
	}
	for _, s := range RequiredSections().Sections {
		switch normalizeHeading(s.Headings[0]) {
		case normalizeHeading("Parent context"), normalizeHeading("Why this is next"),
			normalizeHeading("Working spine"), normalizeHeading("Acceptance gate"),
			normalizeHeading("Closure binding"):
			t.Errorf("contract over-claims non-gating brief section %q as required", s.Field)
		}
	}
}

// TestRequiredSectionsProblemFrameMatchesAssessor asserts the problem-frame
// requirement reflects AssessProblemFrame: a draft carrying the declared
// vocabulary is enforced, and a complete frame is ready.
func TestRequiredSectionsProblemFrameMatchesAssessor(t *testing.T) {
	frame := RequiredSections().ProblemFrame
	if frame.Section != "Value" {
		t.Errorf("problem frame section = %q, want Value", frame.Section)
	}
	if frame.CentralityLine != "Centrality:" {
		t.Errorf("centrality line = %q, want Centrality:", frame.CentralityLine)
	}
	for _, want := range []string{"P1", "P2", "P3", "P4"} {
		found := false
		for _, got := range frame.CheckLabels {
			if got == want {
				found = true
			}
		}
		if !found {
			t.Errorf("check labels missing %q: %v", want, frame.CheckLabels)
		}
	}

	body := "# Value\n\nCentrality: Core\n\nP1: advanced - gate is real\nP2: preserved - no regression\nP3: advanced - new coverage\nP4: N/A - not applicable\n"
	got := AssessProblemFrame(IssueDraft{Title: "t", Body: body})
	if !got.Enforced || !got.Ready {
		t.Errorf("assessor on complete frame: enforced=%t ready=%t reasons=%v", got.Enforced, got.Ready, got.Reasons)
	}
}

// TestRequiredSectionsProjectWorkMatchesGate asserts the project-work headings
// the contract lists are the ones AppendProjectWorkDefaults recognizes, and the
// completion-standard vocabulary matches knownCompletionStandard.
func TestRequiredSectionsProjectWorkMatchesGate(t *testing.T) {
	for _, s := range RequiredSections().ProjectWorkSections {
		if len(s.Headings) == 0 {
			t.Errorf("project-work section %q has no headings", s.Field)
		}
	}
	for _, std := range completionStandards {
		if !knownCompletionStandard(std) {
			t.Errorf("completion standard %q not recognized by gate", std)
		}
	}
	body, err := AppendProjectWorkDefaults("## Parent context\n\n#1\n", ProjectWorkAuthoring{
		EstimatePoints: 3, ParentBaseline: 10, CompletionStandard: "research",
	})
	if err != nil {
		t.Fatalf("AppendProjectWorkDefaults: %v", err)
	}
	for _, s := range RequiredSections().ProjectWorkSections {
		if !headingPresent(body, s.Headings) {
			t.Errorf("authoring path did not emit required heading for %q (%v)", s.Field, s.Headings)
		}
	}
}

// TestRequiredSectionsJSONSchema asserts the JSON encoder yields the declared
// schema key and a stable round-trip.
func TestRequiredSectionsJSONSchema(t *testing.T) {
	b, err := RequiredSectionsJSON()
	if err != nil {
		t.Fatalf("RequiredSectionsJSON: %v", err)
	}
	var decoded RequiredSectionsContract
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("unmarshal contract: %v", err)
	}
	if decoded.Schema != RequiredSectionsSchema {
		t.Errorf("schema = %q, want %q", decoded.Schema, RequiredSectionsSchema)
	}
	if len(decoded.Sections) == 0 {
		t.Error("contract emitted no required sections")
	}
	if decoded.ProblemFrame.Section != "Value" {
		t.Errorf("problem frame section = %q, want Value", decoded.ProblemFrame.Section)
	}
}

// headingPresent reports whether any of names appears as a markdown heading.
func headingPresent(body string, names []string) bool {
	sections := markdownSections(body)
	for _, n := range names {
		if strings.TrimSpace(sections[normalizeHeading(n)]) != "" {
			return true
		}
	}
	return false
}

// removeHeadingBlock drops the block under the normalized heading from the
// rendered canonical body, preserving every other section.
func removeHeadingBlock(body, heading string) string {
	var kept []reachabilitySection
	sections := canonicalDispatchableSections()
	for _, s := range sections {
		if normalizeHeading(s.Heading) == heading {
			continue
		}
		kept = append(kept, s)
	}
	return renderReachabilitySections(kept)
}
