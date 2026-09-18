package issuepolicy

// required_sections_contract_test.go — adversarial, spec-driven conformance tests
// for the canonical required-section contract (fak.issue-required-sections/1).
// Written from the public contract spec, independent of the implementation's
// internal choices: every assertion is checked against live gate behavior.

import (
	"encoding/json"
	"strings"
	"testing"
)

// rsContractHeadingMatches reports whether any of the contract entry's Headings
// equals the normalized canonical heading.
func rsContractHeadingMatches(entry RequiredSection, normHeading string) bool {
	for _, h := range entry.Headings {
		if normalizeHeading(h) == normHeading {
			return true
		}
	}
	return false
}

// rsRenderWithoutSection rebuilds the canonical body with the section whose
// normalized heading matches `heading` omitted entirely (heading + body gone).
// It reuses the in-package canonical section list so the variant differs from
// the canonical body in exactly that one block.
func rsRenderWithoutSection(heading string) string {
	var kept []reachabilitySection
	for _, s := range canonicalDispatchableSections() {
		if normalizeHeading(s.Heading) == heading {
			continue
		}
		kept = append(kept, s)
	}
	return renderReachabilitySections(kept)
}

// rsRenderWithEmptySection rebuilds the canonical body keeping every section
// heading but replacing the matched section's body with an empty string. This
// models a heading present with an EMPTY body.
func rsRenderWithEmptySection(heading string) string {
	var out []reachabilitySection
	for _, s := range canonicalDispatchableSections() {
		if normalizeHeading(s.Heading) == heading {
			out = append(out, reachabilitySection{Heading: s.Heading, Body: ""})
			continue
		}
		out = append(out, s)
	}
	return renderReachabilitySections(out)
}

// rsNumberZeroDispatchableBase is a Number-0 (not-yet-filed) draft that satisfies
// the filing-time predicates (a declared scope-class checklist and a DoD phrase)
// AND the full review gate, so it starts Dispatchable. The filing-time
// satisfiers are placed in a section whose canonical name is neither "done
// condition" nor a review-section heading, so they cannot mask a review-section
// removal. Probing against this base means each required review section is
// checked with the filing-time checks ALSO active, as the spec directs.
func rsNumberZeroSections() []reachabilitySection {
	sections := append([]reachabilitySection{}, canonicalDispatchableSections()...)
	sections = append(sections, reachabilitySection{
		Heading: "Filing checklist",
		// Two task-list items declare an atomic scope class; the phrase
		// "definition of done" satisfies HasDoD without being a heading that
		// aliases to the done-condition review section.
		Body: "- [ ] land the guarded change\n- [ ] close when the definition of done is met",
	})
	return sections
}

func rsNumberZeroDispatchableBase() IssueDraft {
	d := CanonicalDispatchableDraft()
	d.Number = 0
	d.Body = renderReachabilitySections(rsNumberZeroSections())
	return d
}

// rsRemoveSectionFrom renders the Number-0 section list with the section whose
// normalized heading matches `heading` omitted, preserving the filing-time
// satisfiers so only the removed review section can change the verdict.
func rsRemoveSectionFrom(sections []reachabilitySection, heading string) string {
	var kept []reachabilitySection
	for _, s := range sections {
		if normalizeHeading(s.Heading) == heading {
			continue
		}
		kept = append(kept, s)
	}
	return renderReachabilitySections(kept)
}

func rsContains(list []string, want string) bool {
	for _, got := range list {
		if got == want {
			return true
		}
	}
	return false
}

// TestRequiredSectionsContract_JSONRoundTrip pins the wire schema: the encoder
// output unmarshals into the contract type and declares the exact schema token.
func TestRequiredSectionsContract_JSONRoundTrip(t *testing.T) {
	raw, err := RequiredSectionsJSON()
	if err != nil {
		t.Fatalf("RequiredSectionsJSON() error: %v", err)
	}
	var decoded RequiredSectionsContract
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("RequiredSectionsJSON() output does not unmarshal into RequiredSectionsContract: %v", err)
	}
	if decoded.Schema != RequiredSectionsSchema {
		t.Fatalf("round-trip schema = %q, want %q", decoded.Schema, RequiredSectionsSchema)
	}
	if RequiredSectionsSchema != "fak.issue-required-sections/1" {
		t.Fatalf("RequiredSectionsSchema = %q, want the pinned literal %q", RequiredSectionsSchema, "fak.issue-required-sections/1")
	}
	if len(decoded.Sections) == 0 {
		t.Fatal("decoded contract carries zero Sections")
	}
	// The in-memory constructor and the JSON encoder must agree.
	if got := RequiredSections(); got.Schema != decoded.Schema {
		t.Fatalf("RequiredSections().Schema = %q but JSON schema = %q", got.Schema, decoded.Schema)
	}
}

// TestRequiredSectionsContract_RequiredReviewSectionsGateDispatch proves the
// contract is behaviorally true: for every Required review-sourced section, at
// least one advertised heading's removal from the canonical body (with Number 0
// so filing-time checks also apply) makes dispatchability leave Dispatchable.
func TestRequiredSectionsContract_RequiredReviewSectionsGateDispatch(t *testing.T) {
	base := rsNumberZeroDispatchableBase()
	if got := ReviewIssueDraft(base, Options{}).Dispatchability; got != Dispatchable {
		t.Fatalf("Number-0 base is not dispatchable (%s); the contract probe is meaningless", got)
	}

	checked := 0
	for _, entry := range RequiredSections().Sections {
		if !entry.Required || entry.Source != "review" {
			continue
		}
		checked++
		gated := false
		for _, h := range entry.Headings {
			norm := normalizeHeading(h)
			variant := base
			variant.Body = rsRemoveSectionFrom(rsNumberZeroSections(), norm)
			if ReviewIssueDraft(variant, Options{}).Dispatchability != Dispatchable {
				gated = true
				break
			}
		}
		if !gated {
			t.Errorf("required review section %q advertises %v, but no single heading removal leaves Dispatchable", entry.Field, entry.Headings)
		}
	}
	if checked == 0 {
		t.Fatal("contract lists no Required review-sourced sections; nothing was verified")
	}
}

// TestRequiredSectionsContract_DoesNotOverClaimNonGatingSections is the
// converse guard: brief-only headings must neither be advertised as required
// review sections nor gate dispatch when removed.
func TestRequiredSectionsContract_DoesNotOverClaimNonGatingSections(t *testing.T) {
	nonGating := []string{"Parent context", "Why this is next", "Working spine", "Acceptance gate", "Closure binding"}

	base := CanonicalDispatchableDraft()
	for _, h := range nonGating {
		norm := normalizeHeading(h)
		variant := base
		variant.Body = rsRenderWithoutSection(norm)
		if got := ReviewIssueDraft(variant, Options{}).Dispatchability; got != Dispatchable {
			t.Errorf("removing non-gating heading %q changed dispatchability to %s; the contract MUST list it required or the reviewer has a hidden requirement", h, got)
		}
	}

	for _, entry := range RequiredSections().Sections {
		if !entry.Required {
			continue
		}
		for _, h := range nonGating {
			if rsContractHeadingMatches(entry, normalizeHeading(h)) {
				t.Errorf("contract over-claims non-gating heading %q as required under field %q", h, entry.Field)
			}
		}
	}
}

// TestRequiredSectionsContract_ProblemFrameCompleteThenCeremonial pins the
// problem-frame requirement against AssessProblemFrame: a complete frame is
// enforced+ready, and a ceremonial bare check is enforced but not ready.
func TestRequiredSectionsContract_ProblemFrameCompleteThenCeremonial(t *testing.T) {
	complete := "# Value\n\nCentrality: Core\n\nP1: advanced - the gate is load-bearing\nP2: preserved - no regression in existing path\nP3: advanced - adds new coverage\nP4: N/A - orthogonal axis not touched\n"
	got := AssessProblemFrame(IssueDraft{Title: "t", Body: complete})
	if !got.Enforced {
		t.Fatalf("complete frame not Enforced; reasons=%v", got.Reasons)
	}
	if !got.Ready {
		t.Fatalf("complete frame not Ready; reasons=%v", got.Reasons)
	}

	ceremonial := "# Value\n\nCentrality: Core\n\nP1: advanced\nP2: preserved - evidence here\nP3: advanced - evidence here\nP4: N/A - not applicable here\n"
	bad := AssessProblemFrame(IssueDraft{Title: "t", Body: ceremonial})
	if !bad.Enforced {
		t.Fatalf("ceremonial frame not Enforced; reasons=%v", bad.Reasons)
	}
	if bad.Ready {
		t.Fatal("ceremonial bare P1 (no evidence) was accepted as Ready; the gate must refuse a bare label")
	}

	req := RequiredSections().ProblemFrame
	if !rsContains(req.Verdicts, "N/A") {
		t.Errorf("ProblemFrame.Verdicts %v does not contain N/A", req.Verdicts)
	}
	for _, label := range []string{"P1", "P2", "P3", "P4"} {
		if !rsContains(req.CheckLabels, label) {
			t.Errorf("ProblemFrame.CheckLabels %v does not contain %s", req.CheckLabels, label)
		}
	}
}

// TestRequiredSectionsContract_CompletionStandardVocabulary proves the accepted
// completion-standard vocabulary is real: the recognizer accepts every listed
// standard and the authoring path emits a body the gate recognizes, and the
// contract advertises the Completion standard heading.
func TestRequiredSectionsContract_CompletionStandardVocabulary(t *testing.T) {
	if len(completionStandards) == 0 {
		t.Fatal("completionStandards is empty; no vocabulary to verify")
	}
	for _, std := range completionStandards {
		if !knownCompletionStandard(std) {
			t.Errorf("recognizer rejects listed completion standard %q", std)
		}
		body, err := AppendProjectWorkDefaults("## Parent context\n\n#1926\n", ProjectWorkAuthoring{
			EstimatePoints: 3, ParentBaseline: 10, CompletionStandard: std,
			TargetEnvelope:    "CPU, interactive latency",
			WitnessedEnvelope: "CPU, interactive latency observed",
		})
		if err != nil {
			t.Errorf("authoring path rejects completion standard %q: %v", std, err)
			continue
		}
		decoded := CandidateFromIssueDraft(IssueDraft{Number: 1926, Title: "t", Body: body})
		if normalizeCompletionStandard(decoded.CompletionStandard) != normalizeCompletionStandard(std) {
			t.Errorf("authored body for standard %q round-trips to CompletionStandard %q", std, decoded.CompletionStandard)
		}
	}

	foundHeading := false
	for _, entry := range RequiredSections().ProjectWorkSections {
		for _, h := range entry.Headings {
			if normalizeHeading(h) == normalizeHeading("Completion standard") {
				foundHeading = true
			}
		}
	}
	if !foundHeading {
		t.Errorf("ProjectWorkSections %+v never lists a Completion standard heading", RequiredSections().ProjectWorkSections)
	}
}

// TestRequiredSectionsContract_EmptySectionDoesNotSatisfy is the adversarial
// case: a heading present with an EMPTY body must not satisfy the gate. The
// canonical draft is otherwise dispatchable, so only the empty Witness section
// can be responsible.
func TestRequiredSectionsContract_EmptySectionDoesNotSatisfy(t *testing.T) {
	base := CanonicalDispatchableDraft()
	base.Body = rsRenderWithEmptySection(normalizeHeading("Witness"))

	review := ReviewIssueDraft(base, Options{})
	if rsContains(review.MissingSections, "witness") {
		return
	}
	if review.Dispatchability != Dispatchable {
		return
	}
	t.Fatalf("a ## Witness heading with an empty body satisfied the gate: dispatchability=%s missing_sections=%v", review.Dispatchability, review.MissingSections)
}

// TestRequiredSectionsContract_ProjectWorkSectionsNonEmpty guards the structural
// shape of every advertised entry: each carries at least one heading and a
// non-empty field name, so a consumer cannot render an empty requirement.
func TestRequiredSectionsContract_ProjectWorkSectionsNonEmpty(t *testing.T) {
	contract := RequiredSections()
	all := [][]RequiredSection{contract.Sections, contract.ProjectWorkSections, contract.ProductionSections}
	for _, group := range all {
		for _, entry := range group {
			if strings.TrimSpace(entry.Field) == "" {
				t.Errorf("contract entry with empty Field: %+v", entry)
			}
			if len(entry.Headings) == 0 {
				t.Errorf("contract entry %q advertises no headings", entry.Field)
			}
			for _, h := range entry.Headings {
				if strings.TrimSpace(h) == "" {
					t.Errorf("contract entry %q advertises an empty heading", entry.Field)
				}
			}
		}
	}
	if len(contract.DoneConditionPhrases) == 0 {
		t.Error("DoneConditionPhrases is empty")
	}
}
