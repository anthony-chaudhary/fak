package taskmgr

import (
	"encoding/json"
	"strings"
	"testing"
)

// Completion handoffs must name the maturity they actually achieved (#4640):
// under the strict project-work gate a "done" handoff with no achieved maturity
// is refused instead of silently reading as production-ready.
func TestReviewHandoffStrictProjectWorkRequiresAchievedMaturity(t *testing.T) {
	h := verifiedHandoff()
	h.AchievedMaturity = ""
	review := ReviewHandoffWithOptions(h, HandoffReviewOptions{StrictProjectWork: true})
	if review.OK || !contains(review.Reasons, "MISSING_ACHIEVED_MATURITY") {
		t.Fatalf("maturity-less strict review = %+v, want MISSING_ACHIEVED_MATURITY", review)
	}

	h.AchievedMaturity = "demo"
	review = ReviewHandoffWithOptions(h, HandoffReviewOptions{StrictProjectWork: true})
	if contains(review.Reasons, "MISSING_ACHIEVED_MATURITY") || contains(review.Reasons, "BAD_ACHIEVED_MATURITY") {
		t.Fatalf("declared demo maturity refused: %+v", review.Reasons)
	}
	if review.AchievedMaturity != "demo" {
		t.Fatalf("review achieved maturity = %q, want demo", review.AchievedMaturity)
	}
}

// An achieved maturity outside the closed completion-standard vocabulary is a
// refusal on every path (strict or not) — never a guess.
func TestReviewHandoffRefusesUnknownAchievedMaturity(t *testing.T) {
	h := verifiedHandoff()
	h.AchievedMaturity = "done-ish"
	review := ReviewHandoff(h)
	if review.OK || !contains(review.Reasons, "BAD_ACHIEVED_MATURITY") {
		t.Fatalf("unknown maturity review = %+v, want BAD_ACHIEVED_MATURITY", review)
	}
}

// The default (non-strict) path stays backward compatible: an undeclared
// achieved maturity is not refused, and the review JSON keeps stable field
// names for downstream consumers.
func TestReviewHandoffDefaultPathKeepsUndeclaredMaturityCompatible(t *testing.T) {
	review := ReviewHandoff(verifiedHandoff())
	if !review.OK {
		t.Fatalf("default review = %+v, want ok", review)
	}
	if review.AchievedMaturity != "" {
		t.Fatalf("undeclared maturity must stay empty, got %q", review.AchievedMaturity)
	}

	withMaturity := verifiedHandoff()
	withMaturity.AchievedMaturity = "demo complete"
	b, err := json.Marshal(ReviewHandoff(withMaturity))
	if err != nil {
		t.Fatalf("marshal review: %v", err)
	}
	if !strings.Contains(string(b), `"achieved_maturity":"demo"`) {
		t.Fatalf("review JSON missing stable achieved_maturity field:\n%s", b)
	}
}

// The handoff-produced issue body — a status surface the next agent and the
// operator read — names the achieved maturity, and renders an undeclared one
// as explicitly not production-complete, never as a bare "complete" (#4640).
func TestHandoffIssueBodyNamesAchievedMaturity(t *testing.T) {
	h := verifiedHandoff()
	h.AchievedMaturity = "demo"
	body := HandoffIssueBody(h, h.NextSteps[0])
	if !strings.Contains(body, "- Achieved maturity: `demo-complete`") {
		t.Fatalf("issue body missing demo-complete maturity line:\n%s", body)
	}

	h.AchievedMaturity = ""
	body = HandoffIssueBody(h, h.NextSteps[0])
	if !strings.Contains(body, "- Achieved maturity: undeclared (treat as not production-complete)") {
		t.Fatalf("issue body missing undeclared-maturity line:\n%s", body)
	}
}

// The witness fixture from #4640: a closed toy bring-up hands off as
// demo-complete — the handoff cannot be read as the parent's production
// completion because the maturity is named right beside the claimed state.
func TestHandoffToyBringupReadsDemoCompleteNotProduction(t *testing.T) {
	h := verifiedHandoff()
	h.Task.Title = "Toy model bring-up on one prompt"
	h.AchievedMaturity = "demo"
	body := HandoffIssueBody(h, h.NextSteps[0])
	if !strings.Contains(body, "`demo-complete`") {
		t.Fatalf("toy bring-up handoff must read demo-complete:\n%s", body)
	}
	if strings.Contains(body, "`production-complete`") {
		t.Fatalf("toy bring-up handoff must not read production-complete:\n%s", body)
	}
}
