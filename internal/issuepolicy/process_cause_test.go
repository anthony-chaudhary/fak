package issuepolicy

import (
	"strings"
	"testing"
)

func newIssueBodyWithProcessCause(lines ...string) string {
	body := CanonicalDispatchableDraft().Body + "## Definition of done\n\n- [ ] process cause recorded\n\n"
	if len(lines) > 0 {
		body += strings.Join(lines, "\n") + "\n"
	}
	return body
}

func reviewHasProcessCauseFailure(review Review) bool {
	for _, value := range append(append([]string{}, review.Reasons...), review.MissingFields...) {
		normalized := strings.NewReplacer("_", " ", "-", " ").Replace(strings.ToLower(value))
		if strings.Contains(normalized, "process cause") {
			return true
		}
	}
	return false
}

func TestNewIssueProcessCauseClosedVocabulary(t *testing.T) {
	valid := []string{
		"concurrency", "infrastructure-lag", "model-failure", "harness-failure",
		"scoping-failure", "verification-gap", "handoff-failure", "other", "unknown", "none",
	}
	for _, cause := range valid {
		t.Run(cause, func(t *testing.T) {
			lines := []string{"Process cause: " + cause}
			if cause == "concurrency" {
				lines = append(lines, "Process cause detail: shared-state")
			}
			review := ReviewIssueDraft(IssueDraft{Title: "bug(runtime): product symptom", Body: newIssueBodyWithProcessCause(lines...)}, Options{})
			if reviewHasProcessCauseFailure(review) {
				t.Fatalf("valid cause %q was rejected: reasons=%v missing=%v", cause, review.Reasons, review.MissingFields)
			}
		})
	}

	for _, tc := range []struct {
		name  string
		lines []string
	}{
		{name: "missing"},
		{name: "invalid", lines: []string{"Process cause: flaky-ci"}},
		{name: "duplicate", lines: []string{"Process cause: none", "Process cause: none"}},
		{name: "fenced-only-example", lines: []string{"```text", "Process cause: verification-gap", "```"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			review := ReviewIssueDraft(IssueDraft{Title: "bug(runtime): product symptom", Body: newIssueBodyWithProcessCause(tc.lines...)}, Options{})
			if !reviewHasProcessCauseFailure(review) {
				t.Fatalf("invalid process cause was not rejected: reasons=%v missing=%v", review.Reasons, review.MissingFields)
			}
		})
	}
}

func TestConcurrencyProcessCauseDetailContract(t *testing.T) {
	for _, detail := range []string{"shared-state", "lease-contention", "integration-order", "resource-contention", "ownership-overlap"} {
		t.Run(detail, func(t *testing.T) {
			review := ReviewIssueDraft(IssueDraft{
				Title: "bug(runtime): concurrent product symptom",
				Body: newIssueBodyWithProcessCause(
					"Process cause: concurrency",
					"Process cause detail: "+detail,
				),
			}, Options{})
			if reviewHasProcessCauseFailure(review) {
				t.Fatalf("valid concurrency detail %q was rejected: reasons=%v missing=%v", detail, review.Reasons, review.MissingFields)
			}
		})
	}

	for _, tc := range []struct {
		name  string
		lines []string
	}{
		{name: "detail-missing", lines: []string{"Process cause: concurrency"}},
		{name: "detail-invalid", lines: []string{"Process cause: concurrency", "Process cause detail: lock-race"}},
		{name: "detail-on-other-cause", lines: []string{"Process cause: verification-gap", "Process cause detail: shared-state"}},
		{name: "blank-detail-on-other-cause", lines: []string{"Process cause: none", "Process cause detail:"}},
		{name: "duplicate-detail", lines: []string{"Process cause: concurrency", "Process cause detail: shared-state", "Process cause detail: resource-contention"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			review := ReviewIssueDraft(IssueDraft{Title: "bug(runtime): product symptom", Body: newIssueBodyWithProcessCause(tc.lines...)}, Options{})
			if !reviewHasProcessCauseFailure(review) {
				t.Fatalf("invalid detail contract was not rejected: reasons=%v missing=%v", review.Reasons, review.MissingFields)
			}
		})
	}
}

func TestProcessCauseIsIndependentOfProductTypeAndLane(t *testing.T) {
	for _, draft := range []IssueDraft{
		{Title: "bug(runtime): wrong token", Labels: []IssueLabel{{Name: "bug"}}, Body: strings.Replace(newIssueBodyWithProcessCause("Process cause: handoff-failure"), "## Lane\n\ntools", "## Lane\n\ncompute", 1)},
		{Title: "docs(cli): stale example", Labels: []IssueLabel{{Name: "documentation"}}, Body: strings.Replace(newIssueBodyWithProcessCause("Process cause: handoff-failure"), "## Lane\n\ntools", "## Lane\n\ndocs", 1)},
	} {
		review := ReviewIssueDraft(draft, Options{})
		if reviewHasProcessCauseFailure(review) {
			t.Fatalf("product type or lane changed causal classification: title=%q reasons=%v missing=%v", draft.Title, review.Reasons, review.MissingFields)
		}
	}
}

func TestHistoricalNumberedIssueAuditDoesNotRequireProcessCause(t *testing.T) {
	draft := CanonicalDispatchableDraft()
	review := ReviewIssueDraft(draft, Options{})
	if reviewHasProcessCauseFailure(review) {
		t.Fatalf("historical issue #%d was retroactively rejected: reasons=%v missing=%v", draft.Number, review.Reasons, review.MissingFields)
	}
}

func TestTagGeneratedIssuePreservesOrAddsProcessCause(t *testing.T) {
	t.Run("preserves explicit valid cause", func(t *testing.T) {
		body := "generated body\n\nProcess cause: verification-gap\n"
		gotBody, gotLabel, err := TagGeneratedIssue(body, "unknown")
		if err != nil || gotBody != body || gotLabel != "process-cause:verification-gap" {
			t.Fatalf("TagGeneratedIssue() = %q, %q, %v", gotBody, gotLabel, err)
		}
	})

	t.Run("adds honest fallback when absent", func(t *testing.T) {
		gotBody, gotLabel, err := TagGeneratedIssue("generated body", "unknown")
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(gotBody, "Process cause: unknown") || gotLabel != "process-cause:unknown" {
			t.Fatalf("body=%q label=%q", gotBody, gotLabel)
		}
	})

	for _, body := range []string{
		"Process cause: flaky-ci",
		"Process cause: none\nProcess cause: unknown",
	} {
		t.Run("refuses malformed explicit declaration", func(t *testing.T) {
			if _, _, err := TagGeneratedIssue(body, "unknown"); err == nil {
				t.Fatalf("TagGeneratedIssue accepted %q", body)
			}
		})
	}
}
