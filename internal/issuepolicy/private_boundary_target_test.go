package issuepolicy

import "testing"

// privateBoundaryProbeDraft is a filed-shape draft whose body names fak-private:
// the "Boundary notes" section is the documented surface for a private
// reference, and the likely-files line carries a private path. Both the
// private-target and public-target directions must be proven on this ONE draft.
func privateBoundaryProbeDraft() IssueDraft {
	return IssueDraft{
		Number: 1981,
		Title:  "feat(compute): target-repo aware born-routed boundary",
		Body: `## Working spine
Implement target-repo aware born-routed boundary

## Current state
The born-routed gate always treats fak-private as a public leak.

## Why this is next
Unblocks private-tier issue creation.

## Parent context
#100

## Core through-line
Change -> real seam -> observable outcome -> witness.

## Gold-plating boundary
Do not add unrelated polish.

## Done condition / witness
Boundary is target aware.
Witness: go test ./internal/issuepolicy/...

## Definition of done
- [ ] target aware
- [ ] tests pass

## Acceptance gate
All unit tests pass.

## Boundary notes
Requires fak-private platform/dispatch/x.go.

## Lane
compute

## Likely files
- platform/dispatch/x.go

## Expected steps
3
`,
		Labels: []IssueLabel{{Name: "class:dev"}},
	}
}

// TestIssuePrivateBoundaryTargetPrivateAdmits pins the private-target
// direction: a review whose target is the private repo admits a draft that
// names fak-private (no ISSUE_PRIVATE_BOUNDARY reason).
func TestIssuePrivateBoundaryTargetPrivateAdmits(t *testing.T) {
	review := ReviewIssueDraft(privateBoundaryProbeDraft(), Options{TargetPrivate: true})
	if has(review.Reasons, ReasonPrivateBoundary) {
		t.Fatalf("private target must suppress %s; reasons = %+v", ReasonPrivateBoundary, review.Reasons)
	}
	if review.Dispatchability == Refused || review.Verdict == "refused" {
		t.Fatalf("private target must not refuse on the private boundary; review = %+v", review)
	}
}

// TestIssuePrivateBoundaryPublicTargetRefuses pins the public-leak direction:
// the SAME draft is refused with ISSUE_PRIVATE_BOUNDARY when the target is
// public or unspecified (zero-value Options).
func TestIssuePrivateBoundaryPublicTargetRefuses(t *testing.T) {
	for _, tc := range []struct {
		name string
		opt  Options
	}{
		{name: "explicit public", opt: Options{TargetPrivate: false}},
		{name: "unspecified zero value", opt: Options{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			review := ReviewIssueDraft(privateBoundaryProbeDraft(), tc.opt)
			if !has(review.Reasons, ReasonPrivateBoundary) {
				t.Fatalf("public target must keep %s; reasons = %+v", ReasonPrivateBoundary, review.Reasons)
			}
			if review.OK || review.Dispatchability != Refused || review.Verdict != "refused" {
				t.Fatalf("review = %+v, want refused", review)
			}
		})
	}
}

// TestIssuePrivateBoundaryCandidateSuppressesExplicitPrivateFlag pins the
// contract that TargetPrivate suppresses the reason even when the candidate
// carries the explicit Private flag, while a public target still refuses.
func TestIssuePrivateBoundaryCandidateSuppressesExplicitPrivateFlag(t *testing.T) {
	c := completeCandidate()
	c.Private = true
	c.BoundaryNotes = []string{"Requires fak-private Slack control transcript."}

	privateReview := ReviewCandidate(c, Options{TargetPrivate: true})
	if has(privateReview.Reasons, ReasonPrivateBoundary) {
		t.Fatalf("private target must suppress %s even with explicit flag; reasons = %+v", ReasonPrivateBoundary, privateReview.Reasons)
	}

	publicReview := ReviewCandidate(c, Options{})
	if !has(publicReview.Reasons, ReasonPrivateBoundary) {
		t.Fatalf("public target must keep %s; reasons = %+v", ReasonPrivateBoundary, publicReview.Reasons)
	}
	if publicReview.OK || publicReview.Dispatchability != Refused {
		t.Fatalf("public target review = %+v, want refused", publicReview)
	}
}
