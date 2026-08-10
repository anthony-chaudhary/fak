package issuecohort

// wavemember_issue_test.go — issue #5040: a wave member carries the live issue
// number it was planned from, so a downstream reader can bind a landed commit's
// `#N` back to the wave the work was DECIDED in. Without it, the only join
// between a commit and its wave would be a second key invented outside the plan.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// TestWaveMemberCarriesIssueNumber: a cohort planned over EXISTING issues
// (`fak issue cohort --from-issues`) puts each member's issue number on its wave
// row, and the number is the review's own — not re-derived here.
func TestWaveMemberCarriesIssueNumber(t *testing.T) {
	a := fullCandidate("alpha")
	a.IssueNumber = 701
	a.Paths = []string{"internal/gateway/**"}
	b := fullCandidate("beta")
	b.IssueNumber = 702
	b.Paths = []string{"internal/model/**"}

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.NumWaves != 1 || len(plan.Waves[0].Members) != 2 {
		t.Fatalf("waves = %+v, want one wave of 2 disjoint members", plan.Waves)
	}
	got := map[string]int{}
	for _, m := range plan.Waves[0].Members {
		got[m.Key] = m.IssueNumber
	}
	if got["alpha"] != 701 || got["beta"] != 702 {
		t.Fatalf("member issue numbers = %v, want alpha=701 beta=702", got)
	}

	// The number survives the JSON the overlay actually reads.
	raw, err := json.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"issue_number":701`) {
		t.Fatalf("plan json lost the issue number: %s", raw)
	}
}

// TestWaveMemberOmitsUnfiledIssueNumber: a cohort planned over not-yet-filed
// candidates has nothing to carry, and says so by omission rather than by
// inventing a placeholder. A downstream reader must be able to tell "this plan
// cannot bind to commits yet" from "this member is issue 0".
func TestWaveMemberOmitsUnfiledIssueNumber(t *testing.T) {
	plan := Build([]issuepolicy.Candidate{fullCandidate("alpha")}, Options{})
	if len(plan.Waves) != 1 || len(plan.Waves[0].Members) != 1 {
		t.Fatalf("waves = %+v, want one wave of 1", plan.Waves)
	}
	if n := plan.Waves[0].Members[0].IssueNumber; n != 0 {
		t.Fatalf("unfiled member issue number = %d, want 0", n)
	}
	raw, err := json.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(raw), "issue_number") {
		t.Fatalf("unfiled member emitted an issue_number key: %s", raw)
	}
}
