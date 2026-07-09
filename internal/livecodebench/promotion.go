package livecodebench

import "fmt"

// promotion.go pins the promotion contract every LiveCodeBench report carries
// (#2114, epic #2085), copying the terminalbench honesty pattern: every report
// stamps an evidence_class plus the promotion_requirements checklist, a fixture
// run stamps the simulated class (EvidenceFixtureSmoke — LCB's spelling of
// terminalbench's SIMULATED_LOCAL_FIXTURE), and only the official lcb_runner
// grading over the exact saved generations may stamp EvidenceOfficialLCBRunner
// and flip result_claim_allowed.

// PromotionRequirements is the canonical checklist a local LiveCodeBench number
// must clear before it may be promoted to a claimable score: the pinned problem
// ids, the release + contest-date window, both arms' saved generations, the
// official grader's output, and the same-config identity across arms. Every
// report stamps it verbatim; it returns a fresh slice each call so a caller
// cannot mutate the shared contract.
func PromotionRequirements() []string {
	return []string{
		"problem-ids-pinned-and-identical-across-arms",
		"release-version-and-date-window-recorded",
		"both-arms-generations-saved-with-digest",
		"official-lcb-runner-grader-output-recorded",
		"same-config-across-arms",
	}
}

// MarkOfficiallyGraded promotes a report after the official lcb_runner
// evaluator has graded the exact saved generations: it stamps the official
// evidence class and flips result_claim_allowed, then re-validates the whole
// report so a promotion can never produce an invalid artifact. It refuses a
// report with no graded arm results — with nothing graded there is nothing to
// promote, and the report stays at its local evidence class.
func (r *Report) MarkOfficiallyGraded() error {
	graded := false
	for _, arm := range r.Arms {
		if arm.Graded > 0 {
			graded = true
			break
		}
	}
	if !graded {
		return fmt.Errorf("livecodebench report: cannot mark officially graded: no arm carries graded results")
	}
	r.EvidenceClass = EvidenceOfficialLCBRunner
	r.ResultClaimAllowed = true
	return r.Validate()
}
