package dogfoodissues

// Live strict-scope control (#1968).
//
// Every scorecard ACTION row is already graded against the shared issuecontract
// before it can become a public issue, and a row that fails review is returned as
// a SkippedRow instead of being synced. What was missing is the consequence: a
// --live run dropped those rows and still exited 0, so a bypassed review was
// indistinguishable from a clean sync and an aggregate debt row could reach the
// tracker as a broad cleanup ticket. Live sync now grades every row under strict
// root scope — the contract's own scope fields (root-point change, in/out of
// scope, done condition), a non-forgeable witness, a lane AND path hints — and
// the CLI refuses the whole run unless every row is dispatchable. A dry-run keeps
// the advisory behaviour and may still show skipped aggregate rows.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const (
	// ErrorStrictScope tags the Result of a --live run refused because at least
	// one planned row was not dispatchable under strict root scope.
	ErrorStrictScope = "strict_scope"

	// ReasonLaneMissing and ReasonPathHintsMissing split the shared contract's
	// lane-OR-paths route test into the lane-AND-paths pair a live sync demands:
	// a lane alone routes a worker to a leaf but never to the files it may touch,
	// and path hints alone leave the lane lease unnameable.
	ReasonLaneMissing      = "ISSUE_LANE_MISSING"
	ReasonPathHintsMissing = "ISSUE_PATH_HINTS_MISSING"
)

// strictScopeOptions is the issuecontract grading a dogfood row is held to. A
// live run additionally demands a strong (non-forgeable) witness grade: the
// issue's done condition must be provable by an independent oracle rather than
// by the worker's own report.
func strictScopeOptions(opt BuildOptions) issuepolicy.Options {
	return issuepolicy.Options{
		Live:              opt.Live,
		DedupeChecked:     opt.DedupeChecked,
		DedupeCap:         opt.DedupeCap,
		StrictProjectWork: true,
		StrictWitness:     opt.Live,
	}
}

// strictScopeHold returns the extra closed-vocabulary hold reasons a live row
// must clear on top of the shared contract review. It is empty for a dry-run,
// which stays advisory by design.
func strictScopeHold(item ActionItem, opt BuildOptions) []string {
	if !opt.Live {
		return nil
	}
	var hold []string
	if strings.TrimSpace(item.Lane) == "" {
		hold = append(hold, ReasonLaneMissing)
	}
	if !hasPathHint(item.Paths) {
		hold = append(hold, ReasonPathHintsMissing)
	}
	return hold
}

func hasPathHint(paths []string) bool {
	for _, path := range paths {
		if strings.TrimSpace(path) != "" {
			return true
		}
	}
	return false
}

// applyStrictScopeHold folds hold into review, keeping the reason list sorted and
// deduplicated the way issuecontract emits it. A held row can never stay
// dispatchable; it degrades to triage_only unless the contract already refused it
// outright, in which case the stronger refusal verdict is preserved.
func applyStrictScopeHold(review issuepolicy.Review, hold []string) issuepolicy.Review {
	if len(hold) == 0 {
		return review
	}
	seen := map[string]bool{}
	for _, reason := range review.Reasons {
		seen[reason] = true
	}
	reasons := append([]string(nil), review.Reasons...)
	for _, reason := range hold {
		if !seen[reason] {
			seen[reason] = true
			reasons = append(reasons, reason)
		}
	}
	sort.Strings(reasons)
	review.Reasons = reasons
	review.OK = false
	if review.Dispatchability == issuepolicy.Dispatchable {
		review.Verdict = "needs_scope"
		review.Dispatchability = issuepolicy.TriageOnly
	}
	return review
}

// StrictScopeRefusalMessage explains a live refusal in the operator's terms: what
// was held, why, and the two ways forward. It is the stderr counterpart of the
// ErrorStrictScope result tag.
func StrictScopeRefusalMessage(skipped []SkippedRow) string {
	held := make([]string, 0, len(skipped))
	for _, row := range skipped {
		held = append(held, fmt.Sprintf("%s [%s]", row.Key, row.Reason))
	}
	return fmt.Sprintf("live issue sync refused: %d row(s) are not dispatchable under strict root scope: %s; "+
		"scope each row (root-point change, witness, lane, path hints) or re-run without --live",
		len(skipped), strings.Join(held, "; "))
}
