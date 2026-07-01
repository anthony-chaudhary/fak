package safecommit

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// ScoreResult annotates a safecommit outcome with a deterministic 0-100 score and A-F grade.
// The score is an operator-facing summary of commit health: a verified commit is full credit,
// retryable pre-commit refusals retain more credit than post-commit integrity failures, and
// reviewer unavailability is a small penalty on an otherwise verified commit.
func ScoreResult(res Result) Result {
	res.Score, res.ScoreNotes = resultScore(res)
	res.Grade = resultGrade(res.Score)
	return res
}

func resultScore(res Result) (int, []string) {
	if strings.TrimSpace(res.Reason) == "" {
		switch {
		case res.Verified:
			if res.Review != nil && res.Review.Verdict == modelroute.ReviewUnavailable {
				return 94, []string{"verified commit; optional review unavailable"}
			}
			return 100, nil
		case res.Committed:
			return 55, []string{"commit landed but verification did not complete"}
		default:
			return 0, []string{"no commit outcome recorded"}
		}
	}

	switch res.Reason {
	case ReasonNothingStaged:
		return 88, []string{"no pathspec changes to commit"}
	case ReasonLockBusy, ReasonWindowFull:
		return 82, []string{"retryable writer-admission refusal"}
	case ReasonPushRejected:
		if res.Verified {
			return 80, []string{"verified local commit; push rejected"}
		}
		return 60, []string{"push rejected before a verified local result"}
	case ReasonStaleBaseDeletion, ReasonSpuriousStagedDeletion, ReasonPreStagedPathOverlap:
		return 72, []string{"pre-commit guard prevented a likely stale or ambiguous path commit"}
	case ReasonCoreSelfModify:
		return 72, []string{"pre-commit guard refused a hard-self core-lock edit without independent maintenance witness"}
	case ReasonOffTrunk, ReasonMergeInProgress, ReasonNotARepo:
		return 62, []string{"repository state blocks a safe path-scoped commit"}
	case ReasonReviewRefuted:
		return 58, []string{"optional review refuted the diff before commit"}
	case ReasonNoPath, ReasonEmptyMessage:
		return 42, []string{"usage error before git work"}
	case ReasonHookRefused:
		return 35, []string{"git hook or commit command refused the change"}
	case ReasonPathspecRace, ReasonSymlinkEscape:
		return 10, []string{"commit landed but failed post-commit pathspec verification"}
	default:
		return 50, []string{"unclassified safecommit refusal: " + res.Reason}
	}
}

func resultGrade(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}
