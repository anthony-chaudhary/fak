package safecommit

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestScoreResultVerifiedCommit(t *testing.T) {
	res := ScoreResult(Result{Committed: true, Verified: true})
	if res.Score != 100 || res.Value != 1 || res.ValueUnit != "quality_ratio" || res.LegacyScore != 100 || res.LegacyScoreScale != 100 || res.Grade != "A" || len(res.ScoreNotes) != 0 {
		t.Fatalf("verified score = %+v notes=%v, want value 1/A/no notes", res, res.ScoreNotes)
	}
}

func TestScoreResultReviewUnavailablePenalty(t *testing.T) {
	res := ScoreResult(Result{
		Committed: true,
		Verified:  true,
		Review:    &modelroute.ReviewResult{Verdict: modelroute.ReviewUnavailable},
	})
	if res.Score != 94 || res.Value != 0.94 || res.Grade != "A" {
		t.Fatalf("review-unavailable score = %+v, want value 0.94/A", res)
	}
	if len(res.ScoreNotes) != 1 || !strings.Contains(res.ScoreNotes[0], "review unavailable") {
		t.Fatalf("review-unavailable notes = %v", res.ScoreNotes)
	}
}

func TestScoreResultRanksPostCommitRaceBelowPreCommitRefusal(t *testing.T) {
	pre := ScoreResult(Result{Reason: ReasonPreStagedPathOverlap})
	race := ScoreResult(Result{Committed: true, Reason: ReasonPathspecRace})
	if !(race.Score < pre.Score) {
		t.Fatalf("race score should be lower than pre-commit refusal: race=%d pre=%d", race.Score, pre.Score)
	}
	if race.Grade != "F" {
		t.Fatalf("race grade = %s, want F", race.Grade)
	}
}

func TestCommitWithScoresResult(t *testing.T) {
	g := &fakeGit{reply: onTrunkBase()}
	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Score != 100 || res.Value != 1 || res.Grade != "A" {
		t.Fatalf("CommitWith score = %+v, want value 1/A", res)
	}
}
