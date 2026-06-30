package safecommit

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestScoreResultVerifiedCommit(t *testing.T) {
	res := ScoreResult(Result{Committed: true, Verified: true})
	if res.Score != 100 || res.Grade != "A" || len(res.ScoreNotes) != 0 {
		t.Fatalf("verified score = %d/%s notes=%v, want 100/A/no notes", res.Score, res.Grade, res.ScoreNotes)
	}
}

func TestScoreResultReviewUnavailablePenalty(t *testing.T) {
	res := ScoreResult(Result{
		Committed: true,
		Verified:  true,
		Review:    &modelroute.ReviewResult{Verdict: modelroute.ReviewUnavailable},
	})
	if res.Score != 94 || res.Grade != "A" {
		t.Fatalf("review-unavailable score = %d/%s, want 94/A", res.Score, res.Grade)
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
	if res.Score != 100 || res.Grade != "A" {
		t.Fatalf("CommitWith score = %d/%s, want 100/A", res.Score, res.Grade)
	}
}
