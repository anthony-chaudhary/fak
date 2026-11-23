package modelroute

import (
	"context"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy"
)

// The best_of judge and synthesis seams (Plan.JudgeMember, Plan.ScorePlanVotes)
// live in pkg/routepolicy.
const (
	JudgeRole    = routepolicy.JudgeRole
	VerifierRole = routepolicy.VerifierRole
)

type (
	Scorer          = routepolicy.Scorer
	ScorerFunc      = routepolicy.ScorerFunc
	Merger          = routepolicy.Merger
	MergerFunc      = routepolicy.MergerFunc
	SynthesisInputs = routepolicy.SynthesisInputs
)

func ScoreVotes(ctx context.Context, judge Scorer, votes []Vote) ([]Vote, error) {
	return routepolicy.ScoreVotes(ctx, judge, votes)
}

func SynthesisInputsFromVotes(votes []Vote) SynthesisInputs {
	return routepolicy.SynthesisInputsFromVotes(votes)
}

func Synthesize(ctx context.Context, merge Merger, votes []Vote) (Result, error) {
	return routepolicy.Synthesize(ctx, merge, votes)
}
