package safecommit

import (
	"math"
	"time"
)

// Commit velocity is the effect-qualified ship-speed reading for a commit (#4241).
// It is deliberately distinct from the outcome-quality Score in score.go: quality
// answers "how healthy was the outcome", velocity answers "how fast did the effect
// land against its budget". The two axes must never be conflated — a slow verified
// commit is high quality and low velocity; a fast refusal is neither.
//
// The governing rule (see Assumptions on the issue): a numeric velocity is
// meaningful only after the command's own authoritative effect fields qualify it.
// The local leg is scored only when the commit both landed and verified; the push
// leg only when it additionally pushed. Every other outcome — an unverified race, a
// refusal, a no-op, a rejected push — retains its measured timing but carries a nil
// score (UNSCORED), so a fast failure can never earn velocity credit.

// Velocity leg status values. A SCORED leg carries a non-nil budget-relative score;
// an UNSCORED leg carries a nil score with its timing retained.
const (
	VelocityScored   = "SCORED"
	VelocityUnscored = "UNSCORED"
)

// VelocityBudgets declares the separate wall-clock budgets the local and push
// effect boundaries are graded against. Pooling the two (grading a push against the
// local budget or vice versa) is out of scope: the boundaries have different cost
// floors — a local commit is CPU/disk bound, a push additionally pays the network.
type VelocityBudgets struct {
	Local time.Duration
	Push  time.Duration
}

// DefaultVelocityBudgets are the initial declared budgets. They are intentionally
// generous placeholders: #4225's outcome-separated process p50s will supply the
// empirical distribution these should track. Until then they give the score a
// stable, documented denominator rather than a magic literal at each call site.
var DefaultVelocityBudgets = VelocityBudgets{
	Local: 10 * time.Second,
	Push:  60 * time.Second,
}

// CommitVelocityLeg is one effect boundary's ship-speed reading: the measured
// elapsed, the budget it is graded against, and a 0-100 score that is nil until the
// boundary's effect fields qualify it. Timing is retained even when UNSCORED so a
// refusal/race/no-op still reports how long it took without earning a score.
type CommitVelocityLeg struct {
	ElapsedNS int64  `json:"elapsed_ns"`
	BudgetNS  int64  `json:"budget_ns"`
	Score     *int   `json:"score"` // nil == UNSCORED (marshals to JSON null)
	Status    string `json:"status"`
	Note      string `json:"note,omitempty"`
}

// CommitVelocity is the nested, effect-qualified commit-velocity object: a local
// and a push leg, each qualified independently by the command's authoritative
// effect fields (Result.Committed/Verified/Pushed).
type CommitVelocity struct {
	Local CommitVelocityLeg `json:"local"`
	Push  CommitVelocityLeg `json:"push"`
}

// ScoreCommitVelocity grades a commit's ship speed from its authoritative effect
// fields and the measured wall-clock of each boundary. localElapsed is the time to
// a verified local commit; pushElapsed is the time to a verified push. The local
// leg is scored only when Committed && Verified; the push leg only when it
// additionally Pushed. An unqualified leg keeps its timing with a nil score.
//
// The elapsed values are injected by the caller (measured against an authoritative
// clock at the commit/push boundaries) so the scorer stays pure and clock-free —
// which is exactly what makes it injected-clock testable.
func ScoreCommitVelocity(res Result, localElapsed, pushElapsed time.Duration, budgets VelocityBudgets) CommitVelocity {
	localQualified := res.Committed && res.DeliveryVerified()
	pushQualified := localQualified && res.Pushed

	return CommitVelocity{
		Local: velocityLeg(localElapsed, budgets.Local, localQualified, localUnscoredNote(res)),
		Push:  velocityLeg(pushElapsed, budgets.Push, pushQualified, pushUnscoredNote(res)),
	}
}

// requalifyCommitVelocity preserves measured timings while applying a newly attached completion
// contract. CommitWith measures before cmd/fak attaches compile/test evidence, so failing to fold
// the same axes here would leave a skipped-infra receipt with an incorrectly SCORED local leg.
func requalifyCommitVelocity(res *Result) {
	if res == nil || res.Velocity == nil {
		return
	}
	v := ScoreCommitVelocity(*res,
		time.Duration(res.Velocity.Local.ElapsedNS),
		time.Duration(res.Velocity.Push.ElapsedNS),
		VelocityBudgets{
			Local: time.Duration(res.Velocity.Local.BudgetNS),
			Push:  time.Duration(res.Velocity.Push.BudgetNS),
		},
	)
	res.Velocity = &v
}

// velocityLeg builds one leg. When qualified it carries a budget-relative score and
// SCORED status; otherwise the score stays nil (UNSCORED) with the timing retained
// and a one-line note explaining why the boundary did not qualify.
func velocityLeg(elapsed, budget time.Duration, qualified bool, unscoredNote string) CommitVelocityLeg {
	leg := CommitVelocityLeg{
		ElapsedNS: elapsed.Nanoseconds(),
		BudgetNS:  budget.Nanoseconds(),
		Status:    VelocityUnscored,
	}
	if !qualified {
		leg.Note = unscoredNote
		return leg
	}
	s := budgetScore(elapsed, budget)
	leg.Score = &s
	leg.Status = VelocityScored
	return leg
}

// budgetScore maps an elapsed/budget pair to a 0-100 score: full credit at or under
// budget, then a monotonic budget/elapsed decay above it (2x budget -> 50, 10x -> 10),
// floored at 0 and capped at 100. The cap means a faster-than-instant measurement can
// never exceed a genuinely fast commit's score — there is no headroom to game.
func budgetScore(elapsed, budget time.Duration) int {
	if budget <= 0 {
		return 0
	}
	if elapsed <= budget {
		return 100
	}
	ratio := float64(budget) / float64(elapsed)
	return clampScore(int(math.Round(ratio * 100)))
}

func clampScore(s int) int {
	switch {
	case s < 0:
		return 0
	case s > 100:
		return 100
	default:
		return s
	}
}

// localUnscoredNote explains why the local leg did not qualify.
func localUnscoredNote(res Result) string {
	switch {
	case !res.Committed:
		return "no commit landed"
	case !res.DeliveryVerified():
		return "commit did not verify"
	default:
		return ""
	}
}

// pushUnscoredNote explains why the push leg did not qualify.
func pushUnscoredNote(res Result) string {
	switch {
	case !res.Committed || !res.DeliveryVerified():
		return "local commit did not qualify"
	case !res.Pushed:
		if res.Reason == ReasonPushRejected {
			return "push rejected"
		}
		return "no verified push"
	default:
		return ""
	}
}
