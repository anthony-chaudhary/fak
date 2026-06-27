package modelroute

// Judge / verifier free-text scoring for best_of (#602, epic #595).
//
// ReduceBestOf folds an ensemble by max(Vote.Score) with a deterministic
// Model-string tie-break (see Combine). But who FILLS the scores? Today the
// caller must supply them. This file adds the live SCORER half: a judge/verifier
// member (Role "judge" / "verifier") reads each drafter's free-text Output and
// returns a score, so the existing Combine(ReduceBestOf, votes) then picks the
// judge-preferred answer.
//
// THE HONESTY BOUNDARY (load-bearing — mirrors the package doc + specbridge.go).
// The scoring and the fold are kept STRICTLY separate:
//   - ScoreVotes (here) is the NON-deterministic half — it calls a judge model
//     through a bound Scorer closure. Two runs may score the same outputs
//     differently; nothing here claims bit-exactness.
//   - Combine(ReduceBestOf, …) (modelroute.go) is the DETERMINISTIC half — given
//     fixed scores it always folds the same way. That property is unchanged: this
//     file never touches Combine.
// So determinism is scoped to the FOLD, exactly as the package doc scopes it. A
// best_of result over a live judge is "deterministic GIVEN the judge's scores",
// never "deterministic answer".
//
// LANE PURITY (same rule as specbridge.go): this tier-1 leaf stays stdlib-only.
// It does NOT import an engine, a model package, or a provider client. The live
// model call crosses the seam as a CLOSURE the caller binds — a Scorer — exactly
// as SpecAccept takes polymodel.AcceptGreedy as an AcceptFunc. The wiring that
// runs a real judge model on a real engine to PRODUCE the score is DEFERRED
// engine work above this leaf, out of lane.

import (
	"context"
	"fmt"
)

// JudgeRole / VerifierRole are the reserved Member.Role labels (case-insensitive,
// via normalizeRole) that mark the scoring member of a best_of ensemble. Either
// label names a judge; a Plan should carry exactly one so the scorer is unambiguous.
const (
	JudgeRole    = "judge"
	VerifierRole = "verifier"
)

// Scorer is the bound judge-model call — the closure seam that keeps any engine
// or provider client out of this leaf's import graph. Given a drafter's free-text
// Output it returns that output's score; a higher score is a better answer, and
// the absolute scale is the judge's own (Combine only compares scores within one
// vote set, so a judge need only be internally consistent, not calibrated). The
// context carries cancellation/deadline to the live judge call. The caller binds
// this to a real model invocation (a gateway turn, a `fak run` of a judge model);
// a test binds a fixed-score stand-in. ScoreVotes never runs an engine itself.
type Scorer interface {
	Score(ctx context.Context, subjectOutput string) (float64, error)
}

// ScorerFunc adapts a plain func to the Scorer interface, so a caller can bind a
// closure (a gateway turn, a `fak run` call, a test stub) without declaring a type.
type ScorerFunc func(ctx context.Context, subjectOutput string) (float64, error)

// Score satisfies Scorer.
func (f ScorerFunc) Score(ctx context.Context, subjectOutput string) (float64, error) {
	return f(ctx, subjectOutput)
}

// JudgeMember returns the single member tagged Role=="judge" or Role=="verifier"
// (case-insensitive) — the model that SCORES the others — and reports whether
// exactly one such member exists. Zero or multiple judges is not an unambiguous
// best_of-with-judge ensemble and returns ok=false, so a caller never silently
// scores against the wrong member. The judge is identified purely from
// Member.Role, mirroring how SpecRoles reads drafter/verifier.
func (p Plan) JudgeMember() (Member, bool) {
	var judge Member
	n := 0
	for _, m := range p.Members {
		switch normalizeRole(m.Role) {
		case JudgeRole, VerifierRole:
			judge = m
			n++
		}
	}
	if n != 1 {
		return Member{}, false
	}
	return judge, true
}

// DrafterVotes returns the votes whose member is NOT the judge — the free-text
// answers to be scored. It is the complement of JudgeMember: the judge does not
// score (or compete against) itself. Order is preserved (the Combine member-order
// contract), so a downstream best_of tie-break stays deterministic.
func (p Plan) drafterModels() map[string]bool {
	judge, ok := p.JudgeMember()
	skip := map[string]bool{}
	if ok {
		skip[judge.Model] = true
	}
	return skip
}

// ScoreVotes runs a judge over a set of drafter votes and RETURNS a new []Vote
// with each Vote.Score filled by the Scorer — leaving the deterministic fold to
// Combine(ReduceBestOf, …). It is the live SCORER half of best_of:
//
//	scored, err := ScoreVotes(ctx, judge, votes)   // non-deterministic: a model call
//	res, err := Combine(ReduceBestOf, scored)      // deterministic: max(Score), stable tie-break
//	// res.Output is now the judge-preferred drafter's answer.
//
// It does NOT mutate the input slice (the caller's votes keep their original,
// possibly-empty Scores) and preserves member order, so the Combine member-order
// contract holds. A nil Scorer, an empty vote set, or any judge error fails loud
// — a best_of route must never fold un-scored votes and silently fall back to a
// zero-score tie-break. The judge is called once per vote, in order, so a
// cancelled context stops the scan promptly.
func ScoreVotes(ctx context.Context, judge Scorer, votes []Vote) ([]Vote, error) {
	if judge == nil {
		return nil, fmt.Errorf("modelroute: ScoreVotes needs a bound Scorer (the judge model call)")
	}
	if len(votes) == 0 {
		return nil, fmt.Errorf("modelroute: ScoreVotes needs at least one vote to score")
	}
	out := make([]Vote, len(votes))
	copy(out, votes)
	for i := range out {
		s, err := judge.Score(ctx, out[i].Output)
		if err != nil {
			return nil, fmt.Errorf("modelroute: judge failed to score member %q: %w", out[i].Member.Model, err)
		}
		out[i].Score = s
	}
	return out, nil
}

// ScorePlanVotes is the role-aware wrapper: it scores only the DRAFTER votes of a
// judge-bearing Plan (the members that are not the judge), so the judge does not
// score itself, and returns the scored drafter votes ready for Combine. It
// requires the Plan to carry exactly one judge/verifier member (else fails loud,
// the way SpecRoles refuses a non-spec ensemble) and at least one drafter to
// score. The judge VOTE itself (if the dispatcher even produced one) is dropped:
// best_of picks among the drafters' answers, not the judge's.
func (p Plan) ScorePlanVotes(ctx context.Context, judge Scorer, votes []Vote) ([]Vote, error) {
	if _, ok := p.JudgeMember(); !ok {
		return nil, fmt.Errorf("modelroute: ScorePlanVotes needs a plan with exactly one judge/verifier member")
	}
	skip := p.drafterModels()
	drafters := make([]Vote, 0, len(votes))
	for _, v := range votes {
		if skip[v.Member.Model] {
			continue
		}
		drafters = append(drafters, v)
	}
	if len(drafters) == 0 {
		return nil, fmt.Errorf("modelroute: ScorePlanVotes found no drafter votes to score (only the judge?)")
	}
	return ScoreVotes(ctx, judge, drafters)
}

// ---------------------------------------------------------------------------
// SYNTHESIS — the LLM-merge of N free-text answers (the OpenRouter-Fusion
// analogue). STUB + interface seam only; see the scope note below.
// ---------------------------------------------------------------------------

// Merger is the bound synthesis-model call: it reads N free-text answers and
// returns ONE merged answer. Like Scorer it is a closure the caller binds to a
// real model turn, keeping the engine out of this leaf. It is the
// non-deterministic half of synthesis.
type Merger interface {
	Merge(ctx context.Context, answers []string) (string, error)
}

// MergerFunc adapts a plain func to the Merger interface.
type MergerFunc func(ctx context.Context, answers []string) (string, error)

// Merge satisfies Merger.
func (f MergerFunc) Merge(ctx context.Context, answers []string) (string, error) {
	return f(ctx, answers)
}

// SynthesisInputs is the DETERMINISTIC prep half of synthesis: the ordered,
// de-duplicated free-text answers a Merger is asked to fuse, gathered from a vote
// set in member order (the Combine member-order contract). Splitting this out is
// the same honesty boundary as ScoreVotes/Combine — the GATHER is pure and
// order-stable; only the MERGE step (the Merger call) is non-deterministic.
type SynthesisInputs struct {
	Answers []string `json:"answers"`
}

// SynthesisInputsFromVotes gathers the deterministic merge inputs from a vote set
// in member order, dropping empty answers. This is the pure prep step; the live
// merge is Synthesize.
func SynthesisInputsFromVotes(votes []Vote) SynthesisInputs {
	answers := make([]string, 0, len(votes))
	for _, v := range votes {
		if v.Output == "" {
			continue
		}
		answers = append(answers, v.Output)
	}
	return SynthesisInputs{Answers: answers}
}

// Synthesize is the live LLM-merge of N free-text answers — the OpenRouter-Fusion
// analogue. It performs the deterministic GATHER (SynthesisInputsFromVotes) then
// delegates the non-deterministic MERGE to the bound Merger, exactly mirroring the
// ScoreVotes/Combine split: the prep is pure and order-stable, only the model
// merge is non-bit-exact. The merged Output is returned in a Result tagged with
// the new reduction so a caller can treat it like any other fold outcome.
//
// SCOPE NOTE (#602): synthesis is shipped here as a SEPARATE function with its own
// deterministic-prep / non-deterministic-merge split, deliberately NOT as a
// Reduction const + Combine arm. Combine's contract is a PURE, deterministic fold
// given fixed votes; a synthesis "reduction" would have to call a model to produce
// its Output, breaking that contract and the package's honesty boundary. Keeping
// Synthesize out of Combine preserves "Combine is deterministic given fixed votes"
// untouched. The closed Reduction set (first/vote/best_of/all_reduce/concat) is
// therefore unchanged by this file. If a future change does want a ReduceSynthesis
// const, it must carry its model-merge OUTSIDE Combine and only let Combine see the
// already-merged single Output — the const would be a routing label, not a fold.
func Synthesize(ctx context.Context, merge Merger, votes []Vote) (Result, error) {
	if merge == nil {
		return Result{}, fmt.Errorf("modelroute: Synthesize needs a bound Merger (the synthesis model call)")
	}
	in := SynthesisInputsFromVotes(votes)
	if len(in.Answers) == 0 {
		return Result{}, fmt.Errorf("modelroute: Synthesize needs at least one non-empty answer to merge")
	}
	merged, err := merge.Merge(ctx, in.Answers)
	if err != nil {
		return Result{}, fmt.Errorf("modelroute: synthesis merge failed: %w", err)
	}
	return Result{Reduce: "synthesis", Output: merged, Members: len(votes)}, nil
}
