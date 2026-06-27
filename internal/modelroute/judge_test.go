package modelroute

import (
	"context"
	"errors"
	"testing"
)

// fixedScorer is a mock judge: it returns a pre-set score per drafter output, so a
// test asserts the SCORER->Combine wiring without any live model. byOutput maps a
// drafter's free-text Output to the score the judge assigns it.
type fixedScorer struct {
	byOutput map[string]float64
	calls    int
}

func (f *fixedScorer) Score(_ context.Context, subjectOutput string) (float64, error) {
	f.calls++
	return f.byOutput[subjectOutput], nil
}

// TestScoreVotesThenBestOfPicksJudgePreferred is the issue's primary acceptance: a
// best_of ensemble over free-text drafters picks the JUDGE-preferred answer. The
// votes arrive with empty Scores; the mock judge fills them; Combine(ReduceBestOf)
// then selects the highest-scored drafter's Output.
func TestScoreVotesThenBestOfPicksJudgePreferred(t *testing.T) {
	votes := []Vote{
		{Member: Member{Model: "drafter-a"}, Output: "answer A"},
		{Member: Member{Model: "drafter-b"}, Output: "answer B"},
		{Member: Member{Model: "drafter-c"}, Output: "answer C"},
	}
	// The judge prefers B (highest score), though it arrived second.
	judge := &fixedScorer{byOutput: map[string]float64{
		"answer A": 0.20,
		"answer B": 0.91,
		"answer C": 0.55,
	}}
	scored, err := ScoreVotes(context.Background(), judge, votes)
	if err != nil {
		t.Fatalf("ScoreVotes: %v", err)
	}
	if judge.calls != 3 {
		t.Fatalf("judge should be called once per vote, got %d", judge.calls)
	}
	res, err := Combine(ReduceBestOf, scored)
	if err != nil {
		t.Fatalf("Combine best_of: %v", err)
	}
	if res.Output != "answer B" || res.Winner != "drafter-b" {
		t.Fatalf("best_of should pick the judge-preferred answer B, got output=%q winner=%q", res.Output, res.Winner)
	}
}

// TestScoreVotesDoesNotMutateInput proves ScoreVotes returns a NEW slice and leaves
// the caller's votes (with their original empty Scores) untouched — the scoring and
// the fold stay cleanly separable.
func TestScoreVotesDoesNotMutateInput(t *testing.T) {
	votes := []Vote{
		{Member: Member{Model: "a"}, Output: "x"},
		{Member: Member{Model: "b"}, Output: "y"},
	}
	judge := &fixedScorer{byOutput: map[string]float64{"x": 1, "y": 2}}
	scored, err := ScoreVotes(context.Background(), judge, votes)
	if err != nil {
		t.Fatalf("ScoreVotes: %v", err)
	}
	if votes[0].Score != 0 || votes[1].Score != 0 {
		t.Fatalf("input votes should keep their zero scores, got %v %v", votes[0].Score, votes[1].Score)
	}
	if scored[0].Score != 1 || scored[1].Score != 2 {
		t.Fatalf("scored votes should carry judge scores, got %v %v", scored[0].Score, scored[1].Score)
	}
}

// TestScoreVotesThenBestOfDeterministic proves the FOLD is deterministic given the
// judge's (fixed) scores: the same scored votes always fold to the same winner,
// across many runs. Determinism is scoped to the fold, exactly as the package doc
// and judge.go scope it — the judge's scoring is held fixed here by construction.
func TestScoreVotesThenBestOfDeterministic(t *testing.T) {
	votes := []Vote{
		{Member: Member{Model: "a"}, Output: "ans-a"},
		{Member: Member{Model: "b"}, Output: "ans-b"},
	}
	judge := &fixedScorer{byOutput: map[string]float64{"ans-a": 0.4, "ans-b": 0.8}}
	scored, err := ScoreVotes(context.Background(), judge, votes)
	if err != nil {
		t.Fatalf("ScoreVotes: %v", err)
	}
	first, err := Combine(ReduceBestOf, scored)
	if err != nil {
		t.Fatalf("Combine: %v", err)
	}
	for i := 0; i < 50; i++ {
		got, err := Combine(ReduceBestOf, scored)
		if err != nil {
			t.Fatalf("Combine at %d: %v", i, err)
		}
		if got.Output != first.Output || got.Winner != first.Winner {
			t.Fatalf("fold not deterministic at %d: %q/%q != %q/%q", i, got.Output, got.Winner, first.Output, first.Winner)
		}
	}
}

// TestScoreVotesTieBreakIsCombineDeterministic proves a SCORE tie folds via
// Combine's stable Model-string tie-break (judge.go never changes that): equal
// scores pick the lexicographically smaller model id.
func TestScoreVotesTieBreakIsCombineDeterministic(t *testing.T) {
	votes := []Vote{
		{Member: Member{Model: "zeta"}, Output: "out-z"},
		{Member: Member{Model: "alpha"}, Output: "out-a"},
	}
	judge := &fixedScorer{byOutput: map[string]float64{"out-z": 0.5, "out-a": 0.5}}
	scored, err := ScoreVotes(context.Background(), judge, votes)
	if err != nil {
		t.Fatalf("ScoreVotes: %v", err)
	}
	res, err := Combine(ReduceBestOf, scored)
	if err != nil {
		t.Fatalf("Combine: %v", err)
	}
	if res.Winner != "alpha" {
		t.Fatalf("a score tie should break to the smaller model id (alpha), got %q", res.Winner)
	}
}

// TestScoreVotesRejectsNilAndEmpty proves a nil Scorer and an empty vote set both
// fail loud — a best_of route must never fold un-scored votes.
func TestScoreVotesRejectsNilAndEmpty(t *testing.T) {
	if _, err := ScoreVotes(context.Background(), nil, []Vote{{Output: "x"}}); err == nil {
		t.Fatal("a nil Scorer must be refused")
	}
	judge := &fixedScorer{byOutput: map[string]float64{}}
	if _, err := ScoreVotes(context.Background(), judge, nil); err == nil {
		t.Fatal("an empty vote set must be refused")
	}
}

// TestScoreVotesPropagatesJudgeError proves a judge failure fails loud rather than
// silently scoring zero.
func TestScoreVotesPropagatesJudgeError(t *testing.T) {
	boom := errors.New("judge engine down")
	judge := ScorerFunc(func(_ context.Context, _ string) (float64, error) { return 0, boom })
	_, err := ScoreVotes(context.Background(), judge, []Vote{{Member: Member{Model: "a"}, Output: "x"}})
	if err == nil || !errors.Is(err, boom) {
		t.Fatalf("judge error should propagate, got %v", err)
	}
}

// TestJudgeMemberIdentifiesRole proves JudgeMember finds the single judge/verifier
// member (case-insensitive) and refuses an ambiguous shape.
func TestJudgeMemberIdentifiesRole(t *testing.T) {
	p := Plan{Members: []Member{
		{Model: "drafter-a", Role: "drafter"},
		{Model: "drafter-b"},
		{Model: "the-judge", Role: "Judge"}, // case-insensitive
	}}
	j, ok := p.JudgeMember()
	if !ok || j.Model != "the-judge" {
		t.Fatalf("JudgeMember should find the-judge, got %+v ok=%v", j, ok)
	}
	// verifier also names a judge.
	pv := Plan{Members: []Member{{Model: "d"}, {Model: "v", Role: "verifier"}}}
	if jv, ok := pv.JudgeMember(); !ok || jv.Model != "v" {
		t.Fatalf("verifier should be recognized as the judge, got %+v ok=%v", jv, ok)
	}
	// zero judges and two judges are both ambiguous.
	none := Plan{Members: []Member{{Model: "a"}, {Model: "b"}}}
	if _, ok := none.JudgeMember(); ok {
		t.Fatal("a judge-less plan must not report a judge")
	}
	two := Plan{Members: []Member{{Model: "j1", Role: "judge"}, {Model: "j2", Role: "verifier"}}}
	if _, ok := two.JudgeMember(); ok {
		t.Fatal("two judges must be ambiguous")
	}
}

// TestScorePlanVotesExcludesJudge proves the role-aware wrapper scores only the
// DRAFTER votes — the judge does not score (or compete against) itself — and the
// best_of fold then picks the judge-preferred drafter.
func TestScorePlanVotesExcludesJudge(t *testing.T) {
	p := Plan{
		Members: []Member{
			{Model: "drafter-a", Role: "drafter"},
			{Model: "drafter-b", Role: "drafter"},
			{Model: "the-judge", Role: "judge"},
		},
		Reduce: ReduceBestOf,
	}
	// The dispatcher produced a vote for the judge too; it must be dropped from scoring.
	votes := []Vote{
		{Member: Member{Model: "drafter-a"}, Output: "A"},
		{Member: Member{Model: "drafter-b"}, Output: "B"},
		{Member: Member{Model: "the-judge"}, Output: "judge said B is best"},
	}
	judge := &fixedScorer{byOutput: map[string]float64{
		"A":                    0.3,
		"B":                    0.9,
		"judge said B is best": 999, // never scored: would win if not excluded
	}}
	scored, err := p.ScorePlanVotes(context.Background(), judge, votes)
	if err != nil {
		t.Fatalf("ScorePlanVotes: %v", err)
	}
	if len(scored) != 2 {
		t.Fatalf("only the two drafters should be scored, got %d", len(scored))
	}
	if judge.calls != 2 {
		t.Fatalf("judge should score only the 2 drafters, got %d calls", judge.calls)
	}
	res, err := Combine(ReduceBestOf, scored)
	if err != nil {
		t.Fatalf("Combine: %v", err)
	}
	if res.Output != "B" || res.Winner != "drafter-b" {
		t.Fatalf("best_of should pick drafter B, got output=%q winner=%q", res.Output, res.Winner)
	}
}

// TestScorePlanVotesRejectsNonJudgePlan proves the wrapper refuses a plan with no
// (or ambiguous) judge, and a plan with no drafter votes to score.
func TestScorePlanVotesRejectsNonJudgePlan(t *testing.T) {
	judge := &fixedScorer{byOutput: map[string]float64{}}
	noJudge := Plan{Members: []Member{{Model: "a"}, {Model: "b"}}}
	if _, err := noJudge.ScorePlanVotes(context.Background(), judge, []Vote{{Member: Member{Model: "a"}, Output: "x"}}); err == nil {
		t.Fatal("a judge-less plan must be refused by ScorePlanVotes")
	}
	// Judge present but the only vote is the judge's own — no drafter to score.
	onlyJudge := Plan{Members: []Member{{Model: "j", Role: "judge"}}}
	if _, err := onlyJudge.ScorePlanVotes(context.Background(), judge, []Vote{{Member: Member{Model: "j"}, Output: "self"}}); err == nil {
		t.Fatal("a plan with no drafter votes must be refused")
	}
}

// ---------------------------------------------------------------------------
// SYNTHESIS — the LLM-merge stub.
// ---------------------------------------------------------------------------

// TestSynthesizeMergesViaMerger proves the synthesis seam: the deterministic
// GATHER feeds the bound Merger, and the merged answer comes back tagged.
func TestSynthesizeMergesViaMerger(t *testing.T) {
	votes := []Vote{
		{Member: Member{Model: "a"}, Output: "the sky is blue"},
		{Member: Member{Model: "b"}, Output: ""}, // empty dropped from the gather
		{Member: Member{Model: "c"}, Output: "grass is green"},
	}
	var seen []string
	merger := MergerFunc(func(_ context.Context, answers []string) (string, error) {
		seen = answers
		return "merged: sky blue, grass green", nil
	})
	res, err := Synthesize(context.Background(), merger, votes)
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if len(seen) != 2 || seen[0] != "the sky is blue" || seen[1] != "grass is green" {
		t.Fatalf("gather should drop empties and preserve order, got %v", seen)
	}
	if res.Output != "merged: sky blue, grass green" || res.Reduce != "synthesis" {
		t.Fatalf("synthesis result wrong: %+v", res)
	}
}

// TestSynthesizeRejectsNilAndEmpty proves a nil Merger and an all-empty answer set
// both fail loud.
func TestSynthesizeRejectsNilAndEmpty(t *testing.T) {
	if _, err := Synthesize(context.Background(), nil, []Vote{{Output: "x"}}); err == nil {
		t.Fatal("a nil Merger must be refused")
	}
	merger := MergerFunc(func(_ context.Context, _ []string) (string, error) { return "", nil })
	if _, err := Synthesize(context.Background(), merger, []Vote{{Output: ""}}); err == nil {
		t.Fatal("an all-empty answer set must be refused")
	}
}
