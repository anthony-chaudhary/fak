package operatorresolve

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
)

func TestGitDecidableIsolationTakesWitnessedOption(t *testing.T) {
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Harness:  "claude",
		Question: "How should I isolate my commit from peer edits?",
		Options: []operatorquestion.Option{
			{Label: "Commit explicit owned paths", Rationale: "leave peer-dirty files untouched"},
			{Label: "Wait for a clean tree", Rationale: "delay until peers finish"},
		},
	}
	calls := 0
	oracle := OracleFunc{OracleName: "git-status-readonly", InspectFn: func(_ context.Context, _ operatorquestion.OperatorQuestion, o operatorquestion.Option) (Evidence, bool, error) {
		calls++
		if strings.Contains(o.Label, "explicit owned paths") {
			return Evidence{Claim: "git status shows peer dirt outside the owned path set; path-scoped commit preserves it", Witness: "git status --short; git diff -- <owned paths>", Score: 10}, true, nil
		}
		return Evidence{Claim: "waiting adds latency and does not improve path isolation", Witness: "git status --short", Score: -1}, true, nil
	}}
	got, err := (Resolver{Oracles: []Oracle{oracle}}).Resolve(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != choicetriage.TakeObvious || got.Reason != ReasonWitnessedOption || got.Action != q.Options[0].Label {
		t.Fatalf("got %+v", got)
	}
	if calls != len(q.Options) {
		t.Fatalf("oracle calls=%d want %d", calls, len(q.Options))
	}
	if len(got.Options) != 2 || len(got.Options[0].Evidence) != 1 || got.Options[0].Evidence[0].Oracle != "git-status-readonly" {
		t.Fatalf("missing per-option evidence: %+v", got.Options)
	}
}

func TestAuthorityForkEarnsHumanResidualWithOptions(t *testing.T) {
	calls := 0
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "Which product priority should win?",
		Options:  []operatorquestion.Option{{Label: "Reliability"}, {Label: "Launch date"}},
	}
	got, err := (Resolver{Oracles: []Oracle{OracleFunc{OracleName: "must-not-run", InspectFn: func(context.Context, operatorquestion.OperatorQuestion, operatorquestion.Option) (Evidence, bool, error) {
		calls++
		return Evidence{}, false, nil
	}}}}).Resolve(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != choicetriage.HumanResidual || got.Reason != ReasonAuthorityFork || len(got.Options) != 2 {
		t.Fatalf("got %+v", got)
	}
	if calls != 0 {
		t.Fatalf("authority classification should not run repo oracles, calls=%d", calls)
	}
}

func TestEvidenceTieStaysOffHuman(t *testing.T) {
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "Which reversible implementation should I use?",
		Options:  []operatorquestion.Option{{Label: "A"}, {Label: "B"}},
	}
	oracle := OracleFunc{OracleName: "scorecard-readonly", InspectFn: func(context.Context, operatorquestion.OperatorQuestion, operatorquestion.Option) (Evidence, bool, error) {
		return Evidence{Claim: "equal score", Score: 1}, true, nil
	}}
	got, err := (Resolver{Oracles: []Oracle{oracle}}).Resolve(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != choicetriage.FreshContext || got.Reason != ReasonEvidenceTie {
		t.Fatalf("got %+v", got)
	}
}

func TestRejectsPlanQuestion(t *testing.T) {
	_, err := (Resolver{}).Resolve(context.Background(), operatorquestion.OperatorQuestion{Kind: operatorquestion.PlanApproval, Question: "approve?"})
	if err == nil {
		t.Fatal("plan approval belongs to the plan resolver")
	}
}

type recordingRunner struct {
	calls [][]string
}

func (r *recordingRunner) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	call := append([]string{name}, args...)
	r.calls = append(r.calls, call)
	if len(args) > 0 && args[0] == "status" {
		return []byte(" M peer.go\n"), nil
	}
	return []byte("abc123 prior commit\n"), nil
}

func TestGitIsolationOracleRunsOnlyReadOnlyGitCommands(t *testing.T) {
	runner := &recordingRunner{}
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "How should I isolate this commit?",
		Options:  []operatorquestion.Option{{Label: "Commit explicit owned paths"}, {Label: "Wait"}},
	}
	got, err := (Resolver{Oracles: []Oracle{GitIsolationOracle{Runner: runner}}}).Resolve(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != choicetriage.TakeObvious || got.Action != q.Options[0].Label {
		t.Fatalf("got %+v", got)
	}
	if len(runner.calls) != 4 {
		t.Fatalf("calls=%v", runner.calls)
	}
	for _, call := range runner.calls {
		joined := strings.Join(call, " ")
		if joined != "git status --short" && joined != "git log -1 --oneline -- ." {
			t.Fatalf("mutating or undeclared command: %q", joined)
		}
	}
}

func TestClaudeAndCodexQuestionsShareOneResolverPath(t *testing.T) {
	payloads := []operatorquestion.NativeGate{
		{HarnessCommand: "claude", Tool: "AskUserQuestion", Payload: []byte(`{"questions":[{"header":"Isolation","multiSelect":false,"question":"How should I isolate this commit?","options":[{"label":"Commit explicit owned paths","description":"owned files only"},{"label":"Wait","description":"wait for peers"}]}]}`)},
		{HarnessCommand: "codex", Tool: "functions.request_user_input", Payload: []byte(`{"questions":[{"id":"isolation","header":"Isolation","question":"How should I isolate this commit?","options":[{"label":"Commit explicit owned paths","description":"owned files only"},{"label":"Wait","description":"wait for peers"}]}]}`)},
	}
	oracleCalls := 0
	resolver := Resolver{Oracles: []Oracle{OracleFunc{OracleName: "git-readonly", InspectFn: func(_ context.Context, _ operatorquestion.OperatorQuestion, option operatorquestion.Option) (Evidence, bool, error) {
		oracleCalls++
		if strings.Contains(option.Label, "explicit owned paths") {
			return Evidence{Claim: "path-scoped commit preserves peer dirt", Score: 10}, true, nil
		}
		return Evidence{Claim: "no isolation gain", Score: 0}, true, nil
	}}}}
	var decisions []Verdict
	for _, gate := range payloads {
		q, err := operatorquestion.Normalize(gate)
		if err != nil {
			t.Fatal(err)
		}
		got, err := resolver.Resolve(context.Background(), q)
		if err != nil {
			t.Fatal(err)
		}
		decisions = append(decisions, got)
	}
	if decisions[0].Disposition != choicetriage.TakeObvious || decisions[0].Action != decisions[1].Action || decisions[0].Reason != decisions[1].Reason {
		t.Fatalf("shared resolver drift: claude=%+v codex=%+v", decisions[0], decisions[1])
	}
	if oracleCalls != 4 {
		t.Fatalf("shared oracle calls=%d want 4", oracleCalls)
	}
}
