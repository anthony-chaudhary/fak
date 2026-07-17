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

type lsFilesRunner struct {
	tracked map[string]bool
	calls   [][]string
}

func (r *lsFilesRunner) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	r.calls = append(r.calls, append([]string{name}, args...))
	if len(args) == 3 && args[0] == "ls-files" && args[1] == "--" && r.tracked[args[2]] {
		return []byte(args[2] + "\n"), nil
	}
	return nil, nil
}

func TestTrackedArtifactOracleTakesExistingTrackedFile(t *testing.T) {
	runner := &lsFilesRunner{tracked: map[string]bool{"internal/foo/foo.go": true}}
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "Should I create a new file or edit the existing one?",
		Options: []operatorquestion.Option{
			{Label: "Edit `internal/foo/foo.go`", Rationale: "extend the current implementation"},
			{Label: "Create internal/foo/foo_v2.go", Rationale: "a parallel module"},
		},
	}
	got, err := (Resolver{Oracles: []Oracle{TrackedArtifactOracle{Runner: runner}}}).Resolve(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != choicetriage.TakeObvious || got.Reason != ReasonWitnessedOption || got.Action != q.Options[0].Label {
		t.Fatalf("got %+v", got)
	}
	if len(got.Options) != 2 || len(got.Options[0].Evidence) != 1 || got.Options[0].Evidence[0].Oracle != "tracked-artifact-readonly" {
		t.Fatalf("missing per-option evidence: %+v", got.Options)
	}
	for _, call := range runner.calls {
		joined := strings.Join(call, " ")
		if !strings.HasPrefix(joined, "git ls-files -- ") {
			t.Fatalf("mutating or undeclared command: %q", joined)
		}
	}
}

func TestTrackedArtifactOracleAbstainsWithoutCreateEditFraming(t *testing.T) {
	runner := &lsFilesRunner{tracked: map[string]bool{"internal/foo/foo.go": true}}
	// The commit-isolation fold names no create/edit framing, so the artifact oracle
	// must abstain (run no git) and leave the decision to the other oracles / default.
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "How should I isolate this commit?",
		Options:  []operatorquestion.Option{{Label: "Commit explicit owned paths"}, {Label: "Wait"}},
	}
	got, err := (Resolver{Oracles: []Oracle{TrackedArtifactOracle{Runner: runner}}}).Resolve(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != choicetriage.FreshContext || got.Reason != ReasonNeedsInvestigation {
		t.Fatalf("abstain should fall to FreshContext, got %+v", got)
	}
	if len(runner.calls) != 0 {
		t.Fatalf("abstaining oracle must not run git: %v", runner.calls)
	}
}

func TestTrackedArtifactOracleAbstainsWhenNoOptionNamesAPath(t *testing.T) {
	runner := &lsFilesRunner{}
	q := operatorquestion.OperatorQuestion{
		Kind:     operatorquestion.ChooseApproach,
		Question: "Should I create a new helper or edit inline?",
		Options:  []operatorquestion.Option{{Label: "New helper"}, {Label: "Inline it"}},
	}
	if _, err := (Resolver{Oracles: []Oracle{TrackedArtifactOracle{Runner: runner}}}).Resolve(context.Background(), q); err != nil {
		t.Fatal(err)
	}
	if len(runner.calls) != 0 {
		t.Fatalf("no path token means no git call: %v", runner.calls)
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

func TestScorecardAxisOracleTakesReversibleDominantOptionAndRecordsAxis(t *testing.T) {
	q := operatorquestion.OperatorQuestion{
		Kind:    operatorquestion.ChooseApproach,
		Options: []operatorquestion.Option{{Label: "Fast path"}, {Label: "Legacy path"}},
	}
	resolver := Resolver{Oracles: []Oracle{ScorecardAxisOracle{
		Axes: []ScorecardAxis{
			{Name: "quality-score", Scores: map[string]float64{"Fast path": 94, "Legacy path": 81}, Witness: "quality-ledger@r17"},
			{Name: "operator-heaviness-score", Scores: map[string]float64{"Fast path": 2, "Legacy path": 4}, LowerIsBetter: true, Witness: "heaviness-ledger@r9"},
		},
		Reversible: map[string]bool{"Fast path": true, "Legacy path": true},
	}}}

	got, err := resolver.Resolve(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != choicetriage.TakeObvious || got.Action != "Fast path" {
		t.Fatalf("got disposition=%s action=%q, want TAKE_OBVIOUS Fast path", got.Disposition, got.Action)
	}
	if !strings.Contains(got.Options[0].Evidence[0].Claim, "quality-score") || !strings.Contains(got.Options[0].Evidence[0].Witness, "quality-ledger@r17") {
		t.Fatalf("missing deciding axis or witness in evidence: %+v", got.Options[0].Evidence)
	}
}

func TestScorecardAxisOracleAbstainsOnTieOrIrreversibleOption(t *testing.T) {
	q := operatorquestion.OperatorQuestion{
		Kind:    operatorquestion.ChooseApproach,
		Options: []operatorquestion.Option{{Label: "A"}, {Label: "B"}},
	}
	for _, tc := range []struct {
		name       string
		scores     map[string]float64
		reversible map[string]bool
	}{
		{name: "tie", scores: map[string]float64{"A": 90, "B": 90}, reversible: map[string]bool{"A": true, "B": true}},
		{name: "irreversible", scores: map[string]float64{"A": 95, "B": 80}, reversible: map[string]bool{"A": true, "B": false}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resolver := Resolver{Oracles: []Oracle{ScorecardAxisOracle{
				Axes:       []ScorecardAxis{{Name: "steerability-score", Scores: tc.scores, Witness: "steerability-ledger@r4"}},
				Reversible: tc.reversible,
			}}}
			got, err := resolver.Resolve(context.Background(), q)
			if err != nil {
				t.Fatal(err)
			}
			if got.Disposition != choicetriage.FreshContext {
				t.Fatalf("got %s, want FRESH_CONTEXT abstention", got.Disposition)
			}
		})
	}
}
