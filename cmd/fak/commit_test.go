package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// withCommitFn swaps the commitFn seam for the duration of a test.
func withCommitFn(t *testing.T, fn func(context.Context, safecommit.Options) (safecommit.Result, error)) {
	t.Helper()
	prev := commitFn
	commitFn = fn
	t.Cleanup(func() { commitFn = prev })
}

func TestRunCommit_noPathsIsUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"-m", "msg"})
	if code != 2 {
		t.Fatalf("want exit 2 for no paths, got %d (stderr=%q)", code, errb.String())
	}
}

func TestRunCommit_noMessageIsUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go"})
	if code != 2 {
		t.Fatalf("want exit 2 for no message, got %d (stderr=%q)", code, errb.String())
	}
}

func TestRunCommit_dashMAndDashFAreExclusive(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "x", "-F", "f.txt"})
	if code != 2 {
		t.Fatalf("want exit 2 for -m+-F, got %d", code)
	}
	if !strings.Contains(errb.String(), "mutually exclusive") {
		t.Fatalf("stderr should explain the conflict, got %q", errb.String())
	}
}

func TestRunCommit_positionalPathsAfterDashDash(t *testing.T) {
	var gotOpts safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		gotOpts = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"-m", "msg", "--", "internal/x.go", "internal/y.go"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
	}
	if len(gotOpts.Paths) != 2 {
		t.Fatalf("positional paths after -- should reach Options, got %v", gotOpts.Paths)
	}
	if !gotOpts.SignOff {
		t.Fatalf("sign-off should default on")
	}
}

func TestRunCommit_jsonShapeAndRaceExitCode(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed:  true,
			Verified:   false,
			SHA:        "deadbeefcafe",
			Paths:      o.Paths,
			Reason:     safecommit.ReasonPathspecRace,
			RacedExtra: []string{"internal/peer/swept.go"},
			HeadBefore: "0000111122223333",
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--json", "--path", "a.go", "-m", "msg"})
	// PATHSPEC_RACE: the commit ran but is bad -> exit 1 (halt).
	if code != 1 {
		t.Fatalf("race should exit 1, got %d", code)
	}
	var res safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("--json must emit a valid Result: %v\noutput=%q", err, out.String())
	}
	if res.Reason != safecommit.ReasonPathspecRace || len(res.RacedExtra) != 1 {
		t.Fatalf("json result lost the race evidence: %+v", res)
	}
	if res.Score == 0 || res.Grade == "" {
		t.Fatalf("json result should include scored outcome, got %+v", res)
	}
}

func TestRunCommit_offTrunkExit3(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Reason: safecommit.ReasonOffTrunk, Detail: "on feature/x, expected development branch main", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 3 {
		t.Fatalf("a pre-commit refusal should exit 3, got %d", code)
	}
	if !strings.Contains(out.String(), "score:") {
		t.Fatalf("human refusal output should include score, got %q", out.String())
	}
}

func TestRunCommit_infraErrorExit1(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{}, errTest
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 1 {
		t.Fatalf("an infra error should exit 1, got %d", code)
	}
}

func TestRunCommit_messageFromStdin(t *testing.T) {
	var gotMsg string
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		gotMsg = o.Message
		return safecommit.Result{Committed: true, Verified: true, Paths: o.Paths}, nil
	})
	prev := stdin
	stdin = func() io.Reader { return strings.NewReader("from stdin — em-dash ok\n") }
	t.Cleanup(func() { stdin = prev })

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-F", "-"})
	if code != 0 {
		t.Fatalf("want 0, got %d (stderr=%q)", code, errb.String())
	}
	if !strings.Contains(gotMsg, "from stdin") {
		t.Fatalf("message should come from stdin, got %q", gotMsg)
	}
}

func TestRunCommit_reviewModelWiresSafecommitReview(t *testing.T) {
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--path", "a.go",
		"-m", "feat(loop): add review (#1185) (fak cmd)",
		"--review-model", "cheap-scout",
		"--review-objective", "ship issue 1185",
	})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q", code, errb.String())
	}
	if got.Review == nil {
		t.Fatal("--review-model did not wire safecommit Review")
	}
	if got.Review.Model != "cheap-scout" || got.Review.Objective != "ship issue 1185" {
		t.Fatalf("review options = %+v", got.Review)
	}
}

func TestParseCommitReviewScoutLabelAcceptsFencedJSON(t *testing.T) {
	label, err := parseCommitReviewScoutLabel("```json\n{\"verdict\":\"refute\",\"reason\":\"missing test\"}\n```")
	if err != nil {
		t.Fatalf("parseCommitReviewScoutLabel: %v", err)
	}
	if label.Labels["verdict"] != "refute" || label.Labels["reason"] != "missing test" {
		t.Fatalf("label = %+v", label)
	}
}

func TestRunCommitReviewRefuteRecordsLoopEvidenceAndGoalScratch(t *testing.T) {
	tmp := t.TempDir()
	ledger := filepath.Join(tmp, "loops.jsonl")
	goal := filepath.Join(tmp, "GOAL.md")
	if err := os.WriteFile(goal, []byte(`---
loop: issue-1185
witness: commit-audit
---
# Objective
Ship the review rung.

# Plan
- [ ] wire review
`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_GOAL_LOOP", "issue-1185")
	t.Setenv("FAK_GOAL_ITER", "2")
	t.Setenv("FAK_GOAL_SPEC", goal)
	t.Setenv("FAK_LOOP_LEDGER", ledger)

	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Paths:  o.Paths,
			Reason: safecommit.ReasonReviewRefuted,
			Detail: "missing regression test",
			Review: &modelroute.ReviewResult{
				Model:      "cheap-scout",
				Verdict:    modelroute.ReviewRefute,
				Reason:     "missing regression test",
				DiffSHA256: "abc123",
				ScoutCalls: 1,
			},
		}, nil
	})

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "feat(loop): add review (#1185) (fak cmd)", "--review-model", "cheap-scout"})
	if code != 3 {
		t.Fatalf("review refute should exit 3, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load ledger: %v", err)
	}
	if len(events) != 1 || events[0].Kind != loopmgr.EventHeartbeat || events[0].Reason != "REVIEW_REFUTED" {
		t.Fatalf("review event = %+v", events)
	}
	if len(events[0].EvidenceRefs) != 1 || events[0].EvidenceRefs[0].Kind != "review" || events[0].EvidenceRefs[0].Ref != "refute" {
		t.Fatalf("review evidence = %+v", events[0].EvidenceRefs)
	}
	if events[0].EvidenceRefs[0].SHA256 != "abc123" || events[0].Metrics["scout_calls"] != 1 {
		t.Fatalf("review evidence lost digest/metrics: %+v", events[0])
	}
	raw, err := os.ReadFile(goal)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "NOT_YET review refuted") || !strings.Contains(string(raw), "missing regression test") {
		t.Fatalf("goal scratch missing critique:\n%s", string(raw))
	}
}

func TestRunCommitReviewPassRecordsLoopEvidence(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	t.Setenv("FAK_GOAL_LOOP", "issue-1185")
	t.Setenv("FAK_GOAL_RUN", "run-1")
	t.Setenv("FAK_LOOP_LEDGER", ledger)

	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed: true,
			Verified:  true,
			SHA:       "abc",
			Paths:     o.Paths,
			Review: &modelroute.ReviewResult{
				Model:      "cheap-scout",
				Verdict:    modelroute.ReviewPass,
				Reason:     "diff matches objective",
				DiffSHA256: "def456",
				ScoutCalls: 1,
			},
		}, nil
	})

	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "feat(loop): add review (#1185) (fak cmd)"})
	if code != 0 {
		t.Fatalf("review pass should exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	events, err := loopmgr.Load(ledger)
	if err != nil {
		t.Fatalf("load ledger: %v", err)
	}
	if len(events) != 1 || events[0].Reason != "REVIEW_PASS" || events[0].EvidenceRefs[0].Ref != "pass" {
		t.Fatalf("review event = %+v", events)
	}
}

var errTest = errTestErr{}

type errTestErr struct{}

func (errTestErr) Error() string { return "test infra failure" }
