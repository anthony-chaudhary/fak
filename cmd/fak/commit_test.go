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
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitintent"
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

func TestRunCommitSubmit_jsonPersistsIntentWithoutGit(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"submit",
		"--json",
		"--queue-dir", queueDir,
		"--id", "issue-1788-cli",
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"--diff-digest", "SHA256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"--path", `internal\commitintent\store.go`,
		"-m", "feat(commitintent): add submit cli (#1788) (fak commitintent)",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitSubmitResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("submit --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if !res.Queued || res.IntentID != "issue-1788-cli" || res.Sequence != 1 || res.QueueSize != 1 {
		t.Fatalf("submit result = %+v", res)
	}
	if got := res.Record.Intent.Paths; len(got) != 1 || got[0] != "internal/commitintent/store.go" {
		t.Fatalf("paths not normalized: %+v", got)
	}
	if got := res.Record.Intent.DiffDigest; got != "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" {
		t.Fatalf("diff digest = %q", got)
	}
	if _, err := os.Stat(filepath.Join(queueDir, "queue.json")); err != nil {
		t.Fatalf("queue file was not written: %v", err)
	}
}

func TestRunCommitSubmit_refusesMissingStampBeforeWritingQueue(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"submit",
		"--queue-dir", queueDir,
		"--id", "issue-1788-cli",
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		"--path", "internal/commitintent/store.go",
		"-m", "feat(commitintent): add submit cli",
	})
	if code != 3 {
		t.Fatalf("want validation refusal exit 3, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	if _, err := os.Stat(filepath.Join(queueDir, "queue.json")); !os.IsNotExist(err) {
		t.Fatalf("queue should not be written on refusal, stat err=%v", err)
	}
}

func TestRunCommitDrainDryRunPlansRollup(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "intent-a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{`internal\commitintent\a.go`}, "worker-a")
	submitDrainIntent(t, queueDir, "intent-b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/b.go"}, "worker-b")

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--dry-run",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if !res.DryRun || res.Drained || !res.Plan.OK {
		t.Fatalf("drain result = %+v", res)
	}
	assertStrings(t, res.Plan.IntentIDs, []string{"intent-a", "intent-b"})
	assertStrings(t, res.Plan.Submitters, []string{"worker-a", "worker-b"})
	assertStrings(t, res.Plan.UnionPaths, []string{"internal/commitintent/a.go", "internal/commitintent/b.go"})
	if res.Commit != nil {
		t.Fatalf("dry-run should not call commit, got %+v", res.Commit)
	}
	if !strings.Contains(res.Plan.Subject, "intent-a, intent-b") || !strings.Contains(res.Plan.Subject, "(fak commitintent)") {
		t.Fatalf("subject should include ids and stamp, got %q", res.Plan.Subject)
	}
	states := drainQueueStates(t, queueDir)
	if states["intent-a"] != commitintent.StatePending || states["intent-b"] != commitintent.StatePending {
		t.Fatalf("dry-run should leave intents pending, states=%+v", states)
	}
}

func TestRunCommitDrainExecutesRollupWithUnionPathsAndMarksDone(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "intent-a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")
	submitDrainIntent(t, queueDir, "intent-b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/b.go"}, "worker-b")

	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	assertStrings(t, got.Paths, []string{"internal/commitintent/a.go", "internal/commitintent/b.go"})
	if !got.SignOff {
		t.Fatalf("drain commit should sign off by default")
	}
	if !strings.Contains(got.Message, "intent-a, intent-b") || !strings.Contains(got.Message, "(fak commitintent)") {
		t.Fatalf("commit message should include rollup ids and stamp, got %q", got.Message)
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if !res.Drained || res.Pathset == nil || !res.Pathset.OK {
		t.Fatalf("result should be drained with pathset witness, got %+v", res)
	}
	assertStrings(t, res.MarkedDone, []string{"intent-a", "intent-b"})
	states := drainQueueStates(t, queueDir)
	if states["intent-a"] != commitintent.StateDone || states["intent-b"] != commitintent.StateDone {
		t.Fatalf("successful drain should mark intents done, states=%+v", states)
	}
}

func TestRunCommitDrainDryRunRefusesStaleAndOverlap(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")
	submitDrainIntent(t, queueDir, "overlap", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-b")
	submitDrainIntent(t, queueDir, "stale", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", []string{"internal/commitintent/stale.go"}, "worker-c")

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--dry-run",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want dry-run exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	assertStrings(t, res.Plan.IntentIDs, []string{"base"})
	if len(res.Stale) != 1 || res.Stale[0].Intent.ID != "stale" {
		t.Fatalf("stale records = %+v", res.Stale)
	}
	if !drainHasRefusal(res, "overlap", "OVERLAPPING_PATH") || !drainHasRefusal(res, "stale", "STALE_INPUT") {
		t.Fatalf("refusals = %+v", res.Plan.Refusals)
	}
}

func TestRunCommitDrainNoRollupKeepsOneIntentMode(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "first", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")
	submitDrainIntent(t, queueDir, "second", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/b.go"}, "worker-b")

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--dry-run",
		"--no-rollup",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 0 {
		t.Fatalf("want dry-run exit 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if res.Plan.RollupEnabled {
		t.Fatalf("--no-rollup should disable rollup: %+v", res.Plan)
	}
	assertStrings(t, res.Plan.IntentIDs, []string{"first"})
	if !drainHasRefusal(res, "second", "ROLLUP_DISABLED") {
		t.Fatalf("refusals = %+v", res.Plan.Refusals)
	}
}

func TestRunCommitDrainPathsetMismatchDoesNotMarkDone(t *testing.T) {
	queueDir := filepath.Join(t.TempDir(), ".fak", "commit-intents")
	submitDrainIntent(t, queueDir, "intent-a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", []string{"internal/commitintent/a.go"}, "worker-a")

	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed: true,
			Verified:  true,
			SHA:       "abc123",
			Paths:     append(append([]string(nil), o.Paths...), "internal/commitintent/extra.go"),
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{
		"drain",
		"--json",
		"--queue-dir", queueDir,
		"--base", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	})
	if code != 1 {
		t.Fatalf("pathset mismatch should exit 1, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	var res commitDrainResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("drain --json emitted invalid JSON: %v\n%s", err, out.String())
	}
	if res.Drained || res.Pathset == nil || res.Pathset.OK {
		t.Fatalf("pathset mismatch should block drain, got %+v", res)
	}
	states := drainQueueStates(t, queueDir)
	if states["intent-a"] != commitintent.StatePending {
		t.Fatalf("mismatched pathset should leave intent pending, states=%+v", states)
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
			LockHoldNS: 17_000_000,
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
	if res.LockHoldNS != 17_000_000 {
		t.Fatalf("json result lost lock hold duration: %+v", res)
	}
}

func TestRunCommit_humanOutputShowsLockHoldDuration(t *testing.T) {
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{
			Committed:  true,
			Verified:   true,
			SHA:        "deadbeefcafe",
			Paths:      o.Paths,
			LockHoldNS: int64(23 * time.Millisecond),
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "lock hold: 23ms") {
		t.Fatalf("human output should expose lock hold duration, got %q", out.String())
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

func submitDrainIntent(t *testing.T, queueDir, id, base string, paths []string, requester string) {
	t.Helper()
	store := commitintent.Store{
		Dir: queueDir,
		Now: func() time.Time { return time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC) },
	}
	if _, _, err := store.Submit(commitintent.Intent{
		ID:      id,
		BaseSHA: base,
		Paths:   paths,
		Subject: "feat(commitintent): add drain rollup (#1789) (fak commitintent)",
		Metadata: commitintent.StampMetadata{
			Issue:     1789,
			Requester: requester,
		},
	}); err != nil {
		t.Fatalf("submit drain intent %s: %v", id, err)
	}
}

func drainQueueStates(t *testing.T, queueDir string) map[string]commitintent.State {
	t.Helper()
	q, err := (commitintent.Store{Dir: queueDir}).Load()
	if err != nil {
		t.Fatalf("Load queue: %v", err)
	}
	out := map[string]commitintent.State{}
	for _, rec := range q.Records {
		out[rec.Intent.ID] = rec.State
	}
	return out
}

func drainHasRefusal(res commitDrainResult, id, reason string) bool {
	for _, refusal := range res.Plan.Refusals {
		if refusal.IntentID == id && string(refusal.Reason) == reason {
			return true
		}
	}
	return false
}

func assertStrings(t *testing.T, got, want []string) {
	t.Helper()
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("strings = %v, want %v", got, want)
	}
}

var errTest = errTestErr{}

type errTestErr struct{}

func (errTestErr) Error() string { return "test infra failure" }
