package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func withPassedCommitBuildCheck(t *testing.T) {
	t.Helper()
	old := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = old })
	commitBuildCheckGate = func(io.Writer, string, []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckPassed, ""
	}
}

// scoredVelocityResult builds a verified+pushed Result whose local and push boundaries both
// landed under their default budgets, so ScoreCommitVelocity qualifies both legs (SCORED).
// The elapsed values are injected (not measured), keeping the CLI test clock-free.
func scoredVelocityResult(o safecommit.Options) safecommit.Result {
	res := safecommit.Result{
		Committed: true,
		Verified:  true,
		Pushed:    true,
		SHA:       "deadbeefcafe",
		Paths:     o.Paths,
	}
	v := safecommit.ScoreCommitVelocity(res, 1*time.Millisecond, 2*time.Second, safecommit.DefaultVelocityBudgets)
	res.Velocity = &v
	return res
}

// TestRunCommit_humanOutputShowsScoredVelocity: a verified, pushed commit under budget renders
// both velocity legs as SCORED with their budget-relative score, distinct from the quality score
// (#4241).
func TestRunCommit_humanOutputShowsScoredVelocity(t *testing.T) {
	withPassedCommitBuildCheck(t)
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return scoredVelocityResult(o), nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--push", "--path", "a.go", "-m", "msg"})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q stdout=%q", code, errb.String(), out.String())
	}
	s := out.String()
	if !strings.Contains(s, "velocity local: 100/100 (SCORED") {
		t.Fatalf("human output should show SCORED local velocity, got %q", s)
	}
	if !strings.Contains(s, "velocity push: 100/100 (SCORED") {
		t.Fatalf("human output should show SCORED push velocity, got %q", s)
	}
}

// TestRunCommit_jsonCarriesVelocityObject: --json emits the nested velocity object with separate
// local/push legs and a non-nil score on each qualified leg (#4241).
func TestRunCommit_jsonCarriesVelocityObject(t *testing.T) {
	withPassedCommitBuildCheck(t)
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return scoredVelocityResult(o), nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--push", "--json", "--path", "a.go", "-m", "msg"})
	if code != 0 {
		t.Fatalf("want 0, got %d stderr=%q", code, errb.String())
	}
	var res safecommit.Result
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("--json must emit a valid Result: %v\noutput=%q", err, out.String())
	}
	if res.Velocity == nil {
		t.Fatalf("json result lost the velocity object: %+v", res)
	}
	if res.Velocity.Local.Score == nil || *res.Velocity.Local.Score != 100 {
		t.Fatalf("qualified local leg should carry score 100, got %+v", res.Velocity.Local)
	}
	if res.Velocity.Push.Score == nil || res.Velocity.Push.Status != safecommit.VelocityScored {
		t.Fatalf("qualified push leg should be SCORED, got %+v", res.Velocity.Push)
	}
	// The velocity budgets are separate, not pooled: the push leg is graded against the push
	// budget, not the local one.
	if res.Velocity.Push.BudgetNS != safecommit.DefaultVelocityBudgets.Push.Nanoseconds() {
		t.Fatalf("push leg must be graded against the push budget, got %d", res.Velocity.Push.BudgetNS)
	}
}

// TestRunCommit_fastFailureStaysUnscored: a fast PATHSPEC_RACE (committed but unverified) keeps
// its measured timing but MUST NOT earn a velocity score — the anti-gaming rule that a fast
// failure/no-op stays UNSCORED (#4241). Covered in both human and JSON surfaces.
func TestRunCommit_fastFailureStaysUnscored(t *testing.T) {
	fastRace := func(o safecommit.Options) safecommit.Result {
		res := safecommit.Result{
			Committed:  true,
			Verified:   false, // raced: the requested paths did not verify
			SHA:        "deadbeefcafe",
			Paths:      o.Paths,
			Reason:     safecommit.ReasonPathspecRace,
			RacedExtra: []string{"internal/peer/swept.go"},
			LockHoldNS: 500_000, // fast — but must not buy a score
		}
		// Injected sub-millisecond elapsed: a fast failure is still fast, yet UNSCORED.
		v := safecommit.ScoreCommitVelocity(res, 200*time.Microsecond, 0, safecommit.DefaultVelocityBudgets)
		res.Velocity = &v
		return res
	}

	// Human surface: the local leg reports UNSCORED with its retained timing, no numeric score.
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return fastRace(o), nil
	})
	var out, errb bytes.Buffer
	if code := runCommit(&out, &errb, []string{"--path", "a.go", "-m", "msg"}); code != 1 {
		t.Fatalf("a raced commit should exit 1, got %d", code)
	}
	s := out.String()
	if !strings.Contains(s, "velocity local: UNSCORED") {
		t.Fatalf("a fast failure must render UNSCORED, got %q", s)
	}
	if strings.Contains(s, "velocity local: 100/100") || strings.Contains(s, "velocity local: 0/100") {
		t.Fatalf("a fast failure must not render a numeric velocity score, got %q", s)
	}

	// JSON surface: the local leg's score is null (nil) with the timing retained.
	var jout, jerr bytes.Buffer
	if code := runCommit(&jout, &jerr, []string{"--json", "--path", "a.go", "-m", "msg"}); code != 1 {
		t.Fatalf("a raced commit should exit 1, got %d", code)
	}
	var res safecommit.Result
	if err := json.Unmarshal(jout.Bytes(), &res); err != nil {
		t.Fatalf("--json must emit a valid Result: %v", err)
	}
	if res.Velocity == nil {
		t.Fatalf("velocity timing must be retained even when UNSCORED: %+v", res)
	}
	if res.Velocity.Local.Score != nil {
		t.Fatalf("a fast failure must keep score:null, got %v", *res.Velocity.Local.Score)
	}
	if res.Velocity.Local.Status != safecommit.VelocityUnscored {
		t.Fatalf("a fast failure local leg must be UNSCORED, got %q", res.Velocity.Local.Status)
	}
	if res.Velocity.Local.ElapsedNS == 0 {
		t.Fatalf("timing must be retained on an UNSCORED leg, got %+v", res.Velocity.Local)
	}
}

// TestRenderCommitResult_sweepReusesSameVelocity: fak sweep --apply renders through the same
// renderCommitResult path as fak commit, so a swept commit exposes the identical velocity
// evidence without inventing a second score (#4241). This asserts the shared renderer directly,
// which is exactly what runSweepApply calls.
func TestRenderCommitResult_sweepReusesSameVelocity(t *testing.T) {
	res := scoredVelocityResult(safecommit.Options{Paths: []string{"internal/compute/x.go"}})
	var out bytes.Buffer
	renderCommitResult(&out, res)
	s := out.String()
	if !strings.Contains(s, "velocity local: 100/100 (SCORED") || !strings.Contains(s, "velocity push: 100/100 (SCORED") {
		t.Fatalf("sweep's shared renderer must expose the same velocity legs, got %q", s)
	}
}
