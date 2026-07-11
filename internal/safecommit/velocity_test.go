package safecommit

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// testBudgets are small, explicit budgets so the injected elapsed values below read
// unambiguously as under/over budget without depending on DefaultVelocityBudgets.
var testBudgets = VelocityBudgets{Local: 10 * time.Second, Push: 30 * time.Second}

func TestScoreCommitVelocity_verifiedLocalScoresLocalOnly(t *testing.T) {
	v := ScoreCommitVelocity(
		Result{Committed: true, Verified: true},
		4*time.Second, 4*time.Second, testBudgets,
	)

	if v.Local.Status != VelocityScored || v.Local.Score == nil || *v.Local.Score != 100 {
		t.Fatalf("local leg = %+v, want SCORED score 100", v.Local)
	}
	if v.Local.ElapsedNS != int64(4*time.Second) || v.Local.BudgetNS != int64(10*time.Second) {
		t.Fatalf("local timing = %+v, want elapsed 4s / budget 10s retained", v.Local)
	}
	// Verified but not pushed: push retains timing but earns no score.
	if v.Push.Status != VelocityUnscored || v.Push.Score != nil {
		t.Fatalf("push leg = %+v, want UNSCORED nil score", v.Push)
	}
	if v.Push.ElapsedNS != int64(4*time.Second) {
		t.Fatalf("push timing not retained: %+v", v.Push)
	}
	if v.Push.Note != "no verified push" {
		t.Fatalf("push note = %q, want %q", v.Push.Note, "no verified push")
	}
}

func TestScoreCommitVelocity_verifiedPushScoresBothLegs(t *testing.T) {
	v := ScoreCommitVelocity(
		Result{Committed: true, Verified: true, Pushed: true},
		3*time.Second, 12*time.Second, testBudgets,
	)

	if v.Local.Status != VelocityScored || v.Local.Score == nil || *v.Local.Score != 100 {
		t.Fatalf("local leg = %+v, want SCORED 100", v.Local)
	}
	if v.Push.Status != VelocityScored || v.Push.Score == nil || *v.Push.Score != 100 {
		t.Fatalf("push leg = %+v, want SCORED 100 (12s within 30s budget)", v.Push)
	}
}

func TestScoreCommitVelocity_overBudgetDecays(t *testing.T) {
	// Push elapsed at 2x its budget must decay to 50; local within budget stays 100.
	v := ScoreCommitVelocity(
		Result{Committed: true, Verified: true, Pushed: true},
		5*time.Second, 60*time.Second, testBudgets,
	)
	if v.Local.Score == nil || *v.Local.Score != 100 {
		t.Fatalf("local score = %v, want 100", v.Local.Score)
	}
	if v.Push.Score == nil || *v.Push.Score != 50 {
		t.Fatalf("push score = %v, want 50 at 2x budget", v.Push.Score)
	}
}

func TestScoreCommitVelocity_fastNoOpStaysUnscored(t *testing.T) {
	// A no-op that returned almost instantly must NOT earn a velocity score:
	// speed without a qualifying effect is not velocity.
	v := ScoreCommitVelocity(
		Result{Reason: ReasonNothingStaged},
		1*time.Millisecond, 1*time.Millisecond, testBudgets,
	)
	if v.Local.Status != VelocityUnscored || v.Local.Score != nil {
		t.Fatalf("local leg = %+v, want UNSCORED for a no-op", v.Local)
	}
	if v.Push.Status != VelocityUnscored || v.Push.Score != nil {
		t.Fatalf("push leg = %+v, want UNSCORED for a no-op", v.Push)
	}
	if v.Local.Note != "no commit landed" {
		t.Fatalf("local note = %q, want %q", v.Local.Note, "no commit landed")
	}
	// Timing is still retained even though nothing scored.
	if v.Local.ElapsedNS != int64(1*time.Millisecond) {
		t.Fatalf("timing not retained on no-op: %+v", v.Local)
	}
}

func TestScoreCommitVelocity_refusalStaysUnscored(t *testing.T) {
	v := ScoreCommitVelocity(
		Result{Reason: ReasonHookRefused},
		2*time.Second, 2*time.Second, testBudgets,
	)
	if v.Local.Score != nil || v.Push.Score != nil {
		t.Fatalf("refusal must not score: %+v", v)
	}
}

func TestScoreCommitVelocity_racedCommitStaysUnscored(t *testing.T) {
	// A raced commit landed (Committed) but did not verify: local requires Verified,
	// so both legs stay UNSCORED while retaining their timing.
	v := ScoreCommitVelocity(
		Result{Committed: true, Reason: ReasonPathspecRace},
		6*time.Second, 6*time.Second, testBudgets,
	)
	if v.Local.Status != VelocityUnscored || v.Local.Score != nil {
		t.Fatalf("raced local leg = %+v, want UNSCORED", v.Local)
	}
	if v.Local.Note != "commit did not verify" {
		t.Fatalf("raced local note = %q, want %q", v.Local.Note, "commit did not verify")
	}
	if v.Local.ElapsedNS != int64(6*time.Second) {
		t.Fatalf("raced timing not retained: %+v", v.Local)
	}
}

func TestScoreCommitVelocity_pushRejectedScoresLocalNotPush(t *testing.T) {
	// A verified local commit whose push was rejected: local qualifies (Committed &&
	// Verified), push does not (not Pushed) and reports the rejection.
	v := ScoreCommitVelocity(
		Result{Committed: true, Verified: true, Reason: ReasonPushRejected},
		4*time.Second, 8*time.Second, testBudgets,
	)
	if v.Local.Status != VelocityScored || v.Local.Score == nil || *v.Local.Score != 100 {
		t.Fatalf("push-rejected local leg = %+v, want SCORED 100", v.Local)
	}
	if v.Push.Status != VelocityUnscored || v.Push.Score != nil {
		t.Fatalf("push-rejected push leg = %+v, want UNSCORED", v.Push)
	}
	if v.Push.Note != "push rejected" {
		t.Fatalf("push note = %q, want %q", v.Push.Note, "push rejected")
	}
	if v.Push.ElapsedNS != int64(8*time.Second) {
		t.Fatalf("push-rejected timing not retained: %+v", v.Push)
	}
}

func TestCommitVelocity_unscoredMarshalsNullScore(t *testing.T) {
	// The JSON contract: an UNSCORED leg serialises score:null (not 0, not omitted),
	// so a consumer can distinguish "did not qualify" from "scored zero".
	v := ScoreCommitVelocity(Result{}, 2*time.Second, 2*time.Second, DefaultVelocityBudgets)
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"score":null`) {
		t.Fatalf("unscored leg JSON = %s, want a score:null field", b)
	}
	if !strings.Contains(string(b), `"status":"UNSCORED"`) {
		t.Fatalf("unscored leg JSON = %s, want status UNSCORED", b)
	}
}

func TestBudgetScore_boundaryAndFloor(t *testing.T) {
	cases := []struct {
		name    string
		elapsed time.Duration
		budget  time.Duration
		want    int
	}{
		{"under budget", 1 * time.Second, 10 * time.Second, 100},
		{"at budget", 10 * time.Second, 10 * time.Second, 100},
		{"2x budget", 20 * time.Second, 10 * time.Second, 50},
		{"10x budget", 100 * time.Second, 10 * time.Second, 10},
		{"far over floors above zero", time.Hour, time.Millisecond, 0},
		{"zero budget", 5 * time.Second, 0, 0},
	}
	for _, tc := range cases {
		if got := budgetScore(tc.elapsed, tc.budget); got != tc.want {
			t.Errorf("%s: budgetScore(%v,%v) = %d, want %d", tc.name, tc.elapsed, tc.budget, got, tc.want)
		}
	}
}
