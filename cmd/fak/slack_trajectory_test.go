package main

import (
	"bytes"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func trajectoryFixture(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "traj.jsonl")
	rows := []trajctl.Row{
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "healthy", Statement: "ok", Status: trajctl.StatusActive}),
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "stalled", Statement: "stuck", Status: trajctl.StatusActive}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "stalled", Method: "commit-progress", Version: "1", Witness: trajctl.W3, Value: .4, UnixMillis: 1}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "stalled", Method: "commit-progress", Version: "1", Witness: trajctl.W3, Value: .4, UnixMillis: 2}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "stalled", Method: trajctl.ActivityDivergenceScorerMethod, Version: "1", Witness: trajctl.W2, Value: 1, UnixMillis: 2}),
	}
	for _, r := range rows {
		if err := trajctl.Append(p, r); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

func TestSlackTrajectoryDryRunGolden(t *testing.T) {
	var out, errout bytes.Buffer
	code := runSlackTrajectory(&out, &errout, []string{"--dry-run", "--ledger", trajectoryFixture(t)})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	want := "trajectory control - worst first\n- stalled - STALL - 0.00 (+0.00): flat progress (delta +0.00) with an active divergence signal\n"
	if out.String() != want {
		t.Fatalf("dry run:\n%s\nwant:\n%s", out.String(), want)
	}
}

func TestSlackTrajectoryLiveRequiresChannel(t *testing.T) {
	t.Setenv(trajectoryChannelEnv, "")
	var out, errout bytes.Buffer
	if code := runSlackTrajectory(&out, &errout, []string{"--ledger", trajectoryFixture(t), "--token", "x"}); code != 2 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	if !strings.Contains(errout.String(), trajectoryChannelEnv) {
		t.Fatalf("err=%q", errout.String())
	}
}

func TestSlackTrajectorySuppressesUnchangedDigestDuringCooldown(t *testing.T) {
	old := trajectoryPost
	defer func() { trajectoryPost = old }()
	posts := 0
	trajectoryPost = func(_, _, _, _ string) error { posts++; return nil }
	ledger := trajectoryFixture(t)
	state := filepath.Join(t.TempDir(), "state.json")
	args := []string{"--ledger", ledger, "--channel", "C1", "--token", "x", "--state", state, "--cooldown", time.Hour.String()}
	for i := 0; i < 2; i++ {
		var out, errout bytes.Buffer
		if code := runSlackTrajectory(&out, &errout, args); code != 0 {
			t.Fatalf("run %d code=%d err=%s", i, code, errout.String())
		}
	}
	if posts != 1 {
		t.Fatalf("posts=%d want 1", posts)
	}
}
