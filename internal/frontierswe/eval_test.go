package frontierswe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunEvalScoresExistingReward is the C13 acceptance: given a verifier
// reward.json, RunEval reproduces the C3 leaderboard number recorded in
// expected.json — offline, no Docker. It exercises every committed fixture so a
// grade regression in any category fails the build.
func TestRunEvalScoresExistingReward(t *testing.T) {
	expected := loadExpected(t)
	for name, want := range expected {
		name, want := name, want
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(rewardFixtureDir, name+".json"))
			if err != nil {
				t.Fatalf("read fixture: %v", err)
			}
			// Stage the reward as <submission>/reward.json so the default lookup path
			// (the run-produced submission tree) is what gets scored.
			sub := t.TempDir()
			if err := os.WriteFile(filepath.Join(sub, "reward.json"), raw, 0o644); err != nil {
				t.Fatalf("stage reward: %v", err)
			}

			ssim := 0.0
			if want.SSIM != nil {
				ssim = *want.SSIM
			}
			res, err := RunEval(EvalConfig{
				Task:          &Task{Name: want.Task},
				SubmissionDir: sub,
				SSIMThreshold: ssim,
			})
			if err != nil {
				t.Fatalf("RunEval: %v", err)
			}
			if !res.Available {
				t.Fatalf("Available=false for a staged reward.json (reason=%q)", res.Reason)
			}
			if res.Source != "existing-reward" {
				t.Errorf("Source = %q, want existing-reward", res.Source)
			}
			if !floatsEqual(res.Correctness, want.Correctness) {
				t.Errorf("correctness = %v, want %v", res.Correctness, want.Correctness)
			}
			if !ptrEqual(res.Speedup, want.Speedup) {
				t.Errorf("speedup = %v, want %v", res.Speedup, want.Speedup)
			}
			if !floatsEqual(res.Score, want.Score) {
				t.Errorf("leaderboard score = %v, want %v", res.Score, want.Score)
			}
		})
	}
}

// TestRunEvalCapturesReward asserts the raw reward.json + a verifier logs dir are
// captured under --out for traceability when a grade is produced.
func TestRunEvalCapturesReward(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(rewardFixtureDir, "git-to-zig_full.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	sub := t.TempDir()
	if err := os.WriteFile(filepath.Join(sub, "reward.json"), raw, 0o644); err != nil {
		t.Fatalf("stage reward: %v", err)
	}
	out := t.TempDir()
	res, err := RunEval(EvalConfig{
		Task: &Task{Name: "git-to-zig"}, SubmissionDir: sub, OutDir: out,
	})
	if err != nil {
		t.Fatalf("RunEval: %v", err)
	}
	if res.RewardPath == "" || res.VerifierLogsDir == "" {
		t.Fatalf("capture paths empty: reward=%q logs=%q", res.RewardPath, res.VerifierLogsDir)
	}
	got, err := os.ReadFile(res.RewardPath)
	if err != nil {
		t.Fatalf("read captured reward: %v", err)
	}
	if string(got) != string(raw) {
		t.Errorf("captured reward.json != source bytes")
	}
	if fi, err := os.Stat(res.VerifierLogsDir); err != nil || !fi.IsDir() {
		t.Errorf("verifier logs dir not created: %v", err)
	}
}

// TestRunEvalAntiCheatForcesZero asserts a flagged trial's gated leaderboard score
// is forced to 0 while correctness is still reported honestly.
func TestRunEvalAntiCheatForcesZero(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(rewardFixtureDir, "git-to-zig_full.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	sub := t.TempDir()
	if err := os.WriteFile(filepath.Join(sub, "reward.json"), raw, 0o644); err != nil {
		t.Fatalf("stage reward: %v", err)
	}
	res, err := RunEval(EvalConfig{
		Task: &Task{Name: "git-to-zig"}, SubmissionDir: sub, AntiCheatFlagged: true,
	})
	if err != nil {
		t.Fatalf("RunEval: %v", err)
	}
	if !floatsEqual(res.Score, 0) {
		t.Errorf("anti-cheat leaderboard score = %v, want 0", res.Score)
	}
	if !res.AntiCheatFlag {
		t.Errorf("AntiCheatFlag = false, want true")
	}
	if !floatsEqual(res.Correctness, 1.0) {
		t.Errorf("correctness = %v, want 1.0 (reported honestly)", res.Correctness)
	}
}

// TestRunEvalGatedWithoutReward asserts the honest gate: with no reward.json and no
// standable verifier environment, RunEval refuses with the reason token + the exact
// remote command, and never a fabricated score.
func TestRunEvalGatedWithoutReward(t *testing.T) {
	res, err := RunEval(EvalConfig{
		Task:          &Task{Name: "git-to-zig"},
		SubmissionDir: t.TempDir(), // empty: no reward.json to score
	})
	if err != nil {
		t.Fatalf("RunEval: %v", err)
	}
	if res.Available {
		t.Fatalf("Available=true without a reward.json (a grade must never be fabricated)")
	}
	if res.Source != "none" {
		t.Errorf("Source = %q, want none", res.Source)
	}
	if !floatsEqual(res.Score, 0) {
		t.Errorf("gated score = %v, want 0 (never fabricated)", res.Score)
	}
	if !strings.Contains(res.Reason, EvalGatedReason) {
		t.Errorf("Reason = %q, want it to carry %s", res.Reason, EvalGatedReason)
	}
	if !strings.Contains(res.RemoteCommand, "docker run") {
		t.Errorf("RemoteCommand = %q, want a docker run verifier command", res.RemoteCommand)
	}
}

// TestVerifierDockerCommand asserts the emitted verifier command is deterministic
// and faithful: the task's oracle command (not an invented CLI), the task's image +
// resource envelope, and mount/env-conveyed submission + artifact paths.
func TestVerifierDockerCommand(t *testing.T) {
	task := &Task{Name: "git-to-zig"}
	task.Environment.DockerImage = "ghcr.io/proximal-labs/frontier-swe/git-to-zig:v6"
	task.Environment.CPUs = 8
	task.Environment.MemoryMB = 16384
	task.Oracle.Command = "python -m harbor verify job.yaml"

	cmd := verifierDockerCommand(task, "/tmp/sub", verifyCommand(task))
	for _, want := range []string{
		"docker run --rm",
		"--cpus 8",
		"--memory 16384m",
		"/tmp/sub:/submission:ro",
		"./logs/verifier:/logs/verifier",
		"FRONTIERSWE_SUBMISSION=/submission",
		"ghcr.io/proximal-labs/frontier-swe/git-to-zig:v6",
		"python -m harbor verify job.yaml",
	} {
		if !strings.Contains(cmd, want) {
			t.Errorf("command missing %q\ngot: %s", want, cmd)
		}
	}
}

// TestVerifyCommandFallback asserts a task with no oracle command falls back to the
// documented harbor default.
func TestVerifyCommandFallback(t *testing.T) {
	if got := verifyCommand(&Task{Name: "git-to-zig"}); got != DefaultVerifyCommand {
		t.Errorf("verifyCommand fallback = %q, want %q", got, DefaultVerifyCommand)
	}
	if got := verifyCommand(&Task{Oracle: Oracle{Command: "  custom verify  "}}); got != "custom verify" {
		t.Errorf("verifyCommand = %q, want trimmed oracle command", got)
	}
}

// TestParseRewardRejectsGarbage asserts malformed reward bytes surface as an error
// rather than a silent zero grade.
func TestParseRewardRejectsGarbage(t *testing.T) {
	if _, err := ParseReward([]byte("{not json")); err == nil {
		t.Fatal("ParseReward accepted malformed JSON")
	}
	r, err := ParseReward([]byte(`{"reward": 1.0}`))
	if err != nil || r == nil {
		t.Fatalf("ParseReward valid reward: %v", err)
	}
}
