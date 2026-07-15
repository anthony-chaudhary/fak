package frontierswe

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The --run-verifier gate tests pin the local capability via EvalConfig.capability
// so both branches are deterministic offline: no test here ever depends on whether
// the box running it happens to have Docker installed, and none ever spawns a real
// container — the runner seam witnesses the invocation instead.

func runnableCapability() *EnvAdapterCapability {
	return &EnvAdapterCapability{DockerPresent: true, Runnable: true}
}

func gatedCapability() *EnvAdapterCapability {
	return &EnvAdapterCapability{
		DockerPresent: false,
		Runnable:      false,
		Reason:        "Docker not found on this host; run the emitted command on a Docker/GHCR/Modal-capable box",
	}
}

// TestRunEvalRunVerifierRefusedWhenNotCapable is the honest-gate half of the
// #1719 acceptance: --run-verifier on an incapable host REFUSES with the
// capability reason + the exact remote command, never invokes the runner, and
// never fabricates a score.
func TestRunEvalRunVerifierRefusedWhenNotCapable(t *testing.T) {
	invoked := false
	res, err := RunEval(EvalConfig{
		Task:          &Task{Name: "git-to-zig"},
		SubmissionDir: t.TempDir(), // empty: no reward.json to score
		RunVerifier:   true,
		capability:    gatedCapability(),
		runner: func(context.Context, []string, string) error {
			invoked = true
			return nil
		},
	})
	if err != nil {
		t.Fatalf("RunEval: %v", err)
	}
	if invoked {
		t.Fatal("verifier runner invoked on an incapable host")
	}
	if res.Available {
		t.Fatal("Available=true on an incapable host (a grade must never be fabricated)")
	}
	if res.Source != "none" {
		t.Errorf("Source = %q, want none", res.Source)
	}
	if !floatsEqual(res.Score, 0) {
		t.Errorf("gated score = %v, want 0", res.Score)
	}
	if !strings.Contains(res.Reason, EvalGatedReason) || !strings.Contains(res.Reason, "Docker not found") {
		t.Errorf("Reason = %q, want %s + the capability reason", res.Reason, EvalGatedReason)
	}
	if !strings.Contains(res.RemoteCommand, "docker run") {
		t.Errorf("RemoteCommand = %q, want a docker run verifier command", res.RemoteCommand)
	}
}

// TestRunEvalNeverRunsVerifierWithoutConsent nails the no-silent-side-effect
// contract: even on a capable host, a bare grade call with no reward.json gates
// (pointing at --run-verifier) rather than spawning the heavyweight verifier.
func TestRunEvalNeverRunsVerifierWithoutConsent(t *testing.T) {
	invoked := false
	res, err := RunEval(EvalConfig{
		Task:          &Task{Name: "git-to-zig"},
		SubmissionDir: t.TempDir(),
		capability:    runnableCapability(),
		runner: func(context.Context, []string, string) error {
			invoked = true
			return nil
		},
	})
	if err != nil {
		t.Fatalf("RunEval: %v", err)
	}
	if invoked {
		t.Fatal("verifier runner invoked without RunVerifier consent")
	}
	if res.Available {
		t.Fatal("Available=true without a reward.json")
	}
	if !strings.Contains(res.Reason, "--run-verifier") {
		t.Errorf("Reason = %q, want it to point at --run-verifier", res.Reason)
	}
}

// TestRunEvalRunVerifierScoresProducedReward is the available-path wiring: on a
// capable host, --run-verifier invokes the exact verifierDockerArgs vector under
// a [verifier] deadline, and the reward.json the verifier drops in the mounted
// /logs/verifier dir is scored with C3 and captured under --out — a real grade,
// Source=verifier-run.
func TestRunEvalRunVerifierScoresProducedReward(t *testing.T) {
	want, ok := loadExpected(t)["git-to-zig_full"]
	if !ok {
		t.Fatal("expected.json has no git-to-zig_full row")
	}
	raw, err := os.ReadFile(filepath.Join(rewardFixtureDir, "git-to-zig_full.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	sub := t.TempDir()
	out := t.TempDir()
	var gotArgs []string
	var gotLogs string
	res, err := RunEval(EvalConfig{
		Task:          &Task{Name: want.Task},
		SubmissionDir: sub,
		OutDir:        out,
		RunVerifier:   true,
		capability:    runnableCapability(),
		runner: func(ctx context.Context, args []string, logsDir string) error {
			if _, hasDeadline := ctx.Deadline(); !hasDeadline {
				t.Error("verifier runner ctx carries no [verifier] timeout deadline")
			}
			gotArgs, gotLogs = args, logsDir
			// The verifier drops reward.json in the host dir mounted at /logs/verifier.
			return os.WriteFile(filepath.Join(logsDir, "reward.json"), raw, 0o644)
		},
	})
	if err != nil {
		t.Fatalf("RunEval: %v", err)
	}
	if !res.Available {
		t.Fatalf("Available=false after a verifier run produced reward.json (reason=%q)", res.Reason)
	}
	if res.Source != "verifier-run" {
		t.Errorf("Source = %q, want verifier-run", res.Source)
	}
	if !floatsEqual(res.Correctness, want.Correctness) || !floatsEqual(res.Score, want.Score) || !ptrEqual(res.Speedup, want.Speedup) {
		t.Errorf("grade = (%v, %v, %v), want (%v, %v, %v)",
			res.Correctness, res.Speedup, res.Score, want.Correctness, want.Speedup, want.Score)
	}

	// The invocation is the exact verifierDockerArgs vector against the abs
	// submission tree + the --out logs dir.
	if wantLogs := filepath.Join(out, "logs", "verifier"); gotLogs != wantLogs {
		t.Errorf("logs dir = %q, want %q", gotLogs, wantLogs)
	}
	joined := strings.Join(gotArgs, " ")
	for _, part := range []string{
		"docker run --rm",
		sub + ":/submission:ro",
		gotLogs + ":/logs/verifier",
		"FRONTIERSWE_SUBMISSION=/submission",
		"ghcr.io/proximal-labs/frontier-swe/" + want.Task + ":v6",
		DefaultVerifyCommand,
	} {
		if !strings.Contains(joined, part) {
			t.Errorf("invocation missing %q\ngot: %s", part, joined)
		}
	}

	// Traceability: the raw reward.json stays where the verifier dropped it, and
	// the graded copy + verifier logs dir are captured under --out.
	if res.RewardPath != filepath.Join(out, "reward.json") {
		t.Errorf("RewardPath = %q, want the --out capture", res.RewardPath)
	}
	captured, err := os.ReadFile(res.RewardPath)
	if err != nil || string(captured) != string(raw) {
		t.Errorf("captured reward.json != verifier output (err=%v)", err)
	}
	if res.VerifierLogsDir != gotLogs {
		t.Errorf("VerifierLogsDir = %q, want %q", res.VerifierLogsDir, gotLogs)
	}
	if _, err := os.Stat(filepath.Join(gotLogs, "reward.json")); err != nil {
		t.Errorf("raw verifier reward.json missing from logs dir: %v", err)
	}
}

// TestRunEvalRunVerifierFailureNeverScores asserts a failed verifier run is an
// error, never a grade — including when the failed run left a (possibly partial)
// reward.json behind, which stays unscored until handed in with --reward.
func TestRunEvalRunVerifierFailureNeverScores(t *testing.T) {
	t.Run("no_reward", func(t *testing.T) {
		res, err := RunEval(EvalConfig{
			Task:        &Task{Name: "git-to-zig"},
			OutDir:      t.TempDir(),
			RunVerifier: true,
			capability:  runnableCapability(),
			runner: func(context.Context, []string, string) error {
				return errors.New("boom")
			},
		})
		if err == nil || !strings.Contains(err.Error(), "produced no reward.json") {
			t.Fatalf("err = %v, want a verifier-run failure", err)
		}
		if res.Available || !floatsEqual(res.Score, 0) {
			t.Errorf("failed run graded: available=%t score=%v", res.Available, res.Score)
		}
	})
	t.Run("partial_reward", func(t *testing.T) {
		res, err := RunEval(EvalConfig{
			Task:        &Task{Name: "git-to-zig"},
			OutDir:      t.TempDir(),
			RunVerifier: true,
			capability:  runnableCapability(),
			runner: func(_ context.Context, _ []string, logsDir string) error {
				if werr := os.WriteFile(filepath.Join(logsDir, "reward.json"), []byte(`{"reward": 1.0}`), 0o644); werr != nil {
					t.Fatalf("stage partial reward: %v", werr)
				}
				return errors.New("boom")
			},
		})
		if err == nil || !strings.Contains(err.Error(), "unscored") {
			t.Fatalf("err = %v, want the left-unscored refusal", err)
		}
		if res.Available || !floatsEqual(res.Score, 0) {
			t.Errorf("failed run graded: available=%t score=%v", res.Available, res.Score)
		}
	})
}

// TestRunEvalRunVerifierEmptyHandedGates asserts a verifier run that completes
// cleanly but drops no reward.json is an honest empty-handed gate (the swebench
// "completed but produced no report" case), not an error and never a silent 0.
func TestRunEvalRunVerifierEmptyHandedGates(t *testing.T) {
	res, err := RunEval(EvalConfig{
		Task:        &Task{Name: "git-to-zig"},
		OutDir:      t.TempDir(),
		RunVerifier: true,
		capability:  runnableCapability(),
		runner: func(context.Context, []string, string) error {
			return nil // clean exit, nothing produced
		},
	})
	if err != nil {
		t.Fatalf("RunEval: %v", err)
	}
	if res.Available {
		t.Fatal("Available=true for an empty-handed verifier run")
	}
	if res.Source != "none" {
		t.Errorf("Source = %q, want none", res.Source)
	}
	if !strings.Contains(res.Reason, EvalGatedReason) || !strings.Contains(res.Reason, "produced no reward.json") {
		t.Errorf("Reason = %q, want %s + the empty-handed cause", res.Reason, EvalGatedReason)
	}
}
