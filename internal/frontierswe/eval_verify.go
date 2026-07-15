package frontierswe

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// This file is the C13 verifier stand-up seam (#1719): the consented PRODUCE half
// of `fak frontierswe eval`. Given --run-verifier, eval stands the task's official
// verifier up in its [environment] docker_image, captures the reward.json +
// /logs/verifier artifacts it produces, and scores them with C3 — the "runs
// locally where it can" half of the swebench-eval gating shape. The gate stays
// honest in both directions:
//
//   - An incapable host (no Docker / integrity gate failed) REFUSES with the
//     capability reason + the exact remote command — never a fabricated score.
//   - A verifier run that fails, or completes empty-handed, is never scored; a
//     failure surfaces as an error and an empty-handed run as an honest gate.
//
// The invocation is built by the same verifierDockerArgs the gated RemoteCommand
// is rendered from, so the run path and the refusal path are one witness. The
// exec itself sits behind the verifierRunner seam so the available-path wiring is
// testable offline on a box with no Docker at all.

// defaultVerifierTimeoutSec is the canonical FrontierSWE [verifier] timeout_sec
// (24h) applied when a task carries no committed verifier budget, so a real run
// is always deadline-bounded — a wedged verifier container can never hang eval
// forever.
const defaultVerifierTimeoutSec = 86400

// verifierRunner executes one verifier invocation. args is the exact docker
// argument vector (verifierDockerArgs); logsDir is the host directory mounted at
// /logs/verifier, where the verifier drops reward.json. The real runner is
// execVerifier; tests inject a fake to witness the wiring without Docker.
type verifierRunner func(ctx context.Context, args []string, logsDir string) error

// runVerifierAndScore is the consented --run-verifier path: refuse honestly on an
// incapable host, otherwise stand the verifier up under the [verifier] timeout_sec
// budget and score the reward.json it produces into res. The error return is
// reserved for a failed run / malformed reward; an honest gate is a valid result.
func runVerifierAndScore(res *EvalResult, cfg EvalConfig, task *Task, capable EnvAdapterCapability, verifyCmd string) error {
	if !capable.Runnable {
		reason := capable.Reason
		if reason == "" {
			reason = "verifier environment not runnable on this host"
		}
		res.gate(reason)
		return nil
	}

	// The verifier drops its artifacts under the host dir mounted at /logs/verifier:
	// <out>/logs/verifier when --out is given, ./logs/verifier otherwise — the same
	// location the gated remote command mounts. Paths are made absolute for the
	// mount (docker -v rejects relative host paths).
	logsRoot := strings.TrimSpace(cfg.OutDir)
	if logsRoot == "" {
		logsRoot = "."
	}
	logsDir := filepath.Join(logsRoot, "logs", "verifier")
	if abs, err := filepath.Abs(logsDir); err == nil {
		logsDir = abs
	}
	if err := os.MkdirAll(logsDir, 0o755); err != nil {
		return fmt.Errorf("create verifier logs dir %s: %w", logsDir, err)
	}
	sub := strings.TrimSpace(cfg.SubmissionDir)
	if sub != "" {
		if abs, err := filepath.Abs(sub); err == nil {
			sub = abs
		}
	}
	args := verifierDockerArgs(task, sub, logsDir, verifyCmd)

	runner := cfg.runner
	if runner == nil {
		runner = execVerifier
	}
	budget := task.VerifierTimeoutSec()
	if budget <= 0 {
		budget = defaultVerifierTimeoutSec
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(budget*float64(time.Second)))
	defer cancel()
	runErr := runner(ctx, args, logsDir)

	rewardPath := filepath.Join(logsDir, "reward.json")
	raw, readErr := os.ReadFile(rewardPath)
	if runErr != nil {
		// A failed verifier run is never scored — even when it left a (possibly
		// partial) reward.json behind. The operator inspects that file and hands it
		// in explicitly with --reward if it is trusted.
		if readErr == nil {
			return fmt.Errorf("verifier run failed; reward.json at %s left unscored (inspect it and re-run with --reward if trusted): %w", rewardPath, runErr)
		}
		return fmt.Errorf("verifier run failed and produced no reward.json: %w", runErr)
	}
	if readErr != nil {
		// Ran cleanly but produced no reward.json — an honest empty-handed gate
		// (the swebench "completed but produced no report" case), never a silent 0.
		res.gate(fmt.Sprintf("verifier completed but produced no reward.json at %s", rewardPath))
		return nil
	}
	reward, perr := ParseReward(raw)
	if perr != nil {
		return fmt.Errorf("parse verifier reward.json %s: %w", rewardPath, perr)
	}
	res.Source = "verifier-run"
	res.RewardPath = rewardPath
	if err := res.scoreAndCapture(cfg, task, reward, raw); err != nil {
		return err
	}
	// With no --out, scoreAndCapture leaves the capture fields alone: the artifacts
	// live where the run dropped them, and the result says so.
	if res.VerifierLogsDir == "" {
		res.VerifierLogsDir = logsDir
	}
	return nil
}

// execVerifier is the real verifierRunner: it runs the docker invocation bounded
// by the caller's [verifier] timeout_sec context, streaming verifier output to
// stderr live (stdout stays clean eval JSON). The context cancel kills the whole
// process tree (mirrors internal/swebench's harness exec), so a deadline can't
// leave a stray container-runner behind.
func execVerifier(ctx context.Context, args []string, _ string) error {
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = 10 * time.Second
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// verifierDockerArgs builds the exact argument vector that stands the task's
// verifier up in its [environment] docker_image against the submission tree,
// honoring the task's resource envelope. The submission is mounted read-only at
// /submission and the verifier artifact dir (logsDir; "./logs/verifier" when
// empty) at /logs/verifier; both are conveyed by mount + env rather than invented
// CLI flags, so the verify command stays faithful to the task's oracle. It is the
// single source for both the gated RemoteCommand rendering and the --run-verifier
// exec — one witness.
func verifierDockerArgs(task *Task, submissionDir, logsDir, verifyCmd string) []string {
	name := "unknown"
	image := ""
	if task != nil {
		if task.Name != "" {
			name = task.Name
		}
		image = task.Environment.DockerImage
	}
	if image == "" {
		image = "ghcr.io/proximal-labs/frontier-swe/" + name + ":v6"
	}
	sub := strings.TrimSpace(submissionDir)
	if sub == "" {
		sub = "./submission"
	}
	if strings.TrimSpace(logsDir) == "" {
		logsDir = "./logs/verifier"
	}

	args := []string{"docker", "run", "--rm"}
	if task != nil {
		if task.Environment.CPUs > 0 {
			args = append(args, "--cpus", strconv.Itoa(task.Environment.CPUs))
		}
		if task.Environment.MemoryMB > 0 {
			args = append(args, "--memory", fmt.Sprintf("%dm", task.Environment.MemoryMB))
		}
		if task.Environment.GPUs > 0 {
			args = append(args, "--gpus", strconv.Itoa(task.Environment.GPUs))
		}
	}
	args = append(args,
		"-v", sub+":/submission:ro",
		"-v", logsDir+":/logs/verifier",
		"-e", "FRONTIERSWE_SUBMISSION=/submission",
		"-e", "FRONTIERSWE_VERIFIER_OUT=/logs/verifier",
		image, "/bin/sh", "-lc", verifyCmd,
	)
	return args
}
