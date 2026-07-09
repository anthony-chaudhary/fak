package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// runFrontiersweEval is the C13 grade verb (#1706, #1719): the FrontierSWE analogue
// of `fak swebench eval`. It turns a trial's SUBMISSION into the leaderboard number
// via the C3 scorer. Given a verifier reward.json (from a prior run or handed in) it
// scores it offline — RUNNABLE NOW, no Docker — into the correctness, speedup, and
// gated leaderboard score, capturing the raw reward.json + a verifier logs dir for
// traceability. Absent a reward.json it stands the verifier up where this host is
// capable, and otherwise prints an honest GATED result with the exact remote command
// — never a fabricated score.
func runFrontiersweEval(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tasks := fs.String("tasks", frontiersweSampleTasks, "task tree containing <task>/task.toml (+ optional oracle.yaml) for the docker_image / verifier budget overlay")
	taskName := fs.String("task", "", "FrontierSWE task to grade (one of the 17 catalog tasks) — required")
	submission := fs.String("submission", "", "the submission tree to grade (the target the verifier reads)")
	reward := fs.String("reward", "", "explicit reward.json to score (default: <submission>/reward.json)")
	out := fs.String("out", "", "directory to capture the raw reward.json + verifier logs into (optional)")
	anticheat := fs.Bool("anti-cheat", false, "the trial was flagged in scoring/anticheat.json (the gated leaderboard score is forced to 0)")
	ssim := fs.Float64("ssim", 0, "revideo-perf-opt SSIM gate threshold (0 => the library default 0.99)")
	asJSON := fs.Bool("json", false, "emit only the eval JSON on stdout (no human summary on stderr)")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *taskName == "" {
		fmt.Fprintln(stderr, "fak frontierswe eval: --task is required (one of the 17 catalog tasks)")
		return 2
	}
	if _, ok := frontierswe.CategoryOf(*taskName); !ok {
		fmt.Fprintf(stderr, "fak frontierswe eval: unknown task %q (not one of the 17 FrontierSWE tasks)\n", *taskName)
		return 2
	}

	// Overlay the docker_image / [verifier] timeout_sec / oracle command from a
	// committed task.toml where one exists; a task with no committed fixture still
	// grades — the verifier command falls back to the canonical GHCR image + harbor
	// default, so the emitted remote command is always concrete.
	task, err := frontierswe.LoadTask(filepath.Join(*tasks, *taskName))
	if err != nil {
		task = &frontierswe.Task{Name: *taskName}
	}

	res, err := frontierswe.RunEval(frontierswe.EvalConfig{
		Task:             task,
		SubmissionDir:    *submission,
		RewardPath:       *reward,
		OutDir:           *out,
		AntiCheatFlagged: *anticheat,
		SSIMThreshold:    *ssim,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe eval: %v\n", err)
		return 1
	}

	jb, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe eval: marshal: %v\n", err)
		return 1
	}
	if *asJSON {
		fmt.Fprintln(stdout, string(jb))
	} else {
		fmt.Fprintln(stdout, string(jb))
		printFrontiersweEvalSummary(stderr, res)
	}
	if !res.Available {
		return 1
	}
	return 0
}

// printFrontiersweEvalSummary writes the human-readable grade on stderr (keeping
// stdout clean JSON): the C3 leaderboard number when graded, or the honest GATED
// result + exact remote command when the verifier can't be stood up here.
func printFrontiersweEvalSummary(w io.Writer, r frontierswe.EvalResult) {
	fmt.Fprintf(w, "\n== fak frontierswe eval (%s) ==\n", r.Schema)
	fmt.Fprintf(w, "task          : %s  (%s)\n", r.Task, r.GateClass)
	if r.Available {
		fmt.Fprintf(w, "source        : %s\n", r.Source)
		fmt.Fprintf(w, "correctness   : %.4f\n", r.Correctness)
		if r.Speedup != nil {
			fmt.Fprintf(w, "speedup       : %.4fx\n", *r.Speedup)
		} else {
			fmt.Fprintf(w, "speedup       : —\n")
		}
		if r.AntiCheatFlag {
			fmt.Fprintf(w, "anti-cheat    : FLAGGED (leaderboard score forced to 0)\n")
		}
		fmt.Fprintf(w, "LEADERBOARD   : %.4f  (the C3 gated score)\n", r.Score)
		if r.RewardPath != "" {
			fmt.Fprintf(w, "reward.json   : %s\n", r.RewardPath)
		}
		if r.VerifierLogsDir != "" {
			fmt.Fprintf(w, "verifier logs : %s\n", r.VerifierLogsDir)
		}
		return
	}
	fmt.Fprintf(w, "local gate    : docker=%t integrity_ok=%t\n", r.DockerPresent, r.IntegrityOK)
	fmt.Fprintf(w, "GATED         : %s\n", r.Reason)
	fmt.Fprintf(w, "no reward.json to score; produce one on a Docker/Modal box (%ds verifier budget) with:\n  %s\n",
		r.VerifierTimeoutSec, r.RemoteCommand)
	fmt.Fprintf(w, "then re-run:  fak frontierswe eval --task %s --reward <logs/verifier/reward.json>\n", r.Task)
}
