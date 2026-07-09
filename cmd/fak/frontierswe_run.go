package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// runFrontiersweRun is the C9 run verb (#1706, #1715): the FrontierSWE analogue of
// `fak swebench run`. It drives one task end-to-end through the fak-routed agent
// shape and writes the submission artifact, the fak.frontierswe.run.v1 meta, and
// the per-turn TTS trace (turn count, cumulative wall-clock, the C8 reuse series).
// The drive is a deterministic mock against a mocked environment (the acceptance
// permits it) whenever the real C7 environment cannot be stood up on this host;
// the exact remote command is always printed so a Docker/Modal box can run it live.
func runFrontiersweRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tasks := fs.String("tasks", frontiersweSampleTasks, "task tree containing <task>/task.toml (+ optional job.yaml)")
	taskName := fs.String("task", "git-to-zig", "FrontierSWE task to drive")
	agent := fs.String("agent", "claude-code", "harness short name routed through fak (claude-code, codex, gemini-cli, qwen-code)")
	gateway := fs.String("gateway", frontierswe.DefaultGatewayBaseURL, "OpenAI-compatible fak gateway base URL the C6 shim routes through")
	upstream := fs.String("upstream-base-url", frontierswe.DefaultUpstreamBase, "co-resident/pinned model upstream that fak serve fronts")
	model := fs.String("model", frontierswe.DefaultModelEnv, "model id forwarded unchanged to the upstream")
	output := fs.String("output", "", "directory to write the run outputs into (default: frontierswe-run/<task>)")
	trials := fs.Int("trials", 0, "n_concurrent_trials to record (default: the task's declared n_concurrent_trials)")
	turns := fs.Int("turns", 3, "turns the mock agent loop drives (capped at the budget-projected trajectory length)")
	predsOnly := fs.Bool("preds-only", false, "stop before grading (grading is C13); write the submission + meta + TTS trace only")
	asJSON := fs.Bool("json", false, "emit only the run JSON on stdout (no human summary on stderr)")
	if !parseFlags(fs, argv) {
		return 2
	}

	task, err := frontierswe.LoadTask(filepath.Join(*tasks, *taskName))
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe run: load task %q from %s: %v\n", *taskName, *tasks, err)
		return 1
	}

	result := frontierswe.BuildRun(frontierswe.RunConfig{
		Task: task, Agent: *agent, Model: *model,
		GatewayBaseURL: *gateway, UpstreamBase: *upstream,
		Trials: *trials, Turns: *turns, PredsOnly: *predsOnly,
	})

	// The no-internet boundary is a hard refusal, not a mock we paper over: if the
	// task forbids internet and the gateway/upstream are not loopback/pinned, the
	// run refuses rather than driving an integrity-violating environment.
	if !result.Gate.IntegrityOK {
		fmt.Fprintf(stderr, "fak frontierswe run: REFUSED: %s\n", result.Gate.Reason)
		return 1
	}

	outDir := *output
	if outDir == "" {
		outDir = filepath.Join("frontierswe-run", task.Name)
	}
	if err := writeFrontiersweRun(outDir, result); err != nil {
		fmt.Fprintf(stderr, "fak frontierswe run: write outputs to %s: %v\n", outDir, err)
		return 1
	}

	jb, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe run: marshal: %v\n", err)
		return 1
	}
	if *asJSON {
		fmt.Fprintln(stdout, string(jb))
	} else {
		printFrontiersweRunSummary(stderr, result, outDir)
	}
	return 0
}

// writeFrontiersweRun writes the run's three required emissions plus the routing
// witness under outDir: meta.json (the fak.frontierswe.run.v1 meta), tts-trace.json
// (the per-turn TTS trace), submission/ (the per-task submission artifact — one
// placeholder file per collected job.yaml artifact), job.yaml (the C6 shim block),
// and run.json (the full payload). It creates outDir if absent.
func writeFrontiersweRun(outDir string, r frontierswe.RunResult) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(outDir, "meta.json"), r.Meta); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(outDir, "tts-trace.json"), r.Trace); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(outDir, "run.json"), r); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "job.yaml"), []byte(r.JobYAML), 0o644); err != nil {
		return err
	}

	// The submission artifact: the per-task target tree. In the mocked drive it is
	// a directory carrying a SUBMISSION.md manifest (naming the real target the
	// verifier reads) plus one placeholder per collected job.yaml artifact, so the
	// collection shape a real run produces is exercised on disk.
	subDir := filepath.Join(outDir, "submission")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		return err
	}
	manifest := fmt.Sprintf("# FrontierSWE submission (%s)\n\ntarget: %s\nmocked: %t\n\nartifacts:\n",
		r.Meta.Task, r.Submission.Target, r.Submission.Mocked)
	for _, a := range r.Artifacts {
		manifest += fmt.Sprintf("- %s (collected=%t mocked=%t)\n", a.Name, a.Collected, a.Mocked)
		placeholder := fmt.Sprintf("mock artifact %q for FrontierSWE task %s (target %s)\n", a.Name, r.Meta.Task, r.Submission.Target)
		if err := os.WriteFile(filepath.Join(subDir, filepath.Base(a.Name)), []byte(placeholder), 0o644); err != nil {
			return err
		}
	}
	return os.WriteFile(filepath.Join(subDir, "SUBMISSION.md"), []byte(manifest), 0o644)
}

func printFrontiersweRunSummary(w io.Writer, r frontierswe.RunResult, outDir string) {
	fmt.Fprintf(w, "\n== fak frontierswe run (%s) ==\n", r.Meta.Schema)
	fmt.Fprintf(w, "task          : %s\n", r.Meta.Task)
	fmt.Fprintf(w, "agent         : %s  (wrapped %s)\n", r.Meta.Agent, r.WrappedAgent)
	fmt.Fprintf(w, "model         : %s\n", r.Meta.Model)
	fmt.Fprintf(w, "budget        : %.0fh (%d s [agent] timeout_sec — enforced)\n", r.Meta.BudgetHours, r.Meta.BudgetSec)
	fmt.Fprintf(w, "trials        : %d\n", r.Meta.Trials)
	fmt.Fprintf(w, "turns driven  : %d  (projected wall %.0fs of the %ds budget)\n", r.Meta.Turns, r.Trace.TotalWallSec, r.Trace.BudgetSec)
	fmt.Fprintf(w, "preds-only    : %t  (grading is C13)\n", r.Meta.PredsOnly)
	fmt.Fprintf(w, "mocked        : %t\n", r.Meta.Mocked)
	fmt.Fprintf(w, "reuse (C8)    : realized r=%.4f  cache_bit=%t\n", r.Trace.CacheSeries.RealizedReuseRate, r.Trace.CacheSeries.CacheBit)
	fmt.Fprintf(w, "submission    : %s\n", r.Submission.Target)

	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "ARTIFACT\tCOLLECTED\tMOCKED")
	for _, a := range r.Artifacts {
		fmt.Fprintf(tw, "%s\t%t\t%t\n", a.Name, a.Collected, a.Mocked)
	}
	_ = tw.Flush()

	fmt.Fprintf(w, "\nlocal gate    : docker=%t runnable=%t", r.Gate.DockerPresent, r.Gate.Runnable)
	if r.Gate.Reason != "" {
		fmt.Fprintf(w, "  (%s)", r.Gate.Reason)
	}
	fmt.Fprintln(w)
	if r.Gate.RemoteCommand != "" {
		if r.Gate.Runnable {
			// Docker is present, but this run is still a deterministic mock — the live
			// driver is the C7-gated path. Keep the live command visible so a capable
			// host is not left to infer from `mocked: true` alone that no win was measured.
			fmt.Fprintf(w, "\nNOTE: this run is a deterministic MOCK — no live win was measured. This host looks capable; run it live with:\n  %s\n", r.Gate.RemoteCommand)
		} else {
			fmt.Fprintf(w, "\nGATED: the real C7 environment can't be stood up here; run it live with:\n  %s\n", r.Gate.RemoteCommand)
		}
	}
	fmt.Fprintf(w, "\noutputs written: %s\n", outDir)
	fmt.Fprintf(w, "  meta.json       — the fak.frontierswe.run.v1 meta\n")
	fmt.Fprintf(w, "  tts-trace.json  — the per-turn TTS trace (turns, cumulative wall, the C8 reuse series)\n")
	fmt.Fprintf(w, "  submission/     — the per-task submission artifact (target %s)\n", r.Submission.Target)
	fmt.Fprintf(w, "  job.yaml        — the fak-routed C6 shim block\n")
}
