// `fak codex-resume` runs a fresh headless Codex continuation to the rollout's
// typed terminal, even when the upstream process fails to exit (#6414).
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/codexresume"
)

func runCodexResume(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codex-resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rollout := fs.String("rollout", "", "existing rollout JSONL for the resumed thread (required)")
	codexExe := fs.String("codex", "codex", "Codex executable")
	cwd := fs.String("cwd", "", "child working directory (default current directory)")
	deadline := fs.Duration("deadline", 10*time.Minute, "maximum time before pre-terminal stall")
	drain := fs.Duration("drain", time.Second, "process drain time after rollout task_complete")
	asJSON := fs.Bool("json", false, "emit the typed result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *rollout == "" || fs.NArg() < 2 {
		fmt.Fprintln(stderr, "usage: fak codex-resume --rollout FILE [--deadline 10m] [--drain 1s] SESSION_ID PROMPT")
		return 2
	}
	dir := *cwd
	if dir == "" {
		var err error
		dir, err = os.Getwd()
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	absRollout, err := filepath.Abs(*rollout)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	cmd := []string{*codexExe, "exec", "resume", "--json", "--skip-git-repo-check", fs.Arg(0), fs.Arg(1)}
	var captured io.Writer = io.Discard
	if !*asJSON {
		captured = stderr
	}
	r, err := codexresume.Run(context.Background(), codexresume.Config{Command: cmd, Dir: dir, Env: os.Environ(), RolloutPath: absRollout, Deadline: *deadline, Drain: *drain, Stdout: captured, Stderr: stderr})
	if err != nil {
		fmt.Fprintf(stderr, "fak codex-resume: %v\n", err)
		return 1
	}
	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(r)
	} else {
		fmt.Fprintf(stdout, "%s useful_work=%t task_completed=%t process_exit=%t reclaimed=%t duration_ms=%d\n", r.Outcome, r.UsefulWork, r.TaskCompleted, r.ProcessExit, r.ForcedReclaim, r.DurationMS)
	}
	switch r.Outcome {
	case codexresume.OutcomeCompleted, codexresume.OutcomeCompletedReclaimed:
		return 0
	case codexresume.OutcomeExplicitCancelled:
		return 130
	default:
		return 1
	}
}
func cmdCodexResume(args []string) { os.Exit(runCodexResume(os.Stdout, os.Stderr, args)) }
