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
	promptFile := fs.String("prompt-file", "", "read the continuation prompt from a file (keeps it out of argv and ledgers)")
	resultFile := fs.String("result-file", "", "atomically persist the scrub-safe typed result")
	asJSON := fs.Bool("json", false, "emit the typed result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *rollout == "" || fs.NArg() < 1 || (*promptFile == "" && fs.NArg() < 2) {
		fmt.Fprintln(stderr, "usage: fak codex-resume --rollout FILE [--prompt-file FILE] SESSION_ID [PROMPT]")
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
	prompt := ""
	if *promptFile != "" {
		b, readErr := os.ReadFile(*promptFile)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak codex-resume: read prompt file: %v\n", readErr)
			return 1
		}
		prompt = string(b)
	} else {
		prompt = fs.Arg(1)
	}
	cmd := []string{*codexExe, "exec", "resume", "--json", "--skip-git-repo-check", fs.Arg(0), prompt}
	var captured io.Writer = io.Discard
	if !*asJSON {
		captured = stderr
	}
	r, err := codexresume.Run(context.Background(), codexresume.Config{Command: cmd, Dir: dir, Env: os.Environ(), RolloutPath: absRollout, Deadline: *deadline, Drain: *drain, Stdout: captured, Stderr: stderr})
	if err != nil {
		fmt.Fprintf(stderr, "fak codex-resume: %v\n", err)
		return 1
	}
	if *resultFile != "" {
		if err := writeCodexResumeResult(*resultFile, r); err != nil {
			fmt.Fprintf(stderr, "fak codex-resume: write result: %v\n", err)
			return 1
		}
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
func writeCodexResumeResult(path string, result codexresume.Result) error {
	b, err := json.Marshal(result)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + fmt.Sprintf(".tmp-%d", os.Getpid())
	if err := os.WriteFile(tmp, append(b, '\n'), 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func cmdCodexResume(args []string) { os.Exit(runCodexResume(os.Stdout, os.Stderr, args)) }
