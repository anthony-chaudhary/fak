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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/codexresume"
)

type codexResumeThreads []string

func (t *codexResumeThreads) String() string { return strings.Join(*t, ",") }
func (t *codexResumeThreads) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return fmt.Errorf("thread id must not be empty")
	}
	*t = append(*t, value)
	return nil
}

type codexResumeBatchResult struct {
	Schema    string               `json:"schema"`
	Results   []codexresume.Result `json:"results"`
	Succeeded int                  `json:"succeeded"`
	Failed    int                  `json:"failed"`
	ExitCode  int                  `json:"exit_code"`
}

func runCodexResume(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "recover" {
		return runSessionRecover(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("codex-resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rollout := fs.String("rollout", "", "existing rollout JSONL for one resumed thread (otherwise discovered under CODEX_HOME/sessions)")
	codexHome := fs.String("codex-home", "", "Codex state home (default CODEX_HOME or ~/.codex)")
	codexExe := fs.String("codex", "codex", "Codex executable")
	cwd := fs.String("cwd", "", "child working directory (default current directory)")
	deadline := fs.Duration("deadline", 10*time.Minute, "maximum time before pre-terminal stall")
	drain := fs.Duration("drain", time.Second, "process drain time after rollout task_complete")
	promptFile := fs.String("prompt-file", "", "read the continuation prompt from a file (keeps it out of argv and ledgers)")
	resultFile := fs.String("result-file", "", "atomically persist the scrub-safe typed result")
	targetProvider := fs.String("target-provider", "", "target provider id (default persisted session provider)")
	targetWire := fs.String("target-wire", "responses", "target wire format")
	requiredCallPrefix := fs.String("required-call-prefix", "fc_", "required response_item.function_call.id prefix")
	checkOnly := fs.Bool("preflight-only", false, "report typed compatibility/admission without spawning")
	asJSON := fs.Bool("json", false, "emit the typed result as JSON")
	var threadFlags codexResumeThreads
	fs.Var(&threadFlags, "thread", "candidate thread id (repeat for bulk recovery; use --prompt-file when launching)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	threadIDs := append(codexResumeThreads(nil), threadFlags...)
	if len(threadIDs) == 0 && fs.NArg() >= 1 {
		threadIDs = append(threadIDs, fs.Arg(0))
	}
	if len(threadIDs) == 0 {
		fmt.Fprintln(stderr, "usage: fak codex-resume [--rollout FILE] [--prompt-file FILE] SESSION_ID [PROMPT]")
		fmt.Fprintln(stderr, "       fak codex-resume --thread ID [--thread ID...] --prompt-file FILE")
		return 2
	}
	if len(threadFlags) > 0 && fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak codex-resume: positional arguments cannot be combined with --thread")
		return 2
	}
	if *rollout != "" && len(threadIDs) != 1 {
		fmt.Fprintln(stderr, "fak codex-resume: --rollout is valid only for one candidate")
		return 2
	}
	if !*checkOnly && *promptFile == "" && (len(threadFlags) > 0 || fs.NArg() < 2) {
		fmt.Fprintln(stderr, "fak codex-resume: a prompt or --prompt-file is required for launch")
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
	absRollout := ""
	if *rollout != "" {
		var err error
		absRollout, err = filepath.Abs(*rollout)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	prompt := ""
	if *promptFile != "" {
		b, readErr := os.ReadFile(*promptFile)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak codex-resume: read prompt file: %v\n", readErr)
			return 1
		}
		prompt = string(b)
	} else if !*checkOnly {
		prompt = fs.Arg(1)
	}

	results := make([]codexresume.Result, 0, len(threadIDs))
	exitCode := 0
	for _, threadID := range threadIDs {
		checkConfig := codexresume.CheckConfig{
			ThreadID:                       threadID,
			RolloutPath:                    absRollout,
			CodexHome:                      *codexHome,
			TargetProvider:                 *targetProvider,
			TargetWire:                     *targetWire,
			RequiredFunctionCallItemPrefix: *requiredCallPrefix,
		}
		var result codexresume.Result
		if *checkOnly {
			preflight := codexresume.Preflight(checkConfig)
			result = codexresume.Result{ThreadID: threadID, Preflight: &preflight}
			if preflight.Verdict == codexresume.VerdictResumable {
				result.Outcome = codexresume.OutcomeCheckOnly
			} else {
				result.Outcome = codexresume.OutcomeRefused
			}
		} else {
			cmd := []string{*codexExe, "exec", "resume", "--json", "--skip-git-repo-check", threadID, prompt}
			var captured io.Writer = io.Discard
			if !*asJSON {
				captured = stderr
			}
			var err error
			result, err = codexresume.Recover(context.Background(), checkConfig, codexresume.Config{
				Command:      cmd,
				Dir:          dir,
				Env:          os.Environ(),
				Deadline:     *deadline,
				Drain:        *drain,
				Stdout:       captured,
				Stderr:       stderr,
				PollInterval: 100 * time.Millisecond,
			})
			if err != nil {
				fmt.Fprintf(stderr, "fak codex-resume %s: %v\n", threadID, err)
				result.ThreadID = threadID
				if result.Outcome == "" {
					result.Outcome = codexresume.OutcomeExited
				}
			}
		}
		results = append(results, result)
		if code := codexResumeResultExitCode(result); code != 0 {
			if code == 130 {
				exitCode = 130
			} else if exitCode == 0 {
				exitCode = code
			}
		}
	}

	var output any
	if len(results) == 1 {
		output = results[0]
	} else {
		output = summarizeCodexResumeBatch(results)
	}
	if *resultFile != "" {
		if err := writeCodexResumeJSON(*resultFile, output); err != nil {
			fmt.Fprintf(stderr, "fak codex-resume: write result: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(output)
	} else {
		for _, result := range results {
			preflightVerdict := codexresume.PreflightVerdict("")
			if result.Preflight != nil {
				preflightVerdict = result.Preflight.Verdict
			}
			fmt.Fprintf(stdout, "%s preflight=%s outcome=%s launch_pid=%d turn_status=%s useful_work=%t task_completed=%t process_exit=%t reclaimed=%t duration_ms=%d\n",
				result.ThreadID, preflightVerdict, result.Outcome, result.LaunchPID, result.TurnStatus, result.UsefulWork, result.TaskCompleted, result.ProcessExit, result.ForcedReclaim, result.DurationMS)
		}
	}
	return exitCode
}

func writeCodexResumeResult(path string, result codexresume.Result) error {
	return writeCodexResumeJSON(path, result)
}

func writeCodexResumeJSON(path string, value any) error {
	b, err := json.Marshal(value)
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

func summarizeCodexResumeBatch(results []codexresume.Result) codexResumeBatchResult {
	batch := codexResumeBatchResult{Schema: "fak-codex-resume-batch/1", Results: results}
	for _, result := range results {
		code := codexResumeResultExitCode(result)
		if code == 0 {
			batch.Succeeded++
		} else {
			batch.Failed++
		}
		if code == 130 {
			batch.ExitCode = 130
		} else if code != 0 && batch.ExitCode == 0 {
			batch.ExitCode = code
		}
	}
	return batch
}

func codexResumeResultExitCode(result codexresume.Result) int {
	switch result.Outcome {
	case codexresume.OutcomeCompleted, codexresume.OutcomeCompletedReclaimed, codexresume.OutcomeCheckOnly:
		return 0
	case codexresume.OutcomeExplicitCancelled:
		return 130
	default:
		return 1
	}
}

func cmdCodexResume(args []string) { os.Exit(runCodexResume(os.Stdout, os.Stderr, args)) }
