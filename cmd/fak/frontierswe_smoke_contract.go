package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

func runFrontiersweSmokeContract(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe smoke-contract", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tasks := fs.String("tasks", frontiersweSampleTasks, "task tree containing <task>/task.toml")
	taskName := fs.String("task", "git-to-zig", "FrontierSWE task to fix in the raw/fak contract")
	model := fs.String("model", "frontierswe-model", "model id shared by raw and fak arms")
	agent := fs.String("agent", "claude-code", "FrontierSWE agent/harness shared by both arms")
	rawBaseURL := fs.String("raw-base-url", "$env:RAW_FRONTIERSWE_BASE_URL", "raw model OpenAI-compatible base URL or env reference")
	fakBaseURL := fs.String("fak-base-url", frontierswe.DefaultGatewayBaseURL, "fak gateway OpenAI-compatible base URL")
	rawCommand := fs.String("raw-command", "", "copy-pasteable raw-arm command (default: generated from task/model/agent)")
	fakCommand := fs.String("fak-command", "", "copy-pasteable fak-arm command (default: generated from task/model/agent)")
	rawOutput := fs.String("raw-output", "experiments/frontierswe/raw-smoke", "raw arm output directory")
	fakOutput := fs.String("fak-output", "experiments/frontierswe/fak-smoke", "fak arm output directory")
	trials := fs.Int("trials", 0, "trials per arm (default: task job.yaml n_attempts, else 1)")
	out := fs.String("out", "", "write the contract JSON here (default: stdout)")
	md := fs.String("md", "", "write the contract markdown here")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	task, err := frontierswe.LoadTask(filepath.Join(*tasks, *taskName))
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe smoke-contract: load task %q from %s: %v\n", *taskName, *tasks, err)
		return 1
	}
	contract := frontierswe.BuildRawFakContract(frontierswe.RawFakContractInput{
		GeneratedAt:    time.Now().UTC().Format(time.RFC3339),
		Task:           task,
		Source:         *tasks,
		Model:          *model,
		Agent:          *agent,
		RawBaseURL:     *rawBaseURL,
		FakBaseURL:     *fakBaseURL,
		RawCommand:     *rawCommand,
		FakCommand:     *fakCommand,
		RawOutputDir:   *rawOutput,
		FakOutputDir:   *fakOutput,
		Trials:         *trials,
		EvalCapability: frontierswe.DetectFrontierEvalCapability(),
	})
	if *out != "" {
		if err := os.WriteFile(*out, jsonIndent(contract), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak frontierswe smoke-contract: write %s: %v\n", *out, err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, string(jsonIndent(contract)))
	}
	if *md != "" {
		if err := os.WriteFile(*md, []byte(frontierswe.RenderRawFakContractMarkdown(contract)), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak frontierswe smoke-contract: write %s: %v\n", *md, err)
			return 1
		}
	}

	fmt.Fprintf(stderr, "\n== fak frontierswe smoke-contract ==\n")
	fmt.Fprintf(stderr, "status       : %s\n", contract.Status)
	fmt.Fprintf(stderr, "task         : %s\n", contract.TaskSelection.Task)
	fmt.Fprintf(stderr, "model        : %s\n", contract.Model.ModelID)
	fmt.Fprintf(stderr, "agent        : %s\n", contract.Model.Agent)
	fmt.Fprintf(stderr, "claim allowed: %t\n", contract.ResultClaimAllowed)
	fmt.Fprintf(stderr, "grader       : runnable=%t", contract.OfficialGrader.Runnable)
	if contract.OfficialGrader.Reason != "" {
		fmt.Fprintf(stderr, " (%s)", contract.OfficialGrader.Reason)
	}
	fmt.Fprintln(stderr)
	if *out != "" {
		fmt.Fprintf(stderr, "json         : %s\n", *out)
	}
	if *md != "" {
		fmt.Fprintf(stderr, "markdown     : %s\n", *md)
	}
	return 0
}
