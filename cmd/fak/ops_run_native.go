package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func runOpsNative(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("ops run --harness native", flag.ContinueOnError)
	fs.SetOutput(stderr)
	harness := fs.String("harness", "native", "native FAK chat harness")
	provider := fs.String("provider", "openai", "native provider wire")
	model := fs.String("model", "", "explicit upstream model")
	baseURL := fs.String("base-url", "", "explicit upstream endpoint (required)")
	keyEnv := fs.String("api-key-env", "", "environment variable holding upstream key")
	codexAuth := fs.Bool("codex-auth", false, "explicitly reuse a Codex-managed ChatGPT login read-only; Codex owns renewal")
	codexHome := fs.String("codex-home", "", "Codex credential home for --codex-auth (default: existing discovery)")
	promptFile := fs.String("prompt-file", "", "UTF-8 task file")
	receiptPath := fs.String("receipt", "", "metadata-only execution receipt")
	policy := fs.String("policy", "", "native capability floor policy file")
	workspace := fs.String("workspace", "", "root for bounded code tools (default current directory)")
	fs.StringVar(workspace, "code-workspace", "", "alias for --workspace")
	maxTurns := fs.Int("max-turns", 10, "positive native model turn ceiling")
	effort := fs.String("effort", "", "native reasoning effort")
	timeout := fs.Duration("timeout", 5*time.Minute, "positive wall-clock deadline")
	dryRun := fs.Bool("dry-run", false, "validate without launching")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if *harness != "native" || fs.NArg() != 0 || strings.TrimSpace(*model) == "" || strings.TrimSpace(*baseURL) == "" || *promptFile == "" || *timeout <= 0 || *maxTurns <= 0 || (!*dryRun && *receiptPath == "") {
		fmt.Fprintln(stderr, "ops run native: require --model, --base-url, --prompt-file, --receipt, positive --timeout and --max-turns; no positional arguments")
		return 2
	}
	if *effort != "" && *effort != "none" && *effort != "low" && *effort != "medium" && *effort != "balanced" && *effort != "adaptive" && *effort != "high" {
		fmt.Fprintln(stderr, "ops run native: unsupported reasoning effort")
		return 2
	}
	switch *provider {
	case "openai", "openai-responses", "astra", "anthropic", "gemini", "xai":
	default:
		fmt.Fprintln(stderr, "ops run native: unsupported provider wire")
		return 2
	}
	if *codexHome != "" && !*codexAuth {
		fmt.Fprintln(stderr, "ops run native: --codex-home requires --codex-auth")
		return 2
	}
	if *codexAuth && (*provider != "openai-responses" || strings.TrimRight(*baseURL, "/") != guardCodexChatGPTBackendBaseURL || *keyEnv != "") {
		fmt.Fprintf(stderr, "ops run native: --codex-auth requires --provider openai-responses, --base-url %s and no API key environment\n", guardCodexChatGPTBackendBaseURL)
		return 2
	}
	prompt, err := os.ReadFile(*promptFile)
	if err != nil || len(strings.TrimSpace(string(prompt))) == 0 {
		fmt.Fprintln(stderr, "ops run native: prompt file must be readable and nonempty")
		return 2
	}
	if *policy != "" {
		if _, err := os.ReadFile(*policy); err != nil {
			fmt.Fprintln(stderr, "ops run native: policy must be readable")
			return 2
		}
	}
	for _, input := range []string{*promptFile, *policy} {
		if input != "" && *receiptPath != "" && opsRunSamePath(input, *receiptPath) {
			fmt.Fprintln(stderr, "ops run native: receipt must be distinct from prompt and policy")
			return 2
		}
	}
	if *dryRun {
		_ = json.NewEncoder(stdout).Encode(map[string]any{"schema": "fak-ops-run-plan/1", "harness": "native", "provider": *provider, "guarded": true, "prompt_delivery": "file", "timeout": timeout.String(), "max_turns": *maxTurns})
		return 0
	}
	// The native receipt contains task text. Keep it private and ephemeral; the
	// durable Ops receipt intentionally retains execution metadata only.
	nativeReceipt, err := os.CreateTemp("", "fak-ops-native-*.json")
	if err != nil {
		fmt.Fprintln(stderr, "ops run native: create private receipt:", err)
		return 1
	}
	nativePath := nativeReceipt.Name()
	_ = nativeReceipt.Close()
	defer os.Remove(nativePath)
	promptPath, err := filepath.Abs(*promptFile)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	argv := []string{"chat", "--task-file", promptPath, "--provider", *provider, "--model", *model, "--base-url", *baseURL, "--api-key-env", *keyEnv, "--receipt", nativePath, "--max-turns", fmt.Sprint(*maxTurns), "--posture", "fail_closed", "--sys-tools=false", "--mcp-tools=false", "--skills=false", "--memory=false"}
	for _, pair := range [][2]string{{"--policy", *policy}, {"--code-workspace", *workspace}, {"--effort", *effort}} {
		if pair[1] != "" {
			argv = append(argv, pair[0], pair[1])
		}
	}
	if *codexAuth {
		argv = append(argv, "--codex-auth")
		if *codexHome != "" {
			argv = append(argv, "--codex-home", *codexHome)
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	receipt := opsRunReceipt{Schema: "fak-ops-run/1", Harness: "native", Status: "running", Started: time.Now().UTC()}
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	cmd := exec.CommandContext(ctx, tuiExecutable(), argv...)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.WaitDelay = 5 * time.Second
	procguard.ConfigureProcessTreeCancel(cmd)
	windowgate.ConfigureBackgroundCommand(cmd)
	err = cmd.Run()
	code := childprocess.ExitCode(err, 1)
	if err == nil {
		code = 0
	}
	receipt.Status = "failed"
	if ctx.Err() != nil {
		code, receipt.Status = 130, "cancelled"
		if ctx.Err() == context.DeadlineExceeded {
			code, receipt.Status = 124, "timed_out"
		}
	} else if code == 0 {
		data, readErr := os.ReadFile(nativePath)
		var native nativeAgentReceipt
		if readErr == nil && json.Unmarshal(data, &native) == nil && native.Schema == nativeAgentReceiptSchema && native.Status == "completed" && native.Metrics.Arm == "fak" && !native.Metrics.HitTurnCap && !native.Metrics.CircuitBreakerTripped {
			receipt.Status = "succeeded"
		} else {
			code = 1
			fmt.Fprintln(stderr, "ops run native: child did not report a completed native turn")
		}
	}
	receipt.ExitCode, receipt.Finished = code, time.Now().UTC()
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	return code
}
