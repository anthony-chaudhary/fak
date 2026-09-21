package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
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

const opsRunNativeRedactedReceiptSchema = "fak.ops-run.native-child-redacted.v1"

type opsRunNativeToolCapabilities struct {
	System bool `json:"system"`
	MCP    bool `json:"mcp"`
	Skills bool `json:"skills"`
	Memory bool `json:"memory"`
}

type opsRunNativeLaunchIdentityReceipt struct {
	opsRunLaunchIdentityReceipt
	Tools                opsRunNativeToolCapabilities `json:"tools"`
	ChildReceiptRef      string                       `json:"child_receipt_ref"`
	ChildReceiptArtifact string                       `json:"child_receipt_artifact"`
}

type opsRunNativeReceipt struct {
	opsRunReceipt
	LaunchIdentity *opsRunNativeLaunchIdentityReceipt `json:"launch_identity,omitempty"`
}

type opsRunNativeRedactedMetrics struct {
	Arm                   string `json:"arm"`
	Turns                 int    `json:"turns"`
	ToolCalls             int    `json:"tool_calls"`
	ToolErrors            int    `json:"tool_errors"`
	Denies                int    `json:"denies"`
	EngineCalls           int    `json:"engine_calls"`
	TaskCompleted         bool   `json:"task_completed"`
	HitTurnCap            bool   `json:"hit_turn_cap"`
	CircuitBreakerTripped bool   `json:"circuit_breaker_tripped"`
}

type opsRunNativeRedactedReceipt struct {
	Schema             string                      `json:"schema"`
	SourceSchema       string                      `json:"source_schema"`
	Status             string                      `json:"status"`
	TouchedPathDigests []string                    `json:"touched_path_digests,omitempty"`
	GitDiffHash        string                      `json:"git_diff_hash,omitempty"`
	Metrics            opsRunNativeRedactedMetrics `json:"metrics"`
}

func writeOpsRunNativeReceipt(path string, receipt opsRunNativeReceipt) error {
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".ops-run-*.json")
	if err != nil {
		return err
	}
	temp := f.Name()
	defer os.Remove(temp)
	if _, err = f.Write(append(data, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(temp, path)
}

func persistOpsRunNativeChildReceipt(receiptPath string, native nativeAgentReceipt) (string, string, error) {
	pathDigests := make([]string, 0, len(native.TouchedPaths))
	for _, path := range native.TouchedPaths {
		pathDigests = append(pathDigests, opsRunDigest(path))
	}
	redacted := opsRunNativeRedactedReceipt{
		Schema: opsRunNativeRedactedReceiptSchema, SourceSchema: native.Schema,
		Status: native.Status, TouchedPathDigests: pathDigests, GitDiffHash: native.GitDiffHash,
		Metrics: opsRunNativeRedactedMetrics{
			Arm: native.Metrics.Arm, Turns: native.Metrics.Turns, ToolCalls: native.Metrics.ToolCalls,
			ToolErrors: native.Metrics.ToolErrors, Denies: native.Metrics.Denies, EngineCalls: native.Metrics.EngineCalls,
			TaskCompleted: native.Metrics.TaskCompleted, HitTurnCap: native.Metrics.HitTurnCap,
			CircuitBreakerTripped: native.Metrics.CircuitBreakerTripped,
		},
	}
	data, err := json.MarshalIndent(redacted, "", "  ")
	if err != nil {
		return "", "", err
	}
	data = append(data, '\n')
	sum := sha256.Sum256(data)
	hexDigest := hex.EncodeToString(sum[:])
	ref := "sha256:" + hexDigest
	relative := filepath.ToSlash(filepath.Join(".fak-ops-native-receipts", hexDigest+".json"))
	dir := filepath.Join(filepath.Dir(receiptPath), ".fak-ops-native-receipts")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", "", err
	}
	target := filepath.Join(dir, hexDigest+".json")
	if existing, readErr := os.ReadFile(target); readErr == nil {
		if string(existing) != string(data) {
			return "", "", fmt.Errorf("native receipt content-address collision")
		}
		return ref, relative, nil
	}
	f, err := os.CreateTemp(dir, ".native-receipt-*.json")
	if err != nil {
		return "", "", err
	}
	temp := f.Name()
	defer os.Remove(temp)
	if err := f.Chmod(0o600); err != nil {
		_ = f.Close()
		return "", "", err
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return "", "", err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return "", "", err
	}
	if err := f.Close(); err != nil {
		return "", "", err
	}
	if err := os.Rename(temp, target); err != nil {
		return "", "", err
	}
	return ref, relative, nil
}

func failOpsRunNativeInferencePreflight(stderr io.Writer, receiptPath string, receipt opsRunNativeReceipt, preflight opsRunInferencePreflightReceipt) int {
	receipt.InferencePreflight = &preflight
	receipt.Status, receipt.ExitCode, receipt.Finished = "failed", 1, time.Now().UTC()
	if err := writeOpsRunNativeReceipt(receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run native: write receipt: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "ops run native: inference preflight: %s\n", preflight.Reason)
	return 1
}

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
	resolvedWorkspace := strings.TrimSpace(*workspace)
	if resolvedWorkspace == "" {
		cwd, cwdErr := os.Getwd()
		if cwdErr != nil {
			fmt.Fprintln(stderr, "ops run native: resolve workspace:", cwdErr)
			return 2
		}
		resolvedWorkspace = cwd
	}
	resolvedWorkspace, err := resolveOpsRunWorkspace(resolvedWorkspace)
	if err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
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
		_ = json.NewEncoder(stdout).Encode(map[string]any{"schema": "fak-ops-run-plan/1", "harness": "native", "provider": *provider, "workspace": resolvedWorkspace, "guarded": false, "prompt_delivery": "file", "timeout": timeout.String(), "max_turns": *maxTurns})
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
	for _, pair := range [][2]string{{"--policy", *policy}, {"--code-workspace", resolvedWorkspace}, {"--effort", *effort}} {
		if pair[1] != "" {
			argv = append(argv, pair[0], pair[1])
		}
	}
	if *codexAuth {
		argv = append(argv, "--codex-auth")
		selectedCodexHome := strings.TrimSpace(*codexHome)
		if selectedCodexHome == "" {
			selectedCodexHome, _ = resolveCodexHome("", true)
		}
		if selectedCodexHome != "" {
			argv = append(argv, "--codex-home", selectedCodexHome)
		}
	}
	env, cleanupEnv, err := opsRunChildEnvironment("{}", *keyEnv)
	if err != nil {
		fmt.Fprintln(stderr, "ops run native: create isolated child environment:", err)
		return 1
	}
	defer cleanupEnv()
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	var nonce [12]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		fmt.Fprintln(stderr, "ops run native: create run identity failed")
		return 1
	}
	effectiveConfig, _ := json.Marshal(map[string]any{
		"posture": "fail_closed", "sys_tools": false, "mcp_tools": false, "skills": false, "memory": false,
	})
	identity := newOpsRunLaunchIdentity("ops-"+hex.EncodeToString(nonce[:]), "native", resolvedWorkspace, *provider, *baseURL, *model, tuiExecutable(), string(effectiveConfig), *policy, false, true)
	if identity.PolicySource == "builtin" {
		identity.PolicyDigest = opsRunDigest("native-default-fail-closed")
	}
	identity.GuardEffective = "fail_closed"
	identity.GuardEvidenceRef = opsRunDigest(identity.PolicyDigest, string(effectiveConfig))
	identity.CapabilityEvidenceRef = opsRunDigest(string(effectiveConfig))
	nativeIdentity := opsRunNativeLaunchIdentityReceipt{
		opsRunLaunchIdentityReceipt: identity,
		Tools:                       opsRunNativeToolCapabilities{},
		ChildReceiptRef:             "unknown",
		ChildReceiptArtifact:        "unknown",
	}
	configPolicy := opsRunConfigPolicyReceipt{Source: identity.PolicySource, Digest: identity.PolicyDigest, Status: "qualified"}
	receipt := opsRunNativeReceipt{
		opsRunReceipt:  opsRunReceipt{Schema: "fak-ops-run/1", Harness: "native", Workspace: resolvedWorkspace, Status: "running", Started: time.Now().UTC(), ConfigPolicy: &configPolicy},
		LaunchIdentity: &nativeIdentity,
	}
	if err := writeOpsRunNativeReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	if *provider != "openai" {
		preflight := opsRunInferenceRefusal(*provider, *baseURL, *model, "unsupported_provider_protocol")
		return failOpsRunNativeInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	preflight, err := opsRunInferencePreflight(ctx, *baseURL, *model)
	receipt.InferencePreflight = &preflight
	if err != nil {
		return failOpsRunNativeInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	receipt.LaunchIdentity.InferenceProbeRef = preflight.ReceiptRef
	// Persist the qualifying reference before launch. A receipt write failure
	// cannot produce an unqualified native child process.
	if err := writeOpsRunNativeReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	cmd := exec.CommandContext(ctx, tuiExecutable(), argv...)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.Env = env
	cmd.WaitDelay = 5 * time.Second
	procguard.ConfigureProcessTreeCancel(cmd)
	windowgate.ConfigureBackgroundCommand(cmd)
	err = cmd.Run()
	code := childprocess.ExitCode(err, 1)
	if err == nil {
		code = 0
	}
	receipt.Status = "failed"
	data, readErr := os.ReadFile(nativePath)
	var native nativeAgentReceipt
	validNative := readErr == nil && json.Unmarshal(data, &native) == nil && native.Schema == nativeAgentReceiptSchema && native.Metrics.Arm == "fak"
	if validNative {
		ref, artifact, persistErr := persistOpsRunNativeChildReceipt(*receiptPath, native)
		if persistErr != nil {
			code = 1
			fmt.Fprintln(stderr, "ops run native: persist redacted child receipt:", persistErr)
		} else {
			receipt.LaunchIdentity.ChildReceiptRef = ref
			receipt.LaunchIdentity.ChildReceiptArtifact = artifact
		}
	}
	if ctx.Err() != nil {
		code, receipt.Status = 130, "cancelled"
		if ctx.Err() == context.DeadlineExceeded {
			code, receipt.Status = 124, "timed_out"
		}
	} else {
		if code == 0 && validNative && receipt.LaunchIdentity.ChildReceiptRef != "unknown" && native.Status == "completed" && !native.Metrics.HitTurnCap && !native.Metrics.CircuitBreakerTripped {
			receipt.Status = "succeeded"
		} else if code == 0 {
			code = 1
			fmt.Fprintln(stderr, "ops run native: child did not report a completed native turn")
		}
	}
	receipt.ExitCode, receipt.Finished = code, time.Now().UTC()
	if err := writeOpsRunNativeReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintln(stderr, "ops run native:", err)
		return 1
	}
	return code
}
