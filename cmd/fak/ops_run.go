package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Receipts deliberately omit prompts, provider responses and credentials.
type opsRunReceipt struct {
	Schema             string                           `json:"schema"`
	Harness            string                           `json:"harness"`
	Workspace          string                           `json:"workspace"`
	Status             string                           `json:"status"`
	ExitCode           int                              `json:"exit_code"`
	Started            time.Time                        `json:"started_at"`
	Finished           time.Time                        `json:"finished_at"`
	Lifecycle          []opsRunLifecycleRecord          `json:"lifecycle,omitempty"`
	InferencePreflight *opsRunInferencePreflightReceipt `json:"inference_preflight,omitempty"`
}

const (
	opsRunInferencePreflightSchema = "fak.ops-run.inference-preflight.v1"
	opsRunInferenceProbeTool       = "fak_inference_preflight"
	opsRunInferenceProbeTTL        = time.Minute
	opsRunInferenceProbeTimeout    = 5 * time.Second
	opsRunInferenceProbeMaxBytes   = 256 * 1024
)

type opsRunInferencePreflightReceipt struct {
	Schema     string `json:"schema"`
	ReceiptRef string `json:"receipt_ref"`
	Status     string `json:"status"`
	Reason     string `json:"reason,omitempty"`
}

type opsRunInferenceProbeCacheEntry struct {
	RecordedAt time.Time
	Receipt    opsRunInferencePreflightReceipt
	Failure    string
}

var opsRunInferenceProbeCache = struct {
	sync.Mutex
	entries map[string]opsRunInferenceProbeCacheEntry
}{entries: make(map[string]opsRunInferenceProbeCacheEntry)}

func opsRunInferencePreflight(ctx context.Context, baseURL, model string) (opsRunInferencePreflightReceipt, error) {
	configSum := sha256.Sum256([]byte(strings.TrimSpace(baseURL) + "\x00" + strings.TrimSpace(model)))
	cacheKey := hex.EncodeToString(configSum[:])
	now := time.Now().UTC()

	// Holding the lock across the one bounded request is deliberate: concurrent
	// launches with identical configuration share one witnessed probe per TTL.
	opsRunInferenceProbeCache.Lock()
	defer opsRunInferenceProbeCache.Unlock()
	if cached, ok := opsRunInferenceProbeCache.entries[cacheKey]; ok && now.Sub(cached.RecordedAt) < opsRunInferenceProbeTTL {
		if cached.Failure != "" {
			return cached.Receipt, errors.New(cached.Failure)
		}
		return cached.Receipt, nil
	}

	reason := opsRunProbeInferenceRoute(ctx, baseURL, model)
	status := "qualified"
	if reason != "" {
		status = "failed"
	}
	refSum := sha256.Sum256([]byte(opsRunInferencePreflightSchema + "\x00" + cacheKey + "\x00" + status + "\x00" + reason))
	receipt := opsRunInferencePreflightReceipt{
		Schema:     opsRunInferencePreflightSchema,
		ReceiptRef: "sha256:" + hex.EncodeToString(refSum[:]),
		Status:     status,
		Reason:     reason,
	}
	opsRunInferenceProbeCache.entries[cacheKey] = opsRunInferenceProbeCacheEntry{RecordedAt: now, Receipt: receipt, Failure: reason}
	if reason != "" {
		return receipt, errors.New(reason)
	}
	return receipt, nil
}

func opsRunInferenceRefusal(provider, baseURL, model, reason string) opsRunInferencePreflightReceipt {
	configSum := sha256.Sum256([]byte(strings.TrimSpace(provider) + "\x00" + strings.TrimSpace(baseURL) + "\x00" + strings.TrimSpace(model)))
	refSum := sha256.Sum256([]byte(opsRunInferencePreflightSchema + "\x00" + hex.EncodeToString(configSum[:]) + "\x00failed\x00" + reason))
	return opsRunInferencePreflightReceipt{
		Schema:     opsRunInferencePreflightSchema,
		ReceiptRef: "sha256:" + hex.EncodeToString(refSum[:]),
		Status:     "failed",
		Reason:     reason,
	}
}

func failOpsRunInferencePreflight(stderr io.Writer, receiptPath string, receipt opsRunReceipt, preflight opsRunInferencePreflightReceipt) int {
	receipt.InferencePreflight = &preflight
	receipt.Status = "failed"
	receipt.ExitCode = 1
	receipt.Finished = time.Now().UTC()
	if err := writeOpsRunReceipt(receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run: write receipt: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "ops run: inference preflight: %s\n", preflight.Reason)
	return 1
}

func opsRunProbeInferenceRoute(ctx context.Context, baseURL, model string) string {
	endpoint, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || endpoint.Scheme == "" || endpoint.Host == "" {
		return "invalid_endpoint"
	}
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + "/chat/completions"
	endpoint.RawQuery = ""
	endpoint.Fragment = ""
	payload := map[string]any{
		"model":  model,
		"stream": true,
		"messages": []map[string]string{{
			"role":    "user",
			"content": "Return exactly one fak_inference_preflight tool call with {\"ok\":true}; do not execute any tool.",
		}},
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        opsRunInferenceProbeTool,
				"description": "Non-mutating inference route readiness witness.",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"ok": map[string]string{"type": "boolean"}},
					"required":   []string{"ok"},
				},
			},
		}},
		"tool_choice": map[string]any{"type": "function", "function": map[string]string{"name": opsRunInferenceProbeTool}},
		"max_tokens":  32,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "request_encoding"
	}
	requestCtx, cancel := context.WithTimeout(ctx, opsRunInferenceProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return "request_build"
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if requestCtx.Err() != nil {
			return "timeout"
		}
		return "transport"
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Sprintf("http_status_%d", resp.StatusCode)
	}
	if !strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		return "malformed_sse"
	}

	type streamedToolCall struct {
		ID        string
		Type      string
		Name      string
		Arguments strings.Builder
	}
	toolCalls := make(map[int]*streamedToolCall)
	seenModel, finishReason, done := false, "", false
	limited := io.LimitReader(resp.Body, opsRunInferenceProbeMaxBytes+1)
	scanner := bufio.NewScanner(limited)
	scanner.Buffer(make([]byte, 4096), opsRunInferenceProbeMaxBytes+1)
	readBytes := 0
	for scanner.Scan() {
		line := scanner.Text()
		readBytes += len(line) + 1
		if readBytes > opsRunInferenceProbeMaxBytes {
			return "response_too_large"
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			done = true
			continue
		}
		var chunk struct {
			Model   string `json:"model"`
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}
		if data == "" || json.Unmarshal([]byte(data), &chunk) != nil || len(chunk.Choices) == 0 {
			return "malformed_sse"
		}
		if chunk.Model != "" {
			if chunk.Model != model {
				return "model_mismatch"
			}
			seenModel = true
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
			for _, delta := range choice.Delta.ToolCalls {
				call := toolCalls[delta.Index]
				if call == nil {
					call = &streamedToolCall{}
					toolCalls[delta.Index] = call
				}
				if delta.ID != "" {
					call.ID = delta.ID
				}
				if delta.Type != "" {
					call.Type = delta.Type
				}
				if delta.Function.Name != "" {
					call.Name = delta.Function.Name
				}
				call.Arguments.WriteString(delta.Function.Arguments)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		if requestCtx.Err() != nil {
			return "timeout"
		}
		return "malformed_sse"
	}
	if !done || !seenModel || finishReason != "tool_calls" || len(toolCalls) != 1 {
		return "incomplete_sse"
	}
	call := toolCalls[0]
	if call == nil || call.ID == "" || call.Type != "function" || call.Name != opsRunInferenceProbeTool {
		return "malformed_tool_call"
	}
	var arguments struct {
		OK bool `json:"ok"`
	}
	if json.Unmarshal([]byte(call.Arguments.String()), &arguments) != nil || !arguments.OK {
		return "malformed_tool_call"
	}
	return ""
}

const maxOpsRunLifecycleRecords = 16

type opsRunLifecycleRecord struct {
	Reason            string `json:"reason"`
	TerminationReason string `json:"termination_reason,omitempty"`
	ChildState        string `json:"child_state"`
	State             string `json:"state,omitempty"`
	Error             string `json:"error,omitempty"`
	OSError           string `json:"os_error,omitempty"`
	Signal            string `json:"signal,omitempty"`
	ExitCode          *int   `json:"exit_code,omitempty"`
	ElapsedMS         int64  `json:"elapsed_ms"`
	DurationMS        int64  `json:"duration_ms,omitempty"`
	Duration          string `json:"duration,omitempty"`
}

type opsRunReasonKey struct{}
type opsRunSignalCheckerKey struct{}

func withOpsRunTerminationReason(ctx context.Context, reason string) context.Context {
	return context.WithValue(ctx, opsRunReasonKey{}, reason)
}

func withSignalChecker(ctx context.Context, isSignal func() bool) context.Context {
	return context.WithValue(ctx, opsRunSignalCheckerKey{}, isSignal)
}

func opsRunDetermineTerminationReason(ctx context.Context) string {
	if checker, ok := ctx.Value(opsRunSignalCheckerKey{}).(func() bool); ok && checker() {
		return "shutdown"
	}
	if custom, ok := ctx.Value(opsRunReasonKey{}).(string); ok && custom != "" {
		return custom
	}
	if ctx.Err() == context.DeadlineExceeded {
		return "timeout"
	}
	if ctx.Err() == context.Canceled {
		return "cancelled"
	}
	return "cancelled"
}

func probeChildProcessState(cmd *exec.Cmd) string {
	if cmd == nil || cmd.Process == nil {
		return "unknown"
	}
	if cmd.ProcessState != nil {
		return "exited"
	}
	pid := cmd.Process.Pid
	if pid <= 0 {
		return "unknown"
	}
	if processalive.Check(pid) {
		return "alive"
	}
	return "exited"
}

var (
	opsRunProbeChildState = probeChildProcessState
	opsRunCancelProcess   = func(cmd *exec.Cmd, origCancel func() error) error {
		if origCancel != nil {
			return origCancel()
		}
		if cmd == nil || cmd.Process == nil {
			return nil
		}
		if ok, detail := procguard.KillPID(cmd.Process.Pid); !ok {
			if err := cmd.Process.Kill(); err != nil {
				return err
			}
			if detail != "" {
				return errors.New(detail)
			}
		}
		return nil
	}
)

var opsRunExecute = executeOpsRun

type opsRunWorkspaceKey struct{}

func resolveOpsRunWorkspace(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", errors.New("workspace is required")
	}
	abs, err := filepath.Abs(raw)
	if err != nil {
		return "", fmt.Errorf("resolve workspace: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", fmt.Errorf("resolve workspace: %w", err)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("stat workspace: %w", err)
	}
	if !info.IsDir() {
		return "", errors.New("workspace must be a directory")
	}
	return filepath.Clean(resolved), nil
}

func withOpsRunWorkspace(ctx context.Context, workspace string) context.Context {
	return context.WithValue(ctx, opsRunWorkspaceKey{}, workspace)
}

func opsRunWorkspace(ctx context.Context) string {
	workspace, _ := ctx.Value(opsRunWorkspaceKey{}).(string)
	return workspace
}

func runOpsRun(stdout, stderr io.Writer, args []string) int {
	for i, arg := range args {
		if arg == "--harness=native" || arg == "-harness=native" || ((arg == "--harness" || arg == "-harness") && i+1 < len(args) && args[i+1] == "native") {
			return runOpsNative(stdout, stderr, args)
		}
	}
	fs := flag.NewFlagSet("ops run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	harness := fs.String("harness", "opencode", "headless harness: opencode or native (use --harness native --help for native options)")
	provider := fs.String("provider", "openai", "upstream wire: openai or gemini (native Google adapter)")
	promptFile := fs.String("prompt-file", "", "UTF-8 prompt file, delivered over stdin")
	workspace := fs.String("workspace", "", "existing workspace directory for the child process (required except dry-run)")
	timeout := fs.Duration("timeout", 5*time.Minute, "positive wall-clock deadline")
	receiptPath := fs.String("receipt", "", "execution metadata JSON file (required except dry-run)")
	model := fs.String("model", "", "upstream model identifier without an OpenCode provider prefix (required)")
	baseURL := fs.String("base-url", "", "upstream endpoint for guard (Gemini native uses /v1beta)")
	apiKeyEnv := fs.String("api-key-env", "", "environment variable holding the upstream key")
	policy := fs.String("policy", "", "guard capability-floor policy file")
	audit := fs.String("audit", "", "guard audit journal file")
	opencodeBin := fs.String("opencode-bin", "opencode", "OpenCode executable; use the native .exe on Windows")
	auto := fs.Bool("auto", false, "ask OpenCode to approve permissions not explicitly denied")
	pure := fs.Bool("pure", false, "disable OpenCode external plugins")
	dryRun := fs.Bool("dry-run", false, "validate and print metadata without launching")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || *harness != "opencode" || (*provider != "openai" && *provider != "gemini") || *timeout <= 0 || strings.TrimSpace(*model) == "" || *promptFile == "" || (!*dryRun && (*receiptPath == "" || strings.TrimSpace(*workspace) == "")) {
		fmt.Fprintln(stderr, "ops run: require --harness opencode, --provider openai|gemini, --model, --prompt-file, --workspace, --receipt and a positive --timeout; no positional arguments (--workspace and --receipt may be omitted only for --dry-run)")
		return 2
	}
	resolvedWorkspace := ""
	if strings.TrimSpace(*workspace) != "" {
		var err error
		resolvedWorkspace, err = resolveOpsRunWorkspace(*workspace)
		if err != nil {
			fmt.Fprintf(stderr, "ops run: invalid workspace: %v\n", err)
			return 2
		}
	}
	if *provider == "gemini" && (*apiKeyEnv == "" || strings.TrimSpace(os.Getenv(*apiKeyEnv)) == "") {
		fmt.Fprintln(stderr, "ops run: --provider gemini requires --api-key-env naming a nonempty upstream key")
		return 2
	}
	prompt, err := os.ReadFile(*promptFile)
	if err != nil || len(bytes.TrimSpace(prompt)) == 0 {
		fmt.Fprintln(stderr, "ops run: prompt file must be readable and nonempty")
		return 2
	}
	if !*dryRun {
		for _, input := range []string{*promptFile, *policy, *audit} {
			if input != "" && opsRunSamePath(*receiptPath, input) {
				fmt.Fprintln(stderr, "ops run: receipt must be distinct from prompt, policy and audit files")
				return 2
			}
		}
	}
	if runtime.GOOS == "windows" && *opencodeBin == "opencode" {
		// npm exposes a shell shim on PATH; choose its installed native binary.
		if native := filepath.Join(os.Getenv("APPDATA"), "npm", "node_modules", "opencode-ai", "bin", "opencode.exe"); os.Getenv("APPDATA") != "" {
			if info, err := os.Stat(native); err == nil && !info.IsDir() {
				*opencodeBin = native
			}
		}
	}
	if runtime.GOOS != "windows" && *opencodeBin == "opencode" {
		// Cron/launchd children inherit a minimal PATH that misses user-local
		// install dirs. Resolve the well-known binary locations when the bare
		// name is not on PATH so the guarded launch does not fail spuriously;
		// guard leaves explicit paths (containing a separator) to exec.
		if _, lookErr := exec.LookPath(*opencodeBin); lookErr != nil {
			if native := resolvePOSIXOpenCodeBinary(""); native != "" {
				*opencodeBin = native
			}
		}
	}
	argv := []string{tuiExecutable(), "guard", "--provider", *provider, "--split", "off", "--model", *model}
	for _, pair := range [][2]string{{"--base-url", *baseURL}, {"--api-key-env", *apiKeyEnv}, {"--policy", *policy}, {"--audit", *audit}} {
		if pair[1] != "" {
			argv = append(argv, pair[0], pair[1])
		}
	}
	// A fresh provider name prevents deep-merging model/provider overrides from
	// global or project config into the route owned by this invocation.
	var nonce [12]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		fmt.Fprintln(stderr, "ops run: create provider identity failed")
		return 1
	}
	providerID := "fak_ops_" + hex.EncodeToString(nonce[:])
	modelID := providerID + "/" + *model
	argv = append(argv, "--", *opencodeBin, "run", "--format", "json", "--model", modelID)
	if *auto {
		argv = append(argv, "--auto")
	}
	if *pure {
		argv = append(argv, "--pure")
	}
	// Interpolation happens in the child after guard injects its local endpoint.
	config := map[string]any{}
	if existing := os.Getenv("OPENCODE_CONFIG_CONTENT"); strings.TrimSpace(existing) != "" {
		if json.Unmarshal([]byte(existing), &config) != nil || config == nil {
			fmt.Fprintln(stderr, "ops run: OPENCODE_CONFIG_CONTENT must be a JSON object")
			return 2
		}
	}
	providers, _ := config["provider"].(map[string]any)
	if providers == nil {
		providers = map[string]any{}
	}
	npm, childBase, childKey := "@ai-sdk/openai-compatible", "{env:OPENAI_BASE_URL}", "{env:OPENAI_API_KEY}"
	if *provider == "gemini" {
		// Native Google carries thoughtSignature across tool turns. The compatible
		// adapter used above lost that metadata in the witnessed Gemini tool loop.
		npm, childBase, childKey = "@ai-sdk/google", "{env:GOOGLE_GEMINI_BASE_URL}/v1beta", "fak-ops-guard"
	}
	providers[providerID] = map[string]any{"npm": npm, "options": map[string]string{"baseURL": childBase, "apiKey": childKey}, "models": map[string]any{*model: map[string]any{}}}
	config["provider"], config["model"] = providers, modelID
	config["small_model"] = modelID
	config["enabled_providers"] = []string{providerID}
	encoded, _ := json.Marshal(config)
	env := replaceOpsRunEnv(os.Environ(), "OPENCODE_CONFIG_CONTENT", string(encoded))
	if *dryRun {
		_ = json.NewEncoder(stdout).Encode(map[string]any{"schema": "fak-ops-run-plan/1", "harness": *harness, "provider": *provider, "workspace": resolvedWorkspace, "guarded": true, "prompt_delivery": "stdin", "timeout": timeout.String(), "auto": *auto, "pure": *pure})
		return 0
	}
	sigCtx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	ctx, cancel := context.WithTimeout(sigCtx, *timeout)
	defer cancel()
	ctx = withSignalChecker(ctx, func() bool {
		return sigCtx.Err() != nil
	})
	ctx = withOpsRunWorkspace(ctx, resolvedWorkspace)
	receipt := opsRunReceipt{Schema: "fak-ops-run/1", Harness: *harness, Workspace: resolvedWorkspace, Status: "running", Started: time.Now().UTC()}
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run: write receipt: %v\n", err)
		return 1
	}
	if *provider != "openai" {
		preflight := opsRunInferenceRefusal(*provider, *baseURL, *model, "unsupported_provider_protocol")
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	if strings.TrimSpace(*baseURL) == "" {
		preflight := opsRunInferenceRefusal(*provider, *baseURL, *model, "missing_explicit_base_url")
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	preflight, err := opsRunInferencePreflight(ctx, *baseURL, *model)
	receipt.InferencePreflight = &preflight
	if err != nil {
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	// Persist the qualifying reference before launch. A receipt write failure
	// cannot produce an unqualified child process.
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run: write receipt: %v\n", err)
		return 1
	}
	code, complete, eventError, lifecycle := opsRunExecute(ctx, stdout, stderr, argv, env, prompt)
	if len(lifecycle) > 0 {
		receipt.Lifecycle = append(receipt.Lifecycle, lifecycle...)
	}
	receipt.Status = "failed"
	if ctx.Err() != nil {
		if ctx.Err() == context.DeadlineExceeded {
			code = 124
			receipt.Status = "timed_out"
		} else {
			code = 130
			receipt.Status = "cancelled"
		}
		if len(receipt.Lifecycle) == 0 {
			reason := opsRunDetermineTerminationReason(ctx)
			receipt.Lifecycle = append(receipt.Lifecycle, opsRunLifecycleRecord{
				Reason:            reason,
				TerminationReason: reason,
				ChildState:        "unknown",
				State:             "unknown",
				Signal:            "SIGKILL",
				ElapsedMS:         0,
				DurationMS:        0,
				Duration:          "0s",
			})
		}
	} else if code == 0 {
		if eventError || !complete {
			code = 1
			fmt.Fprintln(stderr, "ops run: OpenCode did not report a successful completed turn")
		} else {
			receipt.Status = "succeeded"
		}
	}
	receipt.ExitCode, receipt.Finished = code, time.Now().UTC()
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run: write receipt: %v\n", err)
		return 1
	}
	return code
}

func replaceOpsRunEnv(env []string, key, value string) []string {
	out := make([]string, 0, len(env)+1)
	for _, item := range env {
		name, _, _ := strings.Cut(item, "=")
		if !strings.EqualFold(name, key) {
			out = append(out, item)
		}
	}
	return append(out, key+"="+value)
}

func opsRunSamePath(a, b string) bool {
	left, le := filepath.Abs(a)
	right, re := filepath.Abs(b)
	if le == nil && re == nil && (left == right || (runtime.GOOS == "windows" && strings.EqualFold(left, right))) {
		return true
	}
	li, le := os.Stat(a)
	ri, re := os.Stat(b)
	return le == nil && re == nil && os.SameFile(li, ri)
}

// Forward output while retaining at most one bounded JSON event.
type opsRunEvents struct {
	output   io.Writer
	pending  []byte
	overflow bool
	complete bool
	failed   bool
}

func (w *opsRunEvents) Write(p []byte) (int, error) {
	n, err := w.output.Write(p)
	for _, b := range p[:n] {
		if b == '\n' {
			w.finishLine()
			continue
		}
		if len(w.pending) < 4*1024*1024 && !w.overflow {
			w.pending = append(w.pending, b)
		} else {
			w.overflow = true
			w.pending = nil
		}
	}
	return n, err
}
func (w *opsRunEvents) finishLine() {
	if w.overflow {
		w.failed = true
	}
	if !w.overflow {
		var ev struct {
			Type string `json:"type"`
			Part struct {
				Reason string `json:"reason"`
				State  struct {
					Status string `json:"status"`
				} `json:"state"`
			} `json:"part"`
		}
		if json.Unmarshal(w.pending, &ev) == nil {
			if ev.Type == "error" || (ev.Type == "tool_use" && ev.Part.State.Status == "error") {
				w.failed = true
			}
			if ev.Type == "step_start" {
				w.complete = false
			}
			if ev.Type == "step_finish" {
				w.complete = ev.Part.Reason == "stop"
			}
		} else if bytes.HasPrefix(bytes.TrimSpace(w.pending), []byte("{")) {
			w.failed = true
		}
	}
	w.pending = nil
	w.overflow = false
}

func executeOpsRun(ctx context.Context, stdout, stderr io.Writer, argv, env []string, prompt []byte) (int, bool, bool, []opsRunLifecycleRecord) {
	events := &opsRunEvents{output: stdout}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = opsRunWorkspace(ctx)
	cmd.Env = env
	cmd.Stdin = bytes.NewReader(prompt)
	cmd.Stdout, cmd.Stderr = events, stderr
	cmd.WaitDelay = 5 * time.Second
	procguard.ConfigureProcessTreeCancel(cmd)
	windowgate.ConfigureBackgroundCommand(cmd)

	var (
		mu         sync.Mutex
		lifecycle  []opsRunLifecycleRecord
		termStart  time.Time
		origCancel = cmd.Cancel
	)

	cmd.Cancel = func() error {
		now := time.Now()
		mu.Lock()
		termStart = now
		mu.Unlock()

		reason := opsRunDetermineTerminationReason(ctx)
		childState := opsRunProbeChildState(cmd)

		cancelErr := opsRunCancelProcess(cmd, origCancel)

		elapsed := time.Since(now)
		rec := opsRunLifecycleRecord{
			Reason:            reason,
			TerminationReason: reason,
			ChildState:        childState,
			State:             childState,
			Signal:            "SIGKILL",
			ElapsedMS:         elapsed.Milliseconds(),
			DurationMS:        elapsed.Milliseconds(),
			Duration:          elapsed.String(),
		}
		if cancelErr != nil {
			rec.Error = cancelErr.Error()
			rec.OSError = cancelErr.Error()
		}
		if cmd.ProcessState != nil {
			exitCode := cmd.ProcessState.ExitCode()
			rec.ExitCode = &exitCode
		}

		mu.Lock()
		if len(lifecycle) < maxOpsRunLifecycleRecords {
			lifecycle = append(lifecycle, rec)
		}
		mu.Unlock()

		return cancelErr
	}

	err := cmd.Run()
	events.finishLine()

	mu.Lock()
	if len(lifecycle) > 0 && !termStart.IsZero() {
		totalElapsed := time.Since(termStart)
		last := len(lifecycle) - 1
		lifecycle[last].ElapsedMS = totalElapsed.Milliseconds()
		lifecycle[last].DurationMS = totalElapsed.Milliseconds()
		lifecycle[last].Duration = totalElapsed.String()
		if cmd.ProcessState != nil && lifecycle[last].ExitCode == nil {
			exitCode := cmd.ProcessState.ExitCode()
			lifecycle[last].ExitCode = &exitCode
		}
	}
	resLifecycle := make([]opsRunLifecycleRecord, len(lifecycle))
	copy(resLifecycle, lifecycle)
	mu.Unlock()

	if err != nil {
		fmt.Fprintf(stderr, "ops run: child: %v\n", err)
		return childprocess.ExitCode(err, 1), events.complete, events.failed, resLifecycle
	}
	return 0, events.complete, events.failed, resLifecycle
}

func writeOpsRunReceipt(path string, receipt opsRunReceipt) error {
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

// resolvePOSIXOpenCodeBinary returns the first existing non-directory opencode
// binary among the well-known install locations (official installer, user-local
// bin, Homebrew), or "" when none is present. home may be empty to auto-detect.
func resolvePOSIXOpenCodeBinary(home string) string {
	if strings.TrimSpace(home) == "" {
		if h, err := os.UserHomeDir(); err == nil {
			home = h
		}
	}
	candidates := []string{}
	if home != "" {
		candidates = append(candidates,
			filepath.Join(home, ".opencode", "bin", "opencode"),
			filepath.Join(home, ".local", "bin", "opencode"),
		)
	}
	candidates = append(candidates, "/opt/homebrew/bin/opencode", "/usr/local/bin/opencode")
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return ""
}
