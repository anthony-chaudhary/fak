package main

import (
	"bytes"
	"context"
	"crypto/rand"
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
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ops_run_pi.go — the `fak ops run --harness pi` arm. Pi (earendil-works) is a
// first-class headless coding agent, so Ops runs it through the SAME bounded
// subprocess lifecycle the OpenCode arm owns (bounded deadline, process-tree
// cancellation, run-scoped config, shared receipt) instead of a second driver.
//
// Routing. The launch is wrapped by `fak guard -- pi ...`: guard injects the
// session-scoped `-e` extension (guard_pi.go) that registers the `fak` provider
// at the in-process kernel gateway origin, so every Pi tool call crosses the
// capability floor and inference proxies through the Fak Router (fak serve). Pi
// speaks the OpenAI-completions wire for the `fak` provider (projectassets
// GeneratePiConfig: api "openai-completions"), which is why the inference
// preflight below is the OpenAI-wire probe the OpenCode arm already uses.
//
// Prompt delivery. The task prompt can be tens of KB. On Windows the npm `pi.cmd`
// shim routes argv through cmd.exe, whose command line caps near 8191 bytes; an
// inline prompt therefore launched a worker that exited instantly with 0-byte
// logs (observed 2026-09-21 in the private spawn arm). So the prompt is delivered
// via Pi's `@<file>` inclusion — a short `-p` directive plus `@<abs path>` — and
// never inline.
//
// Pi 1.4.0 JSON completion requires an assistant message_end with stopReason
// stop. A later assistant start or incomplete end clears completion; errors,
// truncation and aborted settlement fail closed. Legacy terminal aliases remain
// supported. Process exit and deadline checks still apply independently.

const (
	opsPiHarness = "pi"

	// opsPiDirective is the short inline `-p` prompt. The task itself arrives via
	// `@<file>`, so this only states the intent and stays well under the Win32
	// argv ceiling.
	opsPiDirective = "Complete the attached task. Follow it exactly; it is your only task."
)

const opsPiAPIKeyMaxBytes = 4096

// opsPiAPIKeyFile holds the admitted descriptor and its directory root. Reads
// verify that the pathname still names the same regular file before and after
// consuming the descriptor, so a replaced path is refused without reopening it.
type opsPiAPIKeyFile struct {
	path string
	base string
	root *os.Root
	file *os.File
	info os.FileInfo
}

func openOpsPiAPIKeyFile(path string) (*opsPiAPIKeyFile, error) {
	if !filepath.IsAbs(path) {
		return nil, fmt.Errorf("--api-key-file must be an absolute path")
	}
	clean := filepath.Clean(path)
	root, err := os.OpenRoot(filepath.Dir(clean))
	if err != nil {
		return nil, fmt.Errorf("open credential directory: %w", err)
	}
	base := filepath.Base(clean)
	info, err := root.Lstat(base)
	if err != nil {
		root.Close()
		return nil, fmt.Errorf("inspect credential file: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		root.Close()
		return nil, fmt.Errorf("credential file must be a regular non-symlink file")
	}
	if info.Size() < 1 || info.Size() > opsPiAPIKeyMaxBytes {
		root.Close()
		return nil, fmt.Errorf("credential file size must be 1..%d bytes", opsPiAPIKeyMaxBytes)
	}
	f, err := root.Open(base)
	if err != nil {
		root.Close()
		return nil, fmt.Errorf("open credential file: %w", err)
	}
	opened, err := f.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		f.Close()
		root.Close()
		return nil, fmt.Errorf("credential path changed during admission")
	}
	k := &opsPiAPIKeyFile{path: clean, base: base, root: root, file: f, info: opened}
	if err := k.verifyCurrent(); err != nil {
		k.Close()
		return nil, err
	}
	return k, nil
}

func (k *opsPiAPIKeyFile) verifyCurrent() error {
	current, err := k.root.Lstat(k.base)
	if err != nil || current.Mode()&os.ModeSymlink != 0 || !current.Mode().IsRegular() || !os.SameFile(k.info, current) {
		return fmt.Errorf("credential path changed after admission")
	}
	return nil
}

func (k *opsPiAPIKeyFile) Read() (string, error) {
	if err := k.verifyCurrent(); err != nil {
		return "", err
	}
	if _, err := k.file.Seek(0, io.SeekStart); err != nil {
		return "", fmt.Errorf("read credential file")
	}
	raw, err := io.ReadAll(io.LimitReader(k.file, opsPiAPIKeyMaxBytes+1))
	if err != nil || len(raw) == 0 || len(raw) > opsPiAPIKeyMaxBytes {
		return "", fmt.Errorf("credential file must contain 1..%d bytes", opsPiAPIKeyMaxBytes)
	}
	if err := k.verifyCurrent(); err != nil {
		return "", err
	}
	if bytes.HasSuffix(raw, []byte("\r\n")) {
		raw = raw[:len(raw)-2]
	} else if bytes.HasSuffix(raw, []byte("\n")) {
		raw = raw[:len(raw)-1]
	}
	if len(raw) == 0 {
		return "", fmt.Errorf("credential token must be nonempty")
	}
	for _, b := range raw {
		if b < 0x21 || b > 0x7e {
			return "", fmt.Errorf("credential token must be one visible ASCII token")
		}
	}
	return string(raw), nil
}

func (k *opsPiAPIKeyFile) Close() error {
	var first error
	if k.file != nil {
		first = k.file.Close()
		k.file = nil
	}
	if k.root != nil {
		if err := k.root.Close(); first == nil {
			first = err
		}
		k.root = nil
	}
	return first
}

// opsPiProviderWire maps an accepted Pi provider id to the guard wire it speaks.
// The `fak` provider is the OpenAI-completions route to the Fak Router; a bare
// `anthropic` selection keeps Pi's native Messages wire through the gateway.
func opsPiProviderWire(provider string) (string, bool) {
	switch provider {
	case projectassets.DefaultPiProviderID:
		return "openai", true
	case "anthropic":
		return "anthropic", true
	default:
		return "", false
	}
}

// findOpsPiBinary resolves the pi executable: an explicit path wins, else PATH is
// searched for the `pi` shim (and the Windows `pi.cmd`/`pi.exe` shims npm
// installs). It fails closed with an actionable error rather than a
// plausible-but-wrong path.
func findOpsPiBinary(explicit string) (string, error) {
	if p := strings.TrimSpace(explicit); p != "" {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p, nil
		}
		if found, err := exec.LookPath(p); err == nil {
			return found, nil
		}
		return "", fmt.Errorf("pi binary %q not found on disk or PATH", explicit)
	}
	for _, name := range []string{"pi", "pi.cmd", "pi.exe"} {
		if found, err := exec.LookPath(name); err == nil {
			return found, nil
		}
	}
	return "", fmt.Errorf("pi binary not found on PATH (install: npm i -g @earendil-works/pi-coding-agent, or pass --pi-bin <path>)")
}

// opsPiChildArgv builds the headless Pi argv. The prompt is delivered via the
// `@<abs>` inclusion, never inline (see the file header).
func opsPiChildArgv(piBin, promptPath, provider, model, thinking string) []string {
	argv := []string{piBin, "-p", opsPiDirective}
	if f := strings.TrimSpace(promptPath); f != "" {
		argv = append(argv, "@"+f)
	}
	argv = append(argv, "--mode", "json", "--no-session")
	if p := strings.TrimSpace(provider); p != "" {
		argv = append(argv, "--provider", p)
	}
	if m := strings.TrimSpace(model); m != "" {
		argv = append(argv, "--model", m)
	}
	if t := strings.TrimSpace(thinking); t != "" {
		argv = append(argv, "--thinking", t)
	}
	return argv
}

// opsPiEvents reads Pi's `--mode json` stream, forwarding it while retaining only
// the terminal classification state (bounded memory): whether a completed turn
// was observed and whether an error event appeared. It mirrors opsRunEvents but
// recognizes Pi's event vocabulary.
type opsPiEvents struct {
	output    io.Writer
	pending   []byte
	overflow  bool
	assistant bool
	failed    bool
}

func (w *opsPiEvents) Write(p []byte) (int, error) {
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

func (w *opsPiEvents) finishLine() {
	if w.overflow {
		w.failed = true
	}
	if !w.overflow {
		switch opsPiClassify(w.pending) {
		case "assistant":
			w.assistant = true
		case "reset":
			w.assistant = false
		case "error":
			w.failed = true
		}
	}
	w.pending = nil
	w.overflow = false
}

// opsPiClassify maps one stdout line to a Pi terminal class. Empty means the
// line is not a recognized terminal-relevant event. The vocabulary is the
// documented assumption in the file header.
func opsPiClassify(line []byte) string {
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 || !bytes.HasPrefix(trimmed, []byte("{")) {
		return ""
	}
	var envelope struct {
		Type    string `json:"type"`
		Aborted bool   `json:"aborted"`
		Message struct {
			Role       string `json:"role"`
			StopReason string `json:"stopReason"`
		} `json:"message"`
	}
	if json.Unmarshal(trimmed, &envelope) != nil {
		var eventType struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(trimmed, &eventType) == nil {
			switch eventType.Type {
			case "error", "run_error", "message_start", "message_end", "agent_settled":
				return "error"
			case "assistant", "message", "result", "run_result", "finish":
				return "assistant"
			}
		}
		return ""
	}
	switch envelope.Type {
	case "error", "run_error":
		return "error"
	case "message_start":
		if envelope.Message.Role == "assistant" {
			return "reset"
		}
	case "message_end":
		if envelope.Message.Role != "assistant" {
			return ""
		}
		switch envelope.Message.StopReason {
		case "stop":
			return "assistant"
		case "error", "aborted", "length":
			return "error"
		default:
			return "reset"
		}
	case "agent_settled":
		if envelope.Aborted {
			return "error"
		}
	case "assistant", "message", "result", "run_result", "finish":
		return "assistant"
	}
	return ""
}

// opsPiComplete reports whether the observed stream is a clean completed turn.
func opsPiComplete(events *opsPiEvents) bool {
	return events.assistant && !events.failed
}

// opsPiExecute is the bounded child lifecycle, separable for tests (mirrors
// opsRunExecute). It returns (exitCode, complete, failed, lifecycle).
func opsPiExecute(ctx context.Context, stdout, stderr io.Writer, argv, env []string, prompt []byte) (int, bool, bool, []opsRunLifecycleRecord) {
	events := &opsPiEvents{output: stdout}
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
		// Wait owns ProcessState; enrich the exit code after Run returns.
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
		fmt.Fprintf(stderr, "ops run pi: child: %v\n", err)
		return childprocess.ExitCode(err, 1), opsPiComplete(events), events.failed, resLifecycle
	}
	return 0, opsPiComplete(events), events.failed, resLifecycle
}

var opsPiRunExecute = opsPiExecute

func runOpsPi(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("ops run --harness pi", flag.ContinueOnError)
	fs.SetOutput(stderr)
	harness := fs.String("harness", opsPiHarness, "headless harness: pi")
	provider := fs.String("provider", projectassets.DefaultPiProviderID, "Pi provider route: fak (Fak Router, OpenAI-completions) or anthropic")
	model := fs.String("model", "", "served model identifier (required)")
	baseURL := fs.String("base-url", "", "Fak Router / gateway endpoint (required except dry-run)")
	apiKeyEnv := fs.String("api-key-env", "", "environment variable holding the upstream key")
	apiKeyFile := fs.String("api-key-file", "", "absolute regular file holding the upstream key (requires --api-key-env)")
	promptFile := fs.String("prompt-file", "", "UTF-8 task file, delivered via Pi's @file inclusion")
	workspace := fs.String("workspace", "", "existing workspace directory for the child process (required except dry-run)")
	timeout := fs.Duration("timeout", 5*time.Minute, "positive wall-clock deadline")
	receiptPath := fs.String("receipt", "", "execution metadata JSON file (required except dry-run)")
	policy := fs.String("policy", "", "guard capability-floor policy file")
	audit := fs.String("audit", "", "guard audit journal file")
	guardModeFlag := fs.String("guard-mode", "enforce", "guard posture: enforce (audit-only/off are unsupported)")
	piBin := fs.String("pi-bin", "pi", "Pi executable; use the native .exe on Windows")
	thinking := fs.String("thinking", "", "Pi thinking effort (omitted when unset)")
	dryRun := fs.Bool("dry-run", false, "validate and print metadata without launching")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if *harness != opsPiHarness || fs.NArg() != 0 || *timeout <= 0 || strings.TrimSpace(*model) == "" || *promptFile == "" || (!*dryRun && (*receiptPath == "" || strings.TrimSpace(*workspace) == "")) {
		fmt.Fprintln(stderr, "ops run pi: require --harness pi, --provider fak|anthropic, --model, --prompt-file, --workspace, --receipt and a positive --timeout; no positional arguments (--workspace and --receipt may be omitted only for --dry-run)")
		return 2
	}
	*apiKeyEnv = strings.TrimSpace(*apiKeyEnv)
	*apiKeyFile = strings.TrimSpace(*apiKeyFile)
	fileMode := *apiKeyFile != ""
	if fileMode && strings.TrimSpace(*apiKeyEnv) == "" {
		fmt.Fprintln(stderr, "ops run pi: --api-key-file requires --api-key-env")
		return 2
	}
	if fileMode {
		name := strings.TrimSpace(*apiKeyEnv)
		canonical := strings.ToUpper(name)
		if !opsRunValidEnvName(name) || opsRunControlledEnv(canonical) || opsRunAllowedPlatformEnv(canonical) {
			fmt.Fprintln(stderr, "ops run pi: --api-key-env name is unsafe")
			return 2
		}
	}
	if fileMode {
		if ambient, ok := os.LookupEnv(strings.TrimSpace(*apiKeyEnv)); ok && ambient != "" {
			fmt.Fprintln(stderr, "ops run pi: --api-key-file refuses a nonempty ambient value for --api-key-env")
			return 2
		}
	}
	var keyFile *opsPiAPIKeyFile
	selectedBearer := ""
	if fileMode {
		var err error
		keyFile, err = openOpsPiAPIKeyFile(strings.TrimSpace(*apiKeyFile))
		if err != nil {
			fmt.Fprintf(stderr, "ops run pi: credential file refused: %v\n", err)
			return 2
		}
		defer keyFile.Close()
		selectedBearer, err = keyFile.Read()
		if err != nil {
			fmt.Fprintf(stderr, "ops run pi: credential file refused: %v\n", err)
			return 2
		}
	} else if strings.TrimSpace(*apiKeyEnv) != "" {
		selectedBearer = os.Getenv(strings.TrimSpace(*apiKeyEnv))
	}
	wire, ok := opsPiProviderWire(strings.ToLower(strings.TrimSpace(*provider)))
	if !ok {
		fmt.Fprintln(stderr, "ops run pi: unsupported --provider; want fak (Fak Router) or anthropic")
		return 2
	}
	resolvedWorkspace := ""
	if strings.TrimSpace(*workspace) != "" {
		var err error
		resolvedWorkspace, err = resolveOpsRunWorkspace(*workspace)
		if err != nil {
			fmt.Fprintf(stderr, "ops run pi: invalid workspace: %v\n", err)
			return 2
		}
	}
	if fileMode {
		for _, other := range []string{*promptFile, *policy, *audit, *receiptPath} {
			if other != "" && opsRunSamePath(*apiKeyFile, other) {
				fmt.Fprintln(stderr, "ops run pi: credential file must be distinct from prompt, policy, audit and receipt files")
				return 2
			}
		}
	}
	prompt, err := os.ReadFile(*promptFile)
	if err != nil || len(bytes.TrimSpace(prompt)) == 0 {
		fmt.Fprintln(stderr, "ops run pi: prompt file must be readable and nonempty")
		return 2
	}
	if !*dryRun {
		for _, input := range []string{*promptFile, *policy, *audit, *apiKeyFile} {
			if input != "" && opsRunSamePath(*receiptPath, input) {
				fmt.Fprintln(stderr, "ops run pi: receipt must be distinct from prompt, policy and audit files")
				return 2
			}
		}
	}
	promptPath, err := filepath.Abs(*promptFile)
	if err != nil {
		fmt.Fprintf(stderr, "ops run pi: resolve prompt file: %v\n", err)
		return 2
	}

	// A dry-run never resolves the binary: the plan is a pure description of what
	// would launch, so a machine without Pi installed can still render it.
	resolvedPi := *piBin
	if !*dryRun {
		resolvedPi, err = findOpsPiBinary(*piBin)
		if err != nil {
			fmt.Fprintf(stderr, "ops run pi: %v\n", err)
			return 2
		}
	}

	var nonce [12]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		fmt.Fprintln(stderr, "ops run pi: create run identity failed")
		return 1
	}
	runID := "ops-" + hex.EncodeToString(nonce[:])

	guardMode, guardPolicy := qualifyOpsRunGuardMode(*guardModeFlag)
	configPolicy := guardPolicy
	piArgv := opsPiChildArgv(resolvedPi, promptPath, *provider, *model, *thinking)
	gatewayProbeURL := strings.TrimSpace(*baseURL)
	// Guard wraps the Pi child: it injects the session-scoped provider extension
	// that repoints Pi at the gateway, and holds the real upstream credential.
	guardArgv := []string{tuiExecutable(), "guard", "--provider", wire, "--split", "off", "--model", *model}
	for _, pair := range [][2]string{{"--base-url", *baseURL}, {"--api-key-env", *apiKeyEnv}, {"--policy", *policy}, {"--audit", *audit}} {
		if pair[1] != "" {
			guardArgv = append(guardArgv, pair[0], pair[1])
		}
	}
	argv := append(append([]string{}, guardArgv...), "--")
	argv = append(argv, piArgv...)

	identity := newOpsRunLaunchIdentity(runID, opsPiHarness, resolvedWorkspace, *provider, *baseURL, *model, resolvedPi, "", *policy, guardMode, false, false)

	if *dryRun {
		_ = json.NewEncoder(stdout).Encode(map[string]any{
			"schema":          "fak-ops-run-plan/1",
			"harness":         opsPiHarness,
			"provider":        *provider,
			"wire":            wire,
			"model":           *model,
			"workspace":       resolvedWorkspace,
			"guarded":         true,
			"guard_mode":      guardMode,
			"prompt_delivery": "file",
			"timeout":         timeout.String(),
			"argv":            argv,
		})
		return 0
	}

	if guardPolicy.Status != "qualified" {
		fmt.Fprintf(stderr, "ops run pi: guard mode refused: %s\n", guardPolicy.Reason)
		now := time.Now().UTC()
		receipt := opsRunReceipt{Schema: "fak-ops-run/1", Harness: opsPiHarness, Workspace: resolvedWorkspace, LaunchIdentity: &identity, Status: "refused", ExitCode: 1, Started: now, Finished: now, ConfigPolicy: &configPolicy}
		if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
			fmt.Fprintf(stderr, "ops run pi: write receipt: %v\n", err)
		}
		return 1
	}

	if keyFile != nil {
		selectedBearer, err = keyFile.Read()
		if err != nil {
			fmt.Fprintln(stderr, "ops run pi: credential path changed before launch")
			return 1
		}
	}
	env, cleanupEnv, err := opsRunChildEnvironmentValue("{}", *apiKeyEnv, selectedBearer)
	if err != nil {
		fmt.Fprintf(stderr, "ops run pi: create isolated child environment: %v\n", err)
		return 1
	}
	defer cleanupEnv()

	receipt := opsRunReceipt{Schema: "fak-ops-run/1", Harness: opsPiHarness, Workspace: resolvedWorkspace, LaunchIdentity: &identity, Status: "running", Started: time.Now().UTC(), ConfigPolicy: &configPolicy}
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run pi: write receipt: %v\n", err)
		return 1
	}

	sigCtx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	ctx, cancel := context.WithTimeout(sigCtx, *timeout)
	defer cancel()
	ctx = withSignalChecker(ctx, func() bool { return sigCtx.Err() != nil })
	ctx = withOpsRunWorkspace(ctx, resolvedWorkspace)

	// Pi's `fak` route is OpenAI-completions; `anthropic` is not a probe surface this
	// arm qualifies. Refuse the unscoped wire rather than launching unverified.
	if wire != "openai" {
		preflight := opsRunInferenceRefusal(wire, gatewayProbeURL, *model, "unsupported_provider_protocol")
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	if strings.TrimSpace(*baseURL) == "" {
		preflight := opsRunInferenceRefusalWithStatus(wire, *baseURL, *model, "missing_explicit_base_url", "refused")
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	var preflight opsRunInferencePreflightReceipt
	if fileMode {
		preflight, err = opsRunInferencePreflightBearer(ctx, guardProbeBaseURLForPi(gatewayProbeURL), *model, selectedBearer)
	} else {
		preflight, err = opsRunInferencePreflight(ctx, guardProbeBaseURLForPi(gatewayProbeURL), *model, *apiKeyEnv)
	}
	if selectedBearer != "" {
		preflight.ObservedModel = strings.ReplaceAll(preflight.ObservedModel, selectedBearer, "[REDACTED]")
	}
	receipt.InferencePreflight = &preflight
	if err != nil {
		return failOpsRunInferencePreflight(stderr, *receiptPath, receipt, preflight)
	}
	opsRunAddCapabilityEvidence(receipt.LaunchIdentity, configPolicy.Digest, preflight.ReceiptRef)
	// Persist the qualifying reference before launch: a receipt write failure
	// cannot produce an unqualified Pi child process.
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run pi: write receipt: %v\n", err)
		return 1
	}

	if keyFile != nil {
		verifiedBearer, readErr := keyFile.Read()
		if readErr != nil || verifiedBearer != selectedBearer {
			fmt.Fprintln(stderr, "ops run pi: credential changed before child launch")
			receipt.Status, receipt.ExitCode, receipt.Finished = "failed", 1, time.Now().UTC()
			if writeErr := writeOpsRunReceipt(*receiptPath, receipt); writeErr != nil {
				fmt.Fprintf(stderr, "ops run pi: write receipt: %v\n", writeErr)
			}
			return 1
		}
	}
	redactedStdout := newOpsRunSecretRedactor(stdout, selectedBearer)
	redactedStderr := newOpsRunSecretRedactor(stderr, selectedBearer)
	code, complete, eventError, lifecycle := opsPiRunExecute(ctx, redactedStdout, redactedStderr, argv, env, prompt)
	stdoutErr, stderrErr := redactedStdout.Flush(), redactedStderr.Flush()
	if stdoutErr != nil || stderrErr != nil {
		eventError = true
	}
	if selectedBearer != "" {
		for i := range lifecycle {
			lifecycle[i].Error = strings.ReplaceAll(lifecycle[i].Error, selectedBearer, "[REDACTED]")
			lifecycle[i].OSError = strings.ReplaceAll(lifecycle[i].OSError, selectedBearer, "[REDACTED]")
		}
	}
	if len(lifecycle) > 0 {
		receipt.Lifecycle = append(receipt.Lifecycle, lifecycle...)
	}
	receipt.Status = "failed"
	if ctx.Err() != nil {
		if ctx.Err() == context.DeadlineExceeded {
			code, receipt.Status = 124, "timed_out"
		} else {
			code, receipt.Status = 130, "cancelled"
		}
		if len(receipt.Lifecycle) == 0 {
			reason := opsRunDetermineTerminationReason(ctx)
			receipt.Lifecycle = append(receipt.Lifecycle, opsRunLifecycleRecord{
				Reason:            reason,
				TerminationReason: reason,
				ChildState:        "unknown",
				State:             "unknown",
				Signal:            "SIGKILL",
			})
		}
	} else if code == 0 {
		if eventError || !complete {
			code = 1
			fmt.Fprintln(stderr, "ops run pi: Pi did not report a successful completed turn")
		} else {
			receipt.Status = "succeeded"
		}
	}
	receipt.ExitCode, receipt.Finished = code, time.Now().UTC()
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run pi: write receipt: %v\n", err)
		return 1
	}
	return code
}

// guardProbeBaseURLForPi derives the /v1 probe origin from the router base URL.
// opsRunInferencePreflight appends /chat/completions itself, so it needs the /v1
// origin Pi's provider config also targets.
func guardProbeBaseURLForPi(baseURL string) string {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		return ""
	}
	if !strings.HasSuffix(base, "/v1") {
		base += "/v1"
	}
	return base
}
