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
// Pi json grammar assumption. `--mode json` streams newline-delimited JSON
// objects each carrying a `type` field. This arm treats a run as a clean success
// only when: the child exit code is 0, at least one assistant/result event was
// observed, and no `"type":"error"` event appeared. Anything else is non-success.
// The assumed type vocabulary is recorded here so a future reader can re-verify
// it against the installed Pi contract; a real installed-Pi qualification is the
// mission's separate live canary (fak-private#2249), not this software contract.

const (
	opsPiHarness = "pi"

	// opsPiDirective is the short inline `-p` prompt. The task itself arrives via
	// `@<file>`, so this only states the intent and stays well under the Win32
	// argv ceiling.
	opsPiDirective = "Complete the attached task. Follow it exactly; it is your only task."
)

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
		Type string `json:"type"`
	}
	if json.Unmarshal(trimmed, &envelope) != nil {
		return ""
	}
	switch envelope.Type {
	case "error", "run_error":
		return "error"
	case "assistant", "message", "result", "run_result", "finish":
		return "assistant"
	default:
		return ""
	}
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
	prompt, err := os.ReadFile(*promptFile)
	if err != nil || len(bytes.TrimSpace(prompt)) == 0 {
		fmt.Fprintln(stderr, "ops run pi: prompt file must be readable and nonempty")
		return 2
	}
	if !*dryRun {
		for _, input := range []string{*promptFile, *policy, *audit} {
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

	env, cleanupEnv, err := opsRunChildEnvironment("{}", *apiKeyEnv)
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
	preflight, err := opsRunInferencePreflight(ctx, guardProbeBaseURLForPi(gatewayProbeURL), *model)
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

	code, complete, eventError, lifecycle := opsPiRunExecute(ctx, stdout, stderr, argv, env, prompt)
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
