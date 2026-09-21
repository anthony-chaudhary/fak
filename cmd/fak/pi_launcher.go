package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

type piLaunchOptions struct {
	dryRun       bool
	probePrompt  string
	addr         string
	baseURL      string
	model        string
	configPath   string
	writeConfig  bool
	checkBackend bool
	thinking     string
	tools        string
	quiet        bool
	command      string
	provider     string
	passthrough  []string
}

var piLaunchRun = execPiLaunchChild

func cmdPi(argv []string) {
	os.Exit(runPi(os.Stdout, os.Stderr, argv))
}

func runPi(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "config" {
		return runPiConfig(stdout, stderr, argv[1:])
	}

	fs := flag.NewFlagSet("pi", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "pi")

	dryRun := fs.Bool("dry-run", false, "print the Pi launch command and exit without executing")
	printEnv := fs.Bool("print-env", false, "print shell export statements for Pi environment and exit")
	probePrompt := fs.String("probe", "", "run a single headless probe turn with this prompt and exit")
	promptFlag := fs.String("prompt", "", "alias for probe prompt (or pass -p)")
	fs.StringVar(promptFlag, "p", "", "alias for probe prompt")
	addr := fs.String("addr", "127.0.0.1:8080", "fak serve gateway listen address (default: 127.0.0.1:8080 or FAK_SERVE_ADDR)")
	baseURL := fs.String("base-url", "", "fak serve provider base URL (default: http://<addr>/v1)")
	model := fs.String("model", "", "model ID (default: --model flag > existing settings.json defaultModel > auto-detect from fak serve /healthz > qwen38:27b-q4)")
	configPath := fs.String("config-path", "", "custom destination path for Pi models.json (default: ~/.pi/agent/models.json)")
	writeConfig := fs.Bool("write-config", true, "ensure ~/.pi/agent/models.json is configured with provider 'fak' before launching")
	checkBackend := fs.Bool("check-backend", true, "verify fak serve backend is reachable before starting Pi")
	thinking := fs.String("thinking", "", "thinking/reasoning level passed to pi (--thinking <level>)")
	tools := fs.String("tools", "", "tool allowlist passed to pi (--tools <list>)")
	window := fs.Int("window", 0, "served model context window in tokens for the SAFE Pi resident budget (default: auto-detect from the backend /v1/models context_length, else the default prior). fak writes contextWindow = min(window, window/2) so Pi auto-compacts inside the safe envelope instead of at the hard cap.")
	safeSettings := fs.Bool("safe-settings", true, "write a safe Pi compaction block (reserveTokens/keepRecentTokens derived from the served window) into Pi's settings.json")
	settingsPath := fs.String("settings-path", "", "destination path or directory for Pi's settings.json (default: ~/.pi/agent/settings.json)")
	quiet := fs.Bool("quiet", false, "suppress launcher banner and diagnostics")
	command := fs.String("command", "pi", "executable name or path for Pi CLI")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak pi [launcher flags] [-- <pi args...>]")
		fmt.Fprintln(stderr, "       fak pi config [--write] [--addr ADDR] [--model MODEL] [--path PATH]")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "First-class support for Pi coding agent as harness with fak serve on Mac as backend.")
		fmt.Fprintln(stderr, "Runs Pi directly targeting fak serve backend (for now raw, without guard).")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "examples:")
		fmt.Fprintln(stderr, "  fak pi                                        # interactive session on local Mac fak serve")
		fmt.Fprintln(stderr, "  fak pi --dry-run                              # preview configuration and command")
		fmt.Fprintln(stderr, "  fak pi --print-env                            # print shell export lines")
		fmt.Fprintln(stderr, "  fak pi --probe \"Explain unified memory\"       # headless probe turn")
		fmt.Fprintln(stderr, "  fak pi config --write                         # write ~/.pi/agent/models.json")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}

	if !parseFlags(fs, argv) {
		return 2
	}

	effectivePrompt := *probePrompt
	if effectivePrompt == "" && *promptFlag != "" {
		effectivePrompt = *promptFlag
	}

	targetAddr := *addr
	if envAddr := os.Getenv("FAK_SERVE_ADDR"); envAddr != "" && targetAddr == "127.0.0.1:8080" {
		targetAddr = envAddr
	}

	targetBaseURL := *baseURL
	if targetBaseURL == "" {
		targetBaseURL = projectassets.NormalizePiBaseURL(targetAddr)
	} else {
		targetBaseURL = projectassets.NormalizePiBaseURL(targetBaseURL)
	}

	targetModel := strings.TrimSpace(*model)
	backendActive := false
	// adoptedDetected records whether the launch model came from the backend's /healthz
	// label rather than a deliberate operator/configured pin. It only gates the diagnostic
	// wording below; the precedence itself is decided by ShouldAdoptDetectedPiModel.
	adoptedDetected := false

	// Probe backend to verify reachability and auto-detect the served model + context window
	detectedModel, reachable, detectedWindow, resolvedBaseURL := probePiBackendWithWindow(targetBaseURL, 1500*time.Millisecond)
	if reachable {
		backendActive = true
		// Adopt the origin that answered: a loopback literal can be refused while
		// the same gateway is live on the other address family (a WSL2 gateway is
		// exposed to the Windows host as [::1] only), so launching Pi at the
		// literal would hand it a dead base URL.
		targetBaseURL = resolvedBaseURL
		// Auto-detect seeds defaultModel only when no deliberate pin exists: an explicit
		// --model wins outright, and an existing non-empty settings.json defaultModel is a
		// deliberate choice the launcher must not clobber with the router's local label.
		if projectassets.ShouldAdoptDetectedPiModel(*settingsPath, *model, detectedModel) {
			targetModel = detectedModel
			adoptedDetected = true
		}
	}

	if targetModel == "" {
		// No explicit --model and auto-detect was not adopted. Honor a deliberate
		// settings.json defaultModel before falling back to the built-in prior, so the
		// default pin below repairs the PROVIDER without re-pointing the MODEL a user
		// chose on purpose.
		if existing, err := projectassets.PiSettingsDefaultModel(*settingsPath); err == nil && existing != "" {
			targetModel = existing
		} else {
			targetModel = projectassets.DefaultPiModelID
		}
	}

	// Safe context budget: an explicit --window wins; otherwise the window the backend
	// advertised; otherwise the default prior. PiSafeContextBudget clamps it and halves it so
	// the resident target is at most 50% of the served window (docs/long-context-defaults.md).
	servedWindow := *window
	if servedWindow <= 0 {
		servedWindow = detectedWindow
	}
	budget := projectassets.PiSafeContextBudget(servedWindow)

	if *printEnv {
		fmt.Fprintln(stdout, "# Environment configuration for Pi with fak serve backend on Mac")
		fmt.Fprintf(stdout, "export PI_PROVIDER=\"%s\"\n", projectassets.DefaultPiProviderID)
		fmt.Fprintf(stdout, "export PI_MODEL=\"%s\"\n", targetModel)
		fmt.Fprintf(stdout, "export OPENAI_BASE_URL=\"%s\"\n", targetBaseURL)
		return 0
	}

	if *checkBackend && !backendActive && !*dryRun {
		fmt.Fprintf(stderr, "fak pi: fak serve backend at %s is not responding.\n", targetBaseURL)
		fmt.Fprintln(stderr, "  To start the backend on macOS with Apple Silicon Metal acceleration:")
		fmt.Fprintf(stderr, "    fak serve --gguf %s --pi\n", targetModel)
		fmt.Fprintln(stderr, "  Or pass --check-backend=false to bypass backend liveness preflight.")
		return 1
	}

	launch := piLaunchOptions{
		dryRun:       *dryRun,
		probePrompt:  effectivePrompt,
		addr:         targetAddr,
		baseURL:      targetBaseURL,
		model:        targetModel,
		configPath:   *configPath,
		writeConfig:  *writeConfig,
		checkBackend: *checkBackend,
		thinking:     *thinking,
		tools:        *tools,
		quiet:        *quiet,
		command:      *command,
		provider:     projectassets.DefaultPiProviderID,
		passthrough:  fs.Args(),
	}

	if launch.writeConfig {
		resolvedPath, modified, err := projectassets.EnsurePiProviderConfigForWindow(launch.configPath, launch.baseURL, launch.model, servedWindow)
		if err != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: warning: could not update Pi config %s: %v\n", resolvedPath, err)
		} else if modified && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: updated %s with provider \"fak\" (baseURL: %s, model: %s, contextWindow: %d = safe 50%% of %d)\n", resolvedPath, launch.baseURL, launch.model, budget.ResidentTarget, budget.ServedWindow)
		}
	}

	if *safeSettings {
		sPath, modified, err := projectassets.EnsurePiSafeCompaction(*settingsPath, budget)
		if err != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: warning: could not update Pi settings %s: %v\n", sPath, err)
		} else if modified && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: wrote safe compaction to %s (reserveTokens: %d, keepRecentTokens: %d)\n", sPath, budget.OutputReserve, budget.KeepRecentTokens)
		}
	}

	// Pin Pi's harness DEFAULT onto the fak router. A plain `pi` launch (no --provider/
	// --model flags) resolves defaultProvider/defaultModel from settings.json, so without
	// this the `fak` provider written above is configured but never used by default. Same
	// non-clobbering discipline as models.json and compaction; idempotent.
	if launch.writeConfig {
		dPath, dModified, dErr := projectassets.EnsurePiDefaultProviderModel(*settingsPath, launch.provider, launch.model)
		if dErr != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: warning: could not pin Pi default provider/model in %s: %v\n", dPath, dErr)
		} else if dModified && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: pinned Pi default to provider %q model %q in %s\n", launch.provider, launch.model, dPath)
		}
	}

	argvOut := buildPiLaunchArgv(launch)

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak pi: dry-run - not launching")
		fmt.Fprintf(stderr, "  backend     = %s (raw without guard)\n", launch.baseURL)
		fmt.Fprintf(stderr, "  provider    = %s\n", launch.provider)
		fmt.Fprintf(stderr, "  model       = %s (%s)\n", launch.model, piModelSource(*model, adoptedDetected))
		fmt.Fprintf(stderr, "  context     = resident target %d tokens (safe 50%% of %d served window, %s)\n", budget.ResidentTarget, budget.ServedWindow, budget.Provenance)
		fmt.Fprintf(stderr, "  compaction  = reserve %d, keep %d (write=%t)\n", budget.OutputReserve, budget.KeepRecentTokens, *safeSettings)
		fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	env := os.Environ()
	if launch.configPath != "" {
		dir := launch.configPath
		if strings.HasSuffix(strings.ToLower(dir), ".json") {
			dir = filepath.Dir(dir)
		}
		env = append(env, "PI_CODING_AGENT_DIR="+dir)
	}

	if !launch.quiet {
		fmt.Fprintf(stderr, "fak pi: launching Pi (raw without guard) -> backend %s (model: %s)\n", launch.baseURL, launch.model)
	}

	return piLaunchRun(stdout, stderr, argvOut, env)
}

// piModelSource names where the effective launch model came from, for the dry-run
// diagnostic: an operator --model wins first, then an adopted /healthz detect, else the
// configured/default fallback. It is display-only and never changes the model.
func piModelSource(explicitModel string, adoptedDetected bool) string {
	switch {
	case strings.TrimSpace(explicitModel) != "":
		return "explicit --model"
	case adoptedDetected:
		return "auto-detected from /healthz"
	default:
		return "configured/existing default"
	}
}

// probePiBackend probes a Pi backend and returns the served model, whether it
// answered, and the base URL that answered. When the supplied base URL is a
// loopback literal that is refused, the probe retries through the loopback
// address-family fallback (127.0.0.1 / [::1] -> localhost) and reports that
// origin so the caller launches Pi against the family that is actually live.
func probePiBackend(baseURL string, timeout time.Duration) (model string, ok bool, resolvedBaseURL string) {
	m, ok, _, resolved := probePiBackendWithWindow(baseURL, timeout)
	return m, ok, resolved
}

// probePiBackendWithWindow probes the backend and also returns the served context window (tokens,
// 0 if unadvertised) plus the base URL that actually answered, so the caller can derive a SAFE
// resident target from the real window instead of an assumed prior (see
// projectassets.PiSafeContextBudget).
func probePiBackendWithWindow(baseURL string, timeout time.Duration) (model string, ok bool, window int, resolvedBaseURL string) {
	client := &http.Client{Timeout: timeout}
	if m, ok, w := probePiBackendWindow(client, baseURL); ok {
		return m, true, w, baseURL
	}
	if fallback, ok := fakclient.LoopbackFallbackURL(baseURL); ok {
		if m, ok, w := probePiBackendWindow(client, fallback); ok {
			return m, true, w, fallback
		}
	}
	return "", false, 0, baseURL
}

// probePiBackendOnce probes a single backend base URL for the served model id.
func probePiBackendOnce(client *http.Client, baseURL string) (string, bool) {
	model, ok, _ := probePiBackendWindow(client, baseURL)
	return model, ok
}

// probePiBackendWindow probes a single backend base URL: GET <root>/healthz, then read
// <base>/models for the served model id AND its advertised context_length (the window the safe
// budget is derived from).
func probePiBackendWindow(client *http.Client, baseURL string) (model string, ok bool, window int) {
	healthy := false
	healthURL := strings.TrimRight(strings.TrimSuffix(baseURL, "/v1"), "/") + "/healthz"
	resp, err := client.Get(healthURL)
	if err == nil && resp.StatusCode == http.StatusOK {
		defer resp.Body.Close()
		var h struct {
			OK    bool   `json:"ok"`
			Model string `json:"model"`
		}
		if json.NewDecoder(resp.Body).Decode(&h) == nil && (h.OK || h.Model != "") {
			model = h.Model
		}
		healthy = true
	}

	// /v1/models carries the served model id AND its advertised context length. Query it even
	// when /healthz answered, because healthz does not report the window.
	modelsURL := strings.TrimRight(baseURL, "/") + "/models"
	mResp, mErr := client.Get(modelsURL)
	if mErr == nil && mResp.StatusCode == http.StatusOK {
		defer mResp.Body.Close()
		var catalog struct {
			Data []struct {
				ID            string `json:"id"`
				ContextLength int    `json:"context_length"`
			} `json:"data"`
		}
		if json.NewDecoder(mResp.Body).Decode(&catalog) == nil && len(catalog.Data) > 0 {
			if model == "" {
				model = catalog.Data[0].ID
			}
			for _, row := range catalog.Data {
				if row.ContextLength > 0 && (row.ID == model || model == "") {
					window = row.ContextLength
					break
				}
			}
			if window == 0 {
				window = catalog.Data[0].ContextLength
			}
			return model, true, window
		}
		return model, healthy, 0
	}
	if healthy {
		return model, true, 0
	}
	return "", false, 0
}

func buildPiLaunchArgv(opts piLaunchOptions) []string {
	cmdName := opts.command
	if cmdName == "" {
		if envBin := os.Getenv("PI_BIN"); envBin != "" {
			cmdName = envBin
		} else {
			cmdName = "pi"
		}
	}
	argv := []string{cmdName}
	if opts.provider != "" {
		argv = append(argv, "--provider", opts.provider)
	}
	if opts.model != "" {
		argv = append(argv, "--model", opts.model)
	}
	if opts.probePrompt != "" {
		argv = append(argv, "-p", opts.probePrompt)
	}
	if opts.thinking != "" {
		argv = append(argv, "--thinking", opts.thinking)
	}
	if opts.tools != "" {
		argv = append(argv, "--tools", opts.tools)
	}
	argv = append(argv, opts.passthrough...)
	return argv
}

func execPiLaunchChild(stdout, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak pi: empty command")
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, stdout, stderr
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	if err := cmd.Run(); err != nil {
		code := childprocess.ExitCode(err, 1)
		if code == 1 {
			fmt.Fprintf(stderr, "fak pi: %v\n", err)
		}
		return code
	}
	return 0
}

func runPiConfig(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("pi config", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "127.0.0.1:8080", "fak serve gateway listen address")
	model := fs.String("model", projectassets.DefaultPiModelID, "served model ID")
	window := fs.Int("window", 0, "served model context window in tokens (default: the default prior). The written contextWindow is min(window, window/2) so Pi auto-compacts inside the safe envelope.")
	write := fs.Bool("write", false, "write or update ~/.pi/agent/models.json (or --path) and the safe Pi compaction settings")
	path := fs.String("path", "", "destination path or directory for models.json")
	settingsPath := fs.String("settings-path", "", "destination path or directory for Pi's settings.json (default: ~/.pi/agent/settings.json)")
	if !parseFlags(fs, argv) {
		return 2
	}
	baseURL := projectassets.NormalizePiBaseURL(*addr)
	budget := projectassets.PiSafeContextBudget(*window)
	if *write {
		resolvedPath, modified, err := projectassets.EnsurePiProviderConfigForWindow(*path, baseURL, *model, *window)
		if err != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", err)
			return 1
		}
		if modified {
			fmt.Fprintf(stdout, "fak pi config: updated %s with provider \"fak\" (baseURL: %s, model: %s, contextWindow: %d = safe 50%% of %d)\n", resolvedPath, baseURL, *model, budget.ResidentTarget, budget.ServedWindow)
		} else {
			fmt.Fprintf(stdout, "fak pi config: %s already has up-to-date provider \"fak\"\n", resolvedPath)
		}
		sPath, sModified, sErr := projectassets.EnsurePiSafeCompaction(*settingsPath, budget)
		if sErr != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", sErr)
			return 1
		}
		if sModified {
			fmt.Fprintf(stdout, "fak pi config: wrote safe compaction to %s (reserveTokens: %d, keepRecentTokens: %d)\n", sPath, budget.OutputReserve, budget.KeepRecentTokens)
		} else {
			fmt.Fprintf(stdout, "fak pi config: %s already has a safe compaction block\n", sPath)
		}
		// Pin the harness default so a bare `pi` launch uses the fak router, not merely
		// having it configured in models.json.
		dPath, dModified, dErr := projectassets.EnsurePiDefaultProviderModel(*settingsPath, projectassets.DefaultPiProviderID, *model)
		if dErr != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", dErr)
			return 1
		}
		if dModified {
			fmt.Fprintf(stdout, "fak pi config: pinned Pi default to provider %q model %q in %s\n", projectassets.DefaultPiProviderID, *model, dPath)
		} else {
			fmt.Fprintf(stdout, "fak pi config: %s already defaults to provider %q model %q\n", dPath, projectassets.DefaultPiProviderID, *model)
		}
		return 0
	}
	out, err := projectassets.GeneratePiConfigForWindow(baseURL, *model, *window)
	if err != nil {
		fmt.Fprintf(stderr, "fak pi config: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	fmt.Fprintf(stderr, "fak pi config: safe context envelope: resident target %d tokens (50%% of %d served window, %s); compaction reserve %d, keep %d\n", budget.ResidentTarget, budget.ServedWindow, budget.Provenance, budget.OutputReserve, budget.KeepRecentTokens)
	return 0
}
