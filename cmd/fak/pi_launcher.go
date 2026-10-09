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
	guarded      bool
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
	skillPack    string
	passthrough  []string
}

var piLaunchRun = execPiLaunchChild

const defaultPiLauncherAddr = "127.0.0.1:8080"

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
	guarded := fs.Bool("guard", false, "launch Pi through the session-scoped fak guard")
	noGuard := fs.Bool("no-guard", false, "launch Pi directly (default; overrides --guard)")
	printEnv := fs.Bool("print-env", false, "print shell export statements for Pi environment and exit")
	probePrompt := fs.String("probe", "", "run a single headless probe turn with this prompt and exit")
	promptFlag := fs.String("prompt", "", "alias for probe prompt (or pass -p)")
	fs.StringVar(promptFlag, "p", "", "alias for probe prompt")
	addr := fs.String("addr", "", "fak serve gateway listen address (default: configured provider URL, then 127.0.0.1:8080 or FAK_SERVE_ADDR)")
	baseURL := fs.String("base-url", "", "fak serve provider base URL (default: http://<addr>/v1)")
	model := fs.String("model", "", "model ID (default: --model flag > existing settings.json defaultModel > auto-detect from fak serve /healthz > qwen38:27b-q4)")
	configPath := fs.String("config-path", "", "custom destination path for Pi models.json (default: ~/.pi/agent/models.json)")
	writeConfig := fs.Bool("write-config", false, "ensure ~/.pi/agent/models.json is configured with provider 'fak' before launching")
	checkBackend := fs.Bool("check-backend", true, "verify fak serve backend is reachable before starting Pi")
	thinking := fs.String("thinking", "", "thinking/reasoning level passed to pi (--thinking <level>)")
	tools := fs.String("tools", "", "tool allowlist passed to pi (--tools <list>)")
	window := fs.Int("window", 0, "served model context window in tokens for the SAFE Pi resident budget (default: auto-detect from the backend /v1/models context_length, else the default prior). fak derives the context envelope once from this RAW window: contextWindow = min(window, quality cap 163840), and compaction reserve/keep sized so a compaction reclaims real room.")
	safeSettings := fs.Bool("safe-settings", false, "write a safe Pi compaction block (reserveTokens/keepRecentTokens derived from the served window) into Pi's settings.json")
	settingsPath := fs.String("settings-path", "", "destination path or directory for Pi's settings.json (default: ~/.pi/agent/settings.json)")
	quiet := fs.Bool("quiet", false, "suppress launcher banner and diagnostics")
	command := fs.String("command", "pi", "executable name or path for Pi CLI")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak pi [launcher flags] [-- <pi args...>]")
		fmt.Fprintln(stderr, "       fak pi config [--write] [--addr ADDR] [--model MODEL] [--path PATH]")
		fmt.Fprintln(stderr, "       fak pi config --from-router [URL] [--write] [--path PATH] [--settings-path PATH]")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "First-class support for Pi coding agent as harness with fak serve on Mac as backend.")
		fmt.Fprintln(stderr, "Runs Pi directly against fak serve by default; pass --guard for session-scoped policy enforcement.")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "examples:")
		fmt.Fprintln(stderr, "  fak pi                                        # interactive session on local Mac fak serve")
		fmt.Fprintln(stderr, "  fak pi --dry-run                              # preview configuration and command")
		fmt.Fprintln(stderr, "  fak pi --print-env                            # print shell export lines")
		fmt.Fprintln(stderr, "  fak pi --probe \"Explain unified memory\"       # headless probe turn")
		fmt.Fprintln(stderr, "  fak pi config --write                         # write ~/.pi/agent/models.json")
		fmt.Fprintln(stderr, "  fak pi config --from-router                   # plan provider \"fak\" models from the router catalog")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}

	if !parseFlags(fs, argv) {
		return 2
	}
	addrSet, baseURLSet := false, false
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "addr":
			addrSet = true
		case "base-url":
			baseURLSet = true
		}
	})

	effectivePrompt := *probePrompt
	if effectivePrompt == "" && *promptFlag != "" {
		effectivePrompt = *promptFlag
	}

	targetAddr := strings.TrimSpace(*addr)
	targetBaseURL := strings.TrimSpace(*baseURL)
	if !baseURLSet && targetBaseURL == "" {
		configured := configuredPiProviderBaseURL(*configPath)
		switch {
		case addrSet:
			targetBaseURL = projectassets.NormalizePiBaseURL(targetAddr)
		case strings.TrimSpace(os.Getenv("FAK_SERVE_ADDR")) != "":
			targetAddr = strings.TrimSpace(os.Getenv("FAK_SERVE_ADDR"))
			targetBaseURL = projectassets.NormalizePiBaseURL(targetAddr)
		case configured != "":
			targetBaseURL = projectassets.NormalizePiBaseURL(configured)
		default:
			targetAddr = defaultPiLauncherAddr
			targetBaseURL = projectassets.NormalizePiBaseURL(targetAddr)
		}
	}
	targetBaseURL = projectassets.NormalizePiBaseURL(targetBaseURL)

	targetModel := strings.TrimSpace(*model)
	modelAutoDetected := false
	// modelReplacedStaleDefault records that targetModel was chosen by REPLACING a
	// configured default the backend does not advertise (a stale placeholder such as
	// `custom-model`). That replacement is authoritative intent: the seed-only writer
	// must not preserve the very value we just decided is wrong, so this forces the
	// authoritative default writer even though the id was auto-detected.
	modelReplacedStaleDefault := false
	backendActive := false
	// adoptedDetected records whether the launch model came from the backend's /healthz
	// label rather than a deliberate operator/configured pin. It only gates the diagnostic
	// wording below; the precedence itself is decided by ShouldAdoptDetectedPiModel.
	adoptedDetected := false

	// Probe backend to verify reachability and auto-detect the served model + context window
	detectedModel, reachable, _, resolvedBaseURL, advertisedModels, advertisedWindows := probePiBackendWithCatalog(targetBaseURL, 1500*time.Millisecond)
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
			modelAutoDetected = true
		}
	}

	if targetModel == "" {
		// No explicit --model and auto-detect was not adopted. Honor a deliberate
		// settings.json defaultModel before falling back to the built-in prior, so the
		// default pin below repairs the PROVIDER without re-pointing the MODEL a user
		// chose on purpose.
		if existing, err := projectassets.PiSettingsDefaultModel(*settingsPath); err == nil && existing != "" && projectassets.PiDefaultModelIsDeliberate(existing, advertisedModels) {
			targetModel = existing
		} else {
			// Either nothing is configured, or the configured default is NOT one of the
			// ids this backend advertises. A non-advertised default (a stale placeholder
			// such as `custom-model`) cannot be a deliberate choice for this router and
			// would otherwise be re-confirmed forever, so replace it with the id the
			// backend actually serves — the detected one when we have it, else the prior.
			if detectedModel != "" && detectedModel != "mock" {
				targetModel = detectedModel
				modelAutoDetected = true
				modelReplacedStaleDefault = true
			} else {
				targetModel = projectassets.DefaultPiModelID
			}
		}
	}

	// Resolve one raw input for the selected model, not the /healthz local engine:
	// explicit --window, then that model's advertised bound, then the registry/prior.
	// Keep the existing per-model registry limits in PiModelContextBudget and pass the
	// same input to the provider writers, compaction settings, and diagnostics.
	servedWindow := *window
	if servedWindow <= 0 {
		servedWindow = piAdvertisedModelWindow(targetModel, advertisedWindows)
	}
	budget := projectassets.PiModelContextBudget(targetModel, servedWindow)

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
		guarded:      *guarded && !*noGuard,
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
		skillPack:    discoverPiSkillPack(""),
		passthrough:  fs.Args(),
	}

	if launch.writeConfig {
		resolvedPath, modified, err := projectassets.EnsurePiProviderConfigForWindow(launch.configPath, launch.baseURL, launch.model, servedWindow)
		if err != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: warning: could not update Pi config %s: %v\n", resolvedPath, err)
		} else if modified && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: updated %s with provider \"fak\" (baseURL: %s, model: %s, contextWindow: %d of %d served)\n", resolvedPath, launch.baseURL, launch.model, budget.ResidentTarget, budget.ServedWindow)
		}
	}

	if *safeSettings {
		sPath, modified, err := projectassets.EnsurePiSafeCompaction(*settingsPath, budget)
		if err != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: warning: could not update Pi settings %s: %v\n", sPath, err)
		} else if modified && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: wrote safe compaction to %s (reserveTokens: %d, keepRecentTokens: %d)\n", sPath, budget.ReserveTokens, budget.KeepRecentTokens)
		}
	}

	// Pin Pi's harness DEFAULT onto the fak router. A plain `pi` launch (no --provider/
	// --model flags) resolves defaultProvider/defaultModel from settings.json, so without
	// this the `fak` provider written above is configured but never used by default. Same
	// non-clobbering discipline as models.json and compaction; idempotent.
	//
	// Precedence: an explicit --model is operator intent and writes authoritatively. A
	// model auto-detected from the backend's /healthz is only a fallback seed — on a
	// routing-mode router /healthz names the local planner engine, not the routed model
	// set, so adopting it would silently clobber the operator's configured route (and
	// make the routing ladder unreachable). Seed when absent, never overwrite.
	if launch.writeConfig {
		var dPath string
		var dModified bool
		var dErr error
		if modelAutoDetected && !modelReplacedStaleDefault {
			dPath, dModified, dErr = projectassets.EnsurePiDefaultProviderModelIfAbsent(*settingsPath, launch.provider, launch.model)
		} else {
			dPath, dModified, dErr = projectassets.EnsurePiDefaultProviderModel(*settingsPath, launch.provider, launch.model)
		}
		if dErr != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: warning: could not pin Pi default provider/model in %s: %v\n", dPath, dErr)
		} else if dModified && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: pinned Pi default to provider %q model %q in %s\n", launch.provider, launch.model, dPath)
		}
	}

	argvOut := buildPiLaunchArgv(launch)
	if launch.guarded {
		argvOut = buildPiGuardLaunchArgv(tuiExecutable(), launch, argvOut)
	}

	// The generated provider advertises the placeholder apiKey "fak", which a
	// keyless gateway ignores but a PROTECTED one rejects (401 invalid gateway key),
	// so resolve the credential for the origin this launch actually selected — the
	// /healthz probe above may have adopted the answering loopback address family —
	// and hand it to the session-only extension below. A guarded launch is skipped:
	// the guard's own in-process gateway authenticates it.
	routerKey, rejectedKeySource := "", ""
	if !launch.guarded {
		routerKey, rejectedKeySource = piRouterCredential(launch.baseURL)
		if rejectedKeySource != "" && !launch.quiet {
			piReportRejectedKey(stderr, rejectedKeySource, launch.baseURL)
		}
	}

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak pi: dry-run - not launching")
		mode := "direct"
		if launch.guarded {
			mode = "guarded"
		}
		fmt.Fprintf(stderr, "  mode        = %s\n", mode)
		fmt.Fprintf(stderr, "  backend     = %s\n", launch.baseURL)
		fmt.Fprintf(stderr, "  provider    = %s\n", launch.provider)
		fmt.Fprintf(stderr, "  model       = %s (%s)\n", launch.model, piModelSource(*model, adoptedDetected))
		if launch.skillPack != "" {
			fmt.Fprintf(stderr, "  skills      = %s\n", launch.skillPack)
		} else {
			fmt.Fprintln(stderr, "  skills      = (no project skill pack discovered)")
		}
		fmt.Fprintf(stderr, "  context     = contextWindow %d tokens of %d served window, maxTokens %d (%s)\n", budget.ResidentTarget, budget.ServedWindow, budget.MaxOutputTokens, budget.Provenance)
		fmt.Fprintf(stderr, "  compaction  = reserve %d, keep %d, fires at %d (write=%t)\n", budget.ReserveTokens, budget.KeepRecentTokens, budget.Envelope.CompactTrigger, *safeSettings)
		if !budget.Viable {
			fmt.Fprintf(stderr, "  warning     = %s\n", budget.Reason)
		}
		fmt.Fprintln(stderr, "  provider-ext= -e <session-provider-extension> (installed and cleaned by the selected launch mode; no persistent config write)")
		if !launch.guarded {
			// Presence only: the resolved value never reaches argv, stdout, stderr, or
			// any persistent Pi config.
			if routerKey != "" {
				fmt.Fprintln(stderr, "  gateway-key = resolved for the selected origin (session-only; value never printed)")
			} else {
				fmt.Fprintln(stderr, "  gateway-key = none resolved (keyless session)")
			}
		}
		fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	if !launch.guarded {
		// Keep an ordinary launch free of persistent Pi config writes while still making the
		// fak provider reachable on a fresh install. Pi loads this provider definition for this
		// child only; explicit user extensions remain supported and follow it in argv order.
		// The resolved router key rides in that 0600 session-only file and nowhere else —
		// not argv, not the dry-run/print-env output, not ~/.pi/agent/models.json — and
		// cleanupProvider removes the file when the child exits.
		var cleanupProvider func()
		var err error
		argvOut, cleanupProvider, err = installPiLaunchProviderExtension(argvOut, launch.baseURL, launch.model, servedWindow, routerKey, newLaunchSessionID("pi"))
		if err != nil {
			fmt.Fprintf(stderr, "fak pi: install session provider: %v\n", err)
			return 1
		}
		defer cleanupProvider()
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
		mode := "direct"
		if launch.guarded {
			mode = "through fak guard"
		}
		fmt.Fprintf(stderr, "fak pi: launching Pi (%s) -> backend %s (model: %s)\n", mode, launch.baseURL, launch.model)
	}

	return piLaunchRun(stdout, stderr, argvOut, env)
}

// piAdvertisedModelWindow prefers an exact model id, then the same qualified/case
// aliases accepted for a deliberate default. Ambiguous aliases use the smallest
// positive bound; another model's window is never a fallback for a missing match.
func piAdvertisedModelWindow(model string, windows map[string]int) int {
	if window := windows[model]; window > 0 {
		return window
	}
	window := 0
	for id, candidate := range windows {
		if candidate > 0 && projectassets.IsPiModelAdvertised(model, []string{id}) && (window == 0 || candidate < window) {
			window = candidate
		}
	}
	return window
}

// configuredPiProviderBaseURL returns the persisted Fak provider endpoint when
// it is readable and valid. The launcher treats a missing or malformed config as
// absent so an explicit flag, FAK_SERVE_ADDR, or the fresh-install default remains usable.
func configuredPiProviderBaseURL(target string) string {
	data, err := os.ReadFile(projectassets.ResolvePiConfigPath(target))
	if err != nil {
		return ""
	}
	var cfg struct {
		Providers map[string]struct {
			BaseURL string `json:"baseUrl"`
		} `json:"providers"`
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(string(data), "\ufeff")), &cfg); err != nil {
		return ""
	}
	return strings.TrimSpace(cfg.Providers[projectassets.DefaultPiProviderID].BaseURL)
}

// piRouterCredential resolves the gateway credential for the origin a `fak pi` launch
// actually selected, through the native auto-router path (resolveRouterAPIKey) — the same
// resolution and origin scoping `fak chat` and `fak agent` auto-connect use. Every
// returned candidate must pass the existing bounded HMAC challenge at the selected
// endpoint before it reaches Pi, including the legacy workspace URL-match fallback.
// This proves endpoint key possession, not authenticated process identity: a proof
// can be relayed. Saved pairings also require the configured origin or its single
// loopback-family fallback to match.
//
// The second result is the SOURCE label of a candidate that failed that proof — never the
// value — so a rejected credential is legible in the default CLI output instead of surfacing
// later as an opaque 401 from the child. It is "" when a key resolved, and also "" when
// nothing was configured for this origin: a keyless launch stays keyless.
func piRouterCredential(origin string) (key, rejectedSource string) {
	key = resolveRouterAPIKey("", false, true, origin)
	if key != "" {
		// Legacy workspace configuration matches the URL but does not prove key
		// possession. Verify every returned candidate before handing it to Pi.
		if verifiedRouterCredential(origin, key) == "" {
			return "", "router configuration"
		}
		return key, ""
	}
	if strings.TrimSpace(os.Getenv(guardRouterKeyEnv)) != "" {
		return "", guardRouterKeyEnv
	}
	if piPairedOriginSelected(origin) {
		return "", guardRouterPairedKeySource
	}
	return "", ""
}

// piPairedOriginSelected reports whether node.json's selected origin is the launch origin or
// its single loopback-family fallback — the same origin match pairedNodeRouterAPIKey performs
// before it proves a saved key. It is a local comparison (no proof round-trip), so a launch
// that resolved no credential can still attribute the rejected candidate's source without a
// second network call. A paired node with no key (or a template placeholder) is keyless, not
// rejected, so it reports false.
func piPairedOriginSelected(origin string) bool {
	cfg, err := nodeReadCfg()
	if err != nil || isTemplateCredential(strings.TrimSpace(cfg.Key)) {
		return false
	}
	selected := strings.TrimSuffix(strings.TrimRight(strings.TrimSpace(origin), "/"), "/v1")
	configured := strings.TrimRight(strings.TrimSpace(cfg.URL), "/")
	if selected == "" || configured == "" {
		return false
	}
	if selected == configured {
		return true
	}
	fallback, ok := fakclient.LoopbackFallbackURL(configured)
	return ok && strings.TrimRight(fallback, "/") == selected
}

// piReportRejectedKey writes the bounded, secret-free diagnostic for a candidate credential
// that did not prove possession at the selected origin. It names the source and the origin
// only: no key material, in any form, is ever written.
func piReportRejectedKey(stderr io.Writer, source, origin string) {
	fmt.Fprintf(stderr, "fak pi: the %s gateway key did not prove possession at %s (unreachable origin, or a stale credential).\n", source, origin)
	fmt.Fprintln(stderr, "  Launching without it; a protected gateway will reject the session. Re-pair with `fak node use`, or export a matching FAK_GATEWAY_KEY, then retry.")
}

// installPiLaunchProviderExtension gives a raw `fak pi` child a launch-local `fak`
// provider without touching ~/.pi/agent/models.json. Unlike the guarded Pi installer,
// it does not reject user -e flags: raw Pi owns its extension policy.
func installPiLaunchProviderExtension(command []string, baseURL, model string, servedWindow int, apiKey, sessionID string) ([]string, func(), error) {
	dir, err := guardSessionTempDir("pi")
	if err != nil {
		return command, func() {}, err
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	source, err := piLaunchProviderExtensionSource(projectassets.NormalizePiBaseURL(baseURL), model, servedWindow, apiKey, sessionID)
	if err != nil {
		cleanup()
		return command, func() {}, err
	}
	extPath := filepath.Join(dir, guardPiExtensionFileName)
	if err := os.WriteFile(extPath, []byte(source), 0o600); err != nil {
		cleanup()
		return command, func() {}, err
	}
	return appendPiExtensionArg(command, extPath), cleanup, nil
}

func piLaunchProviderExtensionSource(baseURL, model string, servedWindow int, apiKey, sessionID string) (string, error) {
	raw, err := projectassets.GeneratePiConfigForWindow(baseURL, model, servedWindow)
	if err != nil {
		return "", err
	}
	var cfg struct {
		Providers map[string]any `json:"providers"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return "", fmt.Errorf("decode generated Pi provider: %w", err)
	}
	provider, ok := cfg.Providers[projectassets.DefaultPiProviderID]
	if !ok {
		return "", fmt.Errorf("generated Pi config omitted provider %q", projectassets.DefaultPiProviderID)
	}
	if key := strings.TrimSpace(apiKey); key != "" {
		providerMap, ok := provider.(map[string]any)
		if !ok {
			return "", fmt.Errorf("generated Pi provider %q is not an object", projectassets.DefaultPiProviderID)
		}
		providerMap["apiKey"] = key
	}
	if sessionID != "" {
		providerMap, ok := provider.(map[string]any)
		if !ok {
			return "", fmt.Errorf("generated Pi provider %q is not an object", projectassets.DefaultPiProviderID)
		}
		// Pi provider config forwards headers on every request: the launch-scoped session id.
		withProviderSessionHeader(providerMap, sessionID)
	}
	literal, err := guardPiTSLiteral(provider)
	if err != nil {
		return "", err
	}
	return "// fak pi: launch-scoped OpenAI-completions route.\n" +
		"export default function (pi) {\n" +
		"  pi.registerProvider(\"" + projectassets.DefaultPiProviderID + "\", " + literal + ");\n" +
		"}\n", nil
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
	model, ok, window, resolved, _, _ := probePiBackendWithCatalog(baseURL, timeout)
	return model, ok, window, resolved
}

// probePiBackendWithCatalog is probePiBackendWithWindow plus the backend's advertised model
// catalog from /v1/models, so the caller can tell a real, servable default from a stale
// placeholder id (see projectassets.PiDefaultModelIsDeliberate) and size the selected model.
// Model ids and positive per-model windows come from whichever origin answered the probe.
func probePiBackendWithCatalog(baseURL string, timeout time.Duration) (model string, ok bool, window int, resolvedBaseURL string, advertised []string, windows map[string]int) {
	client := &http.Client{Timeout: timeout}
	if m, ok, w, ids, windows := probePiBackendWindowFull(client, baseURL); ok {
		return m, true, w, baseURL, ids, windows
	}
	if fallback, ok := fakclient.LoopbackFallbackURL(baseURL); ok {
		if m, ok, w, ids, windows := probePiBackendWindowFull(client, fallback); ok {
			return m, true, w, fallback, ids, windows
		}
	}
	return "", false, 0, baseURL, nil, nil
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
	model, ok, window, _, _ = probePiBackendWindowFull(client, baseURL)
	return model, ok, window
}

// probePiBackendWindowFull is probePiBackendWindow plus every advertised model id in the
// /v1/models catalog and its positive per-model windows, so a caller can validate and
// size a configured default against what the backend actually serves.
func probePiBackendWindowFull(client *http.Client, baseURL string) (model string, ok bool, window int, advertised []string, windows map[string]int) {
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
			windows = make(map[string]int)
			for _, row := range catalog.Data {
				if row.ID != "" {
					advertised = append(advertised, row.ID)
					if row.ContextLength > 0 && (windows[row.ID] == 0 || row.ContextLength < windows[row.ID]) {
						windows[row.ID] = row.ContextLength
					}
				}
			}
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
			return model, true, window, advertised, windows
		}
		return model, healthy, 0, advertised, windows
	}
	if healthy {
		return model, true, 0, advertised, windows
	}
	return "", false, 0, nil, nil
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
	// Wire the project skill pack into Pi. Pi discovers .agents/skills only when the
	// working directory is a project that ships them, so a launch from elsewhere loses
	// the pack. --skill makes it explicit and location-independent; it is repeatable,
	// so emit one flag per discovered skill root.
	if opts.skillPack != "" {
		for _, root := range strings.Split(opts.skillPack, string(os.PathListSeparator)) {
			if root = strings.TrimSpace(root); root != "" {
				argv = append(argv, "--skill", root)
			}
		}
	}
	argv = append(argv, opts.passthrough...)
	return argv
}

func buildPiGuardLaunchArgv(fakBin string, opts piLaunchOptions, child []string) []string {
	argv := []string{
		fakBin,
		"guard",
		"--provider", "openai",
		"--base-url", opts.baseURL,
		"--model", opts.model,
	}
	if opts.probePrompt != "" {
		argv = append(argv, "--probe")
	}
	if opts.quiet {
		argv = append(argv, "--quiet")
	}
	argv = append(argv, "--")
	return append(argv, child...)
}

// defaultPiSkillPackRoots lists the project-asset skill roots Pi understands, in
// preference order: the generated .agents/skills adapters (Pi-native discovery dir)
// and the canonical .claude/skills pack. Both are relative to a project root.
var defaultPiSkillPackRoots = []string{
	filepath.Join(".agents", "skills"),
	filepath.Join(".claude", "skills"),
}

// discoverPiSkillPack walks up from startDir looking for the first directory that
// contains a project skill pack, and returns the discovered skill root directories
// joined by the OS path-list separator (empty when none is found). Walking up means
// `fak pi` launched from a subdirectory of a fak project still resolves the pack.
// The FAK_PI_SKILLS environment variable overrides discovery when set.
func discoverPiSkillPack(startDir string) string {
	if override := strings.TrimSpace(os.Getenv("FAK_PI_SKILLS")); override != "" {
		return override
	}
	dir := startDir
	if dir == "" {
		dir, _ = os.Getwd()
	}
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	for {
		var found []string
		for _, rel := range defaultPiSkillPackRoots {
			candidate := filepath.Join(dir, rel)
			if info, err := os.Stat(candidate); err == nil && info.IsDir() {
				found = append(found, candidate)
			}
		}
		if len(found) > 0 {
			return strings.Join(found, string(os.PathListSeparator))
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
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
	window := fs.Int("window", 0, "served model context window in tokens (default: the default prior). The written contextWindow is min(window, quality cap 163840); compaction reserve/keep come from the same envelope.")
	write := fs.Bool("write", false, "write or update ~/.pi/agent/models.json (or --path) and the safe Pi compaction settings")
	path := fs.String("path", "", "destination path or directory for models.json")
	settingsPath := fs.String("settings-path", "", "destination path or directory for Pi's settings.json (default: ~/.pi/agent/settings.json)")
	fromRouter := &piFromRouterFlag{}
	fs.Var(fromRouter, "from-router", "set provider \"fak\" models from the router's GET /v1/models (optional URL; default: the configured provider baseUrl, else --addr). Prints a plan; --write applies it with a timestamped backup.")
	if !parseFlags(fs, foldPiFromRouterArg(argv)) {
		return 2
	}
	if fromRouter.set {
		return runPiConfigFromRouter(stdout, stderr, fs, fromRouter.url, *addr, *path, *settingsPath, *write)
	}
	baseURL := projectassets.NormalizePiBaseURL(*addr)
	// Per-model budget: the status line and the compaction block must agree with the
	// contextWindow EnsurePiProviderConfigForWindow writes for THIS model. A flat
	// PiSafeContextBudget(*window) reports/writes the wrong envelope for a named model
	// (e.g. DeepSeek's 500k resident target would print as the Qwen 65536).
	budget := projectassets.PiModelContextBudget(*model, *window)
	if *write {
		resolvedPath, modified, err := projectassets.EnsurePiProviderConfigForWindow(*path, baseURL, *model, *window)
		if err != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", err)
			return 1
		}
		if modified {
			fmt.Fprintf(stdout, "fak pi config: updated %s with provider \"fak\" (baseURL: %s, model: %s, contextWindow: %d of %d served)\n", resolvedPath, baseURL, *model, budget.ResidentTarget, budget.ServedWindow)
		} else {
			fmt.Fprintf(stdout, "fak pi config: %s already has up-to-date provider \"fak\"\n", resolvedPath)
		}
		sPath, sModified, sErr := projectassets.EnsurePiSafeCompaction(*settingsPath, budget)
		if sErr != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", sErr)
			return 1
		}
		if sModified {
			fmt.Fprintf(stdout, "fak pi config: wrote safe compaction to %s (reserveTokens: %d, keepRecentTokens: %d)\n", sPath, budget.ReserveTokens, budget.KeepRecentTokens)
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
	fmt.Fprintf(stderr, "fak pi config: context envelope: contextWindow %d of %d served window, maxTokens %d (%s); compaction reserve %d, keep %d, fires at %d\n", budget.ResidentTarget, budget.ServedWindow, budget.MaxOutputTokens, budget.Provenance, budget.ReserveTokens, budget.KeepRecentTokens, budget.Envelope.CompactTrigger)
	if !budget.Viable {
		fmt.Fprintf(stderr, "fak pi config: warning: %s\n", budget.Reason)
	}
	return 0
}
