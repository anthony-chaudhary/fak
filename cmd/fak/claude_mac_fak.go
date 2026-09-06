package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// These defaults are PUBLIC-SAFE PLACEHOLDERS, not a real host. fak is a public
// repo, so the real tailnet gateway/SSH user must never be a tracked default
// (docs/fak/scrubbing-real-values.md). The placeholder intentionally does not
// resolve — supply your own via FAK_MAC_GATEWAY / FAK_MAC_SSH_HOST (or a
// gitignored fak-mac.local.ps1; see fak-mac.local.ps1.example). Do NOT "fix" a
// resolve error by restoring a real value here.
const (
	defaultClaudeMacGateway = "http://node-macos-a.local:8080"
	defaultClaudeMacModel   = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
	defaultClaudeMacSSHHost = "user@node-macos-a.local"
	defaultClaudeMacGrafana = "http://localhost:3000"
)

func cmdClaudeMacFak(argv []string) {
	os.Exit(runClaudeMacFak(os.Stdout, os.Stderr, argv))
}

// claudeMacUnconfiguredGateway reports the refusal text for the ONE failure a
// first-time reader of the README's Mac showcase actually hits: they run
// `fak mac` having configured nothing, so --gateway-url is still the public-safe
// placeholder above. Without this check the run dead-ends five seconds later in
// an ssh error naming `node-macos-a.local` — a host from someone else's lab that
// tells the reader neither that it is a placeholder nor that `fak mac` needs a
// gateway to exist first. Returns "" once the gateway has been supplied.
//
// The distinction this message carries is the one the verb's name hides:
// `fak mac` is a LAUNCHER, not an installer. It points Claude Code at a Mac
// gateway you are already running; standing that gateway up is a separate,
// documented step (#5457).
func claudeMacUnconfiguredGateway(gatewayURL string) string {
	if strings.TrimSpace(gatewayURL) != defaultClaudeMacGateway {
		return ""
	}
	return fmt.Sprintf(`fak claude-mac-fak: no Mac gateway configured.

--gateway-url is still the built-in placeholder %q. That is not a real host and
will not resolve — it stands in for YOUR Mac.

`+"`fak mac`"+` launches Claude Code against a Mac gateway; it does not create one.
The gateway is a local model server with `+"`fak serve`"+` in front of it, running on
the Mac. To get there:

  1. stand the gateway up on the Mac   docs/fak/server-quickstart.md
  2. point fak mac at it               export FAK_MAC_GATEWAY="http://<your-mac>:8080"
                                       (or pass --gateway-url)
  3. re-run                            fak mac

If the gateway needs a bearer token, fak fetches it over ssh from the same Mac;
set FAK_MAC_SSH_HOST to that host, or set FAK_GATEWAY_KEY directly to skip the
fetch. The launcher's flags, preflight panel, and live overlay are documented in
docs/fak/mac-agent-ui.md.
`, defaultClaudeMacGateway)
}

func runClaudeMacFak(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("claude-mac-fak", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gatewayURL := fs.String("gateway-url", envOrDefault("FAK_MAC_GATEWAY", defaultClaudeMacGateway), "fak serve gateway on the Mac")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", defaultClaudeMacModel), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	fetchKey := fs.Bool("fetch-key", true, "when gateway key env is empty, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	configDir := fs.String("claude-config-dir", defaultClaudeMacConfigDir(), "isolated CLAUDE_CONFIG_DIR for this Mac gateway session")
	prompt := fs.String("prompt", "Reply with exactly: OK", "probe prompt; used with --probe")
	probe := fs.Bool("probe", false, "run one headless JSON probe instead of interactive Claude Code")
	interactive := fs.Bool("interactive", false, "explicitly open interactive Claude Code; this is the default")
	dryRun := fs.Bool("dry-run", false, "show the launch plan without starting Claude Code")
	asJSON := fs.Bool("json", false, "emit the launch model as JSON and do not start Claude Code")
	apiTimeoutMS := fs.Int("api-timeout-ms", 1800000, "Claude Code API_TIMEOUT_MS")
	width := fs.Int("width", 120, "dry-run width")
	command := fs.String("command", "claude", "Claude Code command or path to execute")
	debug := fs.Bool("debug", true, "before handoff, probe the gateway (/healthz + /debug/vars) and print a fak debug panel for interactive launches; aborts interactive/probe launches if the gateway is unreachable")
	overlay := fs.Bool("overlay", false, "do not launch Claude; instead poll /debug/vars and print one live fak line per tick (run in a second pane next to the session). Ctrl-C to stop")
	overlayInterval := fs.Duration("overlay-interval", 2*time.Second, "refresh interval for --overlay")
	metrics := fs.Bool("metrics", false, "do not launch Claude; fetch /metrics + /debug/vars using the gateway's own bearer and print them (the panel links 401 from a bare browser click; this needs no token wrangling)")
	grafanaURL := fs.String("grafana-url", envOrDefault("FAK_MAC_GRAFANA", defaultClaudeMacGrafana), "Grafana base URL shown in the debug panel (the shipped tools/grafana stack)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *gatewayURL == "" {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --gateway-url must not be empty")
		return 2
	}
	if *model == "" {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --model must not be empty")
		return 2
	}
	if *apiTimeoutMS < 0 {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --api-timeout-ms must be non-negative")
		return 2
	}
	if *probe && *interactive {
		fmt.Fprintln(stderr, "fak claude-mac-fak: pass either --probe or --interactive, not both")
		return 2
	}
	if *overlayInterval <= 0 {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --overlay-interval must be positive")
		return 2
	}
	// Last of the up-front refusals, deliberately AFTER the flag-shape checks: a
	// malformed argument the caller actually typed is a better thing to report
	// than the configuration they have not reached yet.
	if msg := claudeMacUnconfiguredGateway(*gatewayURL); msg != "" {
		fmt.Fprint(stderr, msg)
		return 2
	}
	if err := os.MkdirAll(*configDir, 0o700); err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: create --claude-config-dir: %v\n", err)
		return 1
	}
	if err := ensureClaudeMacGatewayKey(*keyEnv, *fetchKey, *sshHost, *sshKey, *gatewayURL); err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: %v\n", err)
		return 2
	}

	debugBase, err := normalizeTUIAgentGatewayURL(*gatewayURL)
	if err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: %v\n", err)
		return 2
	}
	dbg := &claudeMacDebugClient{
		base:    debugBase,
		key:     strings.TrimSpace(os.Getenv(strings.TrimSpace(*keyEnv))),
		grafana: strings.TrimSpace(*grafanaURL),
		hc:      &http.Client{Timeout: 10 * time.Second},
	}

	// --overlay is a self-contained watch loop: it never launches Claude. Run it in
	// a second pane next to the interactive session to see fak's live cache/throughput.
	if *overlay {
		return runClaudeMacOverlay(stdout, stderr, dbg, *model, *overlayInterval)
	}

	// --metrics is a self-contained one-shot: it never launches Claude. It is the
	// "easier by default" answer to the panel's metrics/vars links 401ing on a bare
	// click — instead of making the operator carry the bearer to a browser/curl, fak
	// reuses the token it already loaded to fetch both surfaces and print them.
	if *metrics {
		return runClaudeMacMetrics(stdout, stderr, dbg)
	}

	// Preflight debug panel (#claude-mac): probe the gateway before handing the
	// terminal to Claude Code, which makes fak otherwise invisible. Interactive
	// launches render the panel; headless probes keep stdout quiet so Claude's JSON
	// remains parseable, but still abort on an unreachable gateway rather than
	// waiting on Claude Code's long API timeout against a dead backend.
	auth := "gateway-bearer"
	if dbg.key == "" {
		auth = "none"
	}
	if *debug && !*asJSON && !*dryRun {
		health, vars, err := dbg.probe()
		if err != nil {
			fmt.Fprintf(stderr, "fak debug: gateway unreachable: %v\n", err)
			return 1
		}
		if !*probe {
			fmt.Fprint(stdout, renderClaudeMacPreflight(health, vars, dbg.base, *model, auth, dbg.grafana))
		}
	}

	tuiArgs := []string{
		"agent",
		"--command", *command,
		"--claude-config-dir", *configDir,
		"--gateway-url", *gatewayURL,
		"--gateway-key-env", *keyEnv,
		"--model", *model,
		"--api-timeout-ms", strconv.Itoa(*apiTimeoutMS),
		"--width", strconv.Itoa(*width),
	}
	if *dryRun {
		tuiArgs = append(tuiArgs, "--dry-run")
	}
	if *asJSON {
		tuiArgs = append(tuiArgs, "--json")
	}
	passthrough := append([]string(nil), fs.Args()...)
	if *probe {
		// This macOS-only convenience path has a fixed default prompt and accepts an
		// operator override. Its child is launched by runTUI, whose Windows launch
		// seam moves any large -p value to stdin before exec.
		// Claude Code prompt-mode examples put `-p PROMPT` before output flags.
		// The generic TUI appends --prompt after passthrough args, so probe mode
		// injects the prompt directly into Claude's argv to keep the live command
		// deterministic: claude ... -p PROMPT --output-format json.
		if !claudeMacArgsHavePrompt(passthrough) {
			passthrough = append([]string{"-p", *prompt}, passthrough...)
		}
		if !*asJSON && !claudeMacArgsHaveFlag(passthrough, "--output-format") {
			passthrough = append(passthrough, "--output-format", "json")
		}
		passthrough = claudeMacProbeIsolationArgs(passthrough)
	}
	if len(passthrough) > 0 {
		tuiArgs = append(tuiArgs, "--")
		tuiArgs = append(tuiArgs, passthrough...)
	}
	return runTUI(stdout, stderr, tuiArgs)
}

func claudeMacArgsHavePrompt(args []string) bool {
	for _, a := range args {
		if a == "-p" || a == "--prompt" || strings.HasPrefix(a, "--prompt=") {
			return true
		}
	}
	return false
}

func claudeMacProbeIsolationArgs(args []string) []string {
	out := append([]string(nil), args...)
	if !claudeMacArgsHaveFlag(out, "--safe-mode") {
		out = append(out, "--safe-mode")
	}
	if !claudeMacArgsHaveFlag(out, "--tools") {
		out = append(out, "--tools", "")
	}
	if !claudeMacArgsHaveFlag(out, "--disable-slash-commands") {
		out = append(out, "--disable-slash-commands")
	}
	if !claudeMacArgsHaveFlag(out, "--no-session-persistence") {
		out = append(out, "--no-session-persistence")
	}
	return out
}

func claudeMacArgsHaveFlag(args []string, name string) bool {
	for _, a := range args {
		if a == name || strings.HasPrefix(a, name+"=") {
			return true
		}
	}
	return false
}

func envOrDefault(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

func defaultClaudeMacConfigDir() string {
	if v := strings.TrimSpace(os.Getenv("FAK_CLAUDE_CONFIG_DIR")); v != "" {
		return v
	}
	return filepath.Join(os.TempDir(), "fak-claude-mac")
}

func defaultClaudeMacSSHKey() string {
	return envOrHomePath("FAK_MAC_SSH_KEY", ".ssh", "id_ed25519_prod_to_laptop")
}

// execCommand is the indirection seam for the ssh key-fetch so a test can
// substitute a helper process (the os/exec stdlib pattern). Production code
// leaves it as exec.Command.
var execCommand = exec.Command

// gatewayIsLocal reports whether the gateway URL points at this machine
// (loopback host). A local fak serve has no Mac to ssh into and, unless it was
// started with --require-key-env, needs no bearer — so the ssh key-fetch is
// both impossible and unnecessary. Used to skip the fetch for the easy local
// default instead of dead-ending on a doomed ssh call.
func gatewayIsLocal(gatewayURL string) bool {
	u, err := url.Parse(strings.TrimSpace(gatewayURL))
	if err != nil {
		return false
	}
	host := u.Hostname()
	if strings.EqualFold(host, "localhost") {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}

func ensureClaudeMacGatewayKey(envName string, fetch bool, host, keyPath, gatewayURL string) error {
	envName = strings.TrimSpace(envName)
	if envName == "" {
		envName = "FAK_GATEWAY_KEY"
	}
	if strings.TrimSpace(os.Getenv(envName)) != "" {
		return nil
	}
	// Easy local default: a loopback gateway is a local fak serve with no Mac to
	// reach over ssh. Skip the fetch and tolerate an empty bearer — a local serve
	// without --require-key-env accepts unauthenticated requests.
	if gatewayIsLocal(gatewayURL) {
		return nil
	}
	if !fetch {
		return fmt.Errorf("%s is empty; set it or leave --fetch-key enabled", envName)
	}
	host = strings.TrimSpace(host)
	if host == "" {
		return fmt.Errorf("%s is empty and --ssh-host is empty", envName)
	}
	args := []string{"-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ConnectionAttempts=1"}
	if strings.TrimSpace(keyPath) != "" {
		args = append(args, "-i", keyPath)
	}
	args = append(args, host, "cat ~/.fak-gateway-key")
	cmd := execCommand("ssh", args...)
	var errBuf strings.Builder
	cmd.Stderr = &errBuf
	out, err := cmd.Output()
	if err != nil {
		// ssh's exit status (255 = connection-level failure) is opaque on its
		// own; the actionable cause ("Could not resolve hostname",
		// "Permission denied", "Connection refused") goes to stderr, which
		// cmd.Output() otherwise drops. Surface it, and point at the override
		// that skips the fetch entirely.
		detail := strings.TrimSpace(errBuf.String())
		if detail == "" {
			detail = err.Error()
		}
		// A reader who set FAK_MAC_GATEWAY but not FAK_MAC_SSH_HOST is now
		// failing against the OTHER placeholder, and the ssh error names a host
		// they never chose. Say so, rather than leaving them to guess whether
		// `user@node-macos-a.local` is something they were supposed to have (#5457).
		if host == defaultClaudeMacSSHHost {
			return fmt.Errorf("fetch gateway key from %s over ssh: %s\n"+
				"  %s is the built-in PLACEHOLDER ssh host, not your Mac — the gateway URL is set but the ssh host is not.\n"+
				"  set FAK_MAC_SSH_HOST=<user>@<your-mac> so fak can read ~/.fak-gateway-key from it,\n"+
				"  or set %s directly (or pass --fetch-key=false) if the gateway needs no bearer",
				host, detail, defaultClaudeMacSSHHost, envName)
		}
		return fmt.Errorf("fetch gateway key from %s over ssh: %s\n"+
			"  set %s directly (or --gateway-key-env / FAK_MAC_SSH_HOST) to skip the ssh fetch",
			host, detail, envName)
	}
	secret := strings.TrimSpace(string(out))
	if secret == "" {
		return fmt.Errorf("fetch gateway key over ssh: empty key")
	}
	return os.Setenv(envName, secret)
}

// claudeMacHealth is the subset of the gateway's /healthz JSON we render. planner
// names the live /v1/chat/completions backend ("inkernel" | "proxy" | "mock"); a
// "mock" planner means the responses are scripted, not model output (see
// internal/gateway/http.go handleHealth), which the panel surfaces loudly.
type claudeMacHealth struct {
	OK      bool   `json:"ok"`
	Engine  string `json:"engine"`
	Model   string `json:"model"`
	Planner string `json:"planner"`
}

// claudeMacDebugVars is the subset of the gateway's /debug/vars JSON we render. The
// field/JSON-tag names mirror internal/gateway/debug.go debugVarsResponse; JSON
// decode tolerates the many extra fields we do not surface.
type claudeMacDebugVars struct {
	Gateway struct {
		Up               bool    `json:"up"`
		Version          string  `json:"version"`
		Engine           string  `json:"engine"`
		Model            string  `json:"model"`
		VDSO             bool    `json:"vdso"`
		UptimeSeconds    float64 `json:"uptime_seconds"`
		InflightRequests int64   `json:"inflight_requests"`
	} `json:"gateway"`
	Kernel struct {
		Submits      int64   `json:"submits"`
		VDSOHits     int64   `json:"vdso_hits"`
		EngineCalls  int64   `json:"engine_calls"`
		Denies       int64   `json:"denies"`
		Admitted     int64   `json:"admitted"`
		VDSOHitRatio float64 `json:"vdso_hit_ratio"`
	} `json:"kernel"`
	// Inference mirrors internal/gateway debugInferenceVars — the model-generation
	// throughput that makes a busy proxy/chat gateway look busy. The kernel counters
	// above stay 0 on a pure chat/proxy workload (no syscall, no fast-path lookup), so
	// these fields are what an operator actually wants to see: prefill (cold first
	// request) vs decode (steady-state) tok/s, and the oldest in-flight request's age
	// as a slow/wedged-request detector.
	Inference struct {
		Turns                  int64   `json:"turns"`
		PromptTokens           int64   `json:"prompt_tokens"`
		CompletionTokens       int64   `json:"completion_tokens"`
		OutputTokensPerSecond  float64 `json:"output_tokens_per_second"`
		TTFTTurns              int64   `json:"ttft_turns"`
		MeanTTFTSeconds        float64 `json:"mean_ttft_seconds"`
		PrefillTokensPerSecond float64 `json:"prefill_tokens_per_second"`
		DecodeTokensPerSecond  float64 `json:"decode_tokens_per_second"`
		InflightMaxAgeSeconds  float64 `json:"inflight_max_age_seconds"`
	} `json:"inference"`
	Upstream struct {
		ProviderExtraBodySet  bool     `json:"provider_extra_body_set"`
		ProviderExtraBodyKeys []string `json:"provider_extra_body_keys"`
	} `json:"upstream"`
	Runtime struct {
		NumGoroutine int `json:"num_goroutine"`
		Memory       struct {
			HeapAllocBytes uint64 `json:"heap_alloc_bytes"`
		} `json:"memory"`
	} `json:"runtime"`
}

// claudeMacDebugClient reads the Mac gateway's liveness + debug-vars surfaces. It
// reuses the same one-shot bearer-GET shape as sessionClient (session_cmd.go) but
// is kept local so the debug panel/overlay carry no extra coupling.
type claudeMacDebugClient struct {
	base            string
	key             string
	grafana         string
	hc              *http.Client
	workHistoryPath string
	workloadKey     string
	runKey          string
	// infoObservationMode caches whether this gateway serves the versioned
	// /v1/fak/observation endpoint. Zero probes once, 1 keeps using it, and 2
	// skips it for an older gateway after a definitive compatibility miss.
	infoObservationMode uint8
}

// do issues an authenticated GET to c.base+path with the shared client-resolution and
// bearer-auth, returning the live response. The caller owns resp.Body (it must defer
// Close); get and getRaw share this request-building half but diverge on how they read
// the body.
func (c *claudeMacDebugClient) do(path string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, c.base+path, nil)
	if err != nil {
		return nil, err
	}
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	hc := c.hc
	if hc == nil {
		hc = &http.Client{Timeout: 10 * time.Second}
	}
	return hc.Do(req)
}

func (c *claudeMacDebugClient) get(path string, out any) error {
	resp, err := c.do(path)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("GET %s: status %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// getRaw fetches path with the gateway bearer and returns the response body
// verbatim. /metrics is Prometheus text exposition (not JSON), so the JSON-decoding
// get cannot read it; getRaw carries the same auth + 2xx-gate so --metrics speaks
// both the text and JSON surfaces through one client.
func (c *claudeMacDebugClient) getRaw(path string) (string, error) {
	resp, err := c.do(path)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode/100 != 2 {
		return "", fmt.Errorf("GET %s: status %d", path, resp.StatusCode)
	}
	return string(body), nil
}

func (c *claudeMacDebugClient) health() (claudeMacHealth, error) {
	var h claudeMacHealth
	err := c.get("/healthz", &h)
	return h, err
}

func (c *claudeMacDebugClient) vars() (claudeMacDebugVars, error) {
	var v claudeMacDebugVars
	err := c.get("/debug/vars", &v)
	return v, err
}

// probe reads /healthz then /debug/vars. /healthz is the liveness gate; if it
// fails the gateway is unreachable and probe reports that error. A /debug/vars
// failure is non-fatal — health alone still proves the gateway is live — so the
// returned vars is simply zero-valued and the panel renders without counters.
func (c *claudeMacDebugClient) probe() (claudeMacHealth, claudeMacDebugVars, error) {
	h, err := c.health()
	if err != nil {
		return claudeMacHealth{}, claudeMacDebugVars{}, err
	}
	v, _ := c.vars()
	return h, v, nil
}

// renderClaudeMacPreflight renders the debug panel printed before fak hands the
// terminal to Claude Code. It proves the gateway is the live in-kernel fak serve
// (or warns loudly when planner=mock, i.e. scripted responses) and surfaces the
// vDSO cache-hit ratio, inflight requests, and uptime the launch path otherwise
// discards. The bearer is never printed.
func renderClaudeMacPreflight(h claudeMacHealth, v claudeMacDebugVars, gatewayURL, model, auth, grafanaURL string) string {
	var b strings.Builder
	// Signature banner: this panel is the seconds-long "boot screen" a user stares at (and
	// screenshots) before Claude Code takes the terminal, so it opens with one recognizable
	// line naming what fak is doing — pointing Claude Code at the user's own Mac-local model.
	// It is a constant brand line, not a claim; the live gateway/health facts follow
	// immediately below. The `== … ==` framing matches the --metrics output (runClaudeMacMetrics).
	fmt.Fprintln(&b, "== fak · Claude Code -> your own Mac's local model ==")
	ver := appversion.Current()
	if id := guardShortBuildID(); id != "" {
		ver += " (" + id + ")"
	}
	fmt.Fprintf(&b, "fak debug %s · gateway %s\n", ver, gatewayURL)
	healthWord := "ok"
	if !h.OK {
		healthWord = "DOWN"
	}
	engine := firstNonEmpty(h.Engine, v.Gateway.Engine)
	planner := strings.TrimSpace(h.Planner)
	// engine vs planner read as contradictory ("engine=inkernel planner=proxy") unless
	// you know engine is the BUILD (what this binary CAN do) and planner is the LIVE
	// backend actually answering /v1/* this run. Spell that out so the operator reads
	// "proxy" as "this gateway is forwarding to an upstream model", which is the whole
	// reason the first request is slow (cold upstream connect + model load), not a fault.
	fmt.Fprintf(&b, "health: %s  engine(build)=%s  planner(live)=%s\n", healthWord, blankDash(engine), blankDash(planner))
	vdso := "off"
	if v.Gateway.VDSO {
		vdso = "on"
	}
	// cache-hit is the vDSO/kernel fast-path ratio. On a proxy planner the kernel
	// fast path is not exercised (every turn forwards upstream), so 0.00 here is
	// EXPECTED, not a miss — annotate it rather than let it read as a broken cache.
	cacheNote := ""
	if isProxyPlanner(planner) && v.Kernel.VDSOHitRatio == 0 {
		cacheNote = " (proxy: kernel fast-path not exercised — 0 is expected)"
	}
	fmt.Fprintf(&b, "vdso=%s  cache-hit %.2f%s  up %s\n", vdso, v.Kernel.VDSOHitRatio, cacheNote, humanUptime(v.Gateway.UptimeSeconds))
	// Throughput: the model-generation rates the kernel counters cannot show on a
	// proxy/chat workload. Prefill is the cold-ingest rate that dominates a slow FIRST
	// request; decode is steady-state generation. Both read "-" until the first
	// streamed turn measures a TTFT boundary, so an idle gateway never prints a phantom.
	inf := v.Inference
	fmt.Fprintf(&b, "throughput: prefill %s  decode %s  ttft %s  (over %d turn(s))\n",
		tokRate(inf.PrefillTokensPerSecond), tokRate(inf.DecodeTokensPerSecond),
		ttftLabel(inf.MeanTTFTSeconds, inf.TTFTTurns), inf.Turns)
	// In-flight + oldest-request age: the slow/wedged-request detector. A request that
	// is still running is in NO completion histogram yet, so without this a hung first
	// request is invisible. Flag an old one loudly.
	ageWord := inflightAgeLabel(v.Inference.InflightMaxAgeSeconds, v.Gateway.InflightRequests)
	fmt.Fprintf(&b, "inflight %d%s\n", v.Gateway.InflightRequests, ageWord)
	effectiveModel := firstNonEmpty(model, h.Model, v.Gateway.Model)
	fmt.Fprintf(&b, "model %s  auth %s\n", blankDash(effectiveModel), blankDash(auth))
	if v.Upstream.ProviderExtraBodySet {
		fmt.Fprintf(&b, "request tuning: provider extra body set%s\n", providerExtraBodyKeyLabel(v.Upstream.ProviderExtraBodyKeys))
	} else if guardLocalLooksQwen36(effectiveModel) {
		fmt.Fprintf(&b, "WARN: Qwen3.6 request tuning is not visible on this gateway — set FAK_PROVIDER_EXTRA_BODY_JSON=%s before starting fak serve\n", guardQwen36ProviderExtraBodyJSON)
	}
	// metrics/vars are read-only observability endpoints. A bare browser click on
	// these URLs 401s off-box (they are loopback-exempt only — see authExempt), so
	// the panel leads with the zero-friction path: `--metrics` reuses the bearer fak
	// already loaded to fetch both surfaces, no token wrangling. The raw URLs stay
	// printed for scrapers/tunnels, annotated so a 401 reads as expected, not broken.
	fmt.Fprintln(&b, "metrics: run  fak claude-mac-fak --metrics   (fetches /metrics + /debug/vars with the gateway's own bearer)")
	fmt.Fprintf(&b, "  urls: %s/metrics · %s/debug/vars", gatewayURL, gatewayURL)
	if auth == "gateway-bearer" {
		fmt.Fprint(&b, "  (open on the gateway host; off-box needs the bearer)")
	}
	fmt.Fprintln(&b)
	// Grafana is the OPTIONAL local dashboard stack (tools/grafana). The shipped
	// default is http://localhost:3000, which is dead unless that stack is running
	// on THIS box — printing it bare invites a connection-refused click. Show it
	// only when the operator pointed it somewhere real, and label it as the local
	// stack so the localhost host reads as intentional, not a misconfigured link.
	if grafanaURL != "" && grafanaURL != defaultClaudeMacGrafana {
		fmt.Fprintf(&b, "grafana (local stack): %s\n", grafanaURL)
	}
	// Legend: the panel is dense with infra acronyms a fresh operator can't decode on
	// sight. Expand the ones that actually appear above so the panel is self-documenting
	// rather than requiring the docs. Kept to two lines and printed every launch (it is
	// cheap and the panel is read at most once per session).
	b.WriteString(claudeMacPanelLegend())
	if strings.EqualFold(planner, "mock") {
		fmt.Fprintln(&b, "WARN: planner=mock — responses are scripted, not model output")
	}
	fmt.Fprintln(&b, "-> launching claude ...")
	return b.String()
}

// claudeMacPanelLegend expands every acronym/term the preflight panel and the --overlay
// line use, so an operator never has to leave the terminal to decode them. The terms are
// listed in the order they first appear on the surfaces above. Shared by the panel and
// the overlay header so the two can never drift apart.
func claudeMacPanelLegend() string {
	var b strings.Builder
	fmt.Fprintln(&b, "legend:")
	fmt.Fprintln(&b, "  engine(build) = what this fak binary CAN serve · planner(live) = the backend actually answering this run")
	fmt.Fprintln(&b, "    (inkernel = fak runs the model itself · proxy = forwards to an upstream model · mock = scripted, not a model)")
	fmt.Fprintln(&b, "  vDSO = fak's in-process fast-path cache (a served result that skips the model) · cache-hit = its hit ratio")
	fmt.Fprintln(&b, "  prefill = prompt-ingest speed (tok/s; the cold cost of a slow FIRST request) · decode = steady-state generation (tok/s)")
	fmt.Fprintln(&b, "  TTFT = time-to-first-token (the prefill→decode boundary) · tok/s = tokens per second · inflight = requests running now")
	fmt.Fprintln(&b, "  up = gateway uptime · auth = how this client authenticates to the gateway · '-' = not measured yet (no served turn)")
	return b.String()
}

func providerExtraBodyKeyLabel(keys []string) string {
	if len(keys) == 0 {
		return ""
	}
	return fmt.Sprintf(" (keys: %s)", strings.Join(keys, ", "))
}

func isProxyPlanner(planner string) bool {
	return strings.EqualFold(strings.TrimSpace(planner), "proxy")
}

// tokRate renders a tokens/sec value, or "-" when it is 0 (not yet measured) so an idle
// or buffered-only gateway never prints a misleading "0 tok/s" that looks like a stall.
func tokRate(v float64) string {
	if v <= 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f tok/s", v)
}

// ttftLabel renders the mean time-to-first-token over the turns that measured it, or "-"
// when none have (a buffered-only or idle gateway).
func ttftLabel(meanSec float64, turns int64) string {
	if turns <= 0 || meanSec <= 0 {
		return "-"
	}
	if meanSec >= 1 {
		return fmt.Sprintf("%.1fs", meanSec)
	}
	return fmt.Sprintf("%dms", int(meanSec*1000))
}

// inflightAgeLabel annotates the in-flight count with the oldest request's age, loudly
// when that age is high — the signal that the FIRST request is slow / wedged rather than
// simply idle. Empty when nothing is in flight.
func inflightAgeLabel(maxAgeSec float64, inflight int64) string {
	if inflight <= 0 || maxAgeSec <= 0 {
		return ""
	}
	word := fmt.Sprintf("  (oldest %s in flight", humanUptime(maxAgeSec))
	if maxAgeSec >= 30 {
		word += " — SLOW: cold upstream load or a wedged request"
	}
	return word + ")"
}

// renderClaudeMacOverlayLine renders one compact live line for the --overlay watch
// loop: kernel throughput (submits / vDSO hits / engine calls), inflight requests,
// heap, and goroutine count — the fak-specific signal an operator wants visible
// next to a running session.
func renderClaudeMacOverlayLine(v claudeMacDebugVars) string {
	pct := 0.0
	if v.Kernel.Submits > 0 {
		pct = float64(v.Kernel.VDSOHits) / float64(v.Kernel.Submits) * 100
	}
	// Lead with the throughput an operator on a proxy/chat workload actually wants:
	// the kernel counters (submits/hits/engine) stay 0 there, so a line built only
	// from them reads "dead" while the box is decoding tokens. prefill/decode tok/s
	// and the oldest in-flight age make the real work — and a slow first request —
	// visible; "-" until the first streamed turn measures a TTFT boundary.
	inf := v.Inference
	age := ""
	if v.Gateway.InflightRequests > 0 && inf.InflightMaxAgeSeconds > 0 {
		age = fmt.Sprintf(" (oldest %s)", humanUptime(inf.InflightMaxAgeSeconds))
	}
	return fmt.Sprintf("prefill %s  decode %s  turns %d  inflight %d%s  submits %d  hits %d (%.1f%%)  engine %d  heap %s  gor %d",
		tokRate(inf.PrefillTokensPerSecond), tokRate(inf.DecodeTokensPerSecond), inf.Turns,
		v.Gateway.InflightRequests, age,
		v.Kernel.Submits, v.Kernel.VDSOHits, pct, v.Kernel.EngineCalls,
		humanBytes(int64(v.Runtime.Memory.HeapAllocBytes)), v.Runtime.NumGoroutine)
}

// runClaudeMacOverlay polls /debug/vars and prints one live line per tick until
// Ctrl-C, the second-pane companion to the interactive session. It mirrors the
// poll-loop shape of followGuardJournal (signal.NotifyContext + ticker) and never
// launches Claude. A transient fetch error prints a one-line note and keeps polling.
func runClaudeMacOverlay(stdout, stderr io.Writer, c *claudeMacDebugClient, model string, interval time.Duration) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	fmt.Fprintf(stdout, "fak debug overlay · %s  model=%s  (every %s, Ctrl-C to stop)\n", c.base, blankDash(model), interval)
	// Print the legend ONCE in the header (the per-tick line must stay compact). Covers
	// the shared throughput terms plus the overlay-only kernel/runtime fields.
	fmt.Fprint(stdout, claudeMacOverlayLegend())
	emit := func() {
		v, err := c.vars()
		if err != nil {
			fmt.Fprintf(stderr, "fak overlay: %v\n", err)
			return
		}
		fmt.Fprintf(stdout, "  %s\n", renderClaudeMacOverlayLine(v))
	}
	emit()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return 0
		case <-ticker.C:
			emit()
		}
	}
}

// runClaudeMacMetrics is the one-shot answer to "the panel's /metrics and
// /debug/vars links 401 on a bare browser click." Rather than make the operator
// carry the bearer to a browser or hand-write a curl, it reuses the token the
// client already holds, fetches both surfaces, and prints them: /debug/vars as
// indented JSON (the structured diagnostics) and /metrics verbatim (the Prometheus
// exposition). It never launches Claude. The bearer is sent, never printed.
func runClaudeMacMetrics(stdout, stderr io.Writer, c *claudeMacDebugClient) int {
	authNote := "no auth configured"
	if c.key != "" {
		authNote = "using the gateway bearer"
	}
	fmt.Fprintf(stdout, "fak gateway internals · %s  (%s)\n", c.base, authNote)

	// /debug/vars first: it is the JSON diagnostics block the panel summarizes. Print
	// it indented so it is readable as a standalone snapshot. A failure here is worth
	// surfacing but not fatal — /metrics may still answer.
	var vars json.RawMessage
	if err := c.get("/debug/vars", &vars); err != nil {
		fmt.Fprintf(stderr, "fak metrics: /debug/vars: %v%s\n", err, metricsAuthHint(err, c))
	} else {
		var pretty bytes.Buffer
		if err := json.Indent(&pretty, vars, "", "  "); err != nil {
			pretty.Reset()
			pretty.Write(vars)
		}
		fmt.Fprintf(stdout, "\n== /debug/vars ==\n%s\n", pretty.String())
	}

	// /metrics is Prometheus text exposition — print it verbatim (no parsing) so it
	// is pipe-friendly into promtool/grep, exactly what a scrape would see.
	text, err := c.getRaw("/metrics")
	if err != nil {
		fmt.Fprintf(stderr, "fak metrics: /metrics: %v%s\n", err, metricsAuthHint(err, c))
		return 1
	}
	fmt.Fprintf(stdout, "\n== /metrics ==\n%s", text)
	if !strings.HasSuffix(text, "\n") {
		fmt.Fprintln(stdout)
	}
	return 0
}

// metricsAuthHint turns a 401 into one actionable line so a missing/expired bearer
// reads as "set the key", not an opaque status. Empty for any other error (or when a
// key is already configured, where a 401 means the key is wrong rather than absent).
func metricsAuthHint(err error, c *claudeMacDebugClient) string {
	if err == nil || !strings.Contains(err.Error(), "status 401") {
		return ""
	}
	if c.key == "" {
		return " — set the gateway bearer in FAK_GATEWAY_KEY (or pass --gateway-key-env), or run on the gateway host where these are loopback-exempt"
	}
	return " — the configured bearer was rejected; check FAK_GATEWAY_KEY matches the gateway's --require-key"
}

// claudeMacOverlayLegend expands the terms on the --overlay line. It reuses the shared
// panel legend (prefill/decode/TTFT/tok/s/inflight) and adds the overlay-only kernel and
// runtime fields, so a watcher in a second pane can read the line without the docs.
func claudeMacOverlayLegend() string {
	var b strings.Builder
	b.WriteString(claudeMacPanelLegend())
	fmt.Fprintln(&b, "  turns = model turns served · submits = kernel adjudications · hits = vDSO fast-path hits (% of submits) · engine = submits that reached the model")
	fmt.Fprintln(&b, "  heap = Go heap in use · gor = live goroutines · (submits/hits/engine stay 0 on a proxy/chat workload — that is expected, not a stall)")
	return b.String()
}

// firstNonEmpty is strmatch.FirstTrimmed under the name ~50 call sites in this binary
// already read with. The "first value with non-whitespace text, returned TRIMMED" rule has
// ONE definition (internal/strmatch); this used to re-derive it byte for byte.
func firstNonEmpty(vals ...string) string { return strmatch.FirstTrimmed(vals...) }

func blankDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

// humanUptime renders gateway uptime seconds compactly (e.g. "3h12m", "45s").
func humanUptime(sec float64) string {
	if sec <= 0 {
		return "0s"
	}
	d := time.Duration(sec * float64(time.Second)).Round(time.Second)
	switch {
	case d >= time.Hour:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	case d >= time.Minute:
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	default:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
}
